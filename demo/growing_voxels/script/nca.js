import { createBuffer, createEmptyBuffer } from './gpu.js';

// Float32 ↔ Float16 helpers
const f32Buf = new Float32Array(1);
const u32Buf = new Uint32Array(f32Buf.buffer);

function float32ToFloat16(val) {
  f32Buf[0] = val;
  const f = u32Buf[0];
  const sign = (f >>> 16) & 0x8000;
  const exp = ((f >>> 23) & 0xff) - 127 + 15;
  const frac = (f >>> 13) & 0x3ff;
  if (exp <= 0) return sign; // flush to zero
  if (exp >= 31) return sign | 0x7c00; // infinity
  return sign | (exp << 10) | frac;
}

export class NCACompute {
  constructor(device, model, shaderTemplate, hasF16) {
    this.device = device;
    this.model = model;
    this.shaderTemplate = shaderTemplate;
    this.hasF16 = hasF16;
    this.step = 0;
    this._buildPipeline();
  }

  _buildPipeline() {
    const { device, model, hasF16 } = this;
    const { channels, coarseSize, fcDim, numKernels, seedRadius } = model;
    const S = coarseSize;
    const C = channels;
    const K = numKernels; // 5
    const FC = fcDim;     // 256

    // Bytes per element: 2 for f16, 4 for f32
    this.bpe = hasF16 ? 2 : 4;

    // State buffers (ping-pong), layout: flat array [C * S * S * S]
    const stateSize = C * S * S * S * this.bpe;
    this.stateA = createEmptyBuffer(device, stateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.stateB = createEmptyBuffer(device, stateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.current = 0;

    // NOTE: percBuf was previously allocated here (C * K * S^3 * 4 bytes = 356 MB for Pine Tree!)
    // but was never bound to any shader. The perception is computed inline in the NCA shader
    // using local arrays. Removed to save massive GPU memory.

    // Upload weight buffers (these stay f32 — they're small)
    const percW = model.weights.perception['perceive.weight'];
    this.percWeights = createBuffer(device, percW.data, GPUBufferUsage.STORAGE);

    const w1W = model.weights.adaptation['adapt.0.weight'];
    const w1B = model.weights.adaptation['adapt.0.bias'];
    const w2W = model.weights.adaptation['adapt.2.weight'];
    this.adaptW1 = createBuffer(device, w1W.data, GPUBufferUsage.STORAGE);
    this.adaptB1 = createBuffer(device, w1B.data, GPUBufferUsage.STORAGE);
    this.adaptW2 = createBuffer(device, w2W.data, GPUBufferUsage.STORAGE);

    // Uniforms: [S, C, K, FC, seed, step, updateProb padding]
    this.uniformBuf = createEmptyBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // Random seed buffer for stochastic updates
    this.randomBuf = createEmptyBuffer(device, S * S * S * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    // Build shader with f16 support
    const stateType = hasF16 ? 'f16' : 'f32';
    const f16Enable = hasF16 ? 'enable f16;' : '';
    let shaderCode = this.shaderTemplate
      .replace(/{{S}}/g, this.model.coarseSize)
      .replace(/{{C}}/g, this.model.channels)
      .replace(/{{K}}/g, this.model.numKernels)
      .replace(/{{FC}}/g, this.model.fcDim)
      .replace(/{{LC}}/g, this.model.livingChannel)
      .replace(/{{C_K}}/g, this.model.channels * this.model.numKernels)
      .replace(/{{LIVING_THRESHOLD}}/g, `${model.livingThreshold}`)
      .replace(/{{STATE_TYPE}}/g, stateType)
      .replace(/{{F16_ENABLE}}/g, f16Enable);
    const shaderModule = device.createShaderModule({ code: shaderCode });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // state_in
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // state_out
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // perc_weights
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // w1
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b1
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // w2
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // uniforms
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // random
      ],
    });

    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    this.pipeline = device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'nca_step' },
    });

    this.bindGroupLayout = bindGroupLayout;
    this._createBindGroups();
    this.initSeed();
  }

  _createBindGroups() {
    const { device, bindGroupLayout } = this;
    const makeGroup = (stateIn, stateOut) => device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: stateIn } },
        { binding: 1, resource: { buffer: stateOut } },
        { binding: 2, resource: { buffer: this.percWeights } },
        { binding: 3, resource: { buffer: this.adaptW1 } },
        { binding: 4, resource: { buffer: this.adaptB1 } },
        { binding: 5, resource: { buffer: this.adaptW2 } },
        { binding: 6, resource: { buffer: this.uniformBuf } },
        { binding: 7, resource: { buffer: this.randomBuf } },
      ],
    });
    this.bindGroupAB = makeGroup(this.stateA, this.stateB);
    this.bindGroupBA = makeGroup(this.stateB, this.stateA);
  }

  initSeed() {
    const { channels: C, coarseSize: S, seedRadius } = this.model;
    const totalElements = C * S * S * S;

    if (this.hasF16) {
      // Write as Uint16Array with f16-encoded values
      const state = new Uint16Array(totalElements);
      const one_f16 = float32ToFloat16(1.0); // 0x3C00
      const r = seedRadius - 1;
      const center = Math.floor(S / 2);
      for (let z = center - r; z <= center + r; z++)
        for (let y = center - r; y <= center + r; y++)
          for (let x = center - r; x <= center + r; x++) {
            if (z < 0 || y < 0 || x < 0 || z >= S || y >= S || x >= S) continue;
            for (let c = 3; c < C; c++) {
              state[c * S * S * S + z * S * S + y * S + x] = one_f16;
            }
          }
      this.device.queue.writeBuffer(this.stateA, 0, state);
      this.device.queue.writeBuffer(this.stateB, 0, new Uint16Array(totalElements));
    } else {
      const state = new Float32Array(totalElements);
      const r = seedRadius - 1;
      const center = Math.floor(S / 2);
      for (let z = center - r; z <= center + r; z++)
        for (let y = center - r; y <= center + r; y++)
          for (let x = center - r; x <= center + r; x++) {
            if (z < 0 || y < 0 || x < 0 || z >= S || y >= S || x >= S) continue;
            for (let c = 3; c < C; c++) {
              state[c * S * S * S + z * S * S + y * S + x] = 1.0;
            }
          }
      this.device.queue.writeBuffer(this.stateA, 0, state);
      this.device.queue.writeBuffer(this.stateB, 0, new Float32Array(totalElements));
    }
    this.current = 0;
    this.step = 0;
  }

  encode(encoder) {
    const S = this.model.coarseSize;
    // Write uniforms
    const udata = new Float32Array(8);
    udata[0] = S; udata[1] = this.model.channels;
    udata[2] = this.model.numKernels; udata[3] = this.model.fcDim;
    udata[4] = this.step; udata[5] = 0.5; // update prob
    this.device.queue.writeBuffer(this.uniformBuf, 0, udata);

    // Write random values
    const rng = new Float32Array(S * S * S);
    for (let i = 0; i < rng.length; i++) rng[i] = Math.random();
    this.device.queue.writeBuffer(this.randomBuf, 0, rng);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.current === 0 ? this.bindGroupAB : this.bindGroupBA);
    const wg = Math.ceil(S / 4);
    pass.dispatchWorkgroups(wg, wg, wg);
    pass.end();

    this.current = 1 - this.current;
    this.step++;
  }

  get currentStateBuffer() {
    return this.current === 0 ? this.stateA : this.stateB;
  }

  damageAt(x, y, z, radius) {
    const { channels: C, coarseSize: S } = this.model;
    const r = Math.ceil(radius);
    const zero = new Uint32Array([0]);
    const targets = [this.stateA, this.stateB];

    let cleared = 0;
    for (let dz = -r; dz <= r; dz++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy + dz * dz > radius * radius) continue;
          const vx = Math.round(x) + dx;
          const vy = Math.round(y) + dy;
          const vz = Math.round(z) + dz;
          if (vx < 0 || vy < 0 || vz < 0 || vx >= S || vy >= S || vz >= S) continue;
          cleared++;
          for (let c = 0; c < C; c++) {
            const offset = (c * S * S * S + vz * S * S + vy * S + vx) * this.bpe;
            // WebGPU writeBuffer requires 4-byte alignment. If bpe is 2 (F16), offset might be 
            // a multiple of 2. We align to 4 bytes and write a 4-byte zero (which clears 2 F16 voxels).
            // For a damage brush, clearing a neighboring voxel is perfectly fine visually.
            const alignedOffset = offset & ~3;
            for (const target of targets) {
              this.device.queue.writeBuffer(target, alignedOffset, zero, 0, 1);
            }
          }
        }
      }
    }
    console.log(`Damage applied at (${Math.round(x)}, ${Math.round(y)}, ${Math.round(z)}), cleared ${cleared} cells`);
    this.step++; // Increment step to invalidate rendering cache
  }
}
