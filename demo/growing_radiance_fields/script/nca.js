import { createBuffer, createEmptyBuffer } from './gpu.js';

export class NCACompute {
  constructor(device, model, shaderTemplate) {
    this.device = device;
    this.model = model;
    this.shaderTemplate = shaderTemplate;
    this.step = 0;
    this._buildPipeline();
  }

  _buildPipeline() {
    const { device, model } = this;
    const S = model.coarseSize;
    const C = model.channels;
    const K = model.numKernels;
    const FC = model.fcDim;

    const stateSize = C * S * S * S * 4;
    this.stateA = createEmptyBuffer(device, stateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.stateB = createEmptyBuffer(device, stateSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.current = 0;

    this.percWeights = createBuffer(device, model.weights.perception['perceive.weight'].data, GPUBufferUsage.STORAGE);
    this.adaptW1 = createBuffer(device, model.weights.adaptation['adapt.0.weight'].data, GPUBufferUsage.STORAGE);
    this.adaptB1 = createBuffer(device, model.weights.adaptation['adapt.0.bias'].data, GPUBufferUsage.STORAGE);
    this.adaptW2 = createBuffer(device, model.weights.adaptation['adapt.2.weight'].data, GPUBufferUsage.STORAGE);

    this.uniformBuf = createEmptyBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.randomBuf = createEmptyBuffer(device, S * S * S * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    const shaderCode = this.shaderTemplate
      .replace(/{{S}}/g, S)
      .replace(/{{C}}/g, C)
      .replace(/{{K}}/g, K)
      .replace(/{{FC}}/g, FC)
      .replace(/{{LC}}/g, model.livingChannel)
      .replace(/{{C_K}}/g, C * K);
    const shaderModule = device.createShaderModule({ code: shaderCode });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
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
    const state = new Float32Array(C * S * S * S);
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
    this.device.queue.writeBuffer(this.stateB, 0, new Float32Array(C * S * S * S));
    this.current = 0;
    this.step = 0;
  }

  encode(encoder, timestampWrites) {
    const S = this.model.coarseSize;
    const udata = new Float32Array(8);
    udata[0] = S;
    udata[1] = this.model.channels;
    udata[2] = this.model.numKernels;
    udata[3] = this.model.fcDim;
    udata[4] = this.step;
    udata[5] = this.model.updateProb;
    udata[6] = 0.1; // model.livingThreshold is for visuals, NCA needs 0.1
    this.device.queue.writeBuffer(this.uniformBuf, 0, udata);

    const rng = new Float32Array(S * S * S);
    for (let i = 0; i < rng.length; i++) rng[i] = Math.random();
    this.device.queue.writeBuffer(this.randomBuf, 0, rng);

    const passDesc = {};
    if (timestampWrites) {
      passDesc.timestampWrites = timestampWrites;
    }
    const pass = encoder.beginComputePass(passDesc);
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
    const zeros = new Float32Array(C);
    const targets = [this.stateA, this.stateB];

    for (let dz = -r; dz <= r; dz++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy + dz * dz > radius * radius) continue;
          const vx = Math.round(x) + dx;
          const vy = Math.round(y) + dy;
          const vz = Math.round(z) + dz;
          if (vx < 0 || vy < 0 || vz < 0 || vx >= S || vy >= S || vz >= S) continue;
          for (let c = 0; c < C; c++) {
            const offset = (c * S * S * S + vz * S * S + vy * S + vx) * 4;
            for (const target of targets) this.device.queue.writeBuffer(target, offset, zeros, c, 1);
          }
        }
  }
}
