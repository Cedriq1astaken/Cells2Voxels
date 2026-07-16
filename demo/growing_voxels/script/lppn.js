import { createBuffer, createEmptyBuffer } from './gpu.js';

export class LPPNCompute {
  constructor(device, model, shaderTemplate, hasF16) {
    this.device = device;
    this.model = model;
    this.shaderTemplate = shaderTemplate;
    this.hasF16 = hasF16;
    this._buildPipeline();
  }

  _buildPipeline() {
    const { device, model } = this;
    const { channels: C, coarseSize: S, scale, numFrequencies, lppnFirstOmega, lppnHiddenOmega } = model;
    const renderSize = Math.floor(S * scale);
    this.renderSize = renderSize;

    // Output RGBA buffer: [4 * renderSize^3]
    const outSize = 4 * renderSize * renderSize * renderSize * 4;
    this.outputBuf = createEmptyBuffer(device, outSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    // Voxel count buffer (atomic counter for active voxels)
    this.countBuf = createEmptyBuffer(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Upload LPPN weights
    this.lppnBuffers = {};
    for (const [key, info] of Object.entries(model.weights.lppn)) {
      this.lppnBuffers[key] = createBuffer(device, info.data, GPUBufferUsage.STORAGE);
    }

    // Uniforms
    this.uniformBuf = createEmptyBuffer(device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    const { numFrequencies: NF, lppnHiddenDim: HD } = model;
    const coordDim = 3 * Math.max(1, 2 * NF);
    const inputDim = C + coordDim;
    const stateType = this.hasF16 ? 'f16' : 'f32';
    const f16Enable = this.hasF16 ? 'enable f16;' : '';
    const shaderCode = this.shaderTemplate
      .replace(/{{S}}/g, S)
      .replace(/{{C}}/g, C)
      .replace(/{{RS}}/g, this.renderSize)
      .replace(/{{NF}}/g, NF)
      .replace(/{{HD}}/g, HD)
      .replace(/{{LC}}/g, this.model.livingChannel)
      .replace(/{{COORD_DIM}}/g, coordDim)
      .replace(/{{INPUT_DIM}}/g, inputDim)
      .replace(/{{LIVING_THRESHOLD}}/g, `${model.livingThreshold}`)
      .replace(/{{STATE_TYPE}}/g, stateType)
      .replace(/{{F16_ENABLE}}/g, f16Enable);
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // Bind group layout: state_in, output, count, uniforms, then LPPN weights
    const entries = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // state
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // count
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // uniforms
    ];
    // LPPN weight buffers: 8 total (4 layers × weight + bias)
    let bindIdx = 4;
    
    const lppnKeys = Object.keys(model.weights.lppn).sort((a, b) => {
      // First sort by layer number (extracted from e.g. "net.0...")
      const layerA = parseInt(a.split('.')[1]);
      const layerB = parseInt(b.split('.')[1]);
      if (layerA !== layerB) return layerA - layerB;
      // Then ensure weight comes before bias
      if (a.includes('weight') && b.includes('bias')) return -1;
      if (a.includes('bias') && b.includes('weight')) return 1;
      return a.localeCompare(b);
    });
    this.lppnKeyOrder = lppnKeys;
    
    for (const key of lppnKeys) {
      entries.push({ binding: bindIdx++, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    }

    const bindGroupLayout = device.createBindGroupLayout({ entries });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    this.pipeline = device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'lppn_decode' },
    });
    this.bindGroupLayout = bindGroupLayout;
  }

  createBindGroup(stateBuffer) {
    const entries = [
      { binding: 0, resource: { buffer: stateBuffer } },
      { binding: 1, resource: { buffer: this.outputBuf } },
      { binding: 2, resource: { buffer: this.countBuf } },
      { binding: 3, resource: { buffer: this.uniformBuf } },
    ];
    let bindIdx = 4;
    for (const key of this.lppnKeyOrder) {
      entries.push({ binding: bindIdx++, resource: { buffer: this.lppnBuffers[key] } });
    }
    return this.device.createBindGroup({ layout: this.bindGroupLayout, entries });
  }

  encode(encoder, stateBuffer, crossSection) {
    const RS = this.renderSize;
    // Reset counter
    this.device.queue.writeBuffer(this.countBuf, 0, new Uint32Array([0]));

    // Write uniforms
    const u = new Float32Array(12);
    u[0] = this.model.coarseSize;
    u[1] = this.model.channels;
    u[2] = this.model.scale;
    u[3] = RS;
    u[4] = this.model.numFrequencies;
    u[5] = this.model.lppnFirstOmega;
    u[6] = this.model.lppnHiddenOmega;
    u[7] = crossSection?.[0] ?? RS;
    u[8] = crossSection?.[1] ?? RS;
    u[9] = crossSection?.[2] ?? RS;
    this.device.queue.writeBuffer(this.uniformBuf, 0, u);

    const bg = this.createBindGroup(stateBuffer);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    const wg = Math.ceil(RS / 4);
    pass.dispatchWorkgroups(wg, wg, wg);
    pass.end();
  }
}
