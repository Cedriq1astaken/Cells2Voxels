import { createBuffer, createEmptyBuffer } from './gpu.js';

export class LPPNCompute {
  constructor(device, model, shaderTemplate, livingMaskTemplate, activeBlocksTemplate, hasF16) {
    this.device = device;
    this.model = model;
    this.shaderTemplate = shaderTemplate;
    this.livingMaskTemplate = livingMaskTemplate;
    this.activeBlocksTemplate = activeBlocksTemplate;
    this.hasF16 = hasF16;
    this.destroyed = false;
    this._buildPipeline();
  }

  _buildPipeline() {
    const { device, model } = this;
    const { channels: C, coarseSize: S, scale, numFrequencies, lppnFirstOmega, lppnHiddenOmega } = model;
    const renderSize = Math.floor(S * scale);
    this.renderSize = renderSize;

    // Dense fine-grid storage uses f16 when supported; LPPN math remains f32.
    this.decodedBpe = this.hasF16 ? 2 : 4;
    const outSize = 4 * renderSize * renderSize * renderSize * this.decodedBpe;
    this.outputBuf = createEmptyBuffer(device, outSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.fineAlphaBuf = createEmptyBuffer(
      device,
      renderSize * renderSize * renderSize * this.decodedBpe,
      GPUBufferUsage.STORAGE,
    );
    this.blocksPerAxis = Math.ceil(renderSize / 4);
    this.blockCount = this.blocksPerAxis ** 3;
    this.activeBlockBuf = createEmptyBuffer(
      device,
      this.blockCount * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
    );
    this.dispatchArgsBuf = createBuffer(
      device,
      new Uint32Array([0, 1, 1, 0]),
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    );
    this.dispatchResetData = new Uint32Array([0, 1, 1, 0]);


    // Upload LPPN weights
    this.lppnBuffers = {};
    for (const [key, info] of Object.entries(model.weights.lppn)) {
      this.lppnBuffers[key] = createBuffer(device, info.data, GPUBufferUsage.STORAGE);
    }

    // Uniforms and stable bind groups are reused across decodes.
    this.uniformBuf = createEmptyBuffer(device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.uniformData = new Float32Array(12);
    this.bindGroupCache = new Map();

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
      .replace(/{{DECODE_TYPE}}/g, stateType)
      .replace(/{{F16_ENABLE}}/g, f16Enable);
    const shaderModule = device.createShaderModule({ code: shaderCode });
    const livingMaskCode = this.livingMaskTemplate
      .replace(/{{S}}/g, S)
      .replace(/{{RS}}/g, this.renderSize)
      .replace(/{{LC}}/g, this.model.livingChannel)
      .replace(/{{STATE_TYPE}}/g, stateType)
      .replace(/{{DECODE_TYPE}}/g, stateType)
      .replace(/{{F16_ENABLE}}/g, f16Enable);
    const livingMaskModule = device.createShaderModule({ code: livingMaskCode });
    const activeBlocksCode = this.activeBlocksTemplate
      .replace(/{{RS}}/g, this.renderSize)
      .replace(/{{DECODE_TYPE}}/g, stateType)
      .replace(/{{F16_ENABLE}}/g, f16Enable)
      .replace(/{{LIVING_THRESHOLD}}/g, `${model.livingThreshold}`);
    const activeBlocksModule = device.createShaderModule({ code: activeBlocksCode });
    this.livingMaskBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.livingMaskPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.livingMaskBindGroupLayout] }),
      compute: { module: livingMaskModule, entryPoint: 'interpolate_living_alpha' },
    });
    this.activeBlocksBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const activeBlocksLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.activeBlocksBindGroupLayout],
    });
    this.findActiveBlocksPipeline = device.createComputePipeline({
      layout: activeBlocksLayout,
      compute: { module: activeBlocksModule, entryPoint: 'find_active_blocks' },
    });
    this.finalizeDispatchPipeline = device.createComputePipeline({
      layout: activeBlocksLayout,
      compute: { module: activeBlocksModule, entryPoint: 'finalize_dispatch' },
    });
    this.activeBlocksBindGroup = device.createBindGroup({
      layout: this.activeBlocksBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.fineAlphaBuf } },
        { binding: 1, resource: { buffer: this.activeBlockBuf } },
        { binding: 2, resource: { buffer: this.dispatchArgsBuf } },
      ],
    });

    // Binding 2 carries the compacted active-block list; binding 13 exposes
    // the indirect dimensions and exact active count to the decoder.
    const entries = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // state
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // active blocks
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // uniforms
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // fine alpha
    ];
    // LPPN weight buffers: 8 total (4 layers x weight + bias)
    let bindIdx = 5;
    
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

    entries.push({ binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });

    const bindGroupLayout = device.createBindGroupLayout({ entries });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    this.pipeline = device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'lppn_decode' },
    });
    this.bindGroupLayout = bindGroupLayout;
  }

  _getBindGroups(stateBuffer) {
    let groups = this.bindGroupCache.get(stateBuffer);
    if (groups) return groups;

    const decodeEntries = [
      { binding: 0, resource: { buffer: stateBuffer } },
      { binding: 1, resource: { buffer: this.outputBuf } },
      { binding: 2, resource: { buffer: this.activeBlockBuf } },
      { binding: 3, resource: { buffer: this.uniformBuf } },
      { binding: 4, resource: { buffer: this.fineAlphaBuf } },
    ];
    let bindIdx = 5;
    for (const key of this.lppnKeyOrder) {
      decodeEntries.push({ binding: bindIdx++, resource: { buffer: this.lppnBuffers[key] } });
    }

    decodeEntries.push({ binding: 13, resource: { buffer: this.dispatchArgsBuf } });

    groups = {
      mask: this.device.createBindGroup({
        layout: this.livingMaskBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: this.fineAlphaBuf } },
          { binding: 2, resource: { buffer: this.uniformBuf } },
        ],
      }),
      decode: this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: decodeEntries,
      }),
    };
    this.bindGroupCache.set(stateBuffer, groups);
    return groups;
  }

  encode(encoder, stateBuffer, crossSection) {
    const RS = this.renderSize;
    const u = this.uniformData;
    u.fill(0);
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

    const groups = this._getBindGroups(stateBuffer);
    encoder.clearBuffer(this.outputBuf);
    this.device.queue.writeBuffer(this.dispatchArgsBuf, 0, this.dispatchResetData);
    const wg = Math.ceil(RS / 4);
    const maskPass = encoder.beginComputePass();
    maskPass.setPipeline(this.livingMaskPipeline);
    maskPass.setBindGroup(0, groups.mask);
    maskPass.dispatchWorkgroups(wg, wg, wg);
    maskPass.end();

    const activePass = encoder.beginComputePass();
    activePass.setPipeline(this.findActiveBlocksPipeline);
    activePass.setBindGroup(0, this.activeBlocksBindGroup);
    activePass.dispatchWorkgroups(Math.ceil(this.blockCount / 64));
    activePass.end();

    const finalizePass = encoder.beginComputePass();
    finalizePass.setPipeline(this.finalizeDispatchPipeline);
    finalizePass.setBindGroup(0, this.activeBlocksBindGroup);
    finalizePass.dispatchWorkgroups(1);
    finalizePass.end();

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, groups.decode);
    pass.dispatchWorkgroupsIndirect(this.dispatchArgsBuf, 0);
    pass.end();
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.outputBuf?.destroy();
    this.fineAlphaBuf?.destroy();
    this.activeBlockBuf?.destroy();
    this.dispatchArgsBuf?.destroy();
    this.uniformBuf?.destroy();
    for (const buffer of Object.values(this.lppnBuffers)) buffer.destroy();
    this.bindGroupCache.clear();
  }
}
