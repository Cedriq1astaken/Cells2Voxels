import { createBuffer, createEmptyBuffer } from './gpu.js';

const CUBE_VERTS = new Float32Array([
  // pos (xyz), normal (xyz) — 36 vertices for 12 triangles (6 faces)
  // -Z face
  0,0,0, 0,0,-1,  1,1,0, 0,0,-1,  1,0,0, 0,0,-1,
  0,0,0, 0,0,-1,  0,1,0, 0,0,-1,  1,1,0, 0,0,-1,
  // +Z face
  0,0,1, 0,0,1,  1,0,1, 0,0,1,  1,1,1, 0,0,1,
  0,0,1, 0,0,1,  1,1,1, 0,0,1,  0,1,1, 0,0,1,
  // -Y face
  0,0,0, 0,-1,0,  1,0,0, 0,-1,0,  1,0,1, 0,-1,0,
  0,0,0, 0,-1,0,  1,0,1, 0,-1,0,  0,0,1, 0,-1,0,
  // +Y face
  0,1,0, 0,1,0,  1,1,1, 0,1,0,  1,1,0, 0,1,0,
  0,1,0, 0,1,0,  0,1,1, 0,1,0,  1,1,1, 0,1,0,
  // -X face
  0,0,0, -1,0,0,  0,0,1, -1,0,0,  0,1,1, -1,0,0,
  0,0,0, -1,0,0,  0,1,1, -1,0,0,  0,1,0, -1,0,0,
  // +X face
  1,0,0, 1,0,0,  1,1,1, 1,0,0,  1,0,1, 1,0,0,
  1,0,0, 1,0,0,  1,1,0, 1,0,0,  1,1,1, 1,0,0,
]);

const INSTANCE_BYTES = 7 * 4; // XYZ and RGBA as direct f32 values
const COUNT_READ_INTERVAL_HINT_MS = 250;

export class VoxelRenderer {
  constructor(device, format, renderSize, compactShaderCode, renderShaderCode, livingThreshold = 0.1, maxOccupancy = 0.3, rotateModel = true) {
    this.device = device;
    this.format = format;
    this.renderSize = renderSize;
    this.compactShaderCode = compactShaderCode;
    this.renderShaderCode = renderShaderCode;
    this.livingThreshold = livingThreshold;
    this.maxOccupancy = maxOccupancy;
    this.rotateModel = rotateModel;
    this.modelRotation = [0, 0, 0];
    this.voxelCount = 0;
    this.destroyed = false;
    this.countReadIntervalHintMs = COUNT_READ_INTERVAL_HINT_MS;
    this._build();
  }

  _build() {
    const { device, renderSize: RS } = this;
    this.totalVoxels = RS * RS * RS;
    const declaredOccupancy = Number.isFinite(this.maxOccupancy) ? this.maxOccupancy : 0.3;
    const initialFraction = Math.min(1, Math.max(0.1, declaredOccupancy * 1.25));
    const initialCapacity = Math.min(
      this.totalVoxels,
      Math.max(1024, Math.ceil(this.totalVoxels * initialFraction)),
    );

    this.vertexBuf = createBuffer(device, CUBE_VERTS, GPUBufferUsage.VERTEX);
    // count[0] = total visible voxels, count[1] = instances written safely.
    this.countBuf = createEmptyBuffer(device, 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.indirectBuf = createBuffer(device, new Uint32Array([36, 0, 0, 0]), GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.uniformBuf = createEmptyBuffer(device, 160, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.compactParams = createEmptyBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.countResetData = new Uint32Array(2);
    this.compactParamsData = new ArrayBuffer(16);
    this.compactParamsU32 = new Uint32Array(this.compactParamsData);
    this.compactParamsF32 = new Float32Array(this.compactParamsData);
    this.renderUniformData = new Float32Array(40);

    const compactModule = device.createShaderModule({ code: this.compactShaderCode });
    this.compactBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.compactPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.compactBGL] }),
      compute: { module: compactModule, entryPoint: 'compact' },
    });

    const renderModule = device.createShaderModule({ code: this.renderShaderCode });
    this.renderBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderBGL] }),
      vertex: {
        module: renderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module: renderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    });

    this._replaceInstanceBuffer(initialCapacity);
  }

  _replaceInstanceBuffer(capacity) {
    const oldBuffer = this.instanceBuf;
    this.maxInstances = Math.max(1, Math.min(this.totalVoxels, Math.ceil(capacity)));
    this.instanceBuf = createEmptyBuffer(
      this.device,
      this.maxInstances * INSTANCE_BYTES,
      GPUBufferUsage.STORAGE,
    );
    this.renderBG = this.device.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.instanceBuf } },
      ],
    });
    this.compactBG = null;
    this.compactVoxelBuffer = null;
    oldBuffer?.destroy();
  }

  ensureInstanceCapacity(requiredCount) {
    const required = Math.min(this.totalVoxels, Math.max(0, Math.ceil(requiredCount)));
    if (required <= this.maxInstances) return false;
    const nextCapacity = Math.min(
      this.totalVoxels,
      Math.max(
        Math.ceil(this.maxInstances * 1.5),
        Math.ceil(required * 1.25),
      ),
    );
    console.info(`Growing voxel instance capacity: ${this.maxInstances} -> ${nextCapacity}`);
    this._replaceInstanceBuffer(nextCapacity);
    return true;
  }

  createDepthTexture(width, height) {
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return this.depthTexture;
  }

  encodeCompact(encoder, voxelBuffer) {
    const RS = this.renderSize;
    this.device.queue.writeBuffer(this.countBuf, 0, this.countResetData);

    this.compactParamsU32[0] = RS;
    this.compactParamsU32[1] = this.maxInstances;
    this.compactParamsF32[2] = this.livingThreshold;
    this.compactParamsU32[3] = 0;
    this.device.queue.writeBuffer(this.compactParams, 0, this.compactParamsData);

    if (!this.compactBG || this.compactVoxelBuffer !== voxelBuffer) {
      this.compactVoxelBuffer = voxelBuffer;
      this.compactBG = this.device.createBindGroup({
        layout: this.compactBGL,
        entries: [
          { binding: 0, resource: { buffer: voxelBuffer } },
          { binding: 1, resource: { buffer: this.instanceBuf } },
          { binding: 2, resource: { buffer: this.countBuf } },
          { binding: 3, resource: { buffer: this.compactParams } },
        ],
      });
    }

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.compactPipeline);
    pass.setBindGroup(0, this.compactBG);
    const wg = Math.ceil(RS / 4);
    pass.dispatchWorkgroups(wg, wg, wg);
    pass.end();

    // count[1] is clamped by construction, so indirect drawing cannot read
    // beyond the dynamically sized instance buffer.
    encoder.copyBufferToBuffer(this.countBuf, 4, this.indirectBuf, 4, 4);
  }

  encodeRender(encoder, textureView, depthView, mvp, camPos, lightDir) {
    const RS = this.renderSize;
    const f = this.renderUniformData;
    f.fill(0);
    f.set(mvp, 0);
    f[16] = camPos[0];
    f[17] = camPos[1];
    f[18] = camPos[2];
    f[19] = RS;
    f[20] = lightDir[0];
    f[21] = lightDir[1];
    f[22] = lightDir[2];
    f[23] = 1.0 / RS;
    f[24] = this.rotateModel ? 1.0 : 0.0;

    // Precompute Rz * Ry * Rx once on the CPU. WGSL uniform matrices use
    // column-major storage with each vec3 column padded to four floats.
    const [rx, ry, rz] = this.modelRotation;
    const sx = Math.sin(rx), cx = Math.cos(rx);
    const sy = Math.sin(ry), cy = Math.cos(ry);
    const sz = Math.sin(rz), cz = Math.cos(rz);
    f[28] = cz * cy;
    f[29] = sz * cy;
    f[30] = -sy;
    f[32] = cz * sy * sx - sz * cx;
    f[33] = sz * sy * sx + cz * cx;
    f[34] = cy * sx;
    f[36] = cz * sy * cx + sz * sx;
    f[37] = sz * sy * cx - cz * sx;
    f[38] = cy * cx;
    this.device.queue.writeBuffer(this.uniformBuf, 0, f);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.12549, g: 0.16471, b: 0.20784, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.renderPipeline);
    pass.setVertexBuffer(0, this.vertexBuf);
    pass.setBindGroup(0, this.renderBG);
    pass.drawIndirect(this.indirectBuf, 0);
    pass.end();
  }

  async readVoxelCount() {
    if (this.destroyed) return 0;
    const readBuf = createEmptyBuffer(
      this.device,
      4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.countBuf, 0, readBuf, 0, 4);
      this.device.queue.submit([encoder.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const count = new Uint32Array(readBuf.getMappedRange())[0];
      readBuf.unmap();
      this.voxelCount = count;
      return count;
    } finally {
      readBuf.destroy();
    }
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.depthTexture?.destroy();
    this.vertexBuf?.destroy();
    this.instanceBuf?.destroy();
    this.countBuf?.destroy();
    this.indirectBuf?.destroy();
    this.uniformBuf?.destroy();
    this.compactParams?.destroy();
    this.compactBG = null;
    this.renderBG = null;
  }
}
