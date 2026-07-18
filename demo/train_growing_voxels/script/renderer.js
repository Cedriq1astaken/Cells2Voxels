import { createBuffer, createEmptyBuffer } from './gpu.js?v=18';
import { toFloat16Array } from './f16.js?v=18';

const CUBE_VERTICES = new Float32Array([
  0,0,0, 0,0,-1,  1,1,0, 0,0,-1,  1,0,0, 0,0,-1,
  0,0,0, 0,0,-1,  0,1,0, 0,0,-1,  1,1,0, 0,0,-1,
  0,0,1, 0,0,1,   1,0,1, 0,0,1,   1,1,1, 0,0,1,
  0,0,1, 0,0,1,   1,1,1, 0,0,1,   0,1,1, 0,0,1,
  0,0,0, 0,-1,0,  1,0,0, 0,-1,0,  1,0,1, 0,-1,0,
  0,0,0, 0,-1,0,  1,0,1, 0,-1,0,  0,0,1, 0,-1,0,
  0,1,0, 0,1,0,   1,1,1, 0,1,0,   1,1,0, 0,1,0,
  0,1,0, 0,1,0,   0,1,1, 0,1,0,   1,1,1, 0,1,0,
  0,0,0, -1,0,0,  0,0,1, -1,0,0,  0,1,1, -1,0,0,
  0,0,0, -1,0,0,  0,1,1, -1,0,0,  0,1,0, -1,0,0,
  1,0,0, 1,0,0,   1,1,1, 1,0,0,   1,0,1, 1,0,0,
  1,0,0, 1,0,0,   1,1,0, 1,0,0,   1,1,1, 1,0,0,
]);

export class VoxelRenderer {
  constructor(device, format, renderSize, compactCode, renderCode, onCount = () => {}) {
    this.device = device;
    this.format = format;
    this.renderSize = renderSize;
    this.compactCode = compactCode;
    this.renderCode = renderCode;
    this.onCount = onCount;
    this.voxelBuffer = null;
    this.compactBindGroup = null;
    this.depthTexture = null;
    this.depthWidth = 0;
    this.depthHeight = 0;
    this.dirty = false;
    this.countPending = false;
    this.uniformData = new Float32Array(32);
    this.build();
  }

  build() {
    const totalVoxels = this.renderSize ** 3;
    this.maxInstances = totalVoxels;
    this.vertexBuffer = createBuffer(this.device, CUBE_VERTICES, GPUBufferUsage.VERTEX);
    this.instanceBuffer = createEmptyBuffer(
      this.device,
      this.maxInstances * 8 * Uint16Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
    );
    this.countBuffer = createEmptyBuffer(
      this.device,
      4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    this.countReadBuffer = createEmptyBuffer(
      this.device,
      4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    this.indirectBuffer = createBuffer(
      this.device,
      new Uint32Array([36, 0, 0, 0]),
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    );
    this.uniformBuffer = createEmptyBuffer(
      this.device,
      128,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.compactParams = createEmptyBuffer(
      this.device,
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );

    const params = new ArrayBuffer(16);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = this.renderSize;
    paramsU32[1] = this.maxInstances;
    paramsF32[2] = 0.1;
    this.device.queue.writeBuffer(this.compactParams, 0, params);

    const compactModule = this.device.createShaderModule({
      label: 'train_growing_voxels_compact_shader',
      code: this.compactCode,
    });
    compactModule.getCompilationInfo?.().then(info => {
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) console.error('Voxel compaction WGSL errors:', errors);
    });
    this.compactLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.device.pushErrorScope('validation');
    this.compactPipeline = this.device.createComputePipeline({
      label: 'train_growing_voxels_compact_pipeline',
      layout: 'auto',
      compute: { module: compactModule, entryPoint: 'compact' },
    });
    this.device.popErrorScope().then(error => {
      if (error) console.error('Voxel compaction pipeline validation:', error);
    });
    this.compactLayout = this.compactPipeline.getBindGroupLayout(0);

    const renderModule = this.device.createShaderModule({
      label: 'train_growing_voxels_render_shader',
      code: this.renderCode,
    });
    renderModule.getCompilationInfo?.().then(info => {
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) console.error('Voxel render WGSL errors:', errors);
    });
    const renderLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
      vertex: {
        module: renderModule,
        entryPoint: 'vertex_main',
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
        entryPoint: 'fragment_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.instanceBuffer } },
      ],
    });
  }

  resize(width, height) {
    if (width === this.depthWidth && height === this.depthHeight) return;
    this.depthTexture?.destroy();
    this.depthWidth = width;
    this.depthHeight = height;
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  updateVoxelData(data) {
    const halfData = data instanceof Uint16Array ? data : toFloat16Array(data);
    this.ensureVoxelBuffer(halfData.byteLength);
    this.device.queue.writeBuffer(this.voxelBuffer, 0, halfData);
    this.dirty = true;
  }

  updateVoxelBuffer(sourceBuffer, byteLength) {
    this.ensureVoxelBuffer(byteLength);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(sourceBuffer, 0, this.voxelBuffer, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    this.dirty = true;
  }

  ensureVoxelBuffer(byteLength) {
    if (!this.voxelBuffer || this.voxelBuffer.size !== byteLength) {
      this.voxelBuffer?.destroy();
      this.voxelBuffer = createEmptyBuffer(
        this.device,
        byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      this.compactBindGroup = this.device.createBindGroup({
        layout: this.compactLayout,
        entries: [
          { binding: 0, resource: { buffer: this.voxelBuffer } },
          { binding: 1, resource: { buffer: this.instanceBuffer } },
          { binding: 2, resource: { buffer: this.countBuffer } },
          { binding: 3, resource: { buffer: this.compactParams } },
        ],
      });
    }
  }

  render(context, camera) {
    if (!this.voxelBuffer || !this.depthTexture) return;
    const encoder = this.device.createCommandEncoder();
    const compacted = this.dirty;
    const readCount = compacted && !this.countPending;

    if (compacted) {
      this.device.queue.writeBuffer(this.countBuffer, 0, new Uint32Array([0]));
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.compactPipeline);
      pass.setBindGroup(0, this.compactBindGroup);
      const groups = Math.ceil(this.renderSize / 4);
      pass.dispatchWorkgroups(groups, groups, groups);
      pass.end();
      encoder.copyBufferToBuffer(this.countBuffer, 0, this.indirectBuffer, 4, 4);
      if (readCount) {
        encoder.copyBufferToBuffer(this.countBuffer, 0, this.countReadBuffer, 0, 4);
        this.countPending = true;
      }
      this.dirty = false;
    }

    const mvp = camera.getMVP(context.canvas.width / context.canvas.height);
    const cameraPosition = camera.getPosition();
    this.uniformData.fill(0);
    this.uniformData.set(mvp, 0);
    this.uniformData.set(cameraPosition, 16);
    this.uniformData[19] = this.renderSize;
    this.uniformData.set([0.5, 0.8, -0.2], 20);
    this.uniformData[23] = 1 / this.renderSize;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.12549, g: 0.16471, b: 0.20784, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.renderPipeline);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.drawIndirect(this.indirectBuffer, 0);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    if (readCount) {
      this.readCount();
    }
  }

  async readCount() {
    try {
      await this.countReadBuffer.mapAsync(GPUMapMode.READ);
      const count = new Uint32Array(this.countReadBuffer.getMappedRange())[0];
      this.countReadBuffer.unmap();
      this.onCount(count);
    } catch (error) {
      console.warn('Could not read voxel count:', error);
    } finally {
      this.countPending = false;
    }
  }

  destroy() {
    if (this.countReadBuffer?.mapState === 'mapped') this.countReadBuffer.unmap();
    for (const resource of [
      this.vertexBuffer,
      this.instanceBuffer,
      this.countBuffer,
      this.countReadBuffer,
      this.indirectBuffer,
      this.uniformBuffer,
      this.compactParams,
      this.voxelBuffer,
      this.depthTexture,
    ]) {
      resource?.destroy();
    }
  }
}
