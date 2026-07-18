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
    this._build();
  }

  _build() {
    const { device, renderSize: RS } = this;
    const totalVoxels = RS * RS * RS;
    this.maxInstances = totalVoxels;

    this.vertexBuf = createBuffer(device, CUBE_VERTS, GPUBufferUsage.VERTEX);
    this.instanceBuf = createEmptyBuffer(device, this.maxInstances * 7 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX);
    this.countBuf = createEmptyBuffer(device, 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.countReadBuf = createEmptyBuffer(device, 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    // Indirect draw args: [vertexCount=36, instanceCount, firstVertex=0, firstInstance=0]
    this.indirectBuf = createBuffer(device, new Uint32Array([36, 0, 0, 0]), GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.uniformBuf = createEmptyBuffer(device, 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // Compact pipeline
    const compactParams = createEmptyBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.compactParams = compactParams;

    let code = this.compactShaderCode;
    const compactModule = device.createShaderModule({ code });
    const compactBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.compactPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [compactBGL] }),
      compute: { module: compactModule, entryPoint: 'compact' },
    });
    this.compactBGL = compactBGL;

    // Render pipeline
    const renderModule = device.createShaderModule({ code: this.renderShaderCode });
    const renderBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex: {
        module: renderModule, entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module: renderModule, entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
          }
        }],
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
    });
    this.renderBGL = renderBGL;
    this.renderBG = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.instanceBuf } },
      ],
    });
  }

  createDepthTexture(width, height) {
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [width, height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return this.depthTexture;
  }

  encodeCompact(encoder, voxelBuffer) {
    const RS = this.renderSize;
    this.device.queue.writeBuffer(this.countBuf, 0, new Uint32Array([0]));
    
    const paramsData = new ArrayBuffer(16);
    const paramsU32 = new Uint32Array(paramsData);
    const paramsF32 = new Float32Array(paramsData);
    paramsU32[0] = RS;
    paramsU32[1] = this.maxInstances;
    paramsF32[2] = this.livingThreshold;
    paramsU32[3] = 0;
    this.device.queue.writeBuffer(this.compactParams, 0, paramsData);

    const bg = this.device.createBindGroup({
      layout: this.compactBGL,
      entries: [
        { binding: 0, resource: { buffer: voxelBuffer } },
        { binding: 1, resource: { buffer: this.instanceBuf } },
        { binding: 2, resource: { buffer: this.countBuf } },
        { binding: 3, resource: { buffer: this.compactParams } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.compactPipeline);
    pass.setBindGroup(0, bg);
    const wg = Math.ceil(RS / 4);
    pass.dispatchWorkgroups(wg, wg, wg);
    pass.end();

    // Copy instance count into the indirect draw args buffer (offset 4 = instanceCount slot)
    encoder.copyBufferToBuffer(this.countBuf, 0, this.indirectBuf, 4, 4);
  }

  encodeRender(encoder, textureView, depthView, mvp, camPos, lightDir) {
    const RS = this.renderSize;
    const u = new ArrayBuffer(128);
    const f = new Float32Array(u);
    f.set(mvp, 0);      // mat4 at offset 0
    f[16] = camPos[0]; f[17] = camPos[1]; f[18] = camPos[2];
    f[19] = RS;
    f[20] = lightDir[0]; f[21] = lightDir[1]; f[22] = lightDir[2];
    f[23] = 1.0 / RS;
    f[24] = this.rotateModel ? 1.0 : 0.0;
    // vec3 alignment places model_rotation at byte 112 (float index 28).
    f[28] = this.modelRotation[0];
    f[29] = this.modelRotation[1];
    f[30] = this.modelRotation[2];
    this.device.queue.writeBuffer(this.uniformBuf, 0, f);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView, clearValue: { r: 0.12549, g: 0.16471, b: 0.20784, a: 1.0 }, loadOp: 'clear', storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView, depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.renderPipeline);
    pass.setVertexBuffer(0, this.vertexBuf);
    pass.setBindGroup(0, this.renderBG);
    pass.drawIndirect(this.indirectBuf, 0);
    pass.end();
  }

  async readVoxelCount() {
    const { device, countBuf, countReadBuf } = this;
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(countBuf, 0, countReadBuf, 0, 4);
    device.queue.submit([encoder.finish()]);
    await countReadBuf.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(countReadBuf.getMappedRange());
    const count = data[0];
    countReadBuf.unmap();
    this.voxelCount = count;
    return count;
  }
}
