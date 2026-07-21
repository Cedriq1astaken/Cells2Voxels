import { createEmptyBuffer } from './gpu.js';

// Finds the first decoded voxel that the renderer would include for a click ray.
// Only a four-u32 result is read back, so picking does not download the volume.
export class VoxelPicker {
  constructor(device, renderSize, shaderTemplate, hasF16 = false) {
    this.device = device;
    this.renderSize = renderSize;
    this.shaderTemplate = shaderTemplate;
    this.hasF16 = hasF16;
    this.destroyed = false;
    this._build();
  }

  _build() {
    const code = this.shaderTemplate
      .replace(/{{RS}}/g, this.renderSize)
      .replace(/{{F16_ENABLE}}/g, this.hasF16 ? 'enable f16;' : '')
      .replace(/{{DECODE_TYPE}}/g, this.hasF16 ? 'f16' : 'f32');
    const module = this.device.createShaderModule({ code });
    this.uniformBuf = createEmptyBuffer(
      this.device,
      32,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.resultBuf = createEmptyBuffer(
      this.device,
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      compute: { module, entryPoint: 'pick_visible_voxel' },
    });
  }

  async pick(outputBuffer, rawOrigin, rawDirection, alphaThreshold) {
    const params = new Float32Array([
      rawOrigin[0], rawOrigin[1], rawOrigin[2], alphaThreshold,
      rawDirection[0], rawDirection[1], rawDirection[2], 0,
    ]);
    this.device.queue.writeBuffer(this.uniformBuf, 0, params);
    this.device.queue.writeBuffer(this.resultBuf, 0, new Uint32Array(4));

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: outputBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuf } },
        { binding: 2, resource: { buffer: this.resultBuf } },
      ],
    });
    // Each request owns its staging buffer. A reset/model switch can cancel a
    // pending pick and allow another click before the old mapAsync settles.
    const readBuf = createEmptyBuffer(
      this.device,
      16,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    try {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(this.resultBuf, 0, readBuf, 0, 16);
      this.device.queue.submit([encoder.finish()]);

      await readBuf.mapAsync(GPUMapMode.READ);
      const result = new Uint32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      return result[0] === 1 ? { x: result[1], y: result[2], z: result[3] } : null;
    } finally {
      readBuf.destroy();
    }
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniformBuf?.destroy();
    this.resultBuf?.destroy();
  }
}
