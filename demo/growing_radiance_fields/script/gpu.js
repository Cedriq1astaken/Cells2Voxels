export async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not supported in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');

  const requiredFeatures = [];
  if (adapter.features.has('shader-f16')) {
    requiredFeatures.push('shader-f16');
  }
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });

  const hasF16 = device.features.has('shader-f16');
  return { adapter, device, hasF16 };
}

export function configureCanvas(device, canvas) {
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });
  return { ctx, format };
}

export function createBuffer(device, data, usage) {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new (data.constructor)(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

export function createEmptyBuffer(device, size, usage) {
  return device.createBuffer({ size, usage });
}
