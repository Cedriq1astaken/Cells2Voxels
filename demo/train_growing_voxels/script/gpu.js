export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is required for training.');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter was found.');
  if (!adapter.features.has('shader-f16')) {
    throw new Error('This GPU/browser does not support WebGPU shader-f16.');
  }
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });

  device.addEventListener('uncapturederror', event => {
    console.error('WebGPU error:', event.error);
  });
  return { adapter, device };
}

export function configureCanvas(device, canvas) {
  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('Could not create a WebGPU canvas context.');
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  return { context, format };
}

export function createBuffer(device, data, usage) {
  const size = Math.max(8, Math.ceil(data.byteLength / 8) * 8);
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  new data.constructor(buffer.getMappedRange(0, size)).set(data);
  buffer.unmap();
  return buffer;
}

export function createEmptyBuffer(device, size, usage) {
  return device.createBuffer({ size: Math.max(8, Math.ceil(size / 8) * 8), usage });
}
