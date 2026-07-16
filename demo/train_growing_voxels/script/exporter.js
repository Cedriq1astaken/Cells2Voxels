import { buildPerceptionKernels, MODEL_CONSTANTS } from './trainer.js';

export async function exportTrainerModel(trainer, options) {
  if (!window.JSZip) throw new Error('JSZip did not load.');
  const info = await trainer.getExportInfo();
  if (info.iteration < 1) throw new Error('Train at least one iteration before exporting.');

  const name = sanitizeName(options.name || 'Growing Voxel');
  const values = info.variables;
  const sections = { perception: {}, adaptation: {}, lppn: {} };
  const chunks = [];
  let floatOffset = 0;

  const append = (section, key, data, shape) => {
    const copied = data instanceof Float32Array ? new Float32Array(data) : Float32Array.from(data);
    sections[section][key] = {
      offset: floatOffset * Float32Array.BYTES_PER_ELEMENT,
      count: copied.length,
      shape,
    };
    chunks.push(copied);
    floatOffset += copied.length;
  };

  append('perception', 'perceive.weight', buildPerceptionKernels(), [5, 1, 3, 3, 3]);
  append(
    'adaptation',
    'adapt.0.weight',
    transpose2d(values.ncaW1, info.channels * 5, info.fcDim),
    [info.fcDim, info.channels * 5, 1, 1, 1],
  );
  append('adaptation', 'adapt.0.bias', values.ncaB1, [info.fcDim]);
  append(
    'adaptation',
    'adapt.2.weight',
    transpose2d(values.ncaW2, info.fcDim, info.channels),
    [info.channels, info.fcDim, 1, 1, 1],
  );

  const inputDim = info.channels + 6 * MODEL_CONSTANTS.coordinateFrequencies;
  append('lppn', 'net.0.linear.weight', transpose2d(values.l0w, inputDim, info.lppnWidth), [info.lppnWidth, inputDim]);
  append('lppn', 'net.0.linear.bias', values.l0b, [info.lppnWidth]);
  append('lppn', 'net.1.linear.weight', transpose2d(values.l1w, info.lppnWidth, info.lppnWidth), [info.lppnWidth, info.lppnWidth]);
  append('lppn', 'net.1.linear.bias', values.l1b, [info.lppnWidth]);
  append('lppn', 'net.2.linear.weight', transpose2d(values.l2w, info.lppnWidth, info.lppnWidth), [info.lppnWidth, info.lppnWidth]);
  append('lppn', 'net.2.linear.bias', values.l2b, [info.lppnWidth]);
  append('lppn', 'net.3.weight', transpose2d(values.l3w, info.lppnWidth, 4), [4, info.lppnWidth]);
  append('lppn', 'net.3.bias', values.l3b, [4]);

  const packed = new Float32Array(floatOffset);
  let writeOffset = 0;
  for (const chunk of chunks) {
    packed.set(chunk, writeOffset);
    writeOffset += chunk.length;
  }

  const manifest = {
    meta: {
      channels: info.channels,
      coarse_size: info.coarseSize,
      target_size: info.targetSize,
      scale: info.scale,
      living_channel: MODEL_CONSTANTS.livingChannel,
      living_threshold: MODEL_CONSTANTS.livingThreshold,
      seed_radius: info.seedRadius,
      max_occupancy: 1,
      num_frequencies: MODEL_CONSTANTS.coordinateFrequencies,
      lppn_first_omega_0: MODEL_CONSTANTS.firstOmega,
      lppn_hidden_omega_0: MODEL_CONSTANTS.hiddenOmega,
      edge_loss_weight: 0,
      iterations: info.iteration,
      gpu_used: 'WebGPU shader-f16',
      target_path: options.sourceName || name,
      pool_size: info.poolSize,
      browser_export: true,
    },
    ...sections,
  };

  const manifestJson = JSON.stringify(manifest, null, 2);
  const zip = new window.JSZip();
  const folder = zip.folder(name);
  folder.file('nca_manifest.json', manifestJson);
  folder.file('nca_weights.json', manifestJson);
  folder.file('nca_weights.bin', packed.buffer);

  if (options.targetData) {
    const [original, coarse] = await Promise.all([
      projectionPng(options.targetData, info.targetSize, info.targetSize),
      projectionPng(options.targetData, info.targetSize, info.coarseSize),
    ]);
    if (original) folder.file('original.png', original);
    if (coarse) folder.file('coarse.png', coarse);
  }

  const archive = await zip.generateAsync(
    { type: 'blob', compression: 'STORE' },
    metadata => options.onProgress?.(metadata.percent),
  );
  downloadBlob(archive, `${name}.zip`);
  return { name, manifest, byteLength: packed.byteLength };
}

function transpose2d(values, rows, columns) {
  if (values.length !== rows * columns) {
    throw new Error(`Cannot transpose ${values.length} values as ${rows}x${columns}.`);
  }
  const output = new Float32Array(values.length);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      output[column * rows + row] = values[row * columns + column];
    }
  }
  return output;
}

async function projectionPng(volume, size, outputSize) {
  const source = document.createElement('canvas');
  source.width = size;
  source.height = size;
  const context = source.getContext('2d');
  if (!context) return null;
  const image = context.createImageData(size, size);

  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      let bestAlpha = 0;
      let red = 0;
      let green = 0;
      let blue = 0;
      for (let y = 0; y < size; y++) {
        const offset = ((z * size + y) * size + x) * 4;
        const alpha = volume[offset + 3];
        if (alpha > bestAlpha) {
          bestAlpha = alpha;
          red = volume[offset];
          green = volume[offset + 1];
          blue = volume[offset + 2];
        }
      }
      const pixel = ((size - 1 - z) * size + x) * 4;
      image.data[pixel] = clampByte(red * 255);
      image.data[pixel + 1] = clampByte(green * 255);
      image.data[pixel + 2] = clampByte(blue * 255);
      image.data[pixel + 3] = clampByte(bestAlpha * 255);
    }
  }
  context.putImageData(image, 0, 0);

  let output = source;
  if (outputSize !== size) {
    output = document.createElement('canvas');
    output.width = outputSize;
    output.height = outputSize;
    const outputContext = output.getContext('2d');
    outputContext.imageSmoothingEnabled = true;
    outputContext.drawImage(source, 0, 0, outputSize, outputSize);
  }
  return new Promise(resolve => output.toBlob(resolve, 'image/png'));
}

function downloadBlob(blob, filename) {
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function sanitizeName(value) {
  const clean = value.replace(/[^a-z0-9 _.-]/gi, '_').replace(/\s+/g, ' ').trim();
  return clean || 'Growing Voxel';
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}
