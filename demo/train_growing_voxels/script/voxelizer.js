const MAX_BROWSER_RESOLUTION = 64;

export async function parseVox(buffer, options = {}) {
  const data = new DataView(buffer);
  if (data.byteLength < 20 || readId(data, 0) !== 'VOX ') {
    throw new Error('This is not a valid MagicaVoxel VOX file.');
  }

  let offset = 8;
  let currentSize = null;
  let model = null;
  let palette = defaultPalette();
  let hasCustomPalette = false;

  while (offset + 12 <= data.byteLength) {
    const id = readId(data, offset);
    const contentBytes = data.getUint32(offset + 4, true);
    const contentOffset = offset + 12;
    const contentEnd = contentOffset + contentBytes;
    if (contentEnd > data.byteLength) {
      throw new Error(`Corrupt VOX chunk ${id}.`);
    }

    if (id === 'SIZE' && contentBytes >= 12) {
      currentSize = {
        x: data.getUint32(contentOffset, true),
        y: data.getUint32(contentOffset + 4, true),
        z: data.getUint32(contentOffset + 8, true),
      };
    } else if (id === 'XYZI' && currentSize && !model && contentBytes >= 4) {
      const count = data.getUint32(contentOffset, true);
      if (contentOffset + 4 + count * 4 > contentEnd) {
        throw new Error('Corrupt XYZI voxel data.');
      }
      const voxels = new Uint8Array(currentSize.x * currentSize.y * currentSize.z);
      const activeVoxels = new Uint8Array(count * 4);
      let active = 0;
      for (let index = 0; index < count; index++) {
        const voxelOffset = contentOffset + 4 + index * 4;
        const x = data.getUint8(voxelOffset);
        const y = data.getUint8(voxelOffset + 1);
        const z = data.getUint8(voxelOffset + 2);
        const colorIndex = data.getUint8(voxelOffset + 3);
        if (x < currentSize.x && y < currentSize.y && z < currentSize.z) {
          voxels[(z * currentSize.y + y) * currentSize.x + x] = colorIndex;
          const activeOffset = active * 4;
          activeVoxels[activeOffset] = x;
          activeVoxels[activeOffset + 1] = y;
          activeVoxels[activeOffset + 2] = z;
          activeVoxels[activeOffset + 3] = colorIndex;
          active++;
        }
      }
      model = {
        size: { ...currentSize },
        voxels,
        activeVoxels: activeVoxels.subarray(0, active * 4),
        sourceVoxels: active,
      };
    } else if (id === 'RGBA' && contentBytes >= 1024) {
      palette = new Uint32Array(256);
      for (let index = 1; index < 256; index++) {
        palette[index] = data.getUint32(contentOffset + (index - 1) * 4, true);
      }
      hasCustomPalette = true;
    }
    offset = contentEnd;
  }

  if (!model) throw new Error('The VOX file contains no voxel model.');
  const scaleFactor = options.scaleFactor ?? 4;
  const resolution = chooseResolution(model.size, options.resolution, scaleFactor);
  const target = resampleVoxModel(model, palette, resolution);
  return {
    target,
    resolution,
    sourceVoxels: model.sourceVoxels,
    hasCustomPalette,
    format: 'vox',
  };
}

export async function parseObjAndVol(objText, volBuffer, options = {}) {
  const scaleFactor = options.scaleFactor ?? 4;
  const requested = Number(options.resolution) || 32;
  const resolution = clamp(roundUp(requested, scaleFactor), 24, MAX_BROWSER_RESOLUTION);
  const texture = parseVol(volBuffer);
  const { vertices, faces } = parseObj(objText);
  if (vertices.length < 3 || faces.length === 0) {
    throw new Error('The OBJ file has no usable triangle faces.');
  }

  normalizeVertices(vertices);
  const triangles = buildProjectedTriangles(vertices, faces);
  const padding = Math.max(2, Math.round(resolution / 8));
  const innerSize = resolution - padding * 2;
  if (innerSize < 2) throw new Error('The selected resolution is too small for this mesh.');

  const target = new Float32Array(resolution ** 3 * 4);
  const intersections = [];
  for (let gridX = 0; gridX < innerSize; gridX++) {
    const px = (gridX + 0.5) / innerSize - 0.5;
    for (let gridUp = 0; gridUp < innerSize; gridUp++) {
      const py = (gridUp + 0.5) / innerSize - 0.5;
      intersections.length = 0;

      for (const triangle of triangles) {
        if (px < triangle.minX || px > triangle.maxX || py < triangle.minY || py > triangle.maxY) continue;
        const depth = projectedIntersectionDepth(px, py, triangle);
        if (depth !== null) intersections.push(depth);
      }

      if (intersections.length < 2) continue;
      intersections.sort((a, b) => a - b);
      const unique = deduplicateSorted(intersections, 1e-5);
      for (let pair = 0; pair + 1 < unique.length; pair += 2) {
        const near = unique[pair];
        const far = unique[pair + 1];
        const start = clamp(Math.ceil((near + 0.5) * innerSize - 0.5), 0, innerSize - 1);
        const end = clamp(Math.floor((far + 0.5) * innerSize - 0.5), 0, innerSize - 1);
        for (let gridDepth = start; gridDepth <= end; gridDepth++) {
          const x = gridX + padding;
          const y = innerSize - 1 - gridDepth + padding;
          const z = gridUp + padding;
          const outputOffset = ((z * resolution + y) * resolution + x) * 4;
          const color = sampleVolume(texture, gridX, gridUp, gridDepth);
          target[outputOffset] = color[0];
          target[outputOffset + 1] = color[1];
          target[outputOffset + 2] = color[2];
          target[outputOffset + 3] = 1;
        }
      }
    }
    if (gridX % 8 === 7) await yieldToBrowser();
  }

  return { target, resolution, format: 'obj' };
}

function chooseResolution(size, requested, scaleFactor) {
  if (requested === 'native') {
    const padded = Math.max(size.x, size.y, size.z) + 8;
    return clamp(roundUp(Math.max(16, padded), scaleFactor), 16, MAX_BROWSER_RESOLUTION);
  }
  return clamp(roundUp(Number(requested) || 32, scaleFactor), 16, MAX_BROWSER_RESOLUTION);
}

function resampleVoxModel(model, palette, resolution) {
  const { size, voxels } = model;
  const padding = Math.max(2, Math.min(4, Math.floor(resolution / 8)));
  const available = resolution - padding * 2;
  const fit = Math.min(available / size.x, available / size.y, available / size.z);
  const outputSize = {
    x: Math.max(1, Math.round(size.x * fit)),
    y: Math.max(1, Math.round(size.y * fit)),
    z: Math.max(1, Math.round(size.z * fit)),
  };
  const origin = {
    x: Math.floor((resolution - outputSize.x) / 2),
    y: Math.floor((resolution - outputSize.y) / 2),
    z: Math.floor((resolution - outputSize.z) / 2),
  };
  const target = new Float32Array(resolution ** 3 * 4);

  if (fit < 1) {
    downsampleOccupiedVoxels(model, palette, target, resolution, outputSize, origin);
    return target;
  }

  for (let z = 0; z < outputSize.z; z++) {
    const sourceZ = Math.min(size.z - 1, Math.floor((z + 0.5) * size.z / outputSize.z));
    for (let y = 0; y < outputSize.y; y++) {
      const sourceY = Math.min(size.y - 1, Math.floor((y + 0.5) * size.y / outputSize.y));
      for (let x = 0; x < outputSize.x; x++) {
        const sourceX = Math.min(size.x - 1, Math.floor((x + 0.5) * size.x / outputSize.x));
        const colorIndex = voxels[(sourceZ * size.y + sourceY) * size.x + sourceX];
        if (colorIndex === 0) continue;
        const color = palette[colorIndex];
        const outputOffset = (
          ((z + origin.z) * resolution + (y + origin.y)) * resolution + x + origin.x
        ) * 4;
        target[outputOffset] = (color & 0xff) / 255;
        target[outputOffset + 1] = ((color >>> 8) & 0xff) / 255;
        target[outputOffset + 2] = ((color >>> 16) & 0xff) / 255;
        target[outputOffset + 3] = ((color >>> 24) & 0xff) / 255 || 1;
      }
    }
  }
  return target;
}

function downsampleOccupiedVoxels(model, palette, target, resolution, outputSize, origin) {
  const voxelCount = resolution ** 3;
  const colorSums = new Float32Array(voxelCount * 4);
  const sampleCounts = new Uint32Array(voxelCount);
  const { size, activeVoxels } = model;

  for (let offset = 0; offset < activeVoxels.length; offset += 4) {
    const sourceX = activeVoxels[offset];
    const sourceY = activeVoxels[offset + 1];
    const sourceZ = activeVoxels[offset + 2];
    const color = palette[activeVoxels[offset + 3]];
    const x = origin.x + Math.min(outputSize.x - 1, Math.floor((sourceX + 0.5) * outputSize.x / size.x));
    const y = origin.y + Math.min(outputSize.y - 1, Math.floor((sourceY + 0.5) * outputSize.y / size.y));
    const z = origin.z + Math.min(outputSize.z - 1, Math.floor((sourceZ + 0.5) * outputSize.z / size.z));
    const voxelIndex = (z * resolution + y) * resolution + x;
    const colorOffset = voxelIndex * 4;
    colorSums[colorOffset] += color & 0xff;
    colorSums[colorOffset + 1] += (color >>> 8) & 0xff;
    colorSums[colorOffset + 2] += (color >>> 16) & 0xff;
    colorSums[colorOffset + 3] += (color >>> 24) & 0xff;
    sampleCounts[voxelIndex]++;
  }

  for (let voxelIndex = 0; voxelIndex < voxelCount; voxelIndex++) {
    const count = sampleCounts[voxelIndex];
    if (count === 0) continue;
    const offset = voxelIndex * 4;
    target[offset] = colorSums[offset] / (count * 255);
    target[offset + 1] = colorSums[offset + 1] / (count * 255);
    target[offset + 2] = colorSums[offset + 2] / (count * 255);
    target[offset + 3] = colorSums[offset + 3] / (count * 255) || 1;
  }
}

function parseObj(text) {
  const vertices = [];
  const faces = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) {
      const vertex = [Number(parts[1]), Number(parts[2]), Number(parts[3])];
      if (vertex.every(Number.isFinite)) vertices.push(vertex);
    } else if (parts[0] === 'f' && parts.length >= 4) {
      const polygon = parts.slice(1).map(token => {
        const value = Number.parseInt(token.split('/')[0], 10);
        return value < 0 ? vertices.length + value : value - 1;
      });
      if (polygon.every(index => index >= 0 && index < vertices.length)) {
        for (let index = 1; index + 1 < polygon.length; index++) {
          faces.push([polygon[0], polygon[index], polygon[index + 1]]);
        }
      }
    }
  }
  return { vertices, faces };
}

function normalizeVertices(vertices) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (const vertex of vertices) {
    for (let axis = 0; axis < 3; axis++) {
      minimum[axis] = Math.min(minimum[axis], vertex[axis]);
      maximum[axis] = Math.max(maximum[axis], vertex[axis]);
    }
  }
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
  const extent = Math.max(...minimum.map((value, axis) => maximum[axis] - value));
  if (!Number.isFinite(extent) || extent <= 0) throw new Error('The OBJ mesh has zero extent.');
  for (const vertex of vertices) {
    for (let axis = 0; axis < 3; axis++) vertex[axis] = (vertex[axis] - center[axis]) / extent;
  }
}

function buildProjectedTriangles(vertices, faces) {
  const triangles = [];
  for (const face of faces) {
    const a = vertices[face[0]];
    const b = vertices[face[1]];
    const c = vertices[face[2]];
    const denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if (Math.abs(denominator) < 1e-10) continue;
    triangles.push({
      a, b, c,
      inverseDenominator: 1 / denominator,
      minX: Math.min(a[0], b[0], c[0]),
      maxX: Math.max(a[0], b[0], c[0]),
      minY: Math.min(a[1], b[1], c[1]),
      maxY: Math.max(a[1], b[1], c[1]),
    });
  }
  return triangles;
}

function projectedIntersectionDepth(x, y, triangle) {
  const { a, b, c, inverseDenominator } = triangle;
  const wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) * inverseDenominator;
  const wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) * inverseDenominator;
  const wc = 1 - wa - wb;
  const epsilon = 1e-7;
  if (wa < -epsilon || wb < -epsilon || wc < -epsilon) return null;
  return wa * a[2] + wb * b[2] + wc * c[2];
}

function parseVol(buffer) {
  if (!buffer || buffer.byteLength < 4096) throw new Error('The VOL texture is missing or truncated.');
  const data = new DataView(buffer);
  if (readId(data, 0) !== 'VOLU') throw new Error('Invalid VOL texture header.');
  const version = data.getInt32(4, true);
  const size = data.getInt32(268, true);
  const channels = data.getInt32(272, true);
  const bytesPerChannel = data.getInt32(276, true);
  if (version !== 4) throw new Error(`Unsupported VOL version ${version}.`);
  if (size <= 0 || ![3, 4].includes(channels) || bytesPerChannel !== 1) {
    throw new Error('The VOL texture must contain 8-bit RGB or RGBA voxels.');
  }
  const byteLength = size ** 3 * channels;
  if (4096 + byteLength > buffer.byteLength) throw new Error('The VOL texture data is truncated.');
  return { size, channels, values: new Uint8Array(buffer, 4096, byteLength) };
}

function sampleVolume(texture, x, y, z) {
  const tx = modulo(x, texture.size);
  const ty = modulo(y, texture.size);
  const tz = modulo(z, texture.size);
  const offset = ((tz * texture.size + ty) * texture.size + tx) * texture.channels;
  return [
    texture.values[offset] / 255,
    texture.values[offset + 1] / 255,
    texture.values[offset + 2] / 255,
  ];
}

function deduplicateSorted(values, epsilon) {
  const unique = [];
  for (const value of values) {
    if (unique.length === 0 || Math.abs(value - unique[unique.length - 1]) > epsilon) unique.push(value);
  }
  return unique;
}

function defaultPalette() {
  const palette = new Uint32Array(256);
  const levels = [255, 204, 153, 102, 51, 0];
  let index = 1;
  for (const red of levels) {
    for (const green of levels) {
      for (const blue of levels) {
        palette[index++] = packColor(red, green, blue, 255);
      }
    }
  }
  while (index < 256) {
    const value = Math.round(255 * (255 - index) / 39);
    palette[index++] = packColor(value, value, value, 255);
  }
  return palette;
}

function packColor(red, green, blue, alpha) {
  return (red | green << 8 | blue << 16 | alpha << 24) >>> 0;
}

function readId(data, offset) {
  return String.fromCharCode(
    data.getUint8(offset),
    data.getUint8(offset + 1),
    data.getUint8(offset + 2),
    data.getUint8(offset + 3),
  );
}

function roundUp(value, multiple) {
  return Math.ceil(value / multiple) * multiple;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function modulo(value, divisor) {
  return ((value % divisor) + divisor) % divisor;
}

function yieldToBrowser() {
  return new Promise(resolve => setTimeout(resolve, 0));
}
