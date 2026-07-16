const MODEL_BASE_URL = new URL('../model/', import.meta.url);

export async function loadModelIndex() {
  try {
    const resp = await fetch(new URL('index.json', MODEL_BASE_URL), { cache: 'no-store' });
    if (resp.ok) return resp.json();
  } catch (_) {}
  return ['lego'];
}

export async function hasRadianceModel(name) {
  try {
    const dirUrl = new URL(`${encodeURIComponent(name)}/`, MODEL_BASE_URL);
    const resp = await fetch(new URL('radiance_manifest.json', dirUrl), { cache: 'no-store' });
    if (!resp.ok) return false;
    const manifest = await resp.json();
    return manifest.format === 'cells2voxels-radiance-v1' && manifest.radiance;
  } catch (_) {
    return false;
  }
}

export async function loadModel(name) {
  const dirUrl = new URL(`${encodeURIComponent(name)}/`, MODEL_BASE_URL);
  const [manifestResp, weightsResp] = await Promise.all([
    fetch(new URL('radiance_manifest.json', dirUrl), { cache: 'no-store' }),
    fetch(new URL('radiance_weights.bin', dirUrl), { cache: 'no-store' }),
  ]);
  if (!manifestResp.ok) throw new Error(`Missing radiance_manifest.json for ${name}`);
  if (!weightsResp.ok) throw new Error(`Missing radiance_weights.bin for ${name}`);

  const manifest = await manifestResp.json();
  const weightsBuf = await weightsResp.arrayBuffer();
  if (manifest.format !== 'cells2voxels-radiance-v1') {
    throw new Error(`Unsupported radiance format: ${manifest.format || 'unknown'}`);
  }

  const meta = manifest.meta || {};
  const renderer = manifest.renderer || {};
  const siren = manifest.siren || {};
  const channels = meta.channels;
  const gridSize = meta.grid_size || [meta.coarse_size, meta.coarse_size, meta.coarse_size];
  if (!gridSize || gridSize.length !== 3 || gridSize[0] !== gridSize[1] || gridSize[1] !== gridSize[2]) {
    throw new Error('The current WebGPU radiance demo expects a cubic grid.');
  }
  const coarseSize = gridSize[0];
  const livingChannel = meta.living_channel ?? 3;
  const livingThreshold = 0.45; // overridden from meta.living_threshold ?? 0.1
  const seedRadius = meta.seed_radius ?? 3;
  const updateProb = meta.update_prob ?? 0.5;
  const numKernels = meta.perception_kernels ?? manifest.perception?.['perceive.weight']?.shape?.[0] ?? 5;

  const weights = {};
  for (const section of ['perception', 'adaptation', 'radiance']) {
    weights[section] = {};
    for (const [key, info] of Object.entries(manifest[section] || {})) {
      weights[section][key] = {
        data: new Float32Array(weightsBuf, info.offset, info.count),
        shape: info.shape,
      };
    }
  }

  const fcDim = weights.adaptation['adapt.0.weight']?.shape?.[0];
  const l0 = pickTensor(weights.radiance, ['net.0.linear.weight', 'net.0.weight']);
  const l1 = pickTensor(weights.radiance, ['net.1.linear.weight', 'net.1.weight']);
  const l2 = pickTensor(weights.radiance, ['net.2.linear.weight', 'net.2.weight']);
  const l3 = pickTensor(weights.radiance, ['net.3.weight', 'net.3.linear.weight']);
  const l3b = pickTensor(weights.radiance, ['net.3.bias', 'net.3.linear.bias']);
  if (!l0 || !l1 || !l2 || !l3 || !l3b) {
    throw new Error('Radiance decoder must use two hidden SIREN layers plus one output layer.');
  }
  const sirenHiddenDim = l0.shape[0];
  const sirenInputDim = l0.shape[1];
  const sirenOutDim = l3.shape[0];
  if (sirenOutDim !== 4 || (renderer.sh_degree ?? 0) !== 0) {
    throw new Error('This demo supports sh_degree=0 radiance exports with 4 decoder outputs.');
  }

  return {
    name,
    manifest,
    meta,
    renderer,
    siren,
    weights,
    channels,
    coarseSize,
    gridSize,
    livingChannel,
    livingThreshold,
    seedRadius,
    updateProb,
    numKernels,
    fcDim,
    numFrequencies: siren.num_frequencies ?? 1,
    firstOmega: siren.first_omega_0 ?? 10,
    hiddenOmega: siren.hidden_omega_0 ?? 10,
    sirenHiddenDim,
    sirenInputDim,
    sirenOutDim,
    sirenLayers: siren.hidden_layers ?? 2,
    renderSamples: Math.max(16, Math.min(96, renderer.num_samples ?? 64)),
    densityFactor: renderer.density_factor ?? 1,
    colorFactor: renderer.color_factor ?? 1,
    backgroundColor: renderer.background_color ?? 0,
    applyLivingMask: renderer.apply_living_mask ?? true,
    voxelBounds: renderer.voxel_bounds ?? [-1.25, 1.25],
  };
}

function pickTensor(group, names) {
  for (const name of names) {
    if (group[name]) return group[name];
  }
  return null;
}
