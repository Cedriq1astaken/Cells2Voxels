const MODEL_BASE_URL = new URL('../model/', import.meta.url);

export async function loadModelIndex() {
  const resp = await fetch(new URL('index.json', MODEL_BASE_URL));
  return resp.json();
}

export async function loadModel(name) {
  const dirUrl = new URL(`${encodeURIComponent(name)}/`, MODEL_BASE_URL);
  const [manifest, weightsBuf] = await Promise.all([
    fetch(new URL('nca_manifest.json', dirUrl)).then(r => r.json()),
    fetch(new URL('nca_weights.bin', dirUrl)).then(r => r.arrayBuffer()),
  ]);

  const meta = manifest.meta;
  const channels = meta.channels;
  const coarseSize = meta.coarse_size;
  const targetSize = meta.target_size;
  const scale = meta.scale;
  const livingChannel = meta.living_channel ?? 3;
  const livingThreshold = meta.living_threshold ?? 0.1;
  const seedRadius = meta.seed_radius ?? 3;
  const maxOccupancy = meta.max_occupancy ?? 0.3;
  const numFrequencies = meta.num_frequencies ?? 1;
  const lppnFirstOmega = meta.lppn_first_omega_0 ?? 10.0;
  const lppnHiddenOmega = meta.lppn_hidden_omega_0 ?? 10.0;
  const targetPath = String(meta.target_path ?? '').toLowerCase();
  const rotateModel = targetPath.endsWith('.vox') || targetPath.endsWith('.npy');

  const weights = {};
  for (const section of ['perception', 'adaptation', 'lppn']) {
    weights[section] = {};
    for (const [key, info] of Object.entries(manifest[section])) {
      const byteOffset = info.offset;
      const floatCount = info.count;
      const shape = info.shape;
      weights[section][key] = {
        data: new Float32Array(weightsBuf, byteOffset, floatCount),
        shape,
      };
    }
  }

  const percW = weights.perception['perceive.weight'];
  const numKernels = percW.shape[0];
  const fcDim = weights.adaptation['adapt.0.weight'].shape[0];

  const lppnKeys = Object.keys(weights.lppn);
  const lppnWeightKeys = lppnKeys.filter(k => k.includes('weight'));
  const numLppnLayers = lppnWeightKeys.length;
  const lppnHiddenDim = weights.lppn['net.0.linear.weight']?.shape[0]
                     ?? weights.lppn['net.0.weight']?.shape[0]
                     ?? 32;
  const lppnInputDim = weights.lppn['net.0.linear.weight']?.shape[1]
                    ?? weights.lppn['net.0.weight']?.shape[1]
                    ?? (channels + 6);
  const lppnOutDim = 4;

  const modelImg = new URL(`${encodeURIComponent(name)}.png`, dirUrl).href;

  return {
    name, meta, channels, coarseSize, targetSize, scale,
    livingChannel, livingThreshold, seedRadius, maxOccupancy,
    numFrequencies, lppnFirstOmega, lppnHiddenOmega,
    rotateModel,
    numKernels, fcDim,
    numLppnLayers, lppnHiddenDim, lppnInputDim, lppnOutDim,
    weights,
    modelImg,
  };
}
