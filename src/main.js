import Input from './compute/input.js';
import Marcher from './render/marcher.js';
import Renderer from './render/renderer.js';
import Volume from './compute/volume.js';

const ORIENTATION_STORAGE_KEY = 'nca-preview-orientation-v4';
const SCALE_STORAGE_KEY = 'nca-scale-multiplier-v1';
const NCA_SPEED_STORAGE_KEY = 'nca-update-rate-v1';
const LPPN_SPEED_STORAGE_KEY = 'lppn-update-rate-v1';
const MODEL_STORAGE_KEY = 'nca-model-name-v1';
const MODEL_NAME = 'Globe';
const SCALE_MULTIPLIER = null;
const UPDATE_RATE = 1.0;
const LPPN_RATE = 1.0;
const CLICK_DAMAGE_DRAG_THRESHOLD = 6;
const VOXEL_COUNT_REFRESH_SECONDS = 0.6;
const MODEL_ROOT_URL = new URL('../model/', import.meta.url);
const MODEL_INDEX_URL = new URL('./index.json', MODEL_ROOT_URL);

const loadOrientation = () => {
  try {
    const stored = window.localStorage.getItem(ORIENTATION_STORAGE_KEY);
    if (!stored) {
      return undefined;
    }
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed.axisMap) || !Array.isArray(parsed.axisFlip)) {
      return undefined;
    }
    return {
      axisMap: parsed.axisMap,
      axisFlip: parsed.axisFlip,
    };
  } catch {
    return undefined;
  }
};

const saveOrientation = (volume) => {
  window.localStorage.setItem(
    ORIENTATION_STORAGE_KEY,
    JSON.stringify(volume.getOrientationState())
  );
};

const loadScaleMultiplier = () => {
  const urlScale = new URLSearchParams(window.location.search).get('scale');
  if (urlScale) {
    const parsed = Number(urlScale);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }

  try {
    const stored = window.localStorage.getItem(SCALE_STORAGE_KEY);
    if (!stored) {
      return SCALE_MULTIPLIER ?? undefined;
    }
    const parsed = Number(stored);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  } catch {
    return SCALE_MULTIPLIER ?? undefined;
  }

  return SCALE_MULTIPLIER ?? undefined;
};

const saveScaleMultiplier = (value) => {
  window.localStorage.setItem(SCALE_STORAGE_KEY, String(value));
};

const loadUpdateRate = () => {
  const urlSpeed = new URLSearchParams(window.location.search).get('speed');
  if (urlSpeed) {
    const parsed = Number(urlSpeed);
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 1.5) {
      return parsed;
    }
  }

  try {
    const stored = window.localStorage.getItem(NCA_SPEED_STORAGE_KEY);
    if (!stored) {
      return UPDATE_RATE;
    }
    const parsed = Number(stored);
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 1.5) {
      return parsed;
    }
  } catch {
    return UPDATE_RATE;
  }

  return UPDATE_RATE;
};

const saveUpdateRate = (value) => {
  window.localStorage.setItem(NCA_SPEED_STORAGE_KEY, String(value));
};

const loadLppnRate = () => {
  const urlSpeed = new URLSearchParams(window.location.search).get('lppnSpeed');
  if (urlSpeed) {
    const parsed = Number(urlSpeed);
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 1.0) {
      return parsed;
    }
  }

  try {
    const stored = window.localStorage.getItem(LPPN_SPEED_STORAGE_KEY);
    if (!stored) {
      return LPPN_RATE;
    }
    const parsed = Number(stored);
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 1.0) {
      return parsed;
    }
  } catch {
    return LPPN_RATE;
  }

  return LPPN_RATE;
};

const saveLppnRate = (value) => {
  window.localStorage.setItem(LPPN_SPEED_STORAGE_KEY, String(value));
};

const saveModelName = (value) => {
  window.localStorage.setItem(MODEL_STORAGE_KEY, value);
};

const updateModelQueryParam = (value) => {
  const url = new URL(window.location.href);
  url.searchParams.set('model', value);
  window.history.replaceState({}, '', url);
};

const probeModelAsset = async (url) => {
  const response = await fetch(url, { method: 'GET' });
  return response.ok;
};

const discoverModelsFromIndex = async () => {
  const response = await fetch(MODEL_INDEX_URL);
  if (!response.ok) {
    return null;
  }

  const payload = await response.json();
  if (!Array.isArray(payload)) {
    throw new Error(`Invalid model index format at ${MODEL_INDEX_URL}`);
  }

  const uniqueNames = [...new Set(payload
    .filter((entry) => typeof entry === 'string')
    .map((entry) => entry.trim())
    .filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));

  if (!uniqueNames.length) {
    return null;
  }

  return uniqueNames;
};

const discoverModelsFromDirectoryListing = async () => {
  const response = await fetch(MODEL_ROOT_URL);
  if (!response.ok) {
    return null;
  }

  const html = await response.text();
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const directoryNames = Array.from(doc.querySelectorAll('a'))
    .map((anchor) => anchor.getAttribute('href') ?? '')
    .filter((href) => href && href !== '../' && href.endsWith('/'))
    .map((href) => {
      const resolved = new URL(href, MODEL_ROOT_URL);
      const relativePath = resolved.pathname.slice(MODEL_ROOT_URL.pathname.length).replace(/\/$/, '');
      return decodeURIComponent(relativePath);
    })
    .filter((name) => name && !name.includes('/'));

  const uniqueNames = [...new Set(directoryNames)].sort((left, right) => left.localeCompare(right));
  const validated = await Promise.all(uniqueNames.map(async (name) => {
    const manifestUrl = new URL(`./${encodeURIComponent(name)}/nca_manifest.json`, MODEL_ROOT_URL);
    const weightsUrl = new URL(`./${encodeURIComponent(name)}/nca_weights.bin`, MODEL_ROOT_URL);
    const [hasManifest, hasWeights] = await Promise.all([
      probeModelAsset(manifestUrl),
      probeModelAsset(weightsUrl),
    ]);
    return hasManifest && hasWeights ? name : null;
  }));

  const availableModels = validated.filter((name) => typeof name === 'string');
  return availableModels.length ? availableModels : null;
};

const discoverAvailableModels = async () => {
  const indexedModels = await discoverModelsFromIndex();
  if (indexedModels?.length) {
    return indexedModels;
  }

  const listedModels = await discoverModelsFromDirectoryListing();
  if (listedModels?.length) {
    return listedModels;
  }

  throw new Error(`No valid model folders found in ${MODEL_ROOT_URL}. Add model/index.json or enable directory listing on the server.`);
};

const loadModelName = (availableModels) => {
  const model = new URLSearchParams(window.location.search).get('model');
  const stored = model || window.localStorage.getItem(MODEL_STORAGE_KEY) || MODEL_NAME;

  const normalized = stored.trim().toLowerCase();
  const resolved = availableModels.find((candidate) => candidate.toLowerCase() === normalized);
  if (!resolved) {
    throw new Error(`Unknown model "${stored}". Available models: ${availableModels.join(', ')}`);
  }

  return resolved;
};

const resetVolume = (volume) => {
  volume.reset();
};

const formatVoxelCount = (count) => new Intl.NumberFormat('en-US').format(count);
const clampUnifiedRate = (value) => Math.min(1.5, Math.max(0, Number(value)));
const sumManifestCounts = (section = {}) => Object.values(section).reduce(
  (total, entry) => total + (typeof entry?.count === 'number' ? entry.count : 0),
  0
);
const findManifestEntry = (section = {}, matcher) => Object.entries(section).find(([name]) => matcher(name))?.[1];

const buildInfoSections = (manifest, renderInfo) => {
  const perceptionOut = manifest.perception?.['perceive.weight']?.shape?.[0] ?? '-';
  const adaptWidth = manifest.adaptation?.['adapt.0.bias']?.count ?? '-';
  const ncaParams = sumManifestCounts(manifest.perception) + sumManifestCounts(manifest.adaptation);
  const lppnInput = findManifestEntry(manifest.lppn, (name) => name.endsWith('net.0.linear.weight'))?.shape?.[1] ?? '-';
  const lppnWidth = findManifestEntry(manifest.lppn, (name) => name.endsWith('net.0.linear.weight'))?.shape?.[0] ?? '-';
  const lppnOutput = findManifestEntry(manifest.lppn, (name) => name.endsWith('net.6.bias'))?.shape?.[0] ?? 4;
  const lppnParams = sumManifestCounts(manifest.lppn);

  return [
    {
      name: 'NCA',
      rows: [
        ['Domain', '3D voxel grid'],
        ['Coarse grid', `${manifest.meta.coarse_size}^3`],
        ['Target grid', `${manifest.meta.target_size}^3`],
        ['Current render', `${renderInfo.renderSize}^3`],
        ['Channels', `${manifest.meta.channels}`],
        ['Perception out', `${perceptionOut}`],
        ['MLP width', `${adaptWidth}`],
        ['#Params', formatVoxelCount(ncaParams)],
      ],
    },
    {
      name: 'LPPN',
      rows: [
        ['Input dim', `${lppnInput}`],
        ['Output dim (K)', `${lppnOutput}`],
        ['MLP width', `${lppnWidth}`],
        ['Scale', `${manifest.meta.scale}x`],
        ['#Params', formatVoxelCount(lppnParams)],
      ],
    },
  ];
};

const renderInfoTable = (sections) => {
  const rows = sections.map((section) => section.rows.map(([label, value], index) => `
      <tr class="${index === 0 ? 'info-section-start' : ''}">
        ${index === 0 ? `<th class="info-section-cell" rowspan="${section.rows.length}">${section.name}</th>` : ''}
        <th scope="row">${label}</th>
        <td>${value}</td>
      </tr>
    `).join('')).join('');

  return `
    <table class="info-table">
      <thead>
        <tr>
          <th></th>
          <th>Setting</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
};

const setPreviewImage = (image, placeholder, url, missingLabel) => {
  image.onload = () => {
    image.classList.add('is-visible');
    placeholder.classList.add('is-hidden');
  };
  image.onerror = () => {
    image.classList.remove('is-visible');
    placeholder.classList.remove('is-hidden');
    placeholder.innerText = missingLabel;
  };
  image.classList.remove('is-visible');
  placeholder.classList.remove('is-hidden');
  placeholder.innerText = missingLabel;
  image.src = url;
};

const run = async (device) => {
  const dom = document.getElementById('app');
  if (!dom) {
    throw new Error("Couldn't get app DOM node");
  }
  const renderer = await Renderer.create(device);
  const canvas = renderer.getCanvas();
  const input = new Input(canvas);
  const availableModels = await discoverAvailableModels();
  let modelName = loadModelName(availableModels);
  let volume = await Volume.create(
    renderer.getDevice(),
    `../../model/${modelName}`,
    loadOrientation(),
    loadScaleMultiplier()
  );
  const fpsDom = document.getElementById('fps');
  const stepCountDom = document.getElementById('step-count');
  const voxelCountDom = document.getElementById('voxel-count');
  const infoToggleButton = document.getElementById('info-toggle');
  const togglePlayButton = document.getElementById('toggle-play');
  const stepButton = document.getElementById('step-button');
  const controlsToggleButton = document.getElementById('controls-toggle');
  const controlsPanel = document.getElementById('controls-panel');
  const infoPanel = document.getElementById('info-panel');
  const infoCloseButton = document.getElementById('info-close');
  const infoTitleDom = document.getElementById('info-title');
  const infoOriginalImage = document.getElementById('info-original-image');
  const infoOriginalPlaceholder = document.getElementById('info-original-placeholder');
  const infoCoarseImage = document.getElementById('info-coarse-image');
  const infoCoarsePlaceholder = document.getElementById('info-coarse-placeholder');
  const infoTableWrap = document.getElementById('info-table-wrap');
  const modelSelect = document.getElementById('model-select');
  const infoModelNameDom = document.getElementById('info-model-name');
  const scaleRange = document.getElementById('scale-range');
  const scaleValueDom = document.getElementById('scale-value');
  const damageRadiusRange = document.getElementById('damage-radius-range');
  const damageRadiusValueDom = document.getElementById('damage-radius-value');
  const rateRange = document.getElementById('rate-range');
  const rateValueDom = document.getElementById('rate-value');
  const sectionXRange = document.getElementById('section-x-range');
  const sectionXValueDom = document.getElementById('section-x-value');
  const sectionYRange = document.getElementById('section-y-range');
  const sectionYValueDom = document.getElementById('section-y-value');
  const sectionZRange = document.getElementById('section-z-range');
  const sectionZValueDom = document.getElementById('section-z-value');
  if (!(fpsDom instanceof HTMLElement)
    || !(stepCountDom instanceof HTMLElement)
    || !(voxelCountDom instanceof HTMLElement)
    || !(infoToggleButton instanceof HTMLButtonElement)
    || !(togglePlayButton instanceof HTMLButtonElement)
    || !(stepButton instanceof HTMLButtonElement)
    || !(controlsToggleButton instanceof HTMLButtonElement)
    || !(controlsPanel instanceof HTMLElement)
    || !(infoPanel instanceof HTMLElement)
    || !(infoCloseButton instanceof HTMLButtonElement)
    || !(infoTitleDom instanceof HTMLElement)
    || !(infoOriginalImage instanceof HTMLImageElement)
    || !(infoOriginalPlaceholder instanceof HTMLElement)
    || !(infoCoarseImage instanceof HTMLImageElement)
    || !(infoCoarsePlaceholder instanceof HTMLElement)
    || !(infoTableWrap instanceof HTMLElement)
    || !(modelSelect instanceof HTMLSelectElement)
    || !(infoModelNameDom instanceof HTMLElement)
    || !(scaleRange instanceof HTMLInputElement)
    || !(scaleValueDom instanceof HTMLElement)
    || !(damageRadiusRange instanceof HTMLInputElement)
    || !(damageRadiusValueDom instanceof HTMLElement)
    || !(rateRange instanceof HTMLInputElement)
    || !(rateValueDom instanceof HTMLElement)
    || !(sectionXRange instanceof HTMLInputElement)
    || !(sectionXValueDom instanceof HTMLElement)
    || !(sectionYRange instanceof HTMLInputElement)
    || !(sectionYValueDom instanceof HTMLElement)
    || !(sectionZRange instanceof HTMLInputElement)
    || !(sectionZValueDom instanceof HTMLElement)) {
    throw new Error('UI elements are missing from the page.');
  }
  modelSelect.innerHTML = '';
  for (const availableModel of availableModels) {
    const option = document.createElement('option');
    option.value = availableModel;
    option.textContent = availableModel;
    modelSelect.appendChild(option);
  }
  modelSelect.value = modelName;
  let renderInfo = volume.getRenderInfo();
  input.setRenderSize(renderInfo.renderSize, { preserveZoom: false, immediate: true });
  resetVolume(volume);
  let marcher = await Marcher.create(renderer.getDevice(), renderer.getCamera().getBuffer(), volume);
  renderer.add(marcher);
  {
    const view = input.getView();
    renderer.getCamera().setOrbit(view.phi, view.theta, view.radius);
  }
  let isPlaying = true;
  let stepOnce = false;
  let fpsAccumulator = 0;
  let fpsFrames = 0;
  let displayedFps = 0;
  const initialUnifiedRate = clampUnifiedRate(loadUpdateRate());
  let updateRate = initialUnifiedRate;
  let updateAccumulator = 0;
  let lppnRate = initialUnifiedRate;
  let lppnAccumulator = 0;
  let forceRaster = true;
  let isScaling = false;
  let isSwitchingModel = false;
  let controlsOpen = true;
  let infoOpen = false;
  let infoPanelKey = '';
  let damageRadius = volume.getDefaultDamageRadius();
  let crossSection = volume.getCrossSectionState();
  let visibleVoxelCount = 0;
  let voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
  let voxelCountPending = false;
  const pointerState = {
    active: false,
    moved: false,
    x: 0,
    y: 0,
  };
  const syncUnifiedRate = (value, persist = true) => {
    const nextRate = clampUnifiedRate(value);
    updateRate = nextRate;
    lppnRate = nextRate;
    updateAccumulator = 0;
    lppnAccumulator = 0;
    rateRange.value = nextRate.toFixed(2);
    rateValueDom.innerText = `${nextRate.toFixed(2)}x`;
    if (persist) {
      saveUpdateRate(nextRate);
      saveLppnRate(nextRate);
    }
  };
  const syncInfoPanel = () => {
    const manifest = volume.getManifest();
    const nextInfoKey = `${modelName}:${renderInfo.renderSize}:${renderInfo.scaleMultiplier}`;
    infoTitleDom.innerText = modelName;
    infoPanel.classList.toggle('is-open', infoOpen);
    infoPanel.setAttribute('aria-hidden', String(!infoOpen));
    infoToggleButton.classList.toggle('is-active', infoOpen);
    infoToggleButton.setAttribute('aria-expanded', String(infoOpen));
    if (infoPanelKey === nextInfoKey) {
      return;
    }
    infoPanelKey = nextInfoKey;
    setPreviewImage(
      infoOriginalImage,
      infoOriginalPlaceholder,
      new URL(`./${encodeURIComponent(modelName)}/original.png`, MODEL_ROOT_URL).href,
      'Target preview unavailable'
    );
    setPreviewImage(
      infoCoarseImage,
      infoCoarsePlaceholder,
      new URL(`./${encodeURIComponent(modelName)}/coarse.png`, MODEL_ROOT_URL).href,
      'Coarse preview unavailable'
    );
    infoTableWrap.innerHTML = renderInfoTable(buildInfoSections(manifest, renderInfo));
  };
  const syncCrossSectionControls = () => {
    const maxValue = String(renderInfo.renderSize);
    sectionXRange.max = maxValue;
    sectionYRange.max = maxValue;
    sectionZRange.max = maxValue;
    sectionXRange.value = String(crossSection.x);
    sectionYRange.value = String(crossSection.y);
    sectionZRange.value = String(crossSection.z);
    sectionXValueDom.innerText = String(crossSection.x);
    sectionYValueDom.innerText = String(crossSection.y);
    sectionZValueDom.innerText = String(crossSection.z);
  };
  const syncControlLabels = () => {
    infoModelNameDom.innerText = modelName;
    scaleRange.value = renderInfo.scaleMultiplier.toFixed(2);
    scaleValueDom.innerText = `${renderInfo.scaleMultiplier.toFixed(2)}x`;
    damageRadiusRange.value = damageRadius.toFixed(1);
    damageRadiusValueDom.innerText = damageRadius.toFixed(1);
    rateRange.value = updateRate.toFixed(2);
    rateValueDom.innerText = `${updateRate.toFixed(2)}x`;
    syncCrossSectionControls();
    voxelCountDom.innerText = formatVoxelCount(visibleVoxelCount);
  };
  const updateStatus = () => {
    stepCountDom.innerText = String(volume.getTick());
    voxelCountDom.innerText = formatVoxelCount(visibleVoxelCount);
    togglePlayButton.innerText = isPlaying ? 'STOP' : 'START';
    togglePlayButton.classList.toggle('is-active', !isPlaying);
    controlsPanel.classList.toggle('is-collapsed', !controlsOpen);
    controlsToggleButton.classList.toggle('is-active', controlsOpen);
    controlsToggleButton.setAttribute('aria-expanded', String(controlsOpen));
    syncControlLabels();
    syncInfoPanel();
  };
  const refreshVoxelCount = async () => {
    if (voxelCountPending) {
      return;
    }
    voxelCountPending = true;
    const activeVolume = volume;
    try {
      const count = await activeVolume.countVisibleVoxels();
      if (activeVolume !== volume) {
        return;
      }
      visibleVoxelCount = count;
      voxelCountDom.innerText = formatVoxelCount(count);
      updateStatus();
    } finally {
      voxelCountPending = false;
    }
  };
  const damageFromPointer = async (event) => {
    const rect = canvas.getBoundingClientRect();
    const depth = await renderer.readDataDepthAtCanvasPoint(event.clientX, event.clientY);
    if (depth == null) {
      return;
    }
    const ray = renderer.getCamera().getPickRay(event.clientX, event.clientY, rect);
    const hit = [
      ray.origin[0] + ray.direction[0] * depth,
      ray.origin[1] + ray.direction[1] * depth,
      ray.origin[2] + ray.direction[2] * depth,
    ];
    volume.damageAtWorldPosition(hit, damageRadius);
    forceRaster = true;
    voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
    updateStatus(`Damaged ${modelName}`);
  };
  const rebuildVolume = async (nextModelName, nextScale, preserveState = true) => {
    if (isScaling || isSwitchingModel) {
      return;
    }
    isScaling = true;
    isSwitchingModel = true;
    saveScaleMultiplier(nextScale);
    saveModelName(nextModelName);
    updateModelQueryParam(nextModelName);
    try {
      const nextVolume = await Volume.create(
        renderer.getDevice(),
        `../../model/${nextModelName}`,
        volume.getOrientationState(),
        nextScale
      );
      if (preserveState && nextVolume.seedState.byteLength === volume.seedState.byteLength) {
        nextVolume.copySimulationStateFrom(volume);
      } else {
        resetVolume(nextVolume);
      }
      const nextMarcher = await Marcher.create(
        renderer.getDevice(),
        renderer.getCamera().getBuffer(),
        nextVolume
      );
      renderer.remove(marcher);
      renderer.add(nextMarcher);
      const previousRenderSize = renderInfo.renderSize;
      const previousCrossSection = crossSection;
      modelName = nextModelName;
      volume = nextVolume;
      marcher = nextMarcher;
      renderInfo = volume.getRenderInfo();
      damageRadius = Math.max(1, Math.min(24, damageRadius));
      const nextRenderSize = renderInfo.renderSize;
      crossSection = {
        x: Math.max(0, Math.min(nextRenderSize, Math.round((previousCrossSection.x / Math.max(previousRenderSize, 1)) * nextRenderSize))),
        y: Math.max(0, Math.min(nextRenderSize, Math.round((previousCrossSection.y / Math.max(previousRenderSize, 1)) * nextRenderSize))),
        z: Math.max(0, Math.min(nextRenderSize, Math.round((previousCrossSection.z / Math.max(previousRenderSize, 1)) * nextRenderSize))),
      };
      volume.setCrossSection('x', crossSection.x);
      volume.setCrossSection('y', crossSection.y);
      volume.setCrossSection('z', crossSection.z);
      visibleVoxelCount = 0;
      voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
      input.setRenderSize(renderInfo.renderSize, {
        preserveZoom: true,
        immediate: previousRenderSize === renderInfo.renderSize,
      });
      modelSelect.value = modelName;
      updateAccumulator = 0;
      lppnAccumulator = 0;
      forceRaster = true;
      updateStatus();
    } finally {
      isScaling = false;
      isSwitchingModel = false;
    }
  };
  const rebuildForScale = async (nextScale) => rebuildVolume(modelName, nextScale, true);
  modelSelect.addEventListener('change', () => {
    const nextModelName = modelSelect.value;
    if (!availableModels.includes(nextModelName) || nextModelName === modelName) {
      return;
    }
    void rebuildVolume(nextModelName, renderInfo.scaleMultiplier, false);
  });
  let scaleRebuildTimeout = 0;
  scaleRange.addEventListener('input', () => {
    scaleValueDom.innerText = `${Number(scaleRange.value).toFixed(2)}x`;
    window.clearTimeout(scaleRebuildTimeout);
    scaleRebuildTimeout = window.setTimeout(() => {
      const nextScale = Number(scaleRange.value);
      if (!Number.isFinite(nextScale) || Math.abs(nextScale - renderInfo.scaleMultiplier) < 0.001) {
        return;
      }
      void rebuildForScale(nextScale);
    }, 120);
  });
  damageRadiusRange.addEventListener('input', () => {
    damageRadius = Number(damageRadiusRange.value);
    damageRadiusValueDom.innerText = damageRadius.toFixed(1);
    updateStatus();
  });
  rateRange.addEventListener('input', () => {
    syncUnifiedRate(rateRange.value);
    updateStatus();
  });
  sectionXRange.addEventListener('input', () => {
    crossSection.x = Number(sectionXRange.value);
    volume.setCrossSection('x', crossSection.x);
    voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
    forceRaster = true;
    updateStatus();
  });
  sectionYRange.addEventListener('input', () => {
    crossSection.y = Number(sectionYRange.value);
    volume.setCrossSection('y', crossSection.y);
    voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
    forceRaster = true;
    updateStatus();
  });
  sectionZRange.addEventListener('input', () => {
    crossSection.z = Number(sectionZRange.value);
    volume.setCrossSection('z', crossSection.z);
    voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
    forceRaster = true;
    updateStatus();
  });
  infoToggleButton.addEventListener('click', () => {
    infoOpen = !infoOpen;
    updateStatus();
  });
  infoCloseButton.addEventListener('click', () => {
    infoOpen = false;
    updateStatus();
  });
  togglePlayButton.addEventListener('click', () => {
    isPlaying = !isPlaying;
    updateStatus();
  });
  stepButton.addEventListener('click', () => {
    stepOnce = true;
    updateStatus();
  });
  controlsToggleButton.addEventListener('click', () => {
    controlsOpen = !controlsOpen;
    updateStatus();
  });
  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) {
      return;
    }
    pointerState.active = true;
    pointerState.moved = false;
    pointerState.x = event.clientX;
    pointerState.y = event.clientY;
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!pointerState.active || pointerState.moved) {
      return;
    }
    const distance = Math.hypot(event.clientX - pointerState.x, event.clientY - pointerState.y);
    if (distance > CLICK_DAMAGE_DRAG_THRESHOLD) {
      pointerState.moved = true;
    }
  });
  canvas.addEventListener('pointerup', (event) => {
    const isClick = pointerState.active && !pointerState.moved && event.button === 0;
    pointerState.active = false;
    pointerState.moved = false;
    if (!isClick) {
      return;
    }
    void damageFromPointer(event);
  });
  canvas.addEventListener('pointercancel', () => {
    pointerState.active = false;
    pointerState.moved = false;
  });
  syncUnifiedRate(initialUnifiedRate, false);
  updateStatus();
  void refreshVoxelCount();
  window.addEventListener('keydown', (event) => {
    let label = null;
    if (event.key === ' ') {
      event.preventDefault();
      isPlaying = !isPlaying;
      updateStatus();
      return;
    }
    if (event.key === 'n' || event.key === 'N') {
      stepOnce = true;
      updateStatus();
      return;
    }
    if (event.key === '0') {
      resetVolume(volume);
      updateAccumulator = 0;
      lppnAccumulator = 0;
      forceRaster = true;
      voxelCountAccumulator = VOXEL_COUNT_REFRESH_SECONDS;
      updateStatus(`Reset ${modelName}`);
      return;
    }
    if (event.key === '-' || event.key === '_') {
      const nextScale = Math.max(0.25, Number((renderInfo.scaleMultiplier - 0.25).toFixed(4)));
      void rebuildForScale(nextScale);
      return;
    }
    if (event.key === '=' || event.key === '+') {
      const nextScale = Number((renderInfo.scaleMultiplier + 0.25).toFixed(4));
      void rebuildForScale(nextScale);
      return;
    }
    if (event.key === '[' || event.key === '{') {
      syncUnifiedRate(Number((updateRate - 0.25).toFixed(2)));
      updateStatus();
      return;
    }
    if (event.key === ']' || event.key === '}') {
      syncUnifiedRate(Number((updateRate + 0.25).toFixed(2)));
      updateStatus();
      return;
    }
    if (event.key === ',' || event.key === '<') {
      syncUnifiedRate(Number((updateRate - 0.25).toFixed(2)));
      updateStatus();
      return;
    }
    if (event.key === '.' || event.key === '>') {
      syncUnifiedRate(Number((updateRate + 0.25).toFixed(2)));
      updateStatus();
      return;
    }
    if (event.key === 'i' || event.key === 'I') {
      label = volume.rotateQuarterTurn('x', 1);
    } else if (event.key === 'k' || event.key === 'K') {
      label = volume.rotateQuarterTurn('x', -1);
    } else if (event.key === 'j' || event.key === 'J') {
      label = volume.rotateQuarterTurn('y', 1);
    } else if (event.key === 'l' || event.key === 'L') {
      label = volume.rotateQuarterTurn('y', -1);
    } else if (event.key === 'u' || event.key === 'U') {
      label = volume.rotateQuarterTurn('z', 1);
    } else if (event.key === 'o' || event.key === 'O') {
      label = volume.rotateQuarterTurn('z', -1);
    } else if (event.key === 'r' || event.key === 'R') {
      label = volume.cycleOrientation();
    } else if (event.key === 'x' || event.key === 'X') {
      label = volume.toggleFlip(0);
    } else if (event.key === 'y' || event.key === 'Y') {
      label = volume.toggleFlip(1);
    } else if (event.key === 'z' || event.key === 'Z') {
      label = volume.toggleFlip(2);
    }
    if (label) {
      saveOrientation(volume);
      forceRaster = true;
      updateStatus(label);
    }
  });
  renderer.setAnimationLoop((command, delta) => {
    input.update(delta);
    const view = input.getView();
    renderer.getCamera().setOrbit(view.phi, view.theta, view.radius);
    fpsAccumulator += delta;
    fpsFrames += 1;
    if (fpsAccumulator >= 0.25) {
      displayedFps = fpsFrames / fpsAccumulator;
      fpsAccumulator = 0;
      fpsFrames = 0;
      fpsDom.innerText = displayedFps.toFixed(0);
    }
    if (stepOnce) {
      volume.stepSimulationCount(command, 1);
      forceRaster = true;
      stepOnce = false;
    } else if (isPlaying) {
      updateAccumulator += updateRate;
      const stepsThisFrame = Math.floor(updateAccumulator);
      updateAccumulator -= stepsThisFrame;
      if (stepsThisFrame > 0) {
        volume.stepSimulationCount(command, stepsThisFrame);
      }
    }

    let shouldRaster = forceRaster;
    if (!shouldRaster && isPlaying) {
      lppnAccumulator += lppnRate;
      if (lppnAccumulator >= 1.0) {
        lppnAccumulator -= Math.floor(lppnAccumulator);
        shouldRaster = true;
      }
    }

    if (shouldRaster) {
      volume.rasterize(command);
      forceRaster = false;
    }

    stepCountDom.innerText = String(volume.getTick());
    voxelCountAccumulator += delta;
    if ((shouldRaster || forceRaster) && voxelCountAccumulator >= VOXEL_COUNT_REFRESH_SECONDS) {
      voxelCountAccumulator = 0;
      void refreshVoxelCount();
    }
  });
  dom.appendChild(canvas);
};

const createDevice = async () => {
  if (!navigator.gpu) {
    throw new Error("Couldn't load WebGPU");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Couldn't load WebGPU adapter");
  }
  const device = await adapter.requestDevice();
  if (!device) {
    throw new Error("Couldn't load WebGPU device");
  }
  return device;
};

createDevice()
  .then(run)
  .catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
  });
