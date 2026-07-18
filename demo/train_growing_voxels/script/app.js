import { OrbitCamera } from './camera.js?v=18';
import { exportTrainerModel } from './exporter.js?v=19';
import { configureCanvas, initGPU } from './gpu.js?v=18';
import { VoxelRenderer } from './renderer.js?v=19';
import { VoxelTrainer } from './trainer.js?v=20';
import { parseObjAndVol, parseVox } from './voxelizer.js?v=18';

const ui = Object.fromEntries([
  'canvas', 'left-panel', 'loading-overlay', 'loading-text', 'status-banner',
  'file-input', 'vol-input', 'vol-upload-container', 'upload-status', 'vol-status', 'target-name',
  'btn-start', 'btn-stop', 'btn-reset', 'btn-export', 'btn-controls',
  'stat-ips', 'stat-loss', 'stat-iter', 'stat-voxels', 'stat-memory', 'model-shape',
  'cfg-res', 'cfg-native-res', 'val-res', 'cfg-scale', 'cfg-channels', 'val-channels',
  'cfg-width', 'cfg-lr', 'val-lr', 'cfg-step-min', 'cfg-step-max', 'val-steps',
  'cfg-seed-radius', 'val-seed-radius',
  'right-panel', 'loss-chart', 'loss-current', 'loss-min', 'loss-max',
].map(id => [camelCase(id), document.getElementById(id)]));

let device = null;
let context = null;
let format = null;
let shaders = null;
let camera = null;
let renderer = null;
let trainer = null;
let trainerDirty = true;
let parsedTarget = null;
let renderRequested = true;
const lossHistory = [];
const MAX_LOSS_POINTS = 180;

const uploaded = { vox: null, obj: null, vol: null };

async function initialize() {
  setupUI();
  try {
    setStatus('Starting WebGPU f16 training');
    const gpu = await initGPU();
    device = gpu.device;
    ({ context, format } = configureCanvas(device, ui.canvas));
    shaders = await loadShaders();
    camera = new OrbitCamera(requestRender);
    camera.attach(ui.canvas);
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    requestAnimationFrame(renderLoop);
    setStatus('Ready - WebGPU shader-f16', 'ready');
  } catch (error) {
    console.error(error);
    setStatus(error.message, 'error');
  }
}

function setupUI() {
  document.querySelectorAll('[data-file-for]').forEach(button => {
    button.addEventListener('click', () => document.getElementById(button.dataset.fileFor).click());
  });

  ui.btnControls.addEventListener('click', () => {
    const collapsed = ui.leftPanel.classList.toggle('is-collapsed');
    ui.btnControls.classList.toggle('is-active', !collapsed);
    ui.btnControls.setAttribute('aria-expanded', String(!collapsed));
  });

  ui.cfgRes.addEventListener('input', () => {
    ui.valRes.textContent = ui.cfgRes.value;
    ui.cfgNativeRes.checked = false;
  });
  ui.cfgChannels.addEventListener('input', () => {
    ui.valChannels.textContent = ui.cfgChannels.value;
    updateMemoryEstimate();
  });
  ui.cfgLr.addEventListener('input', () => {
    ui.valLr.textContent = Number(ui.cfgLr.value).toFixed(4);
  });
  ui.cfgSeedRadius.addEventListener('input', () => {
    ui.valSeedRadius.textContent = ui.cfgSeedRadius.value;
  });
  const updateSteps = () => {
    const low = Math.min(Number(ui.cfgStepMin.value), Number(ui.cfgStepMax.value));
    const high = Math.max(Number(ui.cfgStepMin.value), Number(ui.cfgStepMax.value));
    ui.valSteps.textContent = `${low}-${high}`;
    updateMemoryEstimate();
  };
  ui.cfgStepMin.addEventListener('input', updateSteps);
  ui.cfgStepMax.addEventListener('input', updateSteps);

  for (const control of [ui.cfgChannels, ui.cfgWidth, ui.cfgLr, ui.cfgStepMin, ui.cfgStepMax, ui.cfgSeedRadius]) {
    control.addEventListener('change', invalidateTrainer);
  }
  for (const control of [ui.cfgRes, ui.cfgNativeRes, ui.cfgScale]) {
    control.addEventListener('change', async () => {
      await invalidateTrainer();
      if (uploaded.vox || (uploaded.obj && uploaded.vol)) await processTarget();
    });
  }

  ui.fileInput.addEventListener('change', handleTargetFile);
  ui.volInput.addEventListener('change', handleVolumeFile);
  ui.btnStart.addEventListener('click', startTraining);
  ui.btnStop.addEventListener('click', stopTraining);
  ui.btnReset.addEventListener('click', resetTraining);
  ui.btnExport.addEventListener('click', exportModel);
}

async function handleTargetFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const lowerName = file.name.toLowerCase();
  if (lowerName.endsWith('.vox')) {
    uploaded.vox = file;
    uploaded.obj = null;
    uploaded.vol = null;
    ui.volUploadContainer.classList.add('hidden');
    ui.uploadStatus.textContent = file.name;
    await processTarget();
  } else if (lowerName.endsWith('.obj')) {
    uploaded.obj = file;
    uploaded.vox = null;
    uploaded.vol = null;
    ui.volUploadContainer.classList.remove('hidden');
    ui.uploadStatus.textContent = file.name;
    ui.volStatus.textContent = 'Select the matching VOL texture';
    await clearTarget();
    setStatus('OBJ selected - waiting for its VOL texture');
  } else {
    setStatus('Choose a VOX or OBJ target.', 'error');
  }
}

async function handleVolumeFile(event) {
  const file = event.target.files?.[0];
  if (!file || !uploaded.obj) return;
  if (!file.name.toLowerCase().endsWith('.vol')) {
    setStatus('Choose a VOL solid texture.', 'error');
    return;
  }
  uploaded.vol = file;
  ui.volStatus.textContent = file.name;
  await processTarget();
}

async function processTarget() {
  if (!device) {
    setStatus('WebGPU is still initializing.');
    return;
  }
  showLoading(uploaded.vox ? 'Reading VOX target' : 'Voxelizing OBJ mesh');
  await invalidateTrainer();
  await nextFrame();

  try {
    const scaleFactor = Number(ui.cfgScale.value);
    const resolution = ui.cfgNativeRes.checked && uploaded.vox ? 'native' : Number(ui.cfgRes.value);
    let parsed;
    let sourceFile;
    if (uploaded.vox) {
      sourceFile = uploaded.vox;
      parsed = await parseVox(await uploaded.vox.arrayBuffer(), { resolution, scaleFactor });
    } else if (uploaded.obj && uploaded.vol) {
      sourceFile = uploaded.obj;
      parsed = await parseObjAndVol(
        await uploaded.obj.text(),
        await uploaded.vol.arrayBuffer(),
        { resolution: Number(ui.cfgRes.value), scaleFactor },
      );
    } else {
      return;
    }

    const active = countActive(parsed.target);
    if (active === 0) throw new Error('The target contains no occupied voxels.');
    parsedTarget = {
      ...parsed,
      name: stripExtension(sourceFile.name),
      sourceName: sourceFile.name,
    };
    ui.targetName.textContent = parsedTarget.name;
    ui.uploadStatus.textContent = `${sourceFile.name} - ${active.toLocaleString()} voxels`;
    ui.valRes.textContent = ui.cfgNativeRes.checked && uploaded.vox
      ? `${parsed.resolution} NATIVE`
      : String(parsed.resolution);
    updateModelShape();
    updateMemoryEstimate();
    createRenderer(parsed.resolution);
    renderer.updateVoxelData(parsed.target);
    requestRender();
    resetStats();
    trainerDirty = true;
    setTrainingControls(false);
    setStatus(`Target ready - ${parsed.resolution} cubed`, 'ready');
  } catch (error) {
    console.error(error);
    await clearTarget();
    setStatus(error.message, 'error');
  } finally {
    hideLoading();
  }
}

function createRenderer(size) {
  renderer?.destroy();
  renderer = new VoxelRenderer(
    device,
    format,
    size,
    shaders.compact,
    shaders.render,
    count => { ui.statVoxels.textContent = count.toLocaleString(); },
  );
  renderer.resize(ui.canvas.width, ui.canvas.height);
}

async function ensureTrainer() {
  if (!parsedTarget) throw new Error('Load a target first.');
  if (trainer && !trainerDirty) return trainer;
  if (trainer) await trainer.dispose();

  const config = readConfig();
  validateTrainingConfig(config);

  trainer = new VoxelTrainer(device, config, parsedTarget.target, shaders, {
    onStep: stats => {
      ui.statIter.textContent = stats.iteration.toLocaleString();
      ui.statLoss.textContent = formatLoss(stats.loss);
      recordLoss(stats.iteration, stats.loss);
      ui.statIps.textContent = stats.iterationsPerSecond.toFixed(2);
      if (stats.preview) {
        renderer.updateVoxelBuffer(stats.preview.buffer, stats.preview.byteLength);
        requestRender();
      }
      ui.btnExport.disabled = stats.iteration < 1 || trainer.running;
    },
    onRunningChange: running => setTrainingControls(running),
    onError: error => {
      console.error(error);
      setStatus(`Training failed: ${error.message}`, 'error');
    },
  });
  trainerDirty = false;
  ui.modelShape.textContent = `${config.coarseSize} CUBED / POOL ${trainer.poolSize}`;
  return trainer;
}

async function startTraining() {
  try {
    const activeTrainer = await ensureTrainer();
    setStatus(`Training on GPU - preview every ${activeTrainer.previewInterval} iterations`, 'ready');
    activeTrainer.start().catch(error => console.error('Training loop stopped:', error));
  } catch (error) {
    setStatus(error.message, 'error');
  }
}

async function stopTraining() {
  if (!trainer) return;
  await trainer.stop();
  setStatus(`Paused at iteration ${trainer.iteration.toLocaleString()}`);
}

async function resetTraining() {
  if (!parsedTarget) return;
  showLoading('Resetting model');
  try {
    if (trainer) await trainer.dispose();
    trainer = null;
    trainerDirty = true;
    renderer.updateVoxelData(parsedTarget.target);
    requestRender();
    resetStats();
    setTrainingControls(false);
    setStatus('Model reset - target preserved', 'ready');
  } finally {
    hideLoading();
  }
}

async function exportModel() {
  if (!trainer || trainer.iteration < 1) return;
  await trainer.stop();
  showLoading('Packing model export');
  try {
    const result = await exportTrainerModel(trainer, {
      name: parsedTarget.name,
      sourceName: parsedTarget.sourceName,
      targetData: parsedTarget.target,
      onProgress: percent => { ui.loadingText.textContent = `Packing export ${Math.round(percent)}%`; },
    });
    setStatus(`Exported ${result.name} - ${(result.byteLength / 1024).toFixed(0)} KiB`, 'ready');
  } catch (error) {
    console.error(error);
    setStatus(`Export failed: ${error.message}`, 'error');
  } finally {
    hideLoading();
    setTrainingControls(false);
  }
}

function readConfig() {
  const stepMin = Math.min(Number(ui.cfgStepMin.value), Number(ui.cfgStepMax.value));
  const stepMax = Math.max(Number(ui.cfgStepMin.value), Number(ui.cfgStepMax.value));
  const targetSize = parsedTarget.resolution;
  const scale = Number(ui.cfgScale.value);
  if (targetSize % scale !== 0) throw new Error(`Resolution ${targetSize} must be divisible by LPPN scale ${scale}.`);
  return {
    targetSize,
    coarseSize: targetSize / scale,
    scale,
    channels: Number(ui.cfgChannels.value),
    fcDim: Number(ui.cfgWidth.value),
    learningRate: Number(ui.cfgLr.value),
    stepMin,
    stepMax: stepMax === stepMin ? stepMax + 1 : stepMax,
    seedRadius: Number(ui.cfgSeedRadius.value),
    updateProbability: 0.5,
    overflowWeight: 100,
  };
}

function estimateTrainingResources(config) {
  const scalarBytes = Uint16Array.BYTES_PER_ELEMENT;
  const alignBufferBytes = bytes => Math.max(8, Math.ceil(bytes / 8) * 8);
  const f16BufferBytes = count => alignBufferBytes(count * scalarBytes);
  const f32BufferBytes = count => alignBufferBytes(count * Float32Array.BYTES_PER_ELEMENT);
  const u32BufferBytes = count => alignBufferBytes(count * Uint32Array.BYTES_PER_ELEMENT);
  const cells = config.coarseSize ** 3;
  const stateElements = cells * config.channels;
  const totalVoxels = config.targetSize ** 3;
  const rows = Math.min(totalVoxels, 8192);
  const inputDim = config.channels + 6;
  const lppnWidth = 64;
  const kernelCount = 5;
  const rawStateBytes = stateElements * scalarBytes;
  const stateBytes = f16BufferBytes(stateElements);
  const stateTapeBytes = f16BufferBytes(stateElements * (config.stepMax + 1));
  const hiddenTapeBytes = f16BufferBytes(cells * config.fcDim * config.stepMax);
  const livingTapeBytes = f16BufferBytes(cells * config.stepMax);
  const targetBytes = f16BufferBytes(totalVoxels * 4);
  const sampleBytes = u32BufferBytes(totalVoxels);
  const previewBytes = targetBytes;
  const lppnInputBytes = f16BufferBytes(rows * inputDim);
  const lppnHiddenBytes = f16BufferBytes(rows * lppnWidth);
  const dNcaHiddenBytes = f16BufferBytes(cells * config.fcDim);
  const dPerceptionBytes = f16BufferBytes(cells * config.channels * kernelCount);
  const poolStateBytes = Math.max(4, Math.ceil(rawStateBytes / 4) * 4);
  const poolSize = Math.max(8, Math.min(64, Math.floor((64 * 1024 ** 2) / rawStateBytes)));
  const poolBytes = alignBufferBytes(poolSize * poolStateBytes);

  const parameterCounts = [
    config.channels * kernelCount * config.fcDim,
    config.fcDim,
    config.fcDim * config.channels,
    inputDim * lppnWidth,
    lppnWidth,
    lppnWidth ** 2,
    lppnWidth,
    lppnWidth ** 2,
    lppnWidth,
    lppnWidth * 4,
    4,
  ];
  const largestParameterBytes = f32BufferBytes(Math.max(...parameterCounts));
  const previewChunks = Math.ceil(totalVoxels / rows);
  const parameterArenaBytes = (config.stepMax * 10 + previewChunks * 6 + 66) * 256;
  const poolRecoveryBytes = alignBufferBytes(4);

  const storageBuffers = {
    'state history': stateTapeBytes,
    'hidden history': hiddenTapeBytes,
    'living-mask history': livingTapeBytes,
    target: targetBytes,
    'sample indices': sampleBytes,
    preview: previewBytes,
    'LPPN input': lppnInputBytes,
    'LPPN hidden activation': lppnHiddenBytes,
    'NCA hidden gradient': dNcaHiddenBytes,
    'NCA perception gradient': dPerceptionBytes,
    'optimizer master/moment': largestParameterBytes,
  };

  const parameterBytes = parameterCounts.reduce(
    (total, count) => total + f16BufferBytes(count) * 2 + f32BufferBytes(count) * 3,
    0,
  );
  const lppnWorkspaceBytes = lppnInputBytes * 2
    + lppnHiddenBytes * 9
    + f16BufferBytes(rows * 4) * 2
    + f16BufferBytes(rows) * 2
    + alignBufferBytes(4) + alignBufferBytes(8);
  const ncaWorkspaceBytes = stateBytes * 6 + dNcaHiddenBytes + dPerceptionBytes;
  const kernelBytes = f16BufferBytes(kernelCount * 27);
  const dummyBytes = alignBufferBytes(4) * 8;
  const trainerGpuBytes = stateTapeBytes + hiddenTapeBytes + livingTapeBytes
    + targetBytes + sampleBytes + previewBytes + poolBytes + poolRecoveryBytes + kernelBytes + dummyBytes
    + lppnWorkspaceBytes + ncaWorkspaceBytes + parameterBytes + parameterArenaBytes;

  const rendererVoxelBytes = targetBytes;
  const rendererInstanceBytes = f16BufferBytes(totalVoxels * 8);
  const rendererAuxiliaryBytes = alignBufferBytes(36 * 6 * Float32Array.BYTES_PER_ELEMENT)
    + alignBufferBytes(4) * 2 + alignBufferBytes(16) + alignBufferBytes(128) + alignBufferBytes(16);
  const rendererDepthBytes = Math.max(1, ui.canvas?.width ?? 1) * Math.max(1, ui.canvas?.height ?? 1) * 4;
  const rendererGpuBytes = rendererVoxelBytes + rendererInstanceBytes + rendererAuxiliaryBytes + rendererDepthBytes;
  const estimatedGpuBytes = trainerGpuBytes + rendererGpuBytes;

  storageBuffers['renderer voxels'] = rendererVoxelBytes;
  storageBuffers['renderer instances'] = rendererInstanceBytes;
  const [largestStorageName, largestStorageBytes] = Object.entries(storageBuffers)
    .reduce((largest, entry) => entry[1] > largest[1] ? entry : largest);
  const largestCreatedBytes = Math.max(
    largestStorageBytes,
    poolBytes,
    parameterArenaBytes,
    rendererVoxelBytes,
    rendererInstanceBytes,
  );
  const largestWorkItems = Math.max(
    cells * config.fcDim,
    cells * config.channels * kernelCount,
    stateElements,
    rows * Math.max(inputDim, lppnWidth),
    ...parameterCounts,
  );

  return {
    estimatedGpuBytes,
    recommendedGpuBytes: Math.ceil(estimatedGpuBytes * 1.25),
    rendererGpuBytes,
    trainerGpuBytes,
    largestCreatedBytes,
    largestStorageBytes,
    largestStorageName,
    largestWorkItems,
  };
}

function validateTrainingConfig(config) {
  const resources = estimateTrainingResources(config);
  const {
    estimatedGpuBytes,
    largestCreatedBytes,
    largestStorageBytes,
    largestStorageName,
    largestWorkItems,
  } = resources;
  if (largestStorageBytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error(
      'The ' + largestStorageName + ' buffer needs ' + formatMiB(largestStorageBytes) + ', '
      + "above this device's " + formatMiB(device.limits.maxStorageBufferBindingSize)
      + ' storage-buffer limit. Increase LPPN scale or reduce resolution, width, or rollout.',
    );
  }

  if (largestCreatedBytes > device.limits.maxBufferSize) {
    throw new Error(
      'These settings require a ' + formatMiB(largestCreatedBytes) + ' GPU buffer, '
      + "above this device's " + formatMiB(device.limits.maxBufferSize) + ' buffer limit.'
    );
  }

  const dispatchCapacity = device.limits.maxComputeWorkgroupsPerDimension * 64;
  if (largestWorkItems > dispatchCapacity) {
    throw new Error(
      'These settings exceed the WebGPU dispatch limit. Increase LPPN scale or reduce resolution, channels, or width.',
    );
  }

  if (estimatedGpuBytes > 768 * 1024 ** 2) {
    throw new Error(
      'These mixed-precision settings need about ' + formatMiB(estimatedGpuBytes) + ' of GPU memory. '
      + 'Increase LPPN scale or reduce resolution, width, channels, or rollout.',
    );
  }
}

function formatMiB(bytes) {
  return `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

async function invalidateTrainer() {
  trainerDirty = true;
  if (trainer) {
    await trainer.dispose();
    trainer = null;
  }
  resetStats();
  updateModelShape();
  updateMemoryEstimate();
  setTrainingControls(false);
}

async function clearTarget() {
  await invalidateTrainer();
  parsedTarget = null;
  renderer?.destroy();
  renderer = null;
  ui.targetName.textContent = 'NO TARGET';
  ui.modelShape.textContent = '-';
  updateMemoryEstimate();
  setTrainingControls(false);
  requestRender();
}

function updateModelShape() {
  if (!parsedTarget) return;
  const scale = Number(ui.cfgScale.value);
  ui.modelShape.textContent = `${parsedTarget.resolution / scale} CUBED TO ${parsedTarget.resolution} CUBED`;
}

function updateMemoryEstimate() {
  if (!parsedTarget) {
    ui.statMemory.textContent = '-';
    ui.statMemory.removeAttribute('title');
    return;
  }
  try {
    const resources = estimateTrainingResources(readConfig());
    ui.statMemory.textContent = formatMiB(resources.estimatedGpuBytes);
    ui.statMemory.title = formatMiB(resources.trainerGpuBytes) + ' training buffers + '
      + formatMiB(resources.rendererGpuBytes) + ' preview/render buffers. '
      + 'Plan for ' + formatMiB(resources.recommendedGpuBytes) + ' with 25% headroom; '
      + 'browser and GPU-driver overhead are additional.';
  } catch (error) {
    ui.statMemory.textContent = 'CHECK CFG';
    ui.statMemory.title = error.message;
  }
}

function setTrainingControls(running) {
  const hasTarget = Boolean(parsedTarget);
  ui.btnStart.disabled = running || !hasTarget;
  ui.btnStop.disabled = !running;
  ui.btnReset.disabled = running || !hasTarget;
  ui.btnExport.disabled = running || !trainer || trainer.iteration < 1;
  ui.btnStart.classList.toggle('is-primary', !running);
  ui.btnStop.classList.toggle('is-primary', running);
  for (const control of [
    ui.fileInput, ui.volInput, ui.cfgRes, ui.cfgNativeRes, ui.cfgScale,
    ui.cfgChannels, ui.cfgWidth, ui.cfgLr, ui.cfgStepMin, ui.cfgStepMax, ui.cfgSeedRadius,
  ]) {
    control.disabled = running;
  }
}

function resetStats() {
  ui.statIps.textContent = '-';
  ui.statLoss.textContent = '-';
  ui.statIter.textContent = '0';
  ui.statVoxels.textContent = parsedTarget ? countActive(parsedTarget.target).toLocaleString() : '0';
  lossHistory.length = 0;
  drawLossChart();
}

function recordLoss(iteration, loss) {
  if (!Number.isFinite(loss)) return;
  lossHistory.push({ iteration, loss });
  if (lossHistory.length > MAX_LOSS_POINTS) lossHistory.shift();
  ui.lossCurrent.textContent = formatLoss(loss);
  drawLossChart();
}

function drawLossChart() {
  const canvas = ui.lossChart;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext('2d');
  if (!context) return;
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, width, height);
  context.scale(dpr, dpr);
  const cssWidth = width / dpr;
  const cssHeight = height / dpr;
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--chart-grid').trim();
  context.lineWidth = 1;
  for (let row = 1; row < 4; row++) {
    const y = row * cssHeight / 4;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(cssWidth, y);
    context.stroke();
  }
  if (!lossHistory.length) {
    ui.lossMin.textContent = '-';
    ui.lossMax.textContent = '-';
    context.setTransform(1, 0, 0, 1, 0, 0);
    return;
  }
  const values = lossHistory.map(point => Math.max(1e-8, point.loss));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const low = Math.log10(min);
  const high = Math.log10(Math.max(max, min * 1.01));
  const span = Math.max(0.001, high - low);
  context.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--teal').trim();
  context.lineWidth = 2;
  context.beginPath();
  values.forEach((value, index) => {
    const x = values.length === 1 ? 0 : index * cssWidth / (values.length - 1);
    const y = cssHeight - (Math.log10(value) - low) / span * cssHeight;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  ui.lossMin.textContent = formatLoss(min);
  ui.lossMax.textContent = formatLoss(max);
  context.setTransform(1, 0, 0, 1, 0, 0);
}

function resizeCanvas() {
  if (!context) return;
  const parent = ui.canvas.parentElement;
  const dpr = Math.min(window.devicePixelRatio || 1, 1);
  const width = Math.max(1, Math.floor(parent.clientWidth * dpr));
  const height = Math.max(1, Math.floor(parent.clientHeight * dpr));
  if (ui.canvas.width !== width || ui.canvas.height !== height) {
    ui.canvas.width = width;
    ui.canvas.height = height;
    renderer?.resize(width, height);
    updateMemoryEstimate();
    requestRender();
  }
}

function requestRender() {
  renderRequested = true;
}

function renderLoop() {
  if (renderRequested && renderer && context && camera) {
    renderer.render(context, camera);
    renderRequested = false;
  }
  requestAnimationFrame(renderLoop);
}

async function loadShaders() {
  const files = {
    compact: 'compact.wgsl',
    render: 'render.wgsl',
    ncaForward: 'nca_forward_f16.wgsl',
    dense: 'dense_f16.wgsl',
    denseBackward: 'dense_backward_f16.wgsl',
    loss: 'loss_f16.wgsl',
    ncaBackward: 'nca_backward_f16.wgsl',
    optimizer: 'optimizer_f16.wgsl',
    poolRecovery: 'pool_recovery_f16.wgsl',
  };
  const entries = await Promise.all(Object.entries(files).map(async ([key, file]) => [
    key,
    await fetch('./shaders/' + file + '?v=20').then(response => {
      if (!response.ok) throw new Error('Could not load shader ' + file + '.');
      return response.text();
    }),
  ]));
  return Object.fromEntries(entries);
}

function showLoading(message) {
  ui.loadingText.textContent = message;
  ui.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
  ui.loadingOverlay.classList.add('hidden');
}

function setStatus(message, type = '') {
  ui.statusBanner.textContent = message;
  ui.statusBanner.classList.toggle('is-error', type === 'error');
  ui.statusBanner.classList.toggle('is-ready', type === 'ready');
}

function countActive(values) {
  let count = 0;
  for (let index = 3; index < values.length; index += 4) {
    if (values[index] > 0.1) count++;
  }
  return count;
}

function formatLoss(value) {
  if (!Number.isFinite(value)) return 'INVALID';
  return value < 0.001 ? value.toExponential(2) : value.toFixed(5);
}

function stripExtension(filename) {
  return filename.replace(/\.[^.]+$/, '');
}

function camelCase(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function nextFrame() {
  return new Promise(resolve => requestAnimationFrame(resolve));
}

initialize();
