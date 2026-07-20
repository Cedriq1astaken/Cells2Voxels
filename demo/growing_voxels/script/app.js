import { initGPU, configureCanvas } from './gpu.js?v=4';
import { loadModel, loadModelIndex } from './model-loader.js?v=5';
import { NCACompute } from './nca.js?v=5';
import { LPPNCompute } from './lppn.js?v=6';
import { VoxelRenderer } from './renderer.js?v=11';
import { OrbitCamera } from './camera.js?v=10';
import { Interaction } from './interaction.js?v=8';
import { VoxelPicker } from './voxel-picker.js?v=2';
import { UI } from './ui.js?v=11';

const SIMULATION_HZ = 30;

class App {
  constructor() {
    this.running = true;
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
    this.fps = 0;
    this.voxelCount = 0;
    this.countPending = false;
    this.lastCountReadTime = -Infinity;
    this._countRequestId = 0;
    this._modelLoadId = 0;
    this._decodeDirty = true;
    this._volumeReady = false;
    this._damagePickPending = false;
    this._damagePreviewPending = false;
    this._damageRequestId = 0;
    this._lastLoopTime = performance.now();
    this._simulationAccumulator = 0;
    this._rebuildId = 0;
    this._rebuilding = false;
  }

  async init() {
    this.ui = new UI();
    this.ui.init();

    // Init WebGPU
    const { device, hasF16 } = await initGPU();
    this.device = device;
    this.hasF16 = hasF16;
    this.ui.setF16(hasF16);

    const canvas = document.getElementById('canvas');
    const { ctx, format } = configureCanvas(device, canvas);
    this.canvas = canvas;
    this.ctx = ctx;
    this.format = format;

    // Camera
    this.camera = new OrbitCamera();
    this.camera.attach(canvas);

    // Fetch shaders
    this.shaders = {
      nca: await fetch(new URL('../shaders/nca.wgsl?v=10', import.meta.url)).then(r => r.text()),
      damage: await fetch(new URL('../shaders/damage.wgsl?v=1', import.meta.url)).then(r => r.text()),
      lppn: await fetch(new URL('../shaders/lppn.wgsl?v=11', import.meta.url)).then(r => r.text()),
      pick: await fetch(new URL('../shaders/pick.wgsl?v=1', import.meta.url)).then(r => r.text()),
      livingMask: await fetch(new URL('../shaders/living_mask.wgsl?v=1', import.meta.url)).then(r => r.text()),
      compact: await fetch(new URL('../shaders/compact.wgsl?v=9', import.meta.url)).then(r => r.text()),
      render: await fetch(new URL('../shaders/render.wgsl?v=11', import.meta.url)).then(r => r.text()),
    };

    // Load models
    const modelNames = await loadModelIndex();
    // Filter to only models that have the standard architecture
    const validModels = [];
    for (const name of modelNames) {
      try {
        const resp = await fetch(new URL(`../model/${encodeURIComponent(name)}/nca_manifest.json`, import.meta.url));
        if (resp.ok) {
          const manifest = await resp.json();
          // Filter to strictly 4-layer LPPN
          if (manifest.lppn && manifest.lppn['net.3.weight'] && !manifest.lppn['net.4.weight']) {
            validModels.push(name);
          }
        }
      } catch (_) {}
    }
    this.ui.populateModels(validModels);

    // Load default model
    await this.loadModel(validModels[0]);

    // UI callbacks
    this.ui.on('toggleRun', () => {
      this.running = !this.running;
      this.ui.setRunning(this.running);
    });
    this.ui.on('reset', () => {
      this._cancelDamagePick();
      this.nca.initSeed();
      this.voxelCount = 0;
      this._decodeDirty = true;
      this._volumeReady = false;
    });
    this.ui.on('modelChange', async (name) => {
      await this.loadModel(name);
    });
    this.ui.on('lppnScaleChange', async (newScale) => {
      if (this.model && this.model.scale !== newScale) {
        const previousScale = this.model.scale;
        this.model.scale = newScale;
        try {
          await this.rebuildLPPN();
        } catch (error) {
          this.model.scale = previousScale;
          this.ui.setLppnScale(previousScale);
          console.error('LPPN scale rebuild failed:', error);
        }
      }
    });
    this.ui.on('crossChange', () => {
      this._decodeDirty = true;
      this._volumeReady = false;
    });

    // Click to damage the first currently visible voxel on the ray.
    canvas.addEventListener('click', (e) => this._handleDamageClick(e));

    // Handle resize
    this._resize();
    window.addEventListener('resize', () => this._resize());

    this.ui.setRunning(this.running);
    this._loop();
  }

  async loadModel(name) {
    this._cancelDamagePick();
    const loadId = ++this._modelLoadId;
    const model = await loadModel(name);
    if (loadId !== this._modelLoadId) return;

    this._cancelDamagePick();
    this.nca?.destroy();
    this.model = model;
    this.nca = new NCACompute(this.device, model, this.shaders.nca, this.shaders.damage, this.hasF16);
    this.ui.setModelInfo(model);
    this.ui.setLppnScale(model.scale);
    await this.rebuildLPPN();
  }

  async rebuildLPPN() {
    const rebuildId = ++this._rebuildId;
    const hadResources = Boolean(this.lppn || this.renderer || this.picker);
    const resumeAfterRebuild = this.running;
    this.running = false;
    this._simulationAccumulator = 0;
    this._rebuilding = true;
    this.ui.setRunning(false);
    this.ui.setLppnScaleBusy(true);

    try {
      // Pausing JavaScript does not drain already submitted GPU work. Stop frame
      // submission above, then wait before destroying buffers still in flight.
      if (hadResources) await this.device.queue.onSubmittedWorkDone();
      if (rebuildId !== this._rebuildId) return;

      this._cancelDamagePick();
      this._invalidateCountRead();
      this.picker?.destroy();
      this.renderer?.destroy();
      this.lppn?.destroy();
      this.picker = null;
      this.renderer = null;
      this.lppn = null;

      this.lppn = new LPPNCompute(this.device, this.model, this.shaders.lppn, this.shaders.livingMask, this.hasF16);
      const renderSize = this.lppn.renderSize;
      const warnsAboutPerformance = this.model.name === 'Frog' || this.model.name === 'Tomato';
      this.ui.setPerformanceWarning(warnsAboutPerformance
        ? `Performance warning: ${this.model.name} is GPU-intensive and may run slowly.`
        : '');
      this.renderer = new VoxelRenderer(this.device, this.format, renderSize, this.shaders.compact, this.shaders.render, this.model.livingThreshold, this.model.maxOccupancy, this.model.rotateModel);
      this.interaction = new Interaction(this.camera, this.nca, renderSize, this.model.rotateModel);
      this.picker = new VoxelPicker(this.device, renderSize, this.shaders.pick);

      this.ui.setCrossSectionMax(renderSize);
      this.voxelCount = 0;
      this.lastNcaStep = undefined;
      this.lastPerfMode = undefined; // Force perf mode sync
      this._decodeDirty = true;
      this._volumeReady = false;
      this._resize();

      this.running = renderSize < 200 && resumeAfterRebuild;
      this._lastLoopTime = performance.now();
      this.ui.setRunning(this.running);
    } finally {
      if (rebuildId === this._rebuildId) {
        this._rebuilding = false;
        this.ui.setLppnScaleBusy(false);
      }
    }
  }
  _invalidateCountRead() {
    this._countRequestId++;
    this.countPending = false;
    this.lastCountReadTime = -Infinity;
  }

  _cancelDamagePick() {
    this._damageRequestId++;
    this._damagePickPending = false;
    this._damagePreviewPending = false;
  }

  async _handleDamageClick(event) {
    if (this.camera.consumeClick() || this._damagePickPending || !this._volumeReady) return;

    const requestId = ++this._damageRequestId;
    this._damagePickPending = true;
    this.interaction.damageRadius = this.ui.damageRadius;
    this.interaction.modelRotation = this.ui.modelRotation;
    const picker = this.picker;
    const interaction = this.interaction;

    try {
      const ray = interaction.getRawRay(this.canvas, event);
      if (!ray) return;
      const fineVoxel = await picker.pick(
        this.lppn.outputBuf,
        ray.origin,
        ray.direction,
        this.renderer.livingThreshold,
      );
      if (requestId !== this._damageRequestId || picker !== this.picker || !fineVoxel) return;

      const damage = interaction.damageFineVoxel(fineVoxel);
      if (damage) {
        this._decodeDirty = true;
        // Reserve the next animation frame for the wound before evolution.
        this._damagePreviewPending = true;
      }
    } catch (error) {
      if (requestId === this._damageRequestId) console.warn('Voxel damage pick failed:', error);
    } finally {
      if (requestId === this._damageRequestId) this._damagePickPending = false;
    }
  }

  _resize() {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    canvas.width = Math.floor(parent.clientWidth * dpr);
    canvas.height = Math.floor(parent.clientHeight * dpr);
    canvas.style.width = parent.clientWidth + 'px';
    canvas.style.height = parent.clientHeight + 'px';
    if (this.renderer) {
      this.renderer.createDepthTexture(canvas.width, canvas.height);
    }
  }

  _loop() {
    requestAnimationFrame(() => this._loop());

    if (this._rebuilding || !this.nca || !this.lppn || !this.renderer) return;

    const now = performance.now();
    // Clamp long gaps so background tabs and GPU stalls do not enqueue a burst.
    const elapsedSeconds = Math.min((now - this._lastLoopTime) / 1000, 0.1);
    this._lastLoopTime = now;

    // Submit each NCA step separately. Its uniform and random buffers are
    // queue-written per step, so batching multiple steps in one command buffer
    // would make every step observe only the final write.
    const previewDamage = this._damagePreviewPending;
    if (this.running && !this._damagePickPending && !previewDamage) {
      const rate = this.ui.simRate;
      this._simulationAccumulator += elapsedSeconds * SIMULATION_HZ * rate;
      const stepsThisFrame = Math.floor(this._simulationAccumulator);
      this._simulationAccumulator -= stepsThisFrame;
      for (let i = 0; i < stepsThisFrame; i++) {
        const ncaEncoder = this.device.createCommandEncoder();
        this.nca.encode(ncaEncoder);
        this.device.queue.submit([ncaEncoder.finish()]);
      }
    } else {
      this._simulationAccumulator = 0;
    }

    const encoder = this.device.createCommandEncoder();
    const ncaStep = this.nca.step;
    const perfMode = this.ui.perfMode;

    let needsDecode = this._decodeDirty;
    if (this.lastNcaStep !== ncaStep) {
      needsDecode = true;
      this.lastNcaStep = ncaStep;
    }
    if (this.lastPerfMode !== perfMode) {
      needsDecode = true;
      this.lastPerfMode = perfMode;
      this.renderer.livingThreshold = perfMode ? 0.45 : this.model.livingThreshold;
    }

    // Decode only when the state or a render control changed. Cross-section
    // invalidation is event-driven, rather than comparing fresh arrays per frame.
    if (needsDecode) {
      this.lppn.encode(encoder, this.nca.currentStateBuffer, this.ui.crossSection);
      this.renderer.encodeCompact(encoder, this.lppn.outputBuf);
      this._decodeDirty = false;
      this._volumeReady = true;
    }

    const textureView = this.ctx.getCurrentTexture().createView();
    const depthView = this.renderer.depthTexture.createView();
    const aspect = this.canvas.width / this.canvas.height;
    const mvp = this.camera.getMVP(aspect);
    const camPos = this.camera.getPosition();
    const lightDir = [0.5, 0.8, 0.6];

    const modelRotation = this.ui.modelRotation;
    this.renderer.modelRotation = modelRotation;
    this.interaction.modelRotation = modelRotation;

    this.renderer.encodeRender(encoder, textureView, depthView, mvp, camPos, lightDir);

    this.device.queue.submit([encoder.finish()]);
    if (previewDamage) this._damagePreviewPending = false;

    // Statistics do not need a per-frame GPU map. The full count also drives
    // overflow-safe growth of the packed instance buffer.
    const countInterval = this.renderer.countReadIntervalHintMs ?? 250;
    if (!this.countPending && now - this.lastCountReadTime >= countInterval) {
      const renderer = this.renderer;
      const requestId = ++this._countRequestId;
      this.countPending = true;
      this.lastCountReadTime = now;
      renderer.readVoxelCount().then(count => {
        if (requestId !== this._countRequestId || renderer !== this.renderer) return;
        this.voxelCount = count;
        if (renderer.ensureInstanceCapacity(count)) {
          this._decodeDirty = true;
          this._volumeReady = false;
        }
      }).catch(error => {
        if (requestId === this._countRequestId) console.warn('Voxel count readback failed:', error);
      }).finally(() => {
        if (requestId === this._countRequestId) this.countPending = false;
      });
    }

    // FPS
    this.frameCount++;
    if (now - this.lastFpsTime >= 500) {
      this.fps = this.frameCount / ((now - this.lastFpsTime) / 1000);
      this.frameCount = 0;
      this.lastFpsTime = now;
    }

    const rs = this.lppn.renderSize;
    this.ui.updateStats(this.fps, this.nca.step, this.voxelCount, rs * rs * rs);
  }
}

const app = new App();
app.init().catch(err => {
  document.body.innerHTML = `<div class="fatal-error">
    <h2>WebGPU Error</h2><p>${err.message}</p>
    <p>Make sure you're using Chrome 113+ or another WebGPU-capable browser.</p>
  </div>`;
  console.error(err);
});

