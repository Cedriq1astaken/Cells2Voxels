import { initGPU, configureCanvas } from './gpu.js?v=4';
import { loadModel, loadModelIndex } from './model-loader.js?v=4';
import { NCACompute } from './nca.js?v=3';
import { LPPNCompute } from './lppn.js?v=5';
import { VoxelRenderer } from './renderer.js?v=8';
import { OrbitCamera } from './camera.js?v=6';
import { Interaction } from './interaction.js?v=7';
import { UI } from './ui.js?v=8';

class App {
  constructor() {
    this.running = true;
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
    this.fps = 0;
    this.voxelCount = 0;
    this.countPending = false;
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
      lppn: await fetch(new URL('../shaders/lppn.wgsl?v=10', import.meta.url)).then(r => r.text()),
      livingMask: await fetch(new URL('../shaders/living_mask.wgsl?v=1', import.meta.url)).then(r => r.text()),
      compact: await fetch(new URL('../shaders/compact.wgsl?v=6', import.meta.url)).then(r => r.text()),
      render: await fetch(new URL('../shaders/render.wgsl?v=7', import.meta.url)).then(r => r.text()),
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
      this.nca.initSeed();
      this.voxelCount = 0;
    });
    this.ui.on('modelChange', async (name) => {
      await this.loadModel(name);
    });
    this.ui.on('lppnScaleChange', (newScale) => {
      if (this.model && this.model.scale !== newScale) {
        this.model.scale = newScale;
        this.rebuildLPPN();
      }
    });

    // Click to damage
    canvas.addEventListener('click', (e) => {
      if (this.camera.consumeClick()) return;
      this.interaction.damageRadius = this.ui.damageRadius;
      this.interaction.handleClick(canvas, e);
    });

    // Handle resize
    this._resize();
    window.addEventListener('resize', () => this._resize());

    this.ui.setRunning(this.running);
    this._loop();
  }

  async loadModel(name) {
    const model = await loadModel(name);
    this.model = model;

    this.nca = new NCACompute(this.device, model, this.shaders.nca, this.hasF16);
    this.ui.setModelInfo(model);
    this.ui.setLppnScale(model.scale);
    
    this.rebuildLPPN();
  }

  rebuildLPPN() {
    if (this.lppn) {
      // Rebuilding, might need to cleanup old buffers if not GC'd, but GC should handle it
    }
    this.lppn = new LPPNCompute(this.device, this.model, this.shaders.lppn, this.shaders.livingMask, this.hasF16);
    const renderSize = this.lppn.renderSize;
    this.renderer = new VoxelRenderer(this.device, this.format, renderSize, this.shaders.compact, this.shaders.render, this.model.livingThreshold, this.model.maxOccupancy, this.model.rotateModel);
    this.interaction = new Interaction(this.camera, this.nca, renderSize, this.model.rotateModel);

    this.ui.setCrossSectionMax(renderSize);
    this.voxelCount = 0;
    this.lastPerfMode = undefined; // Force perf mode sync
    this._resize();
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

    if (!this.nca || !this.lppn || !this.renderer) return;

    // Submit each NCA step separately. Its uniform and random buffers are
    // queue-written per step, so batching multiple steps in one command buffer
    // would make every step observe only the final write.
    if (this.running) {
      const rate = this.ui.simRate;
      this.stepAccumulator = (this.stepAccumulator || 0) + rate;
      const stepsThisFrame = Math.floor(this.stepAccumulator);
      this.stepAccumulator -= stepsThisFrame;
      for (let i = 0; i < stepsThisFrame; i++) {
        const ncaEncoder = this.device.createCommandEncoder();
        this.nca.encode(ncaEncoder);
        this.device.queue.submit([ncaEncoder.finish()]);
      }
    }

    const encoder = this.device.createCommandEncoder();
    const crossSection = this.ui.crossSection;
    const ncaStep = this.nca.step;
    const perfMode = this.ui.perfMode;

    let needsDecode = false;
    if (this.lastNcaStep !== ncaStep) {
      needsDecode = true;
      this.lastNcaStep = ncaStep;
    }
    if (this.lastCrossSection !== crossSection) {
      needsDecode = true;
      this.lastCrossSection = crossSection;
    }
    if (this.lastPerfMode !== perfMode) {
      needsDecode = true;
      this.lastPerfMode = perfMode;
      this.renderer.livingThreshold = perfMode ? 0.45 : this.model.livingThreshold;
    }

    // Only run LPPN decode and Compact passes if the state actually changed
    if (needsDecode) {
      this.lppn.encode(encoder, this.nca.currentStateBuffer, crossSection);
      this.renderer.encodeCompact(encoder, this.lppn.outputBuf);
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

    // Read voxel count asynchronously for stats display only (not used for rendering)
    if (!this.countPending) {
      this.countPending = true;
      this.renderer.readVoxelCount().then(count => {
        this.voxelCount = count;
        this.countPending = false;
      });
    }

    // FPS
    this.frameCount++;
    const now = performance.now();
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

