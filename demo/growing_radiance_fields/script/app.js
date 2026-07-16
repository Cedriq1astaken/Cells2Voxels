import { initGPU, configureCanvas } from './gpu.js?v=1';
import { loadModel, loadModelIndex, hasRadianceModel } from './model-loader.js?v=1';
import { NCACompute } from './nca.js?v=1';
import { RadianceRenderer } from './renderer.js?v=1';
import { OrbitCamera } from './camera.js?v=1';
import { Interaction } from './interaction.js?v=1';
import { UI } from './ui.js?v=1';

class App {
  constructor() {
    this.running = true;
    this.frameCount = 0;
    this.lastFpsTime = performance.now();
    this.fps = 0;
  }

  async init() {
    this.ui = new UI();
    this.ui.init();
    this.ui.setStatus('Loading radiance demo...');

    const { device, hasF16 } = await initGPU();
    this.device = device;
    this.ui.setF16(hasF16);
    this.hasTimestamps = device.features.has('timestamp-query');
    if (this.hasTimestamps) {
      this.querySet = device.createQuerySet({
        type: 'timestamp',
        count: 4,
      });
      this.resolveBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.readBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.profilingPending = false;
    }

    const canvas = document.getElementById('canvas');
    const { ctx, format } = configureCanvas(device, canvas);
    this.canvas = canvas;
    this.ctx = ctx;
    this.format = format;

    this.camera = new OrbitCamera();
    this.camera.distance = 3.1;
    this.camera.attach(canvas);

    this.shaders = {
      nca: await fetch(new URL('../shaders/nca.wgsl?v=1', import.meta.url)).then(r => r.text()),
      render: await fetch(new URL('../shaders/radiance.wgsl?v=1', import.meta.url)).then(r => r.text()),
      mask: await fetch(new URL('../shaders/mask.wgsl?v=1', import.meta.url)).then(r => r.text()),
      pack: await fetch(new URL('../shaders/pack.wgsl?v=1', import.meta.url)).then(r => r.text()),
    };

    const modelNames = await loadModelIndex();
    const validModels = [];
    for (const name of modelNames) {
      if (await hasRadianceModel(name)) validModels.push(name);
    }
    this.ui.populateModels(validModels);

    if (validModels.length) {
      await this.loadModel(validModels[0]);
      this.ui.setStatus('');
    } else {
      this.ui.setStatus('No radiance export found in demo/growing_radiance_fields/model');
    }

    this.ui.on('toggleRun', () => {
      this.running = !this.running;
      this.ui.setRunning(this.running);
    });
    this.ui.on('reset', () => this.nca?.initSeed());
    this.ui.on('modelChange', async (name) => {
      await this.loadModel(name);
      this.ui.setStatus('');
    });

    canvas.addEventListener('click', (e) => {
      if (!this.interaction || this.camera.consumeClick()) return;
      this.interaction.damageRadius = this.ui.damageRadius;
      this.interaction.handleClick(canvas, e);
    });

    this._resize();
    window.addEventListener('resize', () => this._resize());

    this.ui.setRunning(this.running);
    this._loop();
  }

  async loadModel(name) {
    this.ui.setStatus(`Loading ${name}...`);
    const model = await loadModel(name);
    this.model = model;
    this.nca = new NCACompute(this.device, model, this.shaders.nca);
    this.renderer = new RadianceRenderer(this.device, this.format, model, this.shaders.render, this.shaders.mask, this.shaders.pack);
    this.interaction = new Interaction(this.camera, this.nca, model.coarseSize);
    this.ui.setModelInfo(model);
    this.ui.setStatus('');
  }

  _resize() {
    const canvas = this.canvas;
    const parent = canvas.parentElement;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.0);
    canvas.width = Math.max(1, Math.floor(parent.clientWidth * dpr));
    canvas.height = Math.max(1, Math.floor(parent.clientHeight * dpr));
    canvas.style.width = parent.clientWidth + 'px';
    canvas.style.height = parent.clientHeight + 'px';
  }

  _loop() {
    requestAnimationFrame(() => this._loop());
    if (!this.nca || !this.renderer) return;

    const encoder = this.device.createCommandEncoder();

    let hasNcaTimestamps = false;
    let stepsThisFrame = 0;
    if (this.running) {
      stepsThisFrame = Math.max(1, Math.round(this.ui.simRate));
    }

    if (this.hasTimestamps && !this.profilingPending && this.running && stepsThisFrame > 0) {
      hasNcaTimestamps = true;
    }

    if (this.running) {
      for (let i = 0; i < stepsThisFrame; i++) {
        let tw = null;
        if (hasNcaTimestamps) {
          tw = { querySet: this.querySet };
          if (i === 0) tw.beginningOfPassWriteIndex = 0;
          if (i === stepsThisFrame - 1) tw.endOfPassWriteIndex = 1;
        }
        this.nca.encode(encoder, tw);
      }
    }

    let renderTimestampWrites = null;
    if (this.hasTimestamps && !this.profilingPending) {
      renderTimestampWrites = {
        querySet: this.querySet,
        begin: 2,
        end: 3
      };
    }

    const textureView = this.ctx.getCurrentTexture().createView();
    this.renderer.encode(encoder, textureView, this.nca.currentStateBuffer, this.camera, this.canvas, {
      density: this.ui.density,
      exposure: this.ui.exposure,
    }, renderTimestampWrites);

    if (this.hasTimestamps && !this.profilingPending) {
      encoder.resolveQuerySet(this.querySet, 0, 4, this.resolveBuffer, 0);
      encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, 32);
    }

    this.device.queue.submit([encoder.finish()]);

    if (this.hasTimestamps && !this.profilingPending) {
      this.profilingPending = true;
      this.readBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const times = new BigUint64Array(this.readBuffer.getMappedRange());
        const ncaTimeNs = hasNcaTimestamps ? (times[1] - times[0]) : 0n;
        const renderTimeNs = times[3] - times[2];
        const ncaMs = Number(ncaTimeNs) / 1000000;
        const renderMs = Number(renderTimeNs) / 1000000;
        this.ui.updateTimings(ncaMs, renderMs);
        this.readBuffer.unmap();
        this.profilingPending = false;
      }).catch(err => {
        console.error("Mapping error:", err);
        this.profilingPending = false;
      });
    }

    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 500) {
      this.fps = this.frameCount / ((now - this.lastFpsTime) / 1000);
      this.frameCount = 0;
      this.lastFpsTime = now;
    }
    this.ui.updateStats(this.fps, this.nca.step, this.renderer.samples, this.model.coarseSize);
  }
}

const app = new App();
app.init().catch(err => {
  document.body.innerHTML = `<div style="color:#f44;padding:40px;font-size:18px;">
    <h2>WebGPU Error</h2><p>${err.message}</p>
    <p>Make sure you're using Chrome 113+ or another WebGPU-capable browser.</p>
  </div>`;
  console.error(err);
});
