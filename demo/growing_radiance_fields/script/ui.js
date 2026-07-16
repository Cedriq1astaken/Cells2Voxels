export class UI {
  constructor() {
    this.callbacks = {};
    this.els = {};
  }

  init() {
    this.els = {
      modelSelect: document.getElementById('model-select'),
      btnStartStop: document.getElementById('btn-start-stop'),
      btnReset: document.getElementById('btn-reset'),
      btnControls: document.getElementById('btn-controls'),
      leftPanel: document.getElementById('left-panel'),
      valModel: document.getElementById('val-model'),
      sliderSimRate: document.getElementById('slider-sim-rate'),
      sliderDamageR: document.getElementById('slider-damage-radius'),
      sliderDensity: document.getElementById('slider-density'),
      sliderExposure: document.getElementById('slider-exposure'),
      valSimRate: document.getElementById('val-sim-rate'),
      valDamageR: document.getElementById('val-damage-radius'),
      valDensity: document.getElementById('val-density'),
      valExposure: document.getElementById('val-exposure'),
      statFps: document.getElementById('stat-fps'),
      statStep: document.getElementById('stat-step'),
      statSamples: document.getElementById('stat-samples'),
      statF16: document.getElementById('stat-f16'),
      statGpuTime: document.getElementById('stat-gpu-time'),
      statusBanner: document.getElementById('status-banner'),
      infoPanel: document.getElementById('info-panel'),
      infoCh: document.getElementById('info-channels'),
      infoGrid: document.getElementById('info-grid'),
      infoSamples: document.getElementById('info-samples'),
      infoDecoder: document.getElementById('info-decoder'),
      infoGPU: document.getElementById('info-gpu'),
      btnInfo: document.getElementById('btn-info'),
      btnCloseInfo: document.getElementById('btn-close-info'),
    };

    this.els.btnStartStop?.addEventListener('click', () => this._fire('toggleRun'));
    this.els.btnReset?.addEventListener('click', () => this._fire('reset'));
    this.els.modelSelect?.addEventListener('change', () => {
      this._syncModelLabel();
      this._fire('modelChange', this.els.modelSelect.value);
    });

    this.els.btnInfo?.addEventListener('click', () => {
      this.els.infoPanel?.classList.remove('hidden');
      this.els.infoPanel?.setAttribute('aria-hidden', 'false');
    });
    this.els.btnCloseInfo?.addEventListener('click', () => {
      this.els.infoPanel?.classList.add('hidden');
      this.els.infoPanel?.setAttribute('aria-hidden', 'true');
    });

    this.els.btnControls?.addEventListener('click', () => {
      const collapsed = this.els.leftPanel?.classList.toggle('is-collapsed') ?? false;
      this.els.btnControls.classList.toggle('is-active', !collapsed);
      this.els.btnControls.setAttribute('aria-expanded', String(!collapsed));
    });

    for (const name of ['sliderSimRate', 'sliderDamageR', 'sliderDensity', 'sliderExposure']) {
      this.els[name]?.addEventListener('input', () => this._updateSliderValues());
    }
    this._updateSliderValues();
  }

  on(event, cb) { this.callbacks[event] = cb; }
  _fire(event, ...args) { this.callbacks[event]?.(...args); }

  _syncModelLabel() {
    if (this.els.valModel) this.els.valModel.textContent = this.els.modelSelect?.value || '-';
  }

  _updateSliderValues() {
    if (this.els.valSimRate) this.els.valSimRate.textContent = `${this.simRate.toFixed(2)}x`;
    if (this.els.valDamageR) this.els.valDamageR.textContent = String(this.damageRadius);
    if (this.els.valDensity) this.els.valDensity.textContent = `${this.density.toFixed(2)}x`;
    if (this.els.valExposure) this.els.valExposure.textContent = `${this.exposure.toFixed(2)}x`;
  }

  populateModels(names) {
    if (!this.els.modelSelect) return;
    this.els.modelSelect.innerHTML = '';
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      this.els.modelSelect.appendChild(opt);
    }
    this._syncModelLabel();
  }

  setModelInfo(model) {
    if (this.els.infoCh) this.els.infoCh.textContent = model.channels;
    if (this.els.infoGrid) this.els.infoGrid.textContent = `${model.coarseSize}^3`;
    if (this.els.infoSamples) this.els.infoSamples.textContent = model.renderSamples;
    if (this.els.infoDecoder) this.els.infoDecoder.textContent = `${model.sirenHiddenDim} x ${model.sirenLayers}`;
    if (this.els.infoGPU) this.els.infoGPU.textContent = model.meta.gpu_used || 'N/A';
    this._syncModelLabel();
  }

  setStatus(text) {
    if (!this.els.statusBanner) return;
    if (!text) {
      this.els.statusBanner.classList.add('hidden');
      this.els.statusBanner.setAttribute('aria-hidden', 'true');
      this.els.statusBanner.textContent = '';
      return;
    }
    this.els.statusBanner.textContent = text;
    this.els.statusBanner.classList.remove('hidden');
    this.els.statusBanner.setAttribute('aria-hidden', 'false');
  }

  get simRate() { return parseFloat(this.els.sliderSimRate?.value) || 1; }
  get damageRadius() { return parseFloat(this.els.sliderDamageR?.value) || 2; }
  get density() { return parseFloat(this.els.sliderDensity?.value) || 1; }
  get exposure() { return parseFloat(this.els.sliderExposure?.value) || 1; }

  updateStats(fps, step, samples, gridSize) {
    if (this.els.statFps) this.els.statFps.textContent = fps.toFixed(0);
    if (this.els.statStep) this.els.statStep.textContent = step;
    if (this.els.statSamples) this.els.statSamples.textContent = `${samples} @ ${gridSize}^3`;
  }

  updateTimings(ncaMs, renderMs) {
    if (this.els.statGpuTime) {
      const total = ncaMs + renderMs;
      if (ncaMs > 0) {
        this.els.statGpuTime.textContent = `${total.toFixed(1)}ms (NCA: ${ncaMs.toFixed(1)}ms, Render: ${renderMs.toFixed(1)}ms)`;
      } else {
        this.els.statGpuTime.textContent = `${renderMs.toFixed(1)}ms`;
      }
    }
  }

  setF16(supported) {
    if (this.els.statF16) {
      this.els.statF16.textContent = supported ? 'Native' : 'F32';
      this.els.statF16.className = supported ? 'stat-val ok' : 'stat-val warn';
    }
  }

  setRunning(running) {
    if (!this.els.btnStartStop) return;
    this.els.btnStartStop.textContent = running ? 'STOP' : 'START';
    this.els.btnStartStop.classList.toggle('is-active', !running);
  }
}
