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
      cbPerfMode: document.getElementById('cb-perf-mode'),
      sliderSimRate: document.getElementById('slider-sim-rate'),
      sliderDamageR: document.getElementById('slider-damage-radius'),
      sliderCrossX: document.getElementById('slider-cross-x'),
      sliderCrossY: document.getElementById('slider-cross-y'),
      sliderCrossZ: document.getElementById('slider-cross-z'),
      sliderRotateX: document.getElementById('slider-rotate-x'),
      sliderRotateY: document.getElementById('slider-rotate-y'),
      sliderRotateZ: document.getElementById('slider-rotate-z'),
      valSimRate: document.getElementById('val-sim-rate'),
      valDamageR: document.getElementById('val-damage-radius'),
      valCrossX: document.getElementById('val-cross-x'),
      valCrossY: document.getElementById('val-cross-y'),
      valCrossZ: document.getElementById('val-cross-z'),
      valRotateX: document.getElementById('val-rotate-x'),
      valRotateY: document.getElementById('val-rotate-y'),
      valRotateZ: document.getElementById('val-rotate-z'),
      sliderLppnScale: document.getElementById('slider-lppn-scale'),
      valLppnScale: document.getElementById('val-lppn-scale'),
      statFps: document.getElementById('stat-fps'),
      statStep: document.getElementById('stat-step'),
      statVoxels: document.getElementById('stat-voxels'),
      statF16: document.getElementById('stat-f16'),
      infoPanel: document.getElementById('info-panel'),
      infoCh: document.getElementById('info-channels'),
      infoCoarse: document.getElementById('info-coarse'),
      infoTarget: document.getElementById('info-target'),
      infoScale: document.getElementById('info-scale'),
      infoGPU: document.getElementById('info-gpu'),
      imgOriginal: document.getElementById('img-original'),
      imgCoarse: document.getElementById('img-coarse'),
      btnInfo: document.getElementById('btn-info'),
      btnCloseInfo: document.getElementById('btn-close-info'),
    };

    this.els.btnStartStop?.addEventListener('click', () => this._fire('toggleRun'));
    this.els.btnReset?.addEventListener('click', () => this._fire('reset'));
    this.els.modelSelect?.addEventListener('change', () => {
      this._syncModelLabel();
      this._fire('modelChange', this.els.modelSelect.value);
    });

    for (const axis of ['x', 'y', 'z']) {
      const s = this.els[`sliderCross${axis.toUpperCase()}`];
      if (s) {
        s.addEventListener('input', () => {
          this._updateSliderValues();
          this._fire('crossChange');
        });
      }
    }

    if (this.els.sliderLppnScale) {
      this.els.sliderLppnScale.addEventListener('input', () => this._updateSliderValues());
      this.els.sliderLppnScale.addEventListener('change', () => {
        this._fire('lppnScaleChange', parseFloat(this.els.sliderLppnScale.value));
      });
    }

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

    for (const name of ['sliderSimRate', 'sliderDamageR', 'sliderRotateX', 'sliderRotateY', 'sliderRotateZ']) {
      this.els[name]?.addEventListener('input', () => this._updateSliderValues());
    }
    this._updateSliderValues();
  }

  on(event, cb) { this.callbacks[event] = cb; }
  _fire(event, ...args) { this.callbacks[event]?.(...args); }

  _syncModelLabel() {
    if (this.els.valModel) {
      this.els.valModel.textContent = this.els.modelSelect?.value || '-';
    }
  }

  _updateSliderValues() {
    if (this.els.valSimRate) {
      this.els.valSimRate.textContent = `${Number(this.els.sliderSimRate?.value || 1).toFixed(2)}x`;
    }
    if (this.els.valDamageR) this.els.valDamageR.textContent = this.els.sliderDamageR?.value || '2';
    if (this.els.valCrossX) this.els.valCrossX.textContent = this.els.sliderCrossX?.value || '-';
    if (this.els.valCrossY) this.els.valCrossY.textContent = this.els.sliderCrossY?.value || '-';
    if (this.els.valCrossZ) this.els.valCrossZ.textContent = this.els.sliderCrossZ?.value || '-';
    if (this.els.valRotateX) this.els.valRotateX.textContent = `${this.els.sliderRotateX?.value || 0} deg`;
    if (this.els.valRotateY) this.els.valRotateY.textContent = `${this.els.sliderRotateY?.value || 0} deg`;
    if (this.els.valRotateZ) this.els.valRotateZ.textContent = `${this.els.sliderRotateZ?.value || 0} deg`;
    if (this.els.valLppnScale) this.els.valLppnScale.textContent = parseFloat(this.els.sliderLppnScale?.value || 1).toFixed(1);
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
    if (this.els.infoCoarse) this.els.infoCoarse.textContent = `${model.coarseSize}^3`;
    if (this.els.infoTarget) this.els.infoTarget.textContent = `${model.targetSize || '?'}`;
    if (this.els.infoScale) this.els.infoScale.textContent = model.scale;
    if (this.els.infoGPU) this.els.infoGPU.textContent = model.meta.gpu_used || 'N/A';
    this._syncModelLabel();
    if (this.els.imgOriginal && model.originalImg) this.els.imgOriginal.src = model.originalImg;
    if (this.els.imgCoarse && model.coarseImg) this.els.imgCoarse.src = model.coarseImg;
  }

  setCrossSectionMax(rs) {
    for (const slider of [this.els.sliderCrossX, this.els.sliderCrossY, this.els.sliderCrossZ]) {
      if (slider) {
        slider.max = rs;
        slider.value = rs;
      }
    }
    this._updateSliderValues();
  }

  setLppnScale(scale) {
    if (this.els.sliderLppnScale) {
      this.els.sliderLppnScale.value = scale;
      this._updateSliderValues();
    }
  }

  get simRate() { return parseFloat(this.els.sliderSimRate?.value) || 1; }
  get perfMode() { return this.els.cbPerfMode?.checked || false; }
  get damageRadius() { return parseFloat(this.els.sliderDamageR?.value) || 2; }
  get modelRotation() {
    const toRadians = degrees => degrees * Math.PI / 180;
    return [
      toRadians(parseFloat(this.els.sliderRotateX?.value) || 0),
      toRadians(parseFloat(this.els.sliderRotateY?.value) || 0),
      toRadians(parseFloat(this.els.sliderRotateZ?.value) || 0),
    ];
  }
  get crossSection() {
    return [
      parseFloat(this.els.sliderCrossX?.value) || 9999,
      parseFloat(this.els.sliderCrossY?.value) || 9999,
      parseFloat(this.els.sliderCrossZ?.value) || 9999,
    ];
  }

  updateStats(fps, step, voxelCount, totalVoxels) {
    if (this.els.statFps) this.els.statFps.textContent = fps.toFixed(0);
    if (this.els.statStep) this.els.statStep.textContent = step;
    if (this.els.statVoxels) {
      const pct = totalVoxels > 0 ? (voxelCount / totalVoxels * 100).toFixed(1) : '0';
      this.els.statVoxels.textContent = `${voxelCount} (${pct}%)`;
    }
  }

  setF16(supported) {
    if (this.els.statF16) {
      this.els.statF16.textContent = supported ? 'F16' : 'F32';
      this.els.statF16.className = 'stat-val ok';
    }
  }

  setRunning(running) {
    if (!this.els.btnStartStop) return;
    this.els.btnStartStop.textContent = running ? 'STOP' : 'START';
    this.els.btnStartStop.classList.toggle('is-active', !running);
  }
}