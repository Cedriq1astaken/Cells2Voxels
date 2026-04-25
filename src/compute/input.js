import { vec2 } from 'gl-matrix';

class Input {
  constructor(target) {
    this.movement = vec2.create();
    this.view = {
      state: vec2.fromValues(Math.PI * 0.3 - 0.001, Math.PI * 0.125 - 0.001),
      target: vec2.fromValues(Math.PI * 0.3, Math.PI * 0.125),
    };
    this.baseRadius = Input.defaultRadius;
    this.minZoom = Math.log(this.baseRadius * Input.minZoomFactor);
    this.maxZoom = Math.log(this.baseRadius * Input.maxZoomFactor);
    this.zoom = {
      state: this.baseRadius - 0.001,
      target: this.baseRadius,
    };
    this.isDragging = false;

    this.onPointerDown = this.onPointerDown.bind(this);
    this.onPointerMove = this.onPointerMove.bind(this);
    this.onPointerUp = this.onPointerUp.bind(this);
    this.onWheel = this.onWheel.bind(this);

    target.addEventListener('pointerdown', this.onPointerDown);
    target.addEventListener('wheel', this.onWheel);
  }

  updateZoomBounds() {
    this.minZoom = Math.log(Math.max(1, this.baseRadius * Input.minZoomFactor));
    this.maxZoom = Math.log(Math.max(2, this.baseRadius * Input.maxZoomFactor));
  }

  setRenderSize(renderSize, { preserveZoom = true, immediate = false } = {}) {
    const nextBaseRadius = Input.getDefaultRadius(renderSize);
    const ratio = nextBaseRadius / this.baseRadius;
    this.baseRadius = nextBaseRadius;
    this.updateZoomBounds();

    if (preserveZoom) {
      this.zoom.target *= ratio;
      this.zoom.state *= ratio;
    } else {
      this.zoom.target = nextBaseRadius;
      if (immediate) {
        this.zoom.state = nextBaseRadius;
      }
    }

    const minRadius = Math.exp(this.minZoom);
    const maxRadius = Math.exp(this.maxZoom);
    this.zoom.target = Math.min(Math.max(this.zoom.target, minRadius), maxRadius);
    this.zoom.state = Math.min(Math.max(this.zoom.state, minRadius), maxRadius);
  }

  update(delta) {
    const { minPhi, maxPhi } = Input;
    const { movement, view, zoom } = this;

    if (movement[0] !== 0 || movement[1] !== 0) {
      view.target[1] += movement[0];
      view.target[0] = Math.min(Math.max(view.target[0] + movement[1], minPhi), maxPhi);
      vec2.set(movement, 0, 0);
    }

    const damp = 1 - Math.exp(-10 * delta);
    if (Math.max(Math.abs(view.state[0] - view.target[0]), Math.abs(view.state[1] - view.target[1])) > 0.001) {
      vec2.lerp(view.state, view.state, view.target, damp);
    }
    if (Math.abs(zoom.state - zoom.target) > 0.001) {
      zoom.state = zoom.state * (1 - damp) + zoom.target * damp;
    }
  }

  getView() {
    const { view, zoom } = this;
    return {
      phi: view.state[0],
      theta: view.state[1],
      radius: zoom.state,
    };
  }

  onPointerDown(e) {
    const target = e.currentTarget;
    if (!(target instanceof HTMLElement) || e.button !== 0) {
      return;
    }
    this.isDragging = true;
    target.setPointerCapture(e.pointerId);
    target.addEventListener('lostpointercapture', this.onPointerUp);
    target.addEventListener('pointermove', this.onPointerMove);
    target.addEventListener('pointerup', this.onPointerUp);
    target.style.cursor = 'grabbing';
  }

  onPointerMove({ movementX, movementY }) {
    if (!this.isDragging) {
      return;
    }
    const { sensitivity } = Input;
    vec2.set(
      this.movement,
      -movementX * sensitivity.view,
      -movementY * sensitivity.view
    );
  }

  onPointerUp(e) {
    this.isDragging = false;
    const target = e.currentTarget;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    target.removeEventListener('lostpointercapture', this.onPointerUp);
    target.removeEventListener('pointermove', this.onPointerMove);
    target.removeEventListener('pointerup', this.onPointerUp);
    if (target.hasPointerCapture(e.pointerId)) {
      target.releasePointerCapture(e.pointerId);
    }
    target.style.cursor = '';
  }

  onWheel(e) {
    const { sensitivity } = Input;
    const { zoom } = this;
    const zoomRange = this.maxZoom - this.minZoom;
    const logZoom = Math.min(
      Math.max(
        ((Math.log(zoom.target) - this.minZoom) / zoomRange) + (e.deltaY * sensitivity.zoom),
        0
      ),
      1
    );
    zoom.target = Math.exp(this.minZoom + logZoom * zoomRange);
  }
}

Input.sensitivity = {
  view: 0.001,
  zoom: 0.0001,
};
Input.defaultRadiusFactor = 1.76;
Input.defaultRadius = 164;
Input.getDefaultRadius = (renderSize) => Math.max(Input.defaultRadius, renderSize * Input.defaultRadiusFactor);
Input.minPhi = 0.01;
Input.maxPhi = Math.PI - 0.01;
Input.minZoomFactor = 0.35;
Input.maxZoomFactor = 4.0;

export default Input;
