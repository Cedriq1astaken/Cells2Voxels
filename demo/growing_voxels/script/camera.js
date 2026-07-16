export class OrbitCamera {
  constructor() {
    this.azimuth = 35;
    this.elevation = 22;
    this.distance = 1.35;
    this.target = [0, 0, 0];
    this.fov = 50;
    this.near = 0.01;
    this.far = 100;
    this._dragging = false;
    this._pointerId = null;
    this._lastX = 0;
    this._lastY = 0;
    this._dragDistance = 0;
    this._suppressClick = false;
  }

  attach(canvas) {
    canvas.style.touchAction = 'none';

    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || this._dragging) return;
      event.preventDefault();
      this._dragging = true;
      this._pointerId = event.pointerId;
      this._lastX = event.clientX;
      this._lastY = event.clientY;
      this._dragDistance = 0;
      this._suppressClick = false;
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener('pointermove', (event) => {
      if (!this._dragging || event.pointerId !== this._pointerId) return;
      event.preventDefault();
      const dx = event.clientX - this._lastX;
      const dy = event.clientY - this._lastY;
      this._lastX = event.clientX;
      this._lastY = event.clientY;
      this._dragDistance += Math.hypot(dx, dy);

      const sensitivity = 0.35;
      this.azimuth -= dx * sensitivity;
      this.elevation = Math.max(-85, Math.min(85, this.elevation + dy * sensitivity));
    });

    const endDrag = (event) => {
      if (this._pointerId !== null && event.pointerId !== this._pointerId) return;
      this._suppressClick = this._dragDistance > 4;
      this._dragging = false;
      this._pointerId = null;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const zoom = Math.exp(event.deltaY * 0.0015);
      this.distance = Math.max(0.45, Math.min(6, this.distance * zoom));
    }, { passive: false });
  }

  get isDragging() { return this._dragging; }

  consumeClick() {
    const blocked = this._suppressClick;
    this._suppressClick = false;
    return blocked;
  }

  getPosition() {
    const azRad = this.azimuth * Math.PI / 180;
    const elRad = this.elevation * Math.PI / 180;
    const cosEl = Math.cos(elRad);
    return [
      this.target[0] + this.distance * cosEl * Math.sin(azRad),
      this.target[1] + this.distance * Math.sin(elRad),
      this.target[2] + this.distance * cosEl * Math.cos(azRad),
    ];
  }

  getRay(ndcX, ndcY, aspect) {
    const eye = this.getPosition();
    const forward = normalize(sub(this.target, eye));
    let right = normalize(cross(forward, [0, 1, 0]));
    if (Math.abs(dot(right, right)) < 1e-8) right = [1, 0, 0];
    const up = normalize(cross(right, forward));
    const tanHalfFov = Math.tan((this.fov * Math.PI / 180) * 0.5);
    const direction = normalize(add3(
      forward,
      scale3(right, ndcX * aspect * tanHalfFov),
      scale3(up, ndcY * tanHalfFov)
    ));
    return { origin: eye, direction };
  }

  getViewMatrix() {
    const eye = this.getPosition();
    return mat4LookAt(eye, this.target, [0, 1, 0]);
  }

  getProjectionMatrix(aspect) {
    return mat4Perspective(this.fov * Math.PI / 180, aspect, this.near, this.far);
  }

  getMVP(aspect) {
    const view = this.getViewMatrix();
    const proj = this.getProjectionMatrix(aspect);
    return mat4Multiply(proj, view);
  }
}

function mat4LookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function mat4Perspective(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const nf = 1.0 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function mat4Multiply(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      r[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
    }
  return r;
}

function add3(a, b, c) { return [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]; }
function scale3(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function normalize(v) { const l = Math.sqrt(dot(v, v)); return l > 0 ? [v[0]/l, v[1]/l, v[2]/l] : [0,0,1]; }