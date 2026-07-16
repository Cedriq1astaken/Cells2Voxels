export class OrbitCamera {
  constructor(onChange = () => {}) {
    this.azimuth = 35;
    this.elevation = 22;
    this.distance = 1.45;
    this.target = [0, 0, 0];
    this.fov = 50;
    this.near = 0.01;
    this.far = 100;
    this.onChange = onChange;
    this.dragging = false;
    this.pointerId = null;
  }

  attach(canvas) {
    canvas.style.touchAction = 'none';

    canvas.addEventListener('pointerdown', event => {
      if (event.button !== 0 || this.dragging) return;
      event.preventDefault();
      this.dragging = true;
      this.pointerId = event.pointerId;
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener('pointermove', event => {
      if (!this.dragging || event.pointerId !== this.pointerId) return;
      event.preventDefault();
      const dx = event.clientX - this.lastX;
      const dy = event.clientY - this.lastY;
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      this.azimuth -= dx * 0.35;
      this.elevation = clamp(this.elevation + dy * 0.35, -85, 85);
      this.onChange();
    });

    const endDrag = event => {
      if (event.pointerId !== this.pointerId) return;
      this.dragging = false;
      this.pointerId = null;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      this.distance = clamp(this.distance * Math.exp(event.deltaY * 0.0015), 0.45, 6);
      this.onChange();
    }, { passive: false });
  }

  getPosition() {
    const azimuth = this.azimuth * Math.PI / 180;
    const elevation = this.elevation * Math.PI / 180;
    const cosElevation = Math.cos(elevation);
    return [
      this.target[0] + this.distance * cosElevation * Math.sin(azimuth),
      this.target[1] + this.distance * Math.sin(elevation),
      this.target[2] + this.distance * cosElevation * Math.cos(azimuth),
    ];
  }

  getMVP(aspect) {
    return multiply4(
      perspective(this.fov * Math.PI / 180, aspect, this.near, this.far),
      lookAt(this.getPosition(), this.target, [0, 1, 0]),
    );
  }
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function lookAt(eye, target, up) {
  const z = normalize(subtract(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function multiply4(a, b) {
  const result = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let column = 0; column < 4; column++) {
      result[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3];
    }
  }
  return result;
}

function normalize(value) {
  const length = Math.hypot(...value);
  return length > 0 ? value.map(component => component / length) : value;
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
