import { glMatrix, mat4, vec3, vec4 } from 'gl-matrix';

class Camera {
  constructor(
    device,
    aspect = 1,
    fov = 75,
    near = 0.1,
    far = 10000
  ) {
    this.aspect = aspect;
    this.fov = fov;
    this.near = near;
    this.far = far;
    this.position = vec3.create();
    this.target = vec3.create();
    this.projectionMatrix = mat4.create();
    this.inverseProjectionMatrix = mat4.create();
    this.viewMatrix = mat4.create();
    this.inverseViewMatrix = mat4.create();
    this.data = new Float32Array(16 * 2);

    this.buffer = device.createBuffer({
      size: this.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });
    this.device = device;
  }

  getBuffer() {
    return this.buffer;
  }

  getPickRay(clientX, clientY, rect) {
    const x = ((clientX - rect.left) / rect.width) * 2 - 1;
    const y = 1 - (((clientY - rect.top) / rect.height) * 2);
    const clip = vec4.fromValues(x, y, 1, 1);
    const view = vec4.transformMat4(vec4.create(), clip, this.inverseProjectionMatrix);
    view[3] = 0;
    const world = vec4.transformMat4(vec4.create(), view, this.inverseViewMatrix);
    const direction = vec3.normalize(vec3.create(), vec3.fromValues(world[0], world[1], world[2]));
    return {
      origin: vec3.clone(this.position),
      direction,
    };
  }

  setAspect(aspect) {
    this.aspect = aspect;
    this.updateProjection();
  }

  setOrbit(phi, theta, radius) {
    const { offset } = Camera;
    const { position, target } = this;
    const sinPhiRadius = Math.sin(phi) * radius;
    vec3.add(
      position,
      target,
      vec3.set(
        offset,
        sinPhiRadius * Math.sin(theta),
        Math.cos(phi) * radius,
        sinPhiRadius * Math.cos(theta)
      )
    );
    this.updateView();
  }

  updateProjection() {
    const { projectionMatrix, inverseProjectionMatrix, aspect, fov, near, far } = this;
    mat4.perspective(projectionMatrix, glMatrix.toRadian(fov), aspect, near, far);
    mat4.invert(inverseProjectionMatrix, projectionMatrix);
    this.updateBuffer();
  }

  updateView() {
    const { worldUp } = Camera;
    const { viewMatrix, inverseViewMatrix, position, target } = this;
    mat4.lookAt(viewMatrix, position, target, worldUp);
    mat4.invert(inverseViewMatrix, viewMatrix);
    this.updateBuffer();
  }

  updateBuffer() {
    const { device, buffer, data, inverseProjectionMatrix, inverseViewMatrix } = this;
    data.set(inverseProjectionMatrix, 0);
    data.set(inverseViewMatrix, 16);
    device.queue.writeBuffer(buffer, 0, data);
  }
}

Camera.offset = vec3.create();
Camera.worldUp = vec3.fromValues(0, 1, 0);

export default Camera;
