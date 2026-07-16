export class Interaction {
  constructor(camera, nca, renderSize) {
    this.camera = camera;
    this.nca = nca;
    this.renderSize = renderSize;
    this.damageRadius = 2;
  }

  handleClick(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;

    const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((event.clientY - rect.top) / rect.height) * 2;
    const aspect = canvas.width / Math.max(1, canvas.height);
    const ray = this.camera.getRay(ndcX, ndcY, aspect);
    const hit = intersectRayBox(ray.origin, ray.direction, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);
    if (!hit) return false;

    const t = Math.max(hit.tNear, 0) + 1e-4;
    const world = [
      ray.origin[0] + ray.direction[0] * t,
      ray.origin[1] + ray.direction[1] * t,
      ray.origin[2] + ray.direction[2] * t,
    ];
    const S = this.nca.model.coarseSize;
    const gx = worldToCoarse(world[0], S);
    const gy = worldToCoarse(world[1], S);
    const gz = worldToCoarse(world[2], S);
    this.nca.damageAt(gx, gy, gz, this.damageRadius);
    return true;
  }
}

function worldToCoarse(value, size) {
  return Math.max(0, Math.min(size - 1, Math.floor((value + 0.5) * size)));
}

function intersectRayBox(origin, dir, min, max) {
  let tNear = -Infinity;
  let tFar = Infinity;
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(dir[axis]) < 1e-8) {
      if (origin[axis] < min[axis] || origin[axis] > max[axis]) return null;
      continue;
    }
    const inv = 1 / dir[axis];
    let t0 = (min[axis] - origin[axis]) * inv;
    let t1 = (max[axis] - origin[axis]) * inv;
    if (t0 > t1) [t0, t1] = [t1, t0];
    tNear = Math.max(tNear, t0);
    tFar = Math.min(tFar, t1);
    if (tNear > tFar) return null;
  }
  return tFar >= 0 ? { tNear, tFar } : null;
}