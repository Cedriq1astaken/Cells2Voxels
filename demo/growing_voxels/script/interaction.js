export class Interaction {
  constructor(camera, nca, renderSize, rotateModel = true) {
    this.camera = camera;
    this.nca = nca;
    this.renderSize = renderSize;
    this.damageRadius = 2;
    this.rotateModel = rotateModel;
    this.modelRotation = [0, 0, 0];
  }

  handleClick(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;

    const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((event.clientY - rect.top) / rect.height) * 2;
    const aspect = canvas.width / Math.max(1, canvas.height);
    const ray = this.camera.getRay(ndcX, ndcY, aspect);
    // The visual bounding box is 1x1x1, but the models usually only occupy a sphere of radius ~0.35
    // Intersect with a sphere to get a much tighter hit point on the actual model surface
    let hit = intersectRaySphere(ray.origin, ray.direction, 0.4);
    if (!hit) {
      // Fallback to bounding box if somehow they click the extreme edge
      hit = intersectRayBox(ray.origin, ray.direction, [-0.45, -0.45, -0.45], [0.45, 0.45, 0.45]);
    }
    if (!hit) return false;

    const t = Math.max(hit.tNear, 0) + 1e-4;
    const world = [
      ray.origin[0] + ray.direction[0] * t,
      ray.origin[1] + ray.direction[1] * t,
      ray.origin[2] + ray.direction[2] * t,
    ];

    // Undo the user rotation (render order X -> Y -> Z), then undo the
    // model's fixed .vox orientation correction.
    let world_raw = rotateZ(world, -this.modelRotation[2]);
    world_raw = rotateY(world_raw, -this.modelRotation[1]);
    world_raw = rotateX(world_raw, -this.modelRotation[0]);
    if (this.rotateModel) world_raw = [world_raw[0], -world_raw[2], world_raw[1]];

    const S = this.nca.model.coarseSize;
    const gx = worldToCoarse(world_raw[0], S);
    const gy = worldToCoarse(world_raw[1], S);
    const gz = worldToCoarse(world_raw[2], S);
    this.nca.damageAt(gx, gy, gz, this.damageRadius);
    return true;
  }
}

function rotateX(v, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]];
}

function rotateY(v, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
}

function rotateZ(v, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
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

function intersectRaySphere(origin, dir, radius) {
  // Ray: P = origin + t*dir
  // Sphere: |P| = radius  => P.P = radius^2
  // (origin + t*dir).(origin + t*dir) = radius^2
  // dir.dir*t^2 + 2*origin.dir*t + origin.origin - radius^2 = 0
  const a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
  const b = 2 * (origin[0]*dir[0] + origin[1]*dir[1] + origin[2]*dir[2]);
  const c = origin[0]*origin[0] + origin[1]*origin[1] + origin[2]*origin[2] - radius*radius;
  const discriminant = b*b - 4*a*c;
  
  if (discriminant < 0) return null;
  
  const sqrtD = Math.sqrt(discriminant);
  const t0 = (-b - sqrtD) / (2*a);
  const t1 = (-b + sqrtD) / (2*a);
  
  const tNear = Math.min(t0, t1);
  const tFar = Math.max(t0, t1);
  
  return tFar >= 0 ? { tNear, tFar } : null;
}
