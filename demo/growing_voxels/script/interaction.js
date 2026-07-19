export class Interaction {
  constructor(camera, nca, renderSize, rotateModel = true) {
    this.camera = camera;
    this.nca = nca;
    this.renderSize = renderSize;
    this.damageRadius = 2;
    this.rotateModel = rotateModel;
    this.modelRotation = [0, 0, 0];
  }

  getRawRay(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;

    const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((event.clientY - rect.top) / rect.height) * 2;
    const aspect = canvas.width / Math.max(1, canvas.height);
    const ray = this.camera.getRay(ndcX, ndcY, aspect);

    // Match the renderer in reverse: user rotation is X -> Y -> Z, followed
    // by the fixed .vox orientation correction. Apply the inverse to both
    // origin and direction so the picker can traverse the decoded raw volume.
    let origin = undoModelTransform(ray.origin, this.modelRotation, this.rotateModel);
    let direction = undoModelTransform(ray.direction, this.modelRotation, this.rotateModel);
    return { origin, direction };
  }

  damageFineVoxel(fineVoxel) {
    if (!fineVoxel) return null;
    const { coarseSize: S, scale } = this.nca.model;
    const gx = fineToCoarse(fineVoxel.x, scale, S);
    const gy = fineToCoarse(fineVoxel.y, scale, S);
    const gz = fineToCoarse(fineVoxel.z, scale, S);
    const cleared = this.nca.damageAt(gx, gy, gz, this.damageRadius);
    return { coarse: [gx, gy, gz], cleared, fine: fineVoxel };
  }
}

function undoModelTransform(v, rotation, rotateModel) {
  let raw = rotateZ(v, -rotation[2]);
  raw = rotateY(raw, -rotation[1]);
  raw = rotateX(raw, -rotation[0]);
  return rotateModel ? [raw[0], -raw[2], raw[1]] : raw;
}

function fineToCoarse(fineCoordinate, scale, coarseSize) {
  const coordinate = Math.floor((fineCoordinate + 0.5) / scale);
  return Math.max(0, Math.min(coarseSize - 1, coordinate));
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
