enable f16;

struct VoxelInstance {
  px: f16,
  py: f16,
  pz: f16,
  r: f16,
  g: f16,
  b: f16,
  a: f16,
  pad: f16,
}

struct Params {
  size: u32,
  capacity: u32,
  threshold: f32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> voxels: array<f16>;
@group(0) @binding(1) var<storage, read_write> instances: array<VoxelInstance>;
@group(0) @binding(2) var<storage, read_write> count: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn alpha_at(x: u32, y: u32, z: u32) -> f16 {
  let index = (z * params.size * params.size + y * params.size + x) * 4u;
  return voxels[index + 3u];
}

@compute @workgroup_size(4, 4, 4)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let z = gid.z;
  let size = params.size;
  if (x >= size || y >= size || z >= size) { return; }

  let index = (z * size * size + y * size + x) * 4u;
  let alpha = voxels[index + 3u];
  if (alpha <= f16(params.threshold)) { return; }

  if (alpha > f16(0.8) && x > 0u && y > 0u && z > 0u && x + 1u < size && y + 1u < size && z + 1u < size) {
    let hidden =
      alpha_at(x - 1u, y, z) > f16(params.threshold) &&
      alpha_at(x + 1u, y, z) > f16(params.threshold) &&
      alpha_at(x, y - 1u, z) > f16(params.threshold) &&
      alpha_at(x, y + 1u, z) > f16(params.threshold) &&
      alpha_at(x, y, z - 1u) > f16(params.threshold) &&
      alpha_at(x, y, z + 1u) > f16(params.threshold);
    if (hidden) { return; }
  }

  let slot = atomicAdd(&count[0], 1u);
  if (slot >= params.capacity) { return; }
  instances[slot].px = f16(x);
  instances[slot].py = f16(y);
  instances[slot].pz = f16(z);
  instances[slot].r = clamp(voxels[index], f16(0.0), f16(1.0));
  instances[slot].g = clamp(voxels[index + 1u], f16(0.0), f16(1.0));
  instances[slot].b = clamp(voxels[index + 2u], f16(0.0), f16(1.0));
  instances[slot].a = clamp(alpha, f16(0.0), f16(1.0));
  instances[slot].pad = f16(0.0);
}
