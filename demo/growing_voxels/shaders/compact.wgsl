struct VoxelInstance {
  px: f32, py: f32, pz: f32,
  r: f32, g: f32, b: f32, a: f32,
}

struct Params {
  RS: u32,
  maxInst: u32,
  threshold: f32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> voxels: array<f32>;     // [4, RS, RS, RS]
@group(0) @binding(1) var<storage, read_write> instances: array<VoxelInstance>;
@group(0) @binding(2) var<storage, read_write> count: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(4, 4, 4)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let RS = params.RS;
  let maxInst = params.maxInst;
  let threshold = params.threshold;
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x >= RS || y >= RS || z >= RS) { return; }

  let vol = RS * RS * RS;
  let idx = z * RS * RS + y * RS + x;
  let a = voxels[3u * vol + idx];
  if (a <= threshold) { return; }

  // Interior Culling: Only cull voxels that are both opaque AND fully surrounded.
  // Semi-transparent voxels (alpha < 0.8) are always rendered — culling them would
  // expose the hollow interior through their transparent surface.
  let SOLID_THRESHOLD = 0.8;
  var fully_occluded = false;
  if (a > SOLID_THRESHOLD && x > 0u && x < RS - 1u && y > 0u && y < RS - 1u && z > 0u && z < RS - 1u) {
    let a_xp = voxels[3u * vol + z * RS * RS + y * RS + (x + 1u)];
    let a_xn = voxels[3u * vol + z * RS * RS + y * RS + (x - 1u)];
    let a_yp = voxels[3u * vol + z * RS * RS + (y + 1u) * RS + x];
    let a_yn = voxels[3u * vol + z * RS * RS + (y - 1u) * RS + x];
    let a_zp = voxels[3u * vol + (z + 1u) * RS * RS + y * RS + x];
    let a_zn = voxels[3u * vol + (z - 1u) * RS * RS + y * RS + x];

    if (a_xp > threshold && a_xn > threshold &&
        a_yp > threshold && a_yn > threshold &&
        a_zp > threshold && a_zn > threshold) {
      fully_occluded = true;
    }
  }

  if (fully_occluded) { return; }

  let slot = atomicAdd(&count[0], 1u);
  if (slot >= maxInst) { return; } // bounds check

  instances[slot].px = f32(x);
  instances[slot].py = f32(y);
  instances[slot].pz = f32(z);
  instances[slot].r = clamp(voxels[0u * vol + idx], 0.0, 1.0);
  instances[slot].g = clamp(voxels[1u * vol + idx], 0.0, 1.0);
  instances[slot].b = clamp(voxels[2u * vol + idx], 0.0, 1.0);
  instances[slot].a = clamp(a, 0.0, 1.0);

}
