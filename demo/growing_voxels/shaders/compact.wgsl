struct VoxelInstance {
  voxel: u32,
  rgba: u32,
}

struct Params {
  RS: u32,
  maxInst: u32,
  threshold: f32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> voxels: array<f32>; // [4, RS, RS, RS]
@group(0) @binding(1) var<storage, read_write> instances: array<VoxelInstance>;
// count[0] is the full visible count; count[1] is the safely writable draw count.
@group(0) @binding(2) var<storage, read_write> count: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn pack_rgba8(color: vec4<f32>) -> u32 {
  let q = vec4<u32>(round(clamp(color, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
  return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

@compute @workgroup_size(4, 4, 4)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let RS = params.RS;
  let x = gid.x;
  let y = gid.y;
  let z = gid.z;
  if (x >= RS || y >= RS || z >= RS) { return; }

  let vol = RS * RS * RS;
  let idx = z * RS * RS + y * RS + x;
  let a = voxels[3u * vol + idx];
  if (a <= params.threshold) { return; }

  // Cull only opaque voxels that are completely surrounded. Semi-transparent
  // voxels remain visible so their interior does not show through the shell.
  let solid_threshold = 0.8;
  var fully_occluded = false;
  if (a > solid_threshold && x > 0u && x < RS - 1u && y > 0u && y < RS - 1u && z > 0u && z < RS - 1u) {
    let a_xp = voxels[3u * vol + z * RS * RS + y * RS + (x + 1u)];
    let a_xn = voxels[3u * vol + z * RS * RS + y * RS + (x - 1u)];
    let a_yp = voxels[3u * vol + z * RS * RS + (y + 1u) * RS + x];
    let a_yn = voxels[3u * vol + z * RS * RS + (y - 1u) * RS + x];
    let a_zp = voxels[3u * vol + (z + 1u) * RS * RS + y * RS + x];
    let a_zn = voxels[3u * vol + (z - 1u) * RS * RS + y * RS + x];

    if (a_xp > params.threshold && a_xn > params.threshold &&
        a_yp > params.threshold && a_yn > params.threshold &&
        a_zp > params.threshold && a_zn > params.threshold) {
      fully_occluded = true;
    }
  }
  if (fully_occluded) { return; }

  // The first counter retains the true count even if the current packed buffer
  // is too small. JavaScript uses it to grow the buffer on the next decode.
  let slot = atomicAdd(&count[0], 1u);
  if (slot >= params.maxInst) { return; }
  atomicAdd(&count[1], 1u);

  instances[slot].voxel = idx;
  instances[slot].rgba = pack_rgba8(vec4<f32>(
    voxels[0u * vol + idx],
    voxels[1u * vol + idx],
    voxels[2u * vol + idx],
    a,
  ));
}
