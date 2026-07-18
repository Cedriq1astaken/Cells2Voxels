{{F16_ENABLE}}

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const K: u32 = {{K}}u;
const FC: u32 = {{FC}}u;
const VOL: u32 = S * S * S;

@group(0) @binding(0) var<storage, read> state_in: array<{{STATE_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> state_out: array<{{STATE_TYPE}}>;
@group(0) @binding(2) var<storage, read> perc_w: array<f32>;  // [K, 1, 3, 3, 3] = K*27
@group(0) @binding(3) var<storage, read> w1: array<f32>;      // [FC, C*K, 1,1,1]
@group(0) @binding(4) var<storage, read> b1: array<f32>;      // [FC]
@group(0) @binding(5) var<storage, read> w2: array<f32>;      // [C, FC, 1,1,1]
@group(0) @binding(6) var<uniform> params: array<vec4<f32>, 2>; // S, C, K, FC, step, updateProb, ...
@group(0) @binding(7) var<storage, read> rng: array<f32>;     // [VOL]
@group(0) @binding(8) var<storage, read> original_state: array<{{STATE_TYPE}}>;

fn idx3(c: u32, z: u32, y: u32, x: u32) -> u32 {
  return c * VOL + z * S * S + y * S + x;
}

fn sample_state_zero(c: u32, z: i32, y: i32, x: i32) -> f32 {
  if (z < 0 || y < 0 || x < 0 || z >= i32(S) || y >= i32(S) || x >= i32(S)) { return 0.0; }
  return f32(state_in[idx3(c, u32(z), u32(y), u32(x))]);
}

fn sample_original_zero(c: u32, z: i32, y: i32, x: i32) -> f32 {
  if (z < 0 || y < 0 || x < 0 || z >= i32(S) || y >= i32(S) || x >= i32(S)) { return 0.0; }
  return f32(original_state[idx3(c, u32(z), u32(y), u32(x))]);
}

fn wrap_coordinate(value: i32) -> u32 {
  let size = i32(S);
  return u32((value % size + size) % size);
}

fn sample_state_circular(c: u32, z: i32, y: i32, x: i32) -> f32 {
  return f32(state_in[idx3(c, wrap_coordinate(z), wrap_coordinate(y), wrap_coordinate(x))]);
}

@compute @workgroup_size(4, 4, 4)
fn nca_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x >= S || y >= S || z >= S) { return; }
  let voxel_idx = z * S * S + y * S + x;

  // PyTorch's depthwise-convolution reshape produces channel-major features:
  // [channel 0/kernel 0..K, channel 1/kernel 0..K, ...].
  // Perception alone uses circular padding, matching depthwise_conv3d.
  var perceived: array<f32, {{C_K}}>;
  for (var c: u32 = 0u; c < C; c++) {
    for (var k: u32 = 0u; k < K; k++) {
      var val: f32 = 0.0;
      for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
          for (var dx: i32 = -1; dx <= 1; dx++) {
            let k_idx = k * 27u + u32(dz + 1) * 9u + u32(dy + 1) * 3u + u32(dx + 1);
            val += sample_state_circular(c, i32(z) + dz, i32(y) + dy, i32(x) + dx) * perc_w[k_idx];
          }
        }
      }
      perceived[c * K + k] = val;
    }
  }

  // Adaptation: w1 * perceived + b1 -> ReLU -> w2 -> linear delta.
  let CK = C * K;
  var hidden: array<f32, {{FC}}>;
  for (var f: u32 = 0u; f < FC; f++) {
    var val: f32 = b1[f];
    for (var i: u32 = 0u; i < CK; i++) {
      val += w1[f * CK + i] * perceived[i];
    }
    hidden[f] = max(val, 0.0);
  }

  var delta: array<f32, {{C}}>;
  for (var c: u32 = 0u; c < C; c++) {
    var val: f32 = 0.0;
    for (var f: u32 = 0u; f < FC; f++) {
      val += w2[c * FC + f] * hidden[f];
    }
    delta[c] = val;
  }

  let update_mask = select(0.0, 1.0, rng[voxel_idx] < params[1].y);

  // Apply the residual update without activation or clamping.
  for (var c: u32 = 0u; c < C; c++) {
    let old_val = f32(state_in[idx3(c, z, y, x)]);
    state_out[idx3(c, z, y, x)] = {{STATE_TYPE}}(old_val + delta[c] * update_mask);
  }
}

@compute @workgroup_size(4, 4, 4)
fn apply_living_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x >= S || y >= S || z >= S) { return; }

  // GrowingVNCA3D uses hard, zero-padded 3x3x3 masks before and after
  // the residual update. A separate dispatch makes neighbor reads race-free.
  var pre_max: f32 = 0.0;
  var post_max: f32 = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz++) {
    for (var dy: i32 = -1; dy <= 1; dy++) {
      for (var dx: i32 = -1; dx <= 1; dx++) {
        let nz = i32(z) + dz;
        let ny = i32(y) + dy;
        let nx = i32(x) + dx;
        pre_max = max(pre_max, sample_original_zero({{LC}}u, nz, ny, nx));
        post_max = max(post_max, sample_state_zero({{LC}}u, nz, ny, nx));
      }
    }
  }
  let alive = select(0.0, 1.0,
    pre_max > {{LIVING_THRESHOLD}} && post_max > {{LIVING_THRESHOLD}});

  for (var c: u32 = 0u; c < C; c++) {
    state_out[idx3(c, z, y, x)] =
      {{STATE_TYPE}}(f32(state_in[idx3(c, z, y, x)]) * alive);
  }
}
