const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const K: u32 = {{K}}u;
const FC: u32 = {{FC}}u;
const VOL: u32 = S * S * S;

@group(0) @binding(0) var<storage, read_write> state_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(2) var<storage, read> perc_w: array<f32>;
@group(0) @binding(3) var<storage, read> w1: array<f32>;
@group(0) @binding(4) var<storage, read> b1: array<f32>;
@group(0) @binding(5) var<storage, read> w2: array<f32>;
@group(0) @binding(6) var<uniform> params: array<vec4<f32>, 2>;
@group(0) @binding(7) var<storage, read> rng: array<f32>;

fn idx3(c: u32, z: u32, y: u32, x: u32) -> u32 {
  return c * VOL + z * S * S + y * S + x;
}

fn sample_state(c: u32, z: i32, y: i32, x: i32) -> f32 {
  if (z < 0 || y < 0 || x < 0 || z >= i32(S) || y >= i32(S) || x >= i32(S)) { return 0.0; }
  return state_in[idx3(c, u32(z), u32(y), u32(x))];
}

@compute @workgroup_size(4, 4, 4)
fn nca_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let z = gid.z;
  if (x >= S || y >= S || z >= S) { return; }
  let voxel_idx = z * S * S + y * S + x;
  let update_prob = params[1].y;
  let living_threshold = params[1].z;

  var pre_max: f32 = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz++) {
    for (var dy: i32 = -1; dy <= 1; dy++) {
      for (var dx: i32 = -1; dx <= 1; dx++) {
        pre_max = max(pre_max, sample_state({{LC}}u, i32(z) + dz, i32(y) + dy, i32(x) + dx));
      }
    }
  }
  let pre_alive = pre_max > living_threshold;

  var perceived: array<f32, {{C_K}}>;
  for (var c: u32 = 0u; c < C; c++) {
    for (var k: u32 = 0u; k < K; k++) {
      var val: f32 = 0.0;
      for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
          for (var dx: i32 = -1; dx <= 1; dx++) {
            let k_idx = k * 27u + u32(dz + 1) * 9u + u32(dy + 1) * 3u + u32(dx + 1);
            val += sample_state(c, i32(z) + dz, i32(y) + dy, i32(x) + dx) * perc_w[k_idx];
          }
        }
      }
      perceived[c * K + k] = val;
    }
  }

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

  let update_mask = select(0.0, 1.0, rng[voxel_idx] < update_prob);
  for (var c: u32 = 0u; c < C; c++) {
    let old_val = state_in[idx3(c, z, y, x)];
    state_out[idx3(c, z, y, x)] = old_val + delta[c] * update_mask;
  }

  var post_max: f32 = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz++) {
    for (var dy: i32 = -1; dy <= 1; dy++) {
      for (var dx: i32 = -1; dx <= 1; dx++) {
        let nz = i32(z) + dz;
        let ny = i32(y) + dy;
        let nx = i32(x) + dx;
        if (nz < 0 || ny < 0 || nx < 0 || nz >= i32(S) || ny >= i32(S) || nx >= i32(S)) { continue; }
        post_max = max(post_max, state_out[idx3({{LC}}u, u32(nz), u32(ny), u32(nx))]);
      }
    }
  }
  let alive = select(0.0, 1.0, pre_alive && post_max > living_threshold);
  for (var c: u32 = 0u; c < C; c++) {
    state_out[idx3(c, z, y, x)] *= alive;
  }
}
