{{F16_ENABLE}}

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const RS: u32 = {{RS}}u;
const RS_VOL: u32 = RS * RS * RS;
const NF: u32 = {{NF}}u;
const HD: u32 = {{HD}}u;
const COORD_DIM: u32 = {{COORD_DIM}}u;
const INPUT_DIM: u32 = {{INPUT_DIM}}u;
const PI: f32 = 3.141592653589793;

@group(0) @binding(0) var<storage, read> state: array<{{STATE_TYPE}}>;      // [C, S, S, S]
@group(0) @binding(1) var<storage, read_write> output: array<{{DECODE_TYPE}}>; // [4, RS, RS, RS]
@group(0) @binding(2) var<storage, read> active_blocks: array<u32>;
@group(0) @binding(3) var<uniform> params: array<vec4<f32>, 3>;

@group(0) @binding(4) var<storage, read> fine_alpha: array<{{DECODE_TYPE}}>; // [RS, RS, RS]
@group(0) @binding(5) var<storage, read> l0_w: array<f32>;  // [HD, INPUT_DIM]
@group(0) @binding(6) var<storage, read> l0_b: array<f32>;  // [HD]
@group(0) @binding(7) var<storage, read> l1_w: array<f32>;  // [HD, HD]
@group(0) @binding(8) var<storage, read> l1_b: array<f32>;  // [HD]
@group(0) @binding(9) var<storage, read> l2_w: array<f32>;  // [HD, HD]
@group(0) @binding(10) var<storage, read> l2_b: array<f32>; // [HD]
@group(0) @binding(11) var<storage, read> l3_w: array<f32>; // [4, HD]
@group(0) @binding(12) var<storage, read> l3_b: array<f32>; // [4]
// [dispatch x, dispatch y, dispatch z, active block count]
@group(0) @binding(13) var<storage, read> dispatch_args: array<u32>;
fn state_at(c: u32, z: i32, y: i32, x: i32) -> f32 {
  let size = i32(S);
  let wz = (z % size + size) % size;
  let wy = (y % size + size) % size;
  let wx = (x % size + size) % size;
  return f32(state[c * S * S * S + u32(wz) * S * S + u32(wy) * S + u32(wx)]);
}

fn out_idx(c: u32, z: u32, y: u32, x: u32) -> u32 {
  return c * RS_VOL + z * RS * RS + y * RS + x;
}

// Match the trainer: max-pool the circularly interpolated fine alpha field
// over a zero-padded 3x3x3 neighborhood. The field is decoded once in
// living_mask.wgsl instead of redoing 27 trilinear samples here.
fn fine_alpha_at(x: u32, y: u32, z: u32) -> f32 {
  return f32(fine_alpha[z * RS * RS + y * RS + x]);
}

fn fine_living_mask(fx: u32, fy: u32, fz: u32) -> bool {
  var max_alpha: f32 = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz++) {
    for (var dy: i32 = -1; dy <= 1; dy++) {
      for (var dx: i32 = -1; dx <= 1; dx++) {
        let nx = i32(fx) + dx;
        let ny = i32(fy) + dy;
        let nz = i32(fz) + dz;
        if (nx >= 0 && ny >= 0 && nz >= 0 && nx < i32(RS) && ny < i32(RS) && nz < i32(RS)) {
          max_alpha = max(max_alpha, fine_alpha_at(u32(nx), u32(ny), u32(nz)));
        }
      }
    }
  }
  return max_alpha > {{LIVING_THRESHOLD}};
}

@compute @workgroup_size(4, 4, 4)
fn lppn_decode(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let active_slot = workgroup_id.y * dispatch_args[0] + workgroup_id.x;
  if (active_slot >= dispatch_args[3]) { return; }

  let block_index = active_blocks[active_slot];
  let blocks = (RS + 3u) / 4u;
  let block_x = block_index % blocks;
  let block_y = (block_index / blocks) % blocks;
  let block_z = block_index / (blocks * blocks);
  let fx = block_x * 4u + local_id.x;
  let fy = block_y * 4u + local_id.y;
  let fz = block_z * 4u + local_id.z;
  if (fx >= RS || fy >= RS || fz >= RS) { return; }

  let scale = params[0].z;
  let cross_x = params[1].w; let cross_y = params[2].x; let cross_z = params[2].y;
  let first_omega = params[1].y;
  let hidden_omega = params[1].z;

  if (f32(fx) >= cross_x || f32(fy) >= cross_y || f32(fz) >= cross_z) {
    output[out_idx(0u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    output[out_idx(1u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    output[out_idx(2u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    output[out_idx(3u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    return;
  }

  // Map fine voxel to coarse coordinate
  let cf_x = (f32(fx) + 0.5) / scale - 0.5;
  let cf_y = (f32(fy) + 0.5) / scale - 0.5;
  let cf_z = (f32(fz) + 0.5) / scale - 0.5;

  if (!fine_living_mask(fx, fy, fz)) {
    output[out_idx(0u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    output[out_idx(1u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    output[out_idx(2u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    output[out_idx(3u, fz, fy, fx)] = {{DECODE_TYPE}}(0.0);
    return;
  }

  // Trilinear interpolation of state
  let x0 = i32(floor(cf_x)); let y0 = i32(floor(cf_y)); let z0 = i32(floor(cf_z));
  let x1 = x0 + 1; let y1 = y0 + 1; let z1 = z0 + 1;
  let wx = cf_x - f32(x0); let wy = cf_y - f32(y0); let wz = cf_z - f32(z0);

  var interp_state: array<f32, {{C}}>;
  for (var c: u32 = 0u; c < C; c++) {
    let c000 = state_at(c, z0, y0, x0); let c100 = state_at(c, z0, y0, x1);
    let c010 = state_at(c, z0, y1, x0); let c110 = state_at(c, z0, y1, x1);
    let c001 = state_at(c, z1, y0, x0); let c101 = state_at(c, z1, y0, x1);
    let c011 = state_at(c, z1, y1, x0); let c111 = state_at(c, z1, y1, x1);

    let c00 = c000 * (1.0 - wx) + c100 * wx;
    let c01 = c001 * (1.0 - wx) + c101 * wx;
    let c10 = c010 * (1.0 - wx) + c110 * wx;
    let c11 = c011 * (1.0 - wx) + c111 * wx;
    let c0 = c00 * (1.0 - wy) + c10 * wy;
    let c1 = c01 * (1.0 - wy) + c11 * wy;
    interp_state[c] = c0 * (1.0 - wz) + c1 * wz;
  }

  // Local coordinates in [-1, 1]
  let lx = fract(cf_x + 0.5) * 2.0 - 1.0;
  let ly = fract(cf_y + 0.5) * 2.0 - 1.0;
  let lz = fract(cf_z + 0.5) * 2.0 - 1.0;

  // Build input vector: [encoded_coords, interp_state]
  var input_vec: array<f32, {{INPUT_DIM}}>;
  var idx: u32 = 0u;

  // Frequency encoding: sin(pi * i * coord) and cos(pi * i * coord) for i in 1..NF
  for (var freq: u32 = 1u; freq <= NF; freq++) {
    let f_val = f32(freq) * PI;
    input_vec[idx] = sin(f_val * lz); idx++;
    input_vec[idx] = sin(f_val * ly); idx++;
    input_vec[idx] = sin(f_val * lx); idx++;
  }
  for (var freq: u32 = 1u; freq <= NF; freq++) {
    let f_val = f32(freq) * PI;
    input_vec[idx] = cos(f_val * lz); idx++;
    input_vec[idx] = cos(f_val * ly); idx++;
    input_vec[idx] = cos(f_val * lx); idx++;
  }
  if (NF == 0u) {
    input_vec[idx] = lz; idx++;
    input_vec[idx] = ly; idx++;
    input_vec[idx] = lx; idx++;
  }
  // Append cell states
  for (var c: u32 = 0u; c < C; c++) {
    input_vec[idx] = interp_state[c]; idx++;
  }

  // SIREN Layer 0: sin(omega * (W*x + b))
  var h0: array<f32, {{HD}}>;
  for (var j: u32 = 0u; j < HD; j++) {
    var val: f32 = l0_b[j];
    for (var i: u32 = 0u; i < INPUT_DIM; i++) {
      val += l0_w[j * INPUT_DIM + i] * input_vec[i];
    }
    h0[j] = sin(first_omega * val);
  }

  // SIREN Layer 1
  var h1: array<f32, {{HD}}>;
  for (var j: u32 = 0u; j < HD; j++) {
    var val: f32 = l1_b[j];
    for (var i: u32 = 0u; i < HD; i++) {
      val += l1_w[j * HD + i] * h0[i];
    }
    h1[j] = sin(hidden_omega * val);
  }

  // SIREN Layer 2
  var h2: array<f32, {{HD}}>;
  for (var j: u32 = 0u; j < HD; j++) {
    var val: f32 = l2_b[j];
    for (var i: u32 = 0u; i < HD; i++) {
      val += l2_w[j * HD + i] * h1[i];
    }
    h2[j] = sin(hidden_omega * val);
  }

  // Final linear layer (outermost_linear)
  var rgba: array<f32, 4>;
  for (var j: u32 = 0u; j < 4u; j++) {
    var val: f32 = l3_b[j];
    for (var i: u32 = 0u; i < HD; i++) {
      val += l3_w[j * HD + i] * h2[i];
    }
    rgba[j] = val;
  }

  // Preserve the decoder's raw linear RGBA output. The compaction/rendering
  // stage clamps values for display, while training and inference semantics
  // remain identical.
  output[out_idx(0u, fz, fy, fx)] = {{DECODE_TYPE}}(rgba[0]);
  output[out_idx(1u, fz, fy, fx)] = {{DECODE_TYPE}}(rgba[1]);
  output[out_idx(2u, fz, fy, fx)] = {{DECODE_TYPE}}(rgba[2]);
  output[out_idx(3u, fz, fy, fx)] = {{DECODE_TYPE}}(rgba[3]);

}
