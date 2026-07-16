enable f16;

const S: u32 = {{S}}u;
const R: u32 = {{R}}u;
const C: u32 = {{C}}u;
const INPUT_DIM: u32 = C + 6u;
const CELLS: u32 = S * S * S;
const STATE_ELEMENTS: u32 = CELLS * C;
const TOTAL_VOXELS: u32 = R * R * R;
const LIVING_CHANNEL: u32 = 3u;

struct Params {
  rows: u32,
  sample_seed: u32,
  sample_offset: u32,
  sample_mode: u32,
  final_step: u32,
  total_voxels: u32,
  state_elements: u32,
  pad0: u32,
  scale: f32,
  overflow_weight: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
  pad4: f32,
  pad5: f32,
  pad6: f32,
}

@group(0) @binding(0) var<storage, read> input0: array<f16>;
@group(0) @binding(1) var<storage, read> input1: array<f16>;
@group(0) @binding(2) var<storage, read> input2: array<f16>;
@group(0) @binding(3) var<storage, read_write> output0: array<f16>;
@group(0) @binding(4) var<storage, read_write> output1: array<f16>;
@group(0) @binding(5) var<storage, read_write> output2: array<f16>;
@group(0) @binding(6) var<uniform> params: Params;

fn hash(value: u32) -> u32 {
  var x = value;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn sample_index(row: u32) -> u32 {
  if (params.sample_mode == 1u) {
    return min(params.sample_offset + row, TOTAL_VOXELS - 1u);
  }
  return hash(row ^ params.sample_seed) % TOTAL_VOXELS;
}

fn wrap(value: i32) -> u32 {
  let size = i32(S);
  return u32((value % size + size) % size);
}

fn sigmoid(value: f16) -> f16 {
  return f16(1.0) / (f16(1.0) + exp(-value));
}

fn state_at(cell: u32, channel: u32) -> f16 {
  return input0[params.final_step * STATE_ELEMENTS + cell * C + channel];
}

fn fine_coordinates(voxel: u32) -> vec3<u32> {
  let z = voxel / (R * R);
  let remainder = voxel - z * R * R;
  let y = remainder / R;
  let x = remainder - y * R;
  return vec3<u32>(x, y, z);
}

fn interpolate_state(voxel: u32, channel: u32) -> f16 {
  let fine = fine_coordinates(voxel);
  let scale = params.scale;
  let cx = (f32(fine.x) + 0.5) / scale - 0.5;
  let cy = (f32(fine.y) + 0.5) / scale - 0.5;
  let cz = (f32(fine.z) + 0.5) / scale - 0.5;
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let z0 = i32(floor(cz));
  let wx = f16(cx - f32(x0));
  let wy = f16(cy - f32(y0));
  let wz = f16(cz - f32(z0));
  var value = f16(0.0);
  for (var dz = 0; dz <= 1; dz++) {
    for (var dy = 0; dy <= 1; dy++) {
      for (var dx = 0; dx <= 1; dx++) {
        let x = wrap(x0 + dx);
        let y = wrap(y0 + dy);
        let z = wrap(z0 + dz);
        let cell = z * S * S + y * S + x;
        let x_weight = select(f16(1.0) - wx, wx, dx == 1);
        let y_weight = select(f16(1.0) - wy, wy, dy == 1);
        let z_weight = select(f16(1.0) - wz, wz, dz == 1);
        value += state_at(cell, channel) * x_weight * y_weight * z_weight;
      }
    }
  }
  return value;
}

fn interpolation_weight(voxel: u32, coarse_cell: u32) -> f16 {
  let fine = fine_coordinates(voxel);
  let scale = params.scale;
  let cx = (f32(fine.x) + 0.5) / scale - 0.5;
  let cy = (f32(fine.y) + 0.5) / scale - 0.5;
  let cz = (f32(fine.z) + 0.5) / scale - 0.5;
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let z0 = i32(floor(cz));
  let wx = f16(cx - f32(x0));
  let wy = f16(cy - f32(y0));
  let wz = f16(cz - f32(z0));
  let coarse_z = coarse_cell / (S * S);
  let remainder = coarse_cell - coarse_z * S * S;
  let coarse_y = remainder / S;
  let coarse_x = remainder - coarse_y * S;
  var weight = f16(0.0);
  for (var dz = 0; dz <= 1; dz++) {
    for (var dy = 0; dy <= 1; dy++) {
      for (var dx = 0; dx <= 1; dx++) {
        if (wrap(x0 + dx) == coarse_x && wrap(y0 + dy) == coarse_y && wrap(z0 + dz) == coarse_z) {
          let x_weight = select(f16(1.0) - wx, wx, dx == 1);
          let y_weight = select(f16(1.0) - wy, wy, dy == 1);
          let z_weight = select(f16(1.0) - wz, wz, dz == 1);
          weight += x_weight * y_weight * z_weight;
        }
      }
    }
  }
  return weight;
}

@compute @workgroup_size(64)
fn build_lppn_input(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.rows * INPUT_DIM;
  if (index >= total) { return; }
  let row = index / INPUT_DIM;
  let component = index - row * INPUT_DIM;
  let voxel = sample_index(row);
  let fine = fine_coordinates(voxel);
  let local_x = ((f32(fine.x % u32(params.scale)) + 0.5) / params.scale - 0.5) * 2.0;
  let local_y = ((f32(fine.y % u32(params.scale)) + 0.5) / params.scale - 0.5) * 2.0;
  let local_z = ((f32(fine.z % u32(params.scale)) + 0.5) / params.scale - 0.5) * 2.0;
  var value = f16(0.0);
  if (component == 0u) { value = sin(f16(3.14159265 * local_z)); }
  if (component == 1u) { value = sin(f16(3.14159265 * local_y)); }
  if (component == 2u) { value = sin(f16(3.14159265 * local_x)); }
  if (component == 3u) { value = cos(f16(3.14159265 * local_z)); }
  if (component == 4u) { value = cos(f16(3.14159265 * local_y)); }
  if (component == 5u) { value = cos(f16(3.14159265 * local_x)); }
  if (component >= 6u) { value = interpolate_state(voxel, component - 6u); }
  output0[index] = value;
}

@compute @workgroup_size(64)
fn morphology_loss_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) { return; }
  let voxel = sample_index(row);
  let alpha = input0[row * INPUT_DIM + 6u + LIVING_CHANNEL];
  let living = sigmoid((alpha - f16(0.1)) * f16(20.0));
  let rgba_normalizer = f16(1.0 / f32(params.rows * 4u));
  let occupancy_normalizer = f16(1.0 / f32(params.rows));
  var alpha_gradient = f16(0.0);
  var loss = f16(0.0);
  for (var channel = 0u; channel < 4u; channel++) {
    let raw = input1[row * 4u + channel];
    let target_value = input2[voxel * 4u + channel];
    let generated = raw * living;
    let difference = generated - target_value;
    let generated_gradient = (f16(2.0) * difference + sign(difference)) * rgba_normalizer;
    output0[row * 4u + channel] = generated_gradient * living;
    alpha_gradient += generated_gradient * raw * living * (f16(1.0) - living) * f16(20.0);
    loss += (difference * difference + abs(difference)) * rgba_normalizer;
  }
  let target_alpha = input2[voxel * 4u + 3u];
  let alpha_state = alpha * f16(2.0);
  let occupancy_difference = alpha_state - target_alpha;
  alpha_gradient += f16(2.0) * (f16(2.0) * occupancy_difference + sign(occupancy_difference))
    * occupancy_normalizer;
  loss += (occupancy_difference * occupancy_difference + abs(occupancy_difference))
    * occupancy_normalizer;
  output1[row] = alpha_gradient;
  output2[row] = loss;
}

@compute @workgroup_size(1)
fn reduce_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  var loss = f16(0.0);
  for (var row = 0u; row < params.rows; row++) {
    loss += input0[row];
  }
  let state_offset = params.final_step * STATE_ELEMENTS;
  let overflow_normalizer = f16(params.overflow_weight / f32(STATE_ELEMENTS));
  for (var index = 0u; index < STATE_ELEMENTS; index++) {
    let value = input1[state_offset + index];
    let excess = value * value - f16(1.0);
    if (excess > f16(0.0)) {
      loss += excess * overflow_normalizer;
    }
  }
  output0[0] = loss;
}

@compute @workgroup_size(64)
fn coarse_state_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= STATE_ELEMENTS) { return; }
  let coarse_cell = index / C;
  let channel = index - coarse_cell * C;
  var gradient = f16(0.0);
  for (var row = 0u; row < params.rows; row++) {
    let voxel = sample_index(row);
    let weight = interpolation_weight(voxel, coarse_cell);
    gradient += input1[row * INPUT_DIM + 6u + channel] * weight;
    if (channel == LIVING_CHANNEL) {
      gradient += input2[row] * weight * f16(2.0);
    }
  }
  let value = input0[params.final_step * STATE_ELEMENTS + index];
  if (abs(value) > f16(1.0)) {
    gradient += f16(2.0 * params.overflow_weight / f32(STATE_ELEMENTS)) * value;
  }
  output0[index] = gradient;
}

@compute @workgroup_size(64)
fn write_preview(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.rows * 4u;
  if (index >= total) { return; }
  let row = index / 4u;
  let channel = index - row * 4u;
  let alpha = input0[row * INPUT_DIM + 6u + LIVING_CHANNEL];
  let living = sigmoid((alpha - f16(0.1)) * f16(20.0));
  output0[(params.sample_offset + row) * 4u + channel] = input1[index] * living;
}
