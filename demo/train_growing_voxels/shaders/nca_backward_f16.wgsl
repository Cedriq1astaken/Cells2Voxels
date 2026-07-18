enable f16;

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const H: u32 = {{H}}u;
const K: u32 = 5u;
const P: u32 = C * K;
const CELLS: u32 = S * S * S;
const STATE_ELEMENTS: u32 = CELLS * C;

struct Params {
  step: u32,
  rollout_steps: u32,
  random_seed: u32,
  pad0: u32,
  update_probability: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var<storage, read> input0: array<f16>;
@group(0) @binding(1) var<storage, read> input1: array<f16>;
@group(0) @binding(2) var<storage, read> input2: array<f16>;
@group(0) @binding(3) var<storage, read> input3: array<f16>;
@group(0) @binding(4) var<storage, read_write> output0: array<f16>;
@group(0) @binding(5) var<storage, read_write> output1: array<f16>;
@group(0) @binding(6) var<storage, read_write> accumulator: array<f16>;
@group(0) @binding(7) var<uniform> params: Params;

fn hash(value: u32) -> u32 {
  var x = value;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn wrap(value: i32) -> u32 {
  let size = i32(S);
  return u32((value % size + size) % size);
}

fn cell_index(z: u32, y: u32, x: u32) -> u32 {
  return z * S * S + y * S + x;
}

fn history_state(cell: u32, channel: u32) -> f32 {
  return f32(input0[params.step * STATE_ELEMENTS + cell * C + channel]);
}

fn hidden_value(cell: u32, hidden: u32) -> f32 {
  return f32(input1[(params.step * CELLS + cell) * H + hidden]);
}

fn update_mask(cell: u32) -> f32 {
  let random = f32(hash(params.random_seed ^ (params.step * CELLS + cell))) / 4294967295.0;
  return select(0.0, 1.0, random < params.update_probability);
}

fn convolve_state(cell: u32, channel: u32, kernel: u32) -> f32 {
  let z = cell / (S * S);
  let remainder = cell - z * S * S;
  let y = remainder / S;
  let x = remainder - y * S;
  var sum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let neighbor = cell_index(
          wrap(i32(z) + dz),
          wrap(i32(y) + dy),
          wrap(i32(x) + dx),
        );
        let kernel_index = kernel * 27u
          + u32(dz + 1) * 9u
          + u32(dy + 1) * 3u
          + u32(dx + 1);
        sum += history_state(neighbor, channel) * f32(input2[kernel_index]);
      }
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn nca_local_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= STATE_ELEMENTS) { return; }
  let cell = index / C;
  let gate = f32(accumulator[params.step * CELLS + cell]);
  let candidate_gradient = f32(input3[index]) * gate;
  output0[index] = f16(candidate_gradient * update_mask(cell));
  output1[index] = f16(candidate_gradient);
}

@compute @workgroup_size(64)
fn nca_hidden_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= CELLS * H) { return; }
  let cell = index / H;
  let hidden = index - cell * H;
  var gradient = 0.0;
  for (var channel = 0u; channel < C; channel++) {
    gradient += f32(input2[cell * C + channel]) * f32(input1[hidden * C + channel]);
  }
  output0[index] = f16(select(
    0.0,
    gradient,
    f32(input0[(params.step * CELLS + cell) * H + hidden]) > 0.0,
  ));
}

@compute @workgroup_size(64)
fn nca_perception_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= CELLS * P) { return; }
  let cell = index / P;
  let feature = index - cell * P;
  var gradient = 0.0;
  for (var hidden = 0u; hidden < H; hidden++) {
    gradient += f32(input0[cell * H + hidden]) * f32(input1[feature * H + hidden]);
  }
  output0[index] = f16(gradient);
}

@compute @workgroup_size(64)
fn nca_state_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= STATE_ELEMENTS) { return; }
  let cell = index / C;
  let channel = index - cell * C;
  let z = cell / (S * S);
  let remainder = cell - z * S * S;
  let y = remainder / S;
  let x = remainder - y * S;
  var gradient = f32(input0[index]);
  for (var kernel = 0u; kernel < K; kernel++) {
    for (var dz = -1; dz <= 1; dz++) {
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let output_cell = cell_index(
            wrap(i32(z) - dz),
            wrap(i32(y) - dy),
            wrap(i32(x) - dx),
          );
          let kernel_index = kernel * 27u
            + u32(dz + 1) * 9u
            + u32(dy + 1) * 3u
            + u32(dx + 1);
          gradient += f32(input1[output_cell * P + channel * K + kernel])
            * f32(input2[kernel_index]);
        }
      }
    }
  }
  output0[index] = f16(gradient);
}

@compute @workgroup_size(64)
fn accumulate_w2_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= H * C) { return; }
  let hidden = index / C;
  let channel = index - hidden * C;
  var gradient = 0.0;
  for (var cell = 0u; cell < CELLS; cell++) {
    gradient += f32(input0[(params.step * CELLS + cell) * H + hidden])
      * f32(input1[cell * C + channel]);
  }
  accumulator[index] = f16(f32(accumulator[index]) + gradient);
}

@compute @workgroup_size(64)
fn accumulate_w1_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= P * H) { return; }
  let feature = index / H;
  let hidden = index - feature * H;
  let channel = feature / K;
  let kernel = feature - channel * K;
  var gradient = 0.0;
  for (var cell = 0u; cell < CELLS; cell++) {
    gradient += convolve_state(cell, channel, kernel) * f32(input1[cell * H + hidden]);
  }
  accumulator[index] = f16(f32(accumulator[index]) + gradient);
}

@compute @workgroup_size(64)
fn accumulate_b1_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hidden = gid.x;
  if (hidden >= H) { return; }
  var gradient = 0.0;
  for (var cell = 0u; cell < CELLS; cell++) {
    gradient += f32(input0[cell * H + hidden]);
  }
  accumulator[hidden] = f16(f32(accumulator[hidden]) + gradient);
}