enable f16;

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const H: u32 = {{H}}u;
const P: u32 = C * 5u;
const CELLS: u32 = S * S * S;
const STATE_ELEMENTS: u32 = CELLS * C;
const LIVING_CHANNEL: u32 = 3u;

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

fn sigmoid(value: f16) -> f16 {
  return f16(1.0) / (f16(1.0) + exp(-value));
}

fn wrap(value: i32) -> u32 {
  let size = i32(S);
  return u32((value % size + size) % size);
}

fn cell_index(z: u32, y: u32, x: u32) -> u32 {
  return z * S * S + y * S + x;
}

fn history_state(cell: u32, channel: u32) -> f16 {
  return input0[params.step * STATE_ELEMENTS + cell * C + channel];
}

fn hidden_value(cell: u32, hidden: u32) -> f16 {
  return input1[(params.step * CELLS + cell) * H + hidden];
}

fn raw_delta(cell: u32, channel: u32) -> f16 {
  var value = f16(0.0);
  for (var hidden = 0u; hidden < H; hidden++) {
    value += hidden_value(cell, hidden) * input2[hidden * C + channel];
  }
  return value;
}

fn update_mask(cell: u32) -> f16 {
  let random = f32(hash(params.random_seed ^ (params.step * CELLS + cell))) / 4294967295.0;
  return select(f16(0.0), f16(1.0), random < params.update_probability);
}

fn convolve_state(cell: u32, channel: u32, kernel: u32) -> f16 {
  let z = cell / (S * S);
  let remainder = cell - z * S * S;
  let y = remainder / S;
  let x = remainder - y * S;
  var sum = f16(0.0);
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
        sum += history_state(neighbor, channel) * input2[kernel_index];
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
  let channel = index - cell * C;
  let mask = update_mask(cell);
  let old_alpha = history_state(cell, LIVING_CHANNEL);
  let alpha_raw = raw_delta(cell, LIVING_CHANNEL);
  let candidate_alpha = old_alpha + mask * tanh(alpha_raw);
  let pre_living = sigmoid((old_alpha - f16(0.1)) * f16(20.0));
  let post_living = sigmoid((candidate_alpha - f16(0.1)) * f16(20.0));
  let gate = pre_living * post_living;

  var gate_gradient = f16(0.0);
  for (var output_channel = 0u; output_channel < C; output_channel++) {
    let raw = raw_delta(cell, output_channel);
    let candidate = history_state(cell, output_channel)
      + mask * tanh(raw);
    gate_gradient += input3[cell * C + output_channel] * candidate;
  }

  let raw = raw_delta(cell, channel);
  let tanh_raw = tanh(raw);
  var candidate_gradient = input3[index] * gate;
  var state_gradient = f16(0.0);
  if (channel == LIVING_CHANNEL) {
    candidate_gradient += gate_gradient * pre_living
      * post_living * (f16(1.0) - post_living) * f16(20.0);
    state_gradient += gate_gradient * post_living
      * pre_living * (f16(1.0) - pre_living) * f16(20.0);
  }
  state_gradient += candidate_gradient;
  output0[index] = candidate_gradient * mask
    * (f16(1.0) - tanh_raw * tanh_raw);
  output1[index] = state_gradient;
}

@compute @workgroup_size(64)
fn nca_hidden_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= CELLS * H) { return; }
  let cell = index / H;
  let hidden = index - cell * H;
  var gradient = f16(0.0);
  for (var channel = 0u; channel < C; channel++) {
    gradient += input2[cell * C + channel] * input1[hidden * C + channel];
  }
  output0[index] = select(
    f16(0.0),
    gradient,
    input0[(params.step * CELLS + cell) * H + hidden] > f16(0.0),
  );
}

@compute @workgroup_size(64)
fn nca_perception_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= CELLS * P) { return; }
  let cell = index / P;
  let feature = index - cell * P;
  var gradient = f16(0.0);
  for (var hidden = 0u; hidden < H; hidden++) {
    gradient += input0[cell * H + hidden] * input1[feature * H + hidden];
  }
  output0[index] = gradient;
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
  var gradient = input0[index];
  for (var kernel = 0u; kernel < 5u; kernel++) {
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
          gradient += input1[output_cell * P + kernel * C + channel]
            * input2[kernel_index];
        }
      }
    }
  }
  output0[index] = gradient;
}

@compute @workgroup_size(64)
fn accumulate_w2_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= H * C) { return; }
  let hidden = index / C;
  let channel = index - hidden * C;
  var gradient = f16(0.0);
  for (var cell = 0u; cell < CELLS; cell++) {
    gradient += input0[(params.step * CELLS + cell) * H + hidden]
      * input1[cell * C + channel];
  }
  accumulator[index] += gradient;
}

@compute @workgroup_size(64)
fn accumulate_w1_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= P * H) { return; }
  let feature = index / H;
  let hidden = index - feature * H;
  let kernel = feature / C;
  let channel = feature - kernel * C;
  var gradient = f16(0.0);
  for (var cell = 0u; cell < CELLS; cell++) {
    gradient += convolve_state(cell, channel, kernel) * input1[cell * H + hidden];
  }
  accumulator[index] += gradient;
}

@compute @workgroup_size(64)
fn accumulate_b1_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hidden = gid.x;
  if (hidden >= H) { return; }
  var gradient = f16(0.0);
  for (var cell = 0u; cell < CELLS; cell++) {
    gradient += input0[cell * H + hidden];
  }
  accumulator[hidden] += gradient;
}
