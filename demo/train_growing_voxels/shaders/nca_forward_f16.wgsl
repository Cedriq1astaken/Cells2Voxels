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
  state_elements: u32,
  update_probability: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> state_history: array<f16>;
@group(0) @binding(1) var<storage, read_write> hidden_history: array<f16>;
@group(0) @binding(2) var<storage, read> kernels: array<f16>;
@group(0) @binding(3) var<storage, read> w1: array<f16>;
@group(0) @binding(4) var<storage, read> b1: array<f16>;
@group(0) @binding(5) var<storage, read> w2: array<f16>;
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

fn state_at(step: u32, cell: u32, channel: u32) -> f16 {
  return state_history[step * STATE_ELEMENTS + cell * C + channel];
}

fn convolve(step: u32, cell: u32, channel: u32, kernel: u32) -> f16 {
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
        sum += state_at(step, neighbor, channel) * kernels[kernel_index];
      }
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn nca_hidden(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= CELLS * H) { return; }
  let cell = index / H;
  let hidden = index - cell * H;
  var sum = b1[hidden];
  for (var feature = 0u; feature < P; feature++) {
    let kernel = feature / C;
    let channel = feature - kernel * C;
    sum += convolve(params.step, cell, channel, kernel) * w1[feature * H + hidden];
  }
  hidden_history[(params.step * CELLS + cell) * H + hidden] = max(sum, f16(0.0));
}

@compute @workgroup_size(64)
fn nca_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= STATE_ELEMENTS) { return; }
  let cell = index / C;
  let channel = index - cell * C;
  let state_offset = params.step * STATE_ELEMENTS;
  let hidden_offset = (params.step * CELLS + cell) * H;
  let mask_random = f32(hash(params.random_seed ^ (params.step * CELLS + cell))) / 4294967295.0;
  let mask = select(f16(0.0), f16(1.0), mask_random < params.update_probability);

  var raw_delta = f16(0.0);
  var raw_alpha_delta = f16(0.0);
  for (var hidden = 0u; hidden < H; hidden++) {
    let activation = hidden_history[hidden_offset + hidden];
    raw_delta += activation * w2[hidden * C + channel];
    raw_alpha_delta += activation * w2[hidden * C + LIVING_CHANNEL];
  }

  let old_value = state_history[state_offset + index];
  let old_alpha = state_history[state_offset + cell * C + LIVING_CHANNEL];
  let candidate = old_value + mask * tanh(raw_delta);
  let candidate_alpha = old_alpha + mask * tanh(raw_alpha_delta);
  let pre_living = sigmoid((old_alpha - f16(0.1)) * f16(20.0));
  let post_living = sigmoid((candidate_alpha - f16(0.1)) * f16(20.0));
  state_history[state_offset + STATE_ELEMENTS + index] = candidate * pre_living * post_living;
}
