enable f16;

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const H: u32 = {{H}}u;
const K: u32 = 5u;
const P: u32 = C * K;
const CELLS: u32 = S * S * S;
const STATE_ELEMENTS: u32 = CELLS * C;
const LIVING_CHANNEL: u32 = 3u;
const LIVING_THRESHOLD: f32 = 0.1;

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
@group(0) @binding(2) var<storage, read_write> candidate_state: array<f16>;
@group(0) @binding(3) var<storage, read_write> living_history: array<f16>;
@group(0) @binding(4) var<storage, read> kernels: array<f16>;
@group(0) @binding(5) var<storage, read> w1: array<f16>;
@group(0) @binding(6) var<storage, read> b1: array<f16>;
@group(0) @binding(7) var<storage, read> w2: array<f16>;
@group(0) @binding(8) var<uniform> params: Params;

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

fn state_at(step: u32, cell: u32, channel: u32) -> f32 {
  return f32(state_history[step * STATE_ELEMENTS + cell * C + channel]);
}

fn hidden_at(cell: u32, hidden: u32) -> f32 {
  return f32(hidden_history[(params.step * CELLS + cell) * H + hidden]);
}

fn update_mask(cell: u32) -> f32 {
  let random = f32(hash(params.random_seed ^ (params.step * CELLS + cell))) / 4294967295.0;
  return select(0.0, 1.0, random < params.update_probability);
}

fn raw_delta(cell: u32, channel: u32) -> f32 {
  var value = 0.0;
  for (var hidden = 0u; hidden < H; hidden++) {
    value += hidden_at(cell, hidden) * f32(w2[hidden * C + channel]);
  }
  return value;
}

fn convolve(step: u32, cell: u32, channel: u32, kernel: u32) -> f32 {
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
        sum += state_at(step, neighbor, channel) * f32(kernels[kernel_index]);
      }
    }
  }
  return sum;
}

fn living_maxima(cell: u32) -> vec2<f32> {
  let z = cell / (S * S);
  let remainder = cell - z * S * S;
  let y = remainder / S;
  let x = remainder - y * S;
  var pre_max = 0.0;
  var post_max = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let nz = i32(z) + dz;
        let ny = i32(y) + dy;
        let nx = i32(x) + dx;
        if (nz < 0 || ny < 0 || nx < 0 || nz >= i32(S) || ny >= i32(S) || nx >= i32(S)) {
          continue;
        }
        let neighbor = cell_index(u32(nz), u32(ny), u32(nx));
        pre_max = max(pre_max, state_at(params.step, neighbor, LIVING_CHANNEL));
        post_max = max(post_max, f32(candidate_state[neighbor * C + LIVING_CHANNEL]));
      }
    }
  }
  return vec2<f32>(pre_max, post_max);
}

@compute @workgroup_size(64)
fn nca_hidden(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= CELLS * H) { return; }
  let cell = index / H;
  let hidden = index - cell * H;
  var sum = f32(b1[hidden]);
  for (var feature = 0u; feature < P; feature++) {
    let channel = feature / K;
    let kernel = feature - channel * K;
    sum += convolve(params.step, cell, channel, kernel) * f32(w1[feature * H + hidden]);
  }
  hidden_history[(params.step * CELLS + cell) * H + hidden] = f16(max(sum, 0.0));
}

@compute @workgroup_size(64)
fn nca_candidate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= STATE_ELEMENTS) { return; }
  let cell = index / C;
  let channel = index - cell * C;
  candidate_state[index] = f16(state_at(params.step, cell, channel)
    + update_mask(cell) * raw_delta(cell, channel));
}

@compute @workgroup_size(64)
fn nca_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= STATE_ELEMENTS) { return; }
  let cell = index / C;
  let channel = index - cell * C;
  let maxima = living_maxima(cell);
  let alive = maxima.x > LIVING_THRESHOLD && maxima.y > LIVING_THRESHOLD;
  let gate = select(0.0, 1.0, alive);
  state_history[(params.step + 1u) * STATE_ELEMENTS + index] = f16(f32(candidate_state[index]) * gate);
  if (channel == 0u) {
    living_history[params.step * CELLS + cell] = f16(gate);
  }
}