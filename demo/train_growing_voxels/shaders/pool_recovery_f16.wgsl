enable f16;

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const CELLS: u32 = S * S * S;
const STATE_ELEMENTS: u32 = CELLS * C;
const LIVING_CHANNEL: u32 = 3u;
const LIVING_THRESHOLD: f32 = 0.1;

struct Params {
  pool_word_offset: u32,
  state_elements: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> pool: array<f16>;
@group(0) @binding(1) var<storage, read> seed: array<f16>;
@group(0) @binding(2) var<storage, read_write> state_history: array<f16>;
@group(0) @binding(3) var<storage, read_write> pool_alive: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn inspect_pool_alive(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  if (cell >= CELLS) { return; }
  let alpha = f32(pool[params.pool_word_offset + cell * C + LIVING_CHANNEL]);
  if (alpha > LIVING_THRESHOLD) {
    atomicOr(&pool_alive[0], 1u);
  }
}

@compute @workgroup_size(64)
fn restore_or_reseed(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.state_elements) { return; }
  if (atomicLoad(&pool_alive[0]) == 0u) {
    state_history[index] = seed[index];
  } else {
    state_history[index] = pool[params.pool_word_offset + index];
  }
}