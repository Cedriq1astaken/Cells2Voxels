enable f16;

struct Params {
  count: u32,
  iteration: u32,
  normalize: u32,
  pad0: u32,
  learning_rate: f32,
  beta1: f32,
  beta2: f32,
  beta1_correction: f32,
  beta2_correction: f32,
  epsilon: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> weights: array<f16>;
@group(0) @binding(1) var<storage, read_write> gradients: array<f16>;
@group(0) @binding(2) var<storage, read_write> first_moment: array<f16>;
@group(0) @binding(3) var<storage, read_write> second_moment: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(1)
fn normalize_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u || params.normalize == 0u) { return; }
  var squared_norm = f16(0.0);
  for (var index = 0u; index < params.count; index++) {
    let clipped = clamp(gradients[index], f16(-1.0), f16(1.0));
    gradients[index] = clipped;
    squared_norm += clipped * clipped;
  }
  let inverse_norm = f16(1.0) / (sqrt(squared_norm) + f16(0.0001));
  for (var index = 0u; index < params.count; index++) {
    gradients[index] *= inverse_norm;
  }
}

@compute @workgroup_size(64)
fn adam_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.count) { return; }
  let gradient = clamp(gradients[index], f16(-1.0), f16(1.0));
  let beta1 = f16(params.beta1);
  let beta2 = f16(params.beta2);
  let moment1 = beta1 * first_moment[index] + (f16(1.0) - beta1) * gradient;
  let moment2 = beta2 * second_moment[index]
    + (f16(1.0) - beta2) * gradient * gradient;
  first_moment[index] = moment1;
  second_moment[index] = moment2;
  let corrected1 = moment1 / f16(params.beta1_correction);
  let corrected2 = moment2 / f16(params.beta2_correction);
  weights[index] -= f16(params.learning_rate) * corrected1
    / (sqrt(corrected2) + f16(params.epsilon));
}
