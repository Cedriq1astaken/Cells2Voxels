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

// Forward/backward math stays f16. Adam keeps f32 master parameters and
// moments so small gradients do not underflow its second moment to zero.
@group(0) @binding(0) var<storage, read_write> forward_weights: array<f16>;
@group(0) @binding(1) var<storage, read_write> gradients: array<f16>;
@group(0) @binding(2) var<storage, read_write> master_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> first_moment: array<f32>;
@group(0) @binding(4) var<storage, read_write> second_moment: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn normalize_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u || params.normalize == 0u) { return; }
  var squared_norm = f32(0.0);
  for (var index = 0u; index < params.count; index++) {
    let clipped = clamp(f32(gradients[index]), -1.0, 1.0);
    gradients[index] = f16(clipped);
    squared_norm += clipped * clipped;
  }
  let inverse_norm = 1.0 / (sqrt(squared_norm) + 0.0001);
  for (var index = 0u; index < params.count; index++) {
    gradients[index] = f16(f32(gradients[index]) * inverse_norm);
  }
}

@compute @workgroup_size(64)
fn adam_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.count) { return; }
  let gradient = clamp(f32(gradients[index]), -1.0, 1.0);
  let beta1 = params.beta1;
  let beta2 = params.beta2;
  let moment1 = beta1 * f32(first_moment[index]) + (1.0 - beta1) * gradient;
  let moment2 = beta2 * f32(second_moment[index])
    + (1.0 - beta2) * gradient * gradient;
  first_moment[index] = moment1;
  second_moment[index] = moment2;
  let corrected1 = moment1 / params.beta1_correction;
  let corrected2 = moment2 / params.beta2_correction;
  let updated_weight = master_weights[index] - params.learning_rate * corrected1
    / (sqrt(corrected2) + params.epsilon);
  master_weights[index] = updated_weight;
  forward_weights[index] = f16(updated_weight);
}