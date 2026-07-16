enable f16;

struct Params {
  rows: u32,
  input_dim: u32,
  output_dim: u32,
  pad0: u32,
  omega: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var<storage, read> input_values: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read> biases: array<f16>;
@group(0) @binding(3) var<storage, read> upstream_values: array<f16>;
@group(0) @binding(4) var<storage, read_write> output_values: array<f16>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn activation_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.rows * params.output_dim;
  if (index >= total) { return; }
  let row = index / params.output_dim;
  let output_channel = index - row * params.output_dim;
  var preactivation = biases[output_channel];
  for (var input_channel = 0u; input_channel < params.input_dim; input_channel++) {
    preactivation += input_values[row * params.input_dim + input_channel]
      * weights[input_channel * params.output_dim + output_channel];
  }
  let derivative = select(
    f16(1.0),
    f16(params.omega) * cos(f16(params.omega) * preactivation),
    params.omega > 0.0,
  );
  output_values[index] = upstream_values[index] * derivative;
}

@compute @workgroup_size(64)
fn dense_dx(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.rows * params.input_dim;
  if (index >= total) { return; }
  let row = index / params.input_dim;
  let input_channel = index - row * params.input_dim;
  var sum = f16(0.0);
  for (var output_channel = 0u; output_channel < params.output_dim; output_channel++) {
    sum += input_values[row * params.output_dim + output_channel]
      * weights[input_channel * params.output_dim + output_channel];
  }
  output_values[index] = sum;
}

@compute @workgroup_size(64)
fn dense_dw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.input_dim * params.output_dim;
  if (index >= total) { return; }
  let input_channel = index / params.output_dim;
  let output_channel = index - input_channel * params.output_dim;
  var sum = f16(0.0);
  for (var row = 0u; row < params.rows; row++) {
    sum += input_values[row * params.input_dim + input_channel]
      * upstream_values[row * params.output_dim + output_channel];
  }
  output_values[index] = sum;
}

@compute @workgroup_size(64)
fn dense_db(@builtin(global_invocation_id) gid: vec3<u32>) {
  let output_channel = gid.x;
  if (output_channel >= params.output_dim) { return; }
  var sum = f16(0.0);
  for (var row = 0u; row < params.rows; row++) {
    sum += upstream_values[row * params.output_dim + output_channel];
  }
  output_values[output_channel] = sum;
}
