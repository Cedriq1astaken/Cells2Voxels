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
@group(0) @binding(3) var<storage, read_write> output_values: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn dense_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.rows * params.output_dim;
  if (index >= total) { return; }
  let row = index / params.output_dim;
  let output_channel = index - row * params.output_dim;
  var sum = biases[output_channel];
  for (var input_channel = 0u; input_channel < params.input_dim; input_channel++) {
    sum += input_values[row * params.input_dim + input_channel]
      * weights[input_channel * params.output_dim + output_channel];
  }
  output_values[index] = select(
    sum,
    sin(f16(params.omega) * sum),
    params.omega > 0.0,
  );
}
