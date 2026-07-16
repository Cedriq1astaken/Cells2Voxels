enable f16;

struct Uniforms {
  mvp: mat4x4<f32>,
  camera_position: vec3<f32>,
  render_size: f32,
  light_direction: vec3<f32>,
  scale_inverse: f32,
}

struct VoxelInstance {
  px: f16,
  py: f16,
  pz: f16,
  r: f16,
  g: f16,
  b: f16,
  a: f16,
  pad: f16,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<VoxelInstance>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) normal: vec3<f32>,
}

@vertex
fn vertex_main(
  @location(0) vertex_position: vec3<f32>,
  @location(1) vertex_normal: vec3<f32>,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  let instance = instances[instance_index];
  let grid_position = (
    vertex_position
    + vec3<f32>(f32(instance.px), f32(instance.py), f32(instance.pz))
  ) / uniforms.render_size - vec3<f32>(0.5);
  let world_position = vec3<f32>(grid_position.x, grid_position.z, -grid_position.y);

  var output: VertexOutput;
  output.position = uniforms.mvp * vec4<f32>(world_position, 1.0);
  output.color = vec4<f32>(
    f32(instance.r),
    f32(instance.g),
    f32(instance.b),
    f32(instance.a),
  );
  output.normal = vec3<f32>(vertex_normal.x, vertex_normal.z, -vertex_normal.y);
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let diffuse = max(dot(normalize(input.normal), normalize(uniforms.light_direction)), 0.0);
  let lit = input.color.rgb * (0.38 + diffuse * 0.62);
  let visible_alpha = clamp(0.35 + input.color.a * 0.65, 0.0, 1.0);
  return vec4<f32>(lit, visible_alpha);
}
