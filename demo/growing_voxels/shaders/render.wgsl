struct Uniforms {
  mvp: mat4x4<f32>,
  cam_pos: vec3<f32>,
  render_size: f32,
  light_dir: vec3<f32>,
  scale_inv: f32,
  rotate_model: f32,
  model_rotation: vec3<f32>,
}

struct VoxelInstance {
  px: f32, py: f32, pz: f32,
  r: f32, g: f32, b: f32, a: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<VoxelInstance>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) world_pos: vec3<f32>,
}

fn rotate_x(v: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(v.x, c * v.y - s * v.z, s * v.y + c * v.z);
}

fn rotate_y(v: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn rotate_z(v: vec3<f32>, angle: f32) -> vec3<f32> {
  let c = cos(angle);
  let s = sin(angle);
  return vec3<f32>(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

fn apply_model_rotation(v: vec3<f32>) -> vec3<f32> {
  return rotate_z(rotate_y(rotate_x(v, u.model_rotation.x), u.model_rotation.y), u.model_rotation.z);
}

@vertex
fn vs_main(
  @location(0) vert_pos: vec3<f32>,
  @location(1) vert_norm: vec3<f32>,
  @builtin(instance_index) iid: u32,
) -> VSOut {
  let inst = instances[iid];
  let RS = u.render_size;
  // Center the model and normalize to [-0.5, 0.5]
  let world_raw = (vert_pos + vec3<f32>(inst.px, inst.py, inst.pz)) / RS - vec3<f32>(0.5);
  let base_world = select(world_raw, vec3<f32>(world_raw.x, world_raw.z, -world_raw.y), u.rotate_model > 0.5);
  let base_norm = select(vert_norm, vec3<f32>(vert_norm.x, vert_norm.z, -vert_norm.y), u.rotate_model > 0.5);
  let world = apply_model_rotation(base_world);
  let rotated_norm = apply_model_rotation(base_norm);
  
  var out: VSOut;
  out.pos = u.mvp * vec4<f32>(world, 1.0);
  out.color = vec4<f32>(inst.r, inst.g, inst.b, inst.a);
  out.normal = rotated_norm;
  out.world_pos = world;
  return out;
}

@fragment
fn fs_main(v: VSOut) -> @location(0) vec4<f32> {
  let n = normalize(v.normal);
  let l = normalize(u.light_dir);
  let diffuse = max(dot(n, l), 0.0);
  let ambient = 0.35;
  let lit = v.color.rgb * (ambient + diffuse * 0.65);
  return vec4<f32>(lit, v.color.a);
}
