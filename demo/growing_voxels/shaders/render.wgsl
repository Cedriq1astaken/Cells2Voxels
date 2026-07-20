struct Uniforms {
  mvp: mat4x4<f32>,
  cam_pos: vec3<f32>,
  render_size: f32,
  light_dir: vec3<f32>,
  scale_inv: f32,
  rotate_model: f32,
  model_rotation: mat3x3<f32>,
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
}

fn apply_model_rotation(v: vec3<f32>) -> vec3<f32> {
  return u.model_rotation * v;
}

@vertex
fn vs_main(
  @location(0) vert_pos: vec3<f32>,
  @location(1) vert_norm: vec3<f32>,
  @builtin(instance_index) iid: u32,
) -> VSOut {
  let inst = instances[iid];
  let grid_pos = vec3<f32>(inst.px, inst.py, inst.pz);
  let world_raw = (vert_pos + grid_pos) / u.render_size - vec3<f32>(0.5);
  let base_world = select(world_raw, vec3<f32>(world_raw.x, world_raw.z, -world_raw.y), u.rotate_model > 0.5);
  let base_norm = select(vert_norm, vec3<f32>(vert_norm.x, vert_norm.z, -vert_norm.y), u.rotate_model > 0.5);

  var out: VSOut;
  out.pos = u.mvp * vec4<f32>(apply_model_rotation(base_world), 1.0);
  out.color = vec4<f32>(inst.r, inst.g, inst.b, inst.a);
  out.normal = apply_model_rotation(base_norm);
  return out;
}

@fragment
fn fs_main(v: VSOut) -> @location(0) vec4<f32> {
  let n = normalize(v.normal);
  let l = normalize(u.light_dir);
  let diffuse = max(dot(n, l), 0.0);
  let ambient = 0.38;
  let lit = v.color.rgb * (ambient + diffuse * 0.65);
  let luminance = dot(lit, vec3<f32>(0.2126, 0.7152, 0.0722));
  let vivid = mix(vec3<f32>(luminance), lit, 1.06);
  let bright = clamp(vivid * 1.08, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(bright, v.color.a);
}
