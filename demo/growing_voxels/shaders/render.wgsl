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
  voxel: u32,
  rgba: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<VoxelInstance>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) normal: vec3<f32>,
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

fn unpack_rgba8(packed: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(packed & 255u),
    f32((packed >> 8u) & 255u),
    f32((packed >> 16u) & 255u),
    f32((packed >> 24u) & 255u),
  ) / 255.0;
}

@vertex
fn vs_main(
  @location(0) vert_pos: vec3<f32>,
  @location(1) vert_norm: vec3<f32>,
  @builtin(instance_index) iid: u32,
) -> VSOut {
  let inst = instances[iid];
  let RS = u32(u.render_size);
  let plane = RS * RS;
  let x = inst.voxel % RS;
  let y = (inst.voxel / RS) % RS;
  let z = inst.voxel / plane;

  let grid_pos = vec3<f32>(f32(x), f32(y), f32(z));
  let world_raw = (vert_pos + grid_pos) / u.render_size - vec3<f32>(0.5);
  let base_world = select(world_raw, vec3<f32>(world_raw.x, world_raw.z, -world_raw.y), u.rotate_model > 0.5);
  let base_norm = select(vert_norm, vec3<f32>(vert_norm.x, vert_norm.z, -vert_norm.y), u.rotate_model > 0.5);

  var out: VSOut;
  out.pos = u.mvp * vec4<f32>(apply_model_rotation(base_world), 1.0);
  out.color = unpack_rgba8(inst.rgba);
  out.normal = apply_model_rotation(base_norm);
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
