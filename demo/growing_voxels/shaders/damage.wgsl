{{F16_ENABLE}}

const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const VOL: u32 = S * S * S;

struct DamageParams {
  center: vec3<f32>,
  radius: f32,
}

@group(0) @binding(0) var<storage, read_write> state_a: array<{{STATE_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> state_b: array<{{STATE_TYPE}}>;
@group(0) @binding(2) var<uniform> params: DamageParams;

@compute @workgroup_size(4, 4, 4)
fn clear_damage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let z = gid.z;
  if (x >= S || y >= S || z >= S) { return; }

  let delta = vec3<f32>(f32(x), f32(y), f32(z)) - params.center;
  if (dot(delta, delta) > params.radius * params.radius) { return; }

  let voxel = z * S * S + y * S + x;
  for (var c: u32 = 0u; c < C; c++) {
    let index = c * VOL + voxel;
    state_a[index] = {{STATE_TYPE}}(0.0);
    state_b[index] = {{STATE_TYPE}}(0.0);
  }
}
