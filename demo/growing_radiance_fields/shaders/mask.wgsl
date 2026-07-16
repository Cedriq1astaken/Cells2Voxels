const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const VOL: u32 = S * S * S;
const LIVING_THRESHOLD: f32 = {{LIVING_THRESHOLD}};

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var mask_tex: texture_storage_3d<r32float, write>;

fn state_idx(c: u32, z: u32, y: u32, x: u32) -> u32 {
  return c * VOL + z * S * S + y * S + x;
}

fn state_at(c: u32, z: i32, y: i32, x: i32) -> f32 {
  if (z < 0 || y < 0 || x < 0 || z >= i32(S) || y >= i32(S) || x >= i32(S)) { return 0.0; }
  return state[state_idx(c, u32(z), u32(y), u32(x))];
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let z = gid.z;
  
  if (x >= S || y >= S || z >= S) {
    return;
  }
  
  var m: f32 = 0.0;
  for (var dz: i32 = -1; dz <= 1; dz++) {
    for (var dy: i32 = -1; dy <= 1; dy++) {
      for (var dx: i32 = -1; dx <= 1; dx++) {
        m = max(m, state_at({{LC}}u, i32(z) + dz, i32(y) + dy, i32(x) + dx));
      }
    }
  }
  
  let is_alive = select(0.0, 1.0, m > LIVING_THRESHOLD);
  textureStore(mask_tex, vec3<i32>(gid), vec4<f32>(is_alive, 0.0, 0.0, 0.0));
}
