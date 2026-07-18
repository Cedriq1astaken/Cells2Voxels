{{F16_ENABLE}}

const S: u32 = {{S}}u;
const RS: u32 = {{RS}}u;
const RS_VOL: u32 = RS * RS * RS;
const LIVING_CHANNEL: u32 = {{LC}}u;

@group(0) @binding(0) var<storage, read> state: array<{{STATE_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> fine_alpha: array<f32>;
@group(0) @binding(2) var<uniform> params: array<vec4<f32>, 3>;

fn state_at(z: i32, y: i32, x: i32) -> f32 {
  let size = i32(S);
  let wz = (z % size + size) % size;
  let wy = (y % size + size) % size;
  let wx = (x % size + size) % size;
  return f32(state[LIVING_CHANNEL * S * S * S + u32(wz) * S * S + u32(wy) * S + u32(wx)]);
}

fn out_idx(z: u32, y: u32, x: u32) -> u32 {
  return z * RS * RS + y * RS + x;
}

@compute @workgroup_size(4, 4, 4)
fn interpolate_living_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
  let fx = gid.x; let fy = gid.y; let fz = gid.z;
  if (fx >= RS || fy >= RS || fz >= RS) { return; }

  let scale = params[0].z;
  let cf_x = (f32(fx) + 0.5) / scale - 0.5;
  let cf_y = (f32(fy) + 0.5) / scale - 0.5;
  let cf_z = (f32(fz) + 0.5) / scale - 0.5;
  let x0 = i32(floor(cf_x)); let y0 = i32(floor(cf_y)); let z0 = i32(floor(cf_z));
  let x1 = x0 + 1; let y1 = y0 + 1; let z1 = z0 + 1;
  let wx = cf_x - f32(x0); let wy = cf_y - f32(y0); let wz = cf_z - f32(z0);

  let c000 = state_at(z0, y0, x0); let c100 = state_at(z0, y0, x1);
  let c010 = state_at(z0, y1, x0); let c110 = state_at(z0, y1, x1);
  let c001 = state_at(z1, y0, x0); let c101 = state_at(z1, y0, x1);
  let c011 = state_at(z1, y1, x0); let c111 = state_at(z1, y1, x1);
  let c00 = c000 * (1.0 - wx) + c100 * wx;
  let c01 = c001 * (1.0 - wx) + c101 * wx;
  let c10 = c010 * (1.0 - wx) + c110 * wx;
  let c11 = c011 * (1.0 - wx) + c111 * wx;
  let c0 = c00 * (1.0 - wy) + c10 * wy;
  let c1 = c01 * (1.0 - wy) + c11 * wy;
  fine_alpha[out_idx(fz, fy, fx)] = c0 * (1.0 - wz) + c1 * wz;
}