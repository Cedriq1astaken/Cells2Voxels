const S: u32 = {{S}}u;
const VOL: u32 = S * S * S;
const START_CH: u32 = {{START_CH}}u;

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var out_tex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn pack_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let z = gid.z;
  if (x >= S || y >= S || z >= S) { return; }
  
  let voxel_idx = z * S * S + y * S + x;
  
  textureStore(out_tex, vec3<i32>(gid), vec4<f32>(
    state[(START_CH + 0u) * VOL + voxel_idx],
    state[(START_CH + 1u) * VOL + voxel_idx],
    state[(START_CH + 2u) * VOL + voxel_idx],
    state[(START_CH + 3u) * VOL + voxel_idx]
  ));
}
