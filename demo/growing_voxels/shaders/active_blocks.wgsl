{{F16_ENABLE}}

const RS: u32 = {{RS}}u;
const BLOCK_SIZE: u32 = 4u;
const BLOCKS: u32 = (RS + BLOCK_SIZE - 1u) / BLOCK_SIZE;
const BLOCK_COUNT: u32 = BLOCKS * BLOCKS * BLOCKS;
const THRESHOLD: f32 = {{LIVING_THRESHOLD}};

@group(0) @binding(0) var<storage, read> fine_alpha: array<{{DECODE_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> active_blocks: array<u32>;
// [dispatch x, dispatch y, dispatch z, active block count]
@group(0) @binding(2) var<storage, read_write> dispatch_args: array<atomic<u32>>;

fn alpha_at(x: u32, y: u32, z: u32) -> f32 {
  return f32(fine_alpha[z * RS * RS + y * RS + x]);
}

@compute @workgroup_size(64)
fn find_active_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
  let block_index = gid.x;
  if (block_index >= BLOCK_COUNT) { return; }

  let bx = block_index % BLOCKS;
  let by = (block_index / BLOCKS) % BLOCKS;
  let bz = block_index / (BLOCKS * BLOCKS);
  let start_x = bx * BLOCK_SIZE;
  let start_y = by * BLOCK_SIZE;
  let start_z = bz * BLOCK_SIZE;

  // A fine voxel is living when any alpha in its zero-padded 3x3x3
  // neighborhood exceeds the threshold. Scan the whole block plus that
  // one-voxel border so no voxel accepted by the dense path can be skipped.
  let min_x = max(i32(start_x) - 1, 0);
  let min_y = max(i32(start_y) - 1, 0);
  let min_z = max(i32(start_z) - 1, 0);
  let max_x = min(start_x + BLOCK_SIZE, RS - 1u);
  let max_y = min(start_y + BLOCK_SIZE, RS - 1u);
  let max_z = min(start_z + BLOCK_SIZE, RS - 1u);

  var is_active = false;
  for (var z = u32(min_z); z <= max_z && !is_active; z++) {
    for (var y = u32(min_y); y <= max_y && !is_active; y++) {
      for (var x = u32(min_x); x <= max_x; x++) {
        if (alpha_at(x, y, z) > THRESHOLD) {
          is_active = true;
          break;
        }
      }
    }
  }

  if (is_active) {
    let slot = atomicAdd(&dispatch_args[3], 1u);
    active_blocks[slot] = block_index;
  }
}

@compute @workgroup_size(1)
fn finalize_dispatch() {
  let count = atomicLoad(&dispatch_args[3]);
  if (count == 0u) {
    atomicStore(&dispatch_args[0], 0u);
    atomicStore(&dispatch_args[1], 1u);
    atomicStore(&dispatch_args[2], 1u);
    return;
  }

  // Keep both dimensions small while wasting at most one short final row.
  let width = u32(ceil(sqrt(f32(count))));
  let height = (count + width - 1u) / width;
  atomicStore(&dispatch_args[0], width);
  atomicStore(&dispatch_args[1], height);
  atomicStore(&dispatch_args[2], 1u);
}