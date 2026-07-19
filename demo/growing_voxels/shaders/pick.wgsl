const RS: u32 = {{RS}}u;
const RS_F: f32 = f32(RS);
const VOL: u32 = RS * RS * RS;
const MAX_STEPS: u32 = 3u * RS + 3u;
const EPSILON: f32 = 0.00001;
const INF: f32 = 1.0e30;

struct PickParams {
  raw_origin: vec3<f32>,
  alpha_threshold: f32,
  raw_direction: vec3<f32>,
  pad: f32,
}

// The same planar RGBA volume consumed by compact.wgsl.
@group(0) @binding(0) var<storage, read> voxels: array<f32>;
@group(0) @binding(1) var<uniform> params: PickParams;
// [found, x, y, z]
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(1)
fn pick_visible_voxel() {
  result[0] = 0u;
  result[1] = 0u;
  result[2] = 0u;
  result[3] = 0u;

  // Convert the decoded volume's normalized raw coordinates [-0.5, 0.5]
  // into its integer fine-grid coordinates [0, RS].
  let origin = (params.raw_origin + vec3<f32>(0.5)) * RS_F;
  let direction = params.raw_direction * RS_F;
  if (dot(direction, direction) < EPSILON * EPSILON) { return; }

  // Slab intersection against the decoded-volume bounds.
  var t_near = -INF;
  var t_far = INF;
  for (var axis: u32 = 0u; axis < 3u; axis++) {
    let o = origin[axis];
    let d = direction[axis];
    if (abs(d) < EPSILON) {
      if (o < 0.0 || o > RS_F) { return; }
    } else {
      let t0 = (0.0 - o) / d;
      let t1 = (RS_F - o) / d;
      t_near = max(t_near, min(t0, t1));
      t_far = min(t_far, max(t0, t1));
    }
  }
  let start_t = max(t_near, 0.0);
  if (t_far < start_t) { return; }

  // Start just inside the box, so a ray on a boundary selects its entry cell.
  let start = clamp(
    origin + direction * (start_t + EPSILON),
    vec3<f32>(0.0),
    vec3<f32>(RS_F - EPSILON),
  );
  var cell = vec3<i32>(floor(start));

  var t_max_x = INF;
  var t_max_y = INF;
  var t_max_z = INF;
  var t_delta_x = INF;
  var t_delta_y = INF;
  var t_delta_z = INF;
  let step_x: i32 = select(-1, 1, direction.x > 0.0);
  let step_y: i32 = select(-1, 1, direction.y > 0.0);
  let step_z: i32 = select(-1, 1, direction.z > 0.0);

  if (abs(direction.x) >= EPSILON) {
    let boundary = select(f32(cell.x), f32(cell.x + 1), direction.x > 0.0);
    t_max_x = (boundary - origin.x) / direction.x;
    t_delta_x = abs(1.0 / direction.x);
  }
  if (abs(direction.y) >= EPSILON) {
    let boundary = select(f32(cell.y), f32(cell.y + 1), direction.y > 0.0);
    t_max_y = (boundary - origin.y) / direction.y;
    t_delta_y = abs(1.0 / direction.y);
  }
  if (abs(direction.z) >= EPSILON) {
    let boundary = select(f32(cell.z), f32(cell.z + 1), direction.z > 0.0);
    t_max_z = (boundary - origin.z) / direction.z;
    t_delta_z = abs(1.0 / direction.z);
  }

  for (var step: u32 = 0u; step < MAX_STEPS; step++) {
    if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
        cell.x >= i32(RS) || cell.y >= i32(RS) || cell.z >= i32(RS)) {
      return;
    }

    let x = u32(cell.x);
    let y = u32(cell.y);
    let z = u32(cell.z);
    let index = z * RS * RS + y * RS + x;
    // This is exactly compact.wgsl's inclusion predicate.
    if (voxels[3u * VOL + index] > params.alpha_threshold) {
      result[0] = 1u;
      result[1] = x;
      result[2] = y;
      result[3] = z;
      return;
    }

    let next_t = min(t_max_x, min(t_max_y, t_max_z));
    if (next_t > t_far) { return; }
    if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
      cell.x = cell.x + step_x;
      t_max_x = t_max_x + t_delta_x;
    } else if (t_max_y <= t_max_z) {
      cell.y = cell.y + step_y;
      t_max_y = t_max_y + t_delta_y;
    } else {
      cell.z = cell.z + step_z;
      t_max_z = t_max_z + t_delta_z;
    }
  }
}
