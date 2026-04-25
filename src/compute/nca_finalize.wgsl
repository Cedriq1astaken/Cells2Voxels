@group(0) @binding(0) var<storage, read> statePrev: array<f32>;
@group(0) @binding(1) var<storage, read> stateRaw: array<f32>;
@group(0) @binding(2) var<storage, read_write> stateOut: array<f32>;
@group(0) @binding(3) var<storage, read> config: array<u32>;

fn coarseSize() -> u32 {
  return config[0];
}

fn channels() -> u32 {
  return config[1];
}

fn alphaChannel() -> u32 {
  return config[2];
}

fn stateIndex(pos: vec3<u32>, channel: u32) -> u32 {
  let size = coarseSize();
  return (((pos.z * size + pos.y) * size + pos.x) * channels()) + channel;
}

fn clampedCoord(value: i32) -> u32 {
  return u32(clamp(value, 0, i32(coarseSize()) - 1));
}

fn isPrevAlive(center: vec3<u32>) -> bool {
  var value = -1.0;
  for (var z: i32 = -1; z <= 1; z++) {
    for (var y: i32 = -1; y <= 1; y++) {
      for (var x: i32 = -1; x <= 1; x++) {
        let sample = vec3<u32>(
          clampedCoord(i32(center.x) + x),
          clampedCoord(i32(center.y) + y),
          clampedCoord(i32(center.z) + z)
        );
        value = max(value, statePrev[stateIndex(sample, alphaChannel())]);
      }
    }
  }
  return value > 0.1;
}

fn isRawAlive(center: vec3<u32>) -> bool {
  var value = -1.0;
  for (var z: i32 = -1; z <= 1; z++) {
    for (var y: i32 = -1; y <= 1; y++) {
      for (var x: i32 = -1; x <= 1; x++) {
        let sample = vec3<u32>(
          clampedCoord(i32(center.x) + x),
          clampedCoord(i32(center.y) + y),
          clampedCoord(i32(center.z) + z)
        );
        value = max(value, stateRaw[stateIndex(sample, alphaChannel())]);
      }
    }
  }
  return value > 0.1;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = coarseSize();
  if (any(id >= vec3<u32>(size))) {
    return;
  }

  let preAlive = isPrevAlive(id);
  let postAlive = isRawAlive(id);
  let lifeMask = select(0.0, 1.0, preAlive && postAlive);

  for (var channel: u32 = 0u; channel < channels(); channel++) {
    let index = stateIndex(id, channel);
    stateOut[index] = stateRaw[index] * lifeMask;
  }
}
