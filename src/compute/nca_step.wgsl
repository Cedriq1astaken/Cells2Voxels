@group(0) @binding(0) var<storage, read> stateIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> stateOut: array<f32>;
@group(0) @binding(2) var<storage, read> config: array<u32>;
@group(0) @binding(3) var<storage, read> perceptionWeights: array<f32>;
@group(0) @binding(4) var<storage, read> adapt0Weights: array<f32>;
@group(0) @binding(5) var<storage, read> adapt0Bias: array<f32>;
@group(0) @binding(6) var<storage, read> adapt2Weights: array<f32>;
struct StepState {
  tick: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(7) var<uniform> stepState: StepState;

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

fn hash(value: u32) -> u32 {
  var x = value;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn updateMask(id: vec3<u32>) -> ComputeScalar {
  let seed =
    ((id.x * 73856093u)
    ^ (id.y * 19349663u)
    ^ (id.z * 83492791u)
    ^ (stepState.tick * 2654435761u));
  let value = f32(hash(seed)) * (1.0 / 4294967295.0);
  return select(asScalar(0.0), asScalar(1.0), value < 0.5);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = coarseSize();
  if (any(id >= vec3<u32>(size))) {
    return;
  }

  let mask = updateMask(id);

  var features: array<ComputeScalar, NCA_FEATURE_COUNT>;
  for (var channel: u32 = 0; channel < channels(); channel++) {
    for (var kernel: u32 = 0; kernel < 4u; kernel++) {
      var sum: ComputeScalar = asScalar(0.0);
      for (var z: u32 = 0u; z < 3u; z++) {
        for (var y: u32 = 0u; y < 3u; y++) {
          for (var x: u32 = 0u; x < 3u; x++) {
            let sample = vec3<u32>(
              clampedCoord(i32(id.x) + i32(x) - 1),
              clampedCoord(i32(id.y) + i32(y) - 1),
              clampedCoord(i32(id.z) + i32(z) - 1)
            );
            let featureIndex = channel * 4u + kernel;
            let weightIndex = featureIndex * 27u + z * 9u + y * 3u + x;
            sum += asScalar(stateIn[stateIndex(sample, channel)]) * asScalar(perceptionWeights[weightIndex]);
          }
        }
      }
      features[channel * 4u + kernel] = sum;
    }
  }

  var hidden: array<ComputeScalar, NCA_ADAPT_HIDDEN_SIZE>;
  for (var outChannel: u32 = 0u; outChannel < NCA_ADAPT_HIDDEN_SIZE; outChannel++) {
    var sum: ComputeScalar = asScalar(adapt0Bias[outChannel]);
    for (var inChannel: u32 = 0u; inChannel < NCA_FEATURE_COUNT; inChannel++) {
      sum += asScalar(adapt0Weights[outChannel * NCA_FEATURE_COUNT + inChannel]) * features[inChannel];
    }
    hidden[outChannel] = max(sum, asScalar(0.0));
  }

  for (var channel: u32 = 0u; channel < channels(); channel++) {
    var delta: ComputeScalar = asScalar(0.0);
    for (var hiddenChannel: u32 = 0u; hiddenChannel < NCA_ADAPT_HIDDEN_SIZE; hiddenChannel++) {
      delta += asScalar(adapt2Weights[channel * NCA_ADAPT_HIDDEN_SIZE + hiddenChannel]) * hidden[hiddenChannel];
    }
    let current = asScalar(stateIn[stateIndex(id, channel)]);
    let updated = clamp(current + delta * mask, asScalar(-1.0), asScalar(1.0));
    stateOut[stateIndex(id, channel)] = scalarToF32(updated);
  }
}
