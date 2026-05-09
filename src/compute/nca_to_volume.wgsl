const PI: f32 = 3.141592653589793;
const LPPN_OMEGA: f32 = 30.0;

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> colorData: array<u32>;
@group(0) @binding(3) var<storage, read> config: array<u32>;
@group(0) @binding(4) var<storage, read> lppnParams: array<f32>;

fn coarseSize() -> u32 {
  return config[0];
}

fn channels() -> u32 {
  return config[1];
}

fn alphaChannel() -> u32 {
  return config[2];
}

fn targetSize() -> u32 {
  return config[3];
}

fn packedX() -> u32 {
  return config[4];
}

fn axisMapX() -> u32 {
  return config[5];
}

fn axisMapY() -> u32 {
  return config[6];
}

fn axisMapZ() -> u32 {
  return config[7];
}

fn axisFlipX() -> bool {
  return config[8] != 0u;
}

fn axisFlipY() -> bool {
  return config[9] != 0u;
}

fn axisFlipZ() -> bool {
  return config[10] != 0u;
}

fn sectionMaxX() -> u32 {
  return config[11];
}

fn sectionMaxY() -> u32 {
  return config[12];
}

fn sectionMaxZ() -> u32 {
  return config[13];
}

fn stateIndex(pos: vec3<u32>, channel: u32) -> u32 {
  let size = coarseSize();
  return (((pos.z * size + pos.y) * size + pos.x) * channels()) + channel;
}

fn readState(pos: vec3<u32>, channel: u32) -> f32 {
  return state[stateIndex(pos, channel)];
}

fn axisValue(axis: u32, display: vec3<f32>) -> f32 {
  if (axis == 0u) {
    return display.x;
  }
  if (axis == 1u) {
    return display.y;
  }
  return display.z;
}

fn maybeFlip(value: f32, flip: bool) -> f32 {
  if (flip) {
    return f32(targetSize() - 1u) - value;
  }
  return value;
}

fn modelRenderCoord(display: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    maybeFlip(axisValue(axisMapX(), display), axisFlipX()),
    maybeFlip(axisValue(axisMapY(), display), axisFlipY()),
    maybeFlip(axisValue(axisMapZ(), display), axisFlipZ())
  );
}

fn sourceCoord(coord: f32) -> f32 {
  let coarse = f32(coarseSize());
  let render = f32(targetSize());
  return clamp(((coord + 0.5) * coarse / render) - 0.5, 0.0, coarse - 1.0);
}

fn sampleStateChannel(display: vec3<f32>, channel: u32) -> f32 {
  let renderCoord = modelRenderCoord(display);
  let source = vec3<f32>(
    sourceCoord(renderCoord.x),
    sourceCoord(renderCoord.y),
    sourceCoord(renderCoord.z)
  );
  let base = vec3<u32>(
    u32(floor(source.x)),
    u32(floor(source.y)),
    u32(floor(source.z))
  );
  let next = min(base + vec3<u32>(1u), vec3<u32>(coarseSize() - 1u));
  let frac = source - vec3<f32>(f32(base.x), f32(base.y), f32(base.z));

  let c000 = readState(vec3<u32>(base.x, base.y, base.z), channel);
  let c100 = readState(vec3<u32>(next.x, base.y, base.z), channel);
  let c010 = readState(vec3<u32>(base.x, next.y, base.z), channel);
  let c110 = readState(vec3<u32>(next.x, next.y, base.z), channel);
  let c001 = readState(vec3<u32>(base.x, base.y, next.z), channel);
  let c101 = readState(vec3<u32>(next.x, base.y, next.z), channel);
  let c011 = readState(vec3<u32>(base.x, next.y, next.z), channel);
  let c111 = readState(vec3<u32>(next.x, next.y, next.z), channel);

  let c00 = mix(f32(c000), f32(c100), frac.x);
  let c10 = mix(f32(c010), f32(c110), frac.x);
  let c01 = mix(f32(c001), f32(c101), frac.x);
  let c11 = mix(f32(c011), f32(c111), frac.x);
  let c0 = mix(c00, c10, frac.y);
  let c1 = mix(c01, c11, frac.y);
  return mix(c0, c1, frac.z);
}

fn encodeCoordinates(coord: vec3<f32>) -> array<f32, LPPN_COORD_FEATURES> {
  var features: array<f32, LPPN_COORD_FEATURES>;
  for (var harmonic: u32 = 0u; harmonic < LPPN_HARMONICS; harmonic++) {
    let n = f32(harmonic + 1u);
    let base = harmonic * 6u;
    features[base + 0u] = sin(PI * n * coord.x);
    features[base + 1u] = sin(PI * n * coord.y);
    features[base + 2u] = sin(PI * n * coord.z);
    features[base + 3u] = cos(PI * n * coord.x);
    features[base + 4u] = cos(PI * n * coord.y);
    features[base + 5u] = cos(PI * n * coord.z);
  }
  return features;
}

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}

fn decodeLppn(display: vec3<f32>) -> vec4<f32> {
  let renderCoord = modelRenderCoord(display);
  let denom = max(f32(targetSize() - 1u), 1.0);
  let uCoord = (renderCoord / vec3<f32>(denom)) * 2.0 - vec3<f32>(1.0);
  let encoded = encodeCoordinates(uCoord);

  var input: array<ComputeScalar, LPPN_INPUT_SIZE>;
  for (var channel: u32 = 0u; channel < LPPN_CHANNELS; channel++) {
    input[channel] = asScalar(sampleStateChannel(display, channel));
  }
  for (var feature: u32 = 0u; feature < LPPN_COORD_FEATURES; feature++) {
    input[LPPN_CHANNELS + feature] = asScalar(encoded[feature]);
  }

  var hiddenA: array<ComputeScalar, LPPN_HIDDEN_SIZE>;
  var hiddenB: array<ComputeScalar, LPPN_HIDDEN_SIZE>;
  for (var out0: u32 = 0u; out0 < LPPN_HIDDEN_SIZE; out0++) {
    var sum0: ComputeScalar = asScalar(lppnParams[LPPN_LAYER_BIAS_OFFSETS[0u] + out0]);
    for (var in0: u32 = 0u; in0 < LPPN_INPUT_SIZE; in0++) {
      sum0 += asScalar(lppnParams[LPPN_LAYER_WEIGHT_OFFSETS[0u] + out0 * LPPN_INPUT_SIZE + in0]) * input[in0];
    }
    hiddenA[out0] = sin(asScalar(LPPN_OMEGA) * sum0);
  }

  var latestIsB = false;
  for (var layer: u32 = 1u; layer < LPPN_SINE_LAYERS; layer++) {
    if (!latestIsB) {
      for (var outA: u32 = 0u; outA < LPPN_HIDDEN_SIZE; outA++) {
        var sumA: ComputeScalar = asScalar(lppnParams[LPPN_LAYER_BIAS_OFFSETS[layer] + outA]);
        for (var inA: u32 = 0u; inA < LPPN_HIDDEN_SIZE; inA++) {
          sumA += asScalar(lppnParams[LPPN_LAYER_WEIGHT_OFFSETS[layer] + outA * LPPN_HIDDEN_SIZE + inA]) * hiddenA[inA];
        }
        hiddenB[outA] = sin(asScalar(LPPN_OMEGA) * sumA);
      }
      latestIsB = true;
    } else {
      for (var outB: u32 = 0u; outB < LPPN_HIDDEN_SIZE; outB++) {
        var sumB: ComputeScalar = asScalar(lppnParams[LPPN_LAYER_BIAS_OFFSETS[layer] + outB]);
        for (var inB: u32 = 0u; inB < LPPN_HIDDEN_SIZE; inB++) {
          sumB += asScalar(lppnParams[LPPN_LAYER_WEIGHT_OFFSETS[layer] + outB * LPPN_HIDDEN_SIZE + inB]) * hiddenB[inB];
        }
        hiddenA[outB] = sin(asScalar(LPPN_OMEGA) * sumB);
      }
      latestIsB = false;
    }
  }

  var raw: array<ComputeScalar, 4>;
  for (var out3: u32 = 0u; out3 < 4u; out3++) {
    var sum3: ComputeScalar = asScalar(lppnParams[LPPN_HEAD_BIAS_OFFSET + out3]);
    for (var in3: u32 = 0u; in3 < LPPN_HIDDEN_SIZE; in3++) {
      let hidden = select(hiddenA[in3], hiddenB[in3], latestIsB);
      sum3 += asScalar(lppnParams[LPPN_HEAD_WEIGHT_OFFSET + out3 * LPPN_HIDDEN_SIZE + in3]) * hidden;
    }
    raw[out3] = sum3;
  }

  return vec4<f32>(
    sigmoid(scalarToF32(raw[0])),
    sigmoid(scalarToF32(raw[1])),
    sigmoid(scalarToF32(raw[2])),
    sigmoid(scalarToF32(raw[3]))
  );
}

fn packColor(color: vec4<f32>) -> u32 {
  let quantized = vec4<u32>(round(clamp(color, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
  return quantized.x
    | (quantized.y << 8u)
    | (quantized.z << 16u)
    | (quantized.w << 24u);
}

fn clearPackedEntry(id: vec3<u32>) {
  let packedIndex = id.z * packedX() * targetSize() + id.y * packedX() + id.x;
  data[packedIndex] = 0u;
  for (var offset: u32 = 0u; offset < 4u; offset++) {
    let x = id.x * 4u + offset;
    if (x < targetSize()) {
      let colorIndex = id.z * targetSize() * targetSize() + id.y * targetSize() + x;
      colorData[colorIndex] = 0u;
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= packedX() || id.y >= targetSize() || id.z >= targetSize()) {
    return;
  }

  if (id.y > sectionMaxY() || id.z > sectionMaxZ()) {
    clearPackedEntry(id);
    return;
  }

  var coarseAlphas: array<f32, 4>;
  var hasCoarseContent = false;
  for (var offset: u32 = 0u; offset < 4u; offset++) {
    let x = id.x * 4u + offset;
    var coarseAlpha = 0.0;
    if (x < targetSize() && x <= sectionMaxX()) {
      let display = vec3<f32>(f32(x), f32(id.y), f32(id.z));
      coarseAlpha = sampleStateChannel(display, alphaChannel());
      hasCoarseContent = hasCoarseContent || coarseAlpha > 0.1;
    }
    coarseAlphas[offset] = coarseAlpha;
  }

  if (!hasCoarseContent) {
    clearPackedEntry(id);
    return;
  }

  var packed = 0u;
  for (var offset: u32 = 0u; offset < 4u; offset++) {
    let x = id.x * 4u + offset;
    var value = 0u;
    if (x < targetSize()) {
      let colorIndex = id.z * targetSize() * targetSize() + id.y * targetSize() + x;
      var packedColor = 0u;
      if (x <= sectionMaxX() && coarseAlphas[offset] > 0.1) {
        let display = vec3<f32>(f32(x), f32(id.y), f32(id.z));
        let decoded = decodeLppn(display);
        packedColor = packColor(decoded);
        value = select(0u, u32(round(decoded.a * 255.0)), decoded.a > 0.1);
      }
      colorData[colorIndex] = packedColor;
    }
    packed |= value << (offset * 8u);
  }

  let index = id.z * packedX() * targetSize() + id.y * packedX() + id.x;
  data[index] = packed;
}
