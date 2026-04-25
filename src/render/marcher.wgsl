const PI: f32 = 3.141592653589793;
const LPPN_OMEGA: f32 = 30.0;
const SKY_COLOR: vec3<f32> = vec3<f32>(0.28, 0.55, 0.9);

struct VertexInput {
  @builtin(vertex_index) index: u32,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) ray: vec3<f32>,
}

struct FragmentInput {
  @location(0) ray: vec3<f32>,
}

struct FragmentOutput {
  @location(0) color: vec4<f32>,
  @location(1) data: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<uniform> size: vec4<u32>;
@group(0) @binding(3) var<storage, read> state: array<f32>;
@group(0) @binding(4) var<storage, read> config: array<u32>;
@group(0) @binding(5) var<storage, read> lppnParams: array<f32>;

fn coarseSize() -> u32 {
  return config[0];
}

fn channels() -> u32 {
  return config[1];
}

fn renderSize() -> u32 {
  return config[3];
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

fn stateIndex(pos: vec3<u32>, channel: u32) -> u32 {
  let size3 = coarseSize();
  return (((pos.z * size3 + pos.y) * size3 + pos.x) * channels()) + channel;
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
    return f32(renderSize() - 1u) - value;
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
  let render = f32(renderSize());
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
  let denom = max(f32(renderSize() - 1u), 1.0);
  let uCoord = (renderCoord / vec3<f32>(denom)) * 2.0 - vec3<f32>(1.0);
  let encoded = encodeCoordinates(uCoord);

  var input: array<f32, LPPN_INPUT_SIZE>;
  for (var channel: u32 = 0u; channel < LPPN_CHANNELS; channel++) {
    input[channel] = sampleStateChannel(display, channel);
  }
  for (var feature: u32 = 0u; feature < LPPN_COORD_FEATURES; feature++) {
    input[LPPN_CHANNELS + feature] = encoded[feature];
  }

  var hiddenA: array<f32, LPPN_HIDDEN_SIZE>;
  var hiddenB: array<f32, LPPN_HIDDEN_SIZE>;
  for (var out0: u32 = 0u; out0 < LPPN_HIDDEN_SIZE; out0++) {
    var sum0 = lppnParams[LPPN_LAYER_BIAS_OFFSETS[0u] + out0];
    for (var in0: u32 = 0u; in0 < LPPN_INPUT_SIZE; in0++) {
      sum0 += lppnParams[LPPN_LAYER_WEIGHT_OFFSETS[0u] + out0 * LPPN_INPUT_SIZE + in0] * input[in0];
    }
    hiddenA[out0] = sin(LPPN_OMEGA * sum0);
  }

  var latestIsB = false;
  for (var layer: u32 = 1u; layer < LPPN_SINE_LAYERS; layer++) {
    if (!latestIsB) {
      for (var outA: u32 = 0u; outA < LPPN_HIDDEN_SIZE; outA++) {
        var sumA = lppnParams[LPPN_LAYER_BIAS_OFFSETS[layer] + outA];
        for (var inA: u32 = 0u; inA < LPPN_HIDDEN_SIZE; inA++) {
          sumA += lppnParams[LPPN_LAYER_WEIGHT_OFFSETS[layer] + outA * LPPN_HIDDEN_SIZE + inA] * hiddenA[inA];
        }
        hiddenB[outA] = sin(LPPN_OMEGA * sumA);
      }
      latestIsB = true;
    } else {
      for (var outB: u32 = 0u; outB < LPPN_HIDDEN_SIZE; outB++) {
        var sumB = lppnParams[LPPN_LAYER_BIAS_OFFSETS[layer] + outB];
        for (var inB: u32 = 0u; inB < LPPN_HIDDEN_SIZE; inB++) {
          sumB += lppnParams[LPPN_LAYER_WEIGHT_OFFSETS[layer] + outB * LPPN_HIDDEN_SIZE + inB] * hiddenB[inB];
        }
        hiddenA[outB] = sin(LPPN_OMEGA * sumB);
      }
      latestIsB = false;
    }
  }

  var raw: array<f32, 4>;
  for (var out3: u32 = 0u; out3 < 4u; out3++) {
    var sum3 = lppnParams[LPPN_HEAD_BIAS_OFFSET + out3];
    for (var in3: u32 = 0u; in3 < LPPN_HIDDEN_SIZE; in3++) {
      let hidden = select(hiddenA[in3], hiddenB[in3], latestIsB);
      sum3 += lppnParams[LPPN_HEAD_WEIGHT_OFFSET + out3 * LPPN_HIDDEN_SIZE + in3] * hidden;
    }
    raw[out3] = sum3;
  }

  return vec4<f32>(
    sigmoid(raw[0]),
    sigmoid(raw[1]),
    sigmoid(raw[2]),
    sigmoid(raw[3])
  );
}

@vertex fn vertexMain(vertex: VertexInput) -> VertexOutput {
  const quad = array(
    vec2<f32>( 1,  1),
    vec2<f32>( 1, -1),
    vec2<f32>(-1, -1),
    vec2<f32>( 1,  1),
    vec2<f32>(-1, -1),
    vec2<f32>(-1,  1)
  );

  var out: VertexOutput;
  out.position = vec4<f32>(quad[vertex.index], 0, 1);
  out.ray = normalize(
    (camera.view * camera.projection * vec4<f32>(out.position.xy, 1, 1)).xyz
  );
  return out;
}

@fragment fn fragmentMain(fragment: FragmentInput) -> FragmentOutput {
  let rayOrigin = vec3<f32>(
    camera.view[3][0],
    camera.view[3][1],
    camera.view[3][2]
  );
  let rayDirection = normalize(fragment.ray) * 10000.0;

  var position: vec3<f32>;
  var normal: vec3<f32>;
  var voxel: vec3<u32>;
  var value: u32;
  let rayIntersectsVoxel = rayVoxelIntersection(
    rayOrigin,
    rayDirection,
    &position,
    &normal,
    &voxel,
    &value
  );

  var output: FragmentOutput;
  output.color = vec4<f32>(SKY_COLOR, 1);
  output.data = vec4<f32>(0, 0, 0, 10000);

  if (rayIntersectsVoxel) {
    let decoded = decodeLppn(vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z)));
    let shade = 0.2 + 0.8 * max(dot(normal, normalize(vec3<f32>(0.4, 1.0, 0.3))), 0.0);
    output.color = vec4<f32>(decoded.rgb * decoded.a * shade, 1);
    output.data = vec4<f32>(
      normal,
      distance(position, rayOrigin)
    );
  }

  return output;
}
