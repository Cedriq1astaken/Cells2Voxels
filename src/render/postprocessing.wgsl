@group(0) @binding(0) var colorTexture: texture_2d<f32>;
@group(0) @binding(1) var dataTexture: texture_2d<f32>;

const edgeColor : vec3<f32> = vec3<f32>(0, 0, 0);
const edgeIntensity : f32 = 0.18;
const depthScale : f32 = 0.18;
const normalScale : f32 = 0.18;
const backgroundDepth : f32 = 9999.0;
const offset : vec3<i32> = vec3<i32>(1, 1, 0);

fn isBackground(sample: vec4<f32>) -> bool {
  return sample.w >= backgroundDepth;
}

fn clampPixel(pixel: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
  return clamp(pixel, vec2<i32>(0), size - vec2<i32>(1));
}

fn getDataSample(pixel: vec2<i32>, size: vec2<i32>) -> vec4<f32> {
  return textureLoad(dataTexture, clampPixel(pixel, size), 0);
}

fn getEdge(pixel : vec2<i32>, size: vec2<i32>) -> f32 {
  let pixelCenter : vec4<f32> = getDataSample(pixel, size);
  if (isBackground(pixelCenter)) {
    return 0.0;
  }
  let pixelLeft : vec4<f32> = getDataSample(pixel - offset.xz, size);
  let pixelRight : vec4<f32> = getDataSample(pixel + offset.xz, size);
  let pixelUp : vec4<f32> = getDataSample(pixel + offset.zy, size);
  let pixelDown : vec4<f32> = getDataSample(pixel - offset.zy, size);
  let edge : vec4<f32> = (
    abs(pixelLeft    - pixelCenter)
    + abs(pixelRight - pixelCenter) 
    + abs(pixelUp    - pixelCenter) 
    + abs(pixelDown  - pixelCenter)
  );
  return clamp(max((edge.x + edge.y + edge.z) * normalScale, edge.w * depthScale), 0, 1);
}

fn linearTosRGB(linear: vec3<f32>) -> vec3<f32> {
  if (all(linear <= vec3<f32>(0.0031308))) {
    return linear * 12.92;
  }
  return (pow(abs(linear), vec3<f32>(1.0/2.4)) * 1.055) - vec3<f32>(0.055);
}

@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  const quad = array(
    vec2<f32>( 1,  1),
    vec2<f32>( 1, -1),
    vec2<f32>(-1, -1),
    vec2<f32>( 1,  1),
    vec2<f32>(-1, -1),
    vec2<f32>(-1,  1)
  );

  return vec4<f32>(quad[index], 0, 1);
}

@fragment fn fragmentMain(@builtin(position) uv: vec4<f32>) -> @location(0) vec4<f32> {
  let sourceSizeU = textureDimensions(dataTexture);
  let sourceSize = vec2<i32>(sourceSizeU);
  let sourcePixel = clampPixel(vec2<i32>(floor(uv.xy)), sourceSize);
  let color = textureLoad(colorTexture, sourcePixel, 0).xyz;
  return vec4<f32>(linearTosRGB(
    mix(color, edgeColor, getEdge(sourcePixel, sourceSize) * edgeIntensity)
  ), 1);
}
