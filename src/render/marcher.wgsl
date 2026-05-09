const SKY_COLOR: vec3<f32> = vec3<f32>(0.784, 0.851, 0.902);

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
@group(0) @binding(3) var<storage, read> colorData: array<u32>;

fn unpackColor(packed: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(packed & 0xFFu),
    f32((packed >> 8u) & 0xFFu),
    f32((packed >> 16u) & 0xFFu),
    f32((packed >> 24u) & 0xFFu)
  ) / 255.0;
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
    let colorIndex = voxel.z * size.x * size.y + voxel.y * size.x + voxel.x;
    let decoded = unpackColor(colorData[colorIndex]);
    let shade = 0.2 + 0.8 * max(dot(normal, normalize(vec3<f32>(0.4, 1.0, 0.3))), 0.0);
    output.color = vec4<f32>(decoded.rgb * decoded.a * shade, 1);
    output.data = vec4<f32>(
      normal,
      distance(position, rayOrigin)
    );
  }

  return output;
}
