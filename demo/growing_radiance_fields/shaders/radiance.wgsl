const S: u32 = {{S}}u;
const C: u32 = {{C}}u;
const HD: u32 = {{HD}}u;
const INPUT_DIM: u32 = {{INPUT_DIM}}u;
const NF: u32 = {{NF}}u;
const NUM_SAMPLES: u32 = {{NUM_SAMPLES}}u;
const VOL: u32 = S * S * S;
const PI: f32 = 3.141592653589793;
const LIVING_THRESHOLD: f32 = {{LIVING_THRESHOLD}};
const APPLY_LIVING: bool = {{APPLY_LIVING}};

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<uniform> u: array<vec4<f32>, 8>;
@group(0) @binding(2) var<storage, read> l0_w: array<f32>;
@group(0) @binding(3) var<storage, read> l1_w: array<f32>;
@group(0) @binding(4) var<storage, read> l2_w: array<f32>;
@group(0) @binding(5) var<storage, read> l3_w: array<f32>;
@group(0) @binding(6) var<storage, read> biases: array<f32>;
@group(0) @binding(7) var<storage, read> mask: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(p[vid], 0.0, 1.0);
  return out;
}

fn grid_coord(p: vec3<f32>, bmin: f32, bmax: f32) -> vec3<f32> {
  return ((p - vec3<f32>(bmin)) / (bmax - bmin)) * f32(S - 1u);
}

{{SAMPLE_ALL_FEATURES}}

fn relative_coord(p: vec3<f32>, bmin: f32, bmax: f32) -> vec3<f32> {
  let q = grid_coord(p, bmin, bmax);
  return fract(q) * 2.0 - vec3<f32>(1.0);
}

fn softplus(x: f32) -> f32 {
  return log(1.0 + exp(-abs(x))) + max(x, 0.0);
}

fn eval_siren(features: array<f32, {{C}}>, coord: vec3<f32>, first_omega: f32, hidden_omega: f32) -> vec4<f32> {
  var input: array<f32, {{INPUT_DIM}}>;
  var at: u32 = 0u;
  if (NF > 0u) {
    for (var freq: u32 = 1u; freq <= NF; freq++) {
      let f = f32(freq) * PI;
      input[at] = sin(coord.x * f); at++;
      input[at] = sin(coord.y * f); at++;
      input[at] = sin(coord.z * f); at++;
    }
    for (var freq: u32 = 1u; freq <= NF; freq++) {
      let f = f32(freq) * PI;
      input[at] = cos(coord.x * f); at++;
      input[at] = cos(coord.y * f); at++;
      input[at] = cos(coord.z * f); at++;
    }
  } else {
    input[at] = coord.x; at++;
    input[at] = coord.y; at++;
    input[at] = coord.z; at++;
  }
  for (var c: u32 = 0u; c < C; c++) {
    input[at] = features[c];
    at++;
  }

  var h0: array<f32, {{HD}}>;
  for (var j: u32 = 0u; j < HD; j++) {
    var v = biases[j];
    for (var i: u32 = 0u; i < INPUT_DIM; i++) {
      v += l0_w[j * INPUT_DIM + i] * input[i];
    }
    h0[j] = sin(first_omega * v);
  }

  var h1: array<f32, {{HD}}>;
  for (var j: u32 = 0u; j < HD; j++) {
    var v = biases[HD + j];
    for (var i: u32 = 0u; i < HD; i++) {
      v += l1_w[j * HD + i] * h0[i];
    }
    h1[j] = sin(hidden_omega * v);
  }

  var h2: array<f32, {{HD}}>;
  for (var j: u32 = 0u; j < HD; j++) {
    var v = biases[HD * 2u + j];
    for (var i: u32 = 0u; i < HD; i++) {
      v += l2_w[j * HD + i] * h1[i];
    }
    h2[j] = sin(hidden_omega * v);
  }

  var outv: array<f32, 4>;
  for (var j: u32 = 0u; j < 4u; j++) {
    var v = biases[HD * 3u + j];
    for (var i: u32 = 0u; i < HD; i++) {
      v += l3_w[j * HD + i] * h2[i];
    }
    outv[j] = v;
  }
  return vec4<f32>(outv[0], outv[1], outv[2], outv[3]);
}

fn intersect_box(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let safe_rd = select(vec3<f32>(0.000001), rd, abs(rd) > vec3<f32>(0.000001));
  let inv = vec3<f32>(1.0) / safe_rd;
  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let lo = min(t0, t1);
  let hi = max(t0, t1);
  let tnear = max(max(lo.x, lo.y), lo.z);
  let tfar = min(min(hi.x, hi.y), hi.z);
  return vec2<f32>(tnear, tfar);
}

fn background(rd: vec3<f32>, strength: f32) -> vec3<f32> {
  let t = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  let sky = mix(vec3<f32>(0.70, 0.80, 0.86), vec3<f32>(0.93, 0.95, 0.94), pow(t, 1.4));
  return sky * (0.75 + 0.25 * strength);
}

@fragment
fn fs_main(v: VSOut) -> @location(0) vec4<f32> {
  let canvas = max(u[0].xy, vec2<f32>(1.0));
  let tan_half_fov = u[0].z;
  let ro = u[1].xyz;
  let forward = normalize(u[2].xyz);
  let right = normalize(u[3].xyz);
  let up = normalize(u[4].xyz);
  let bmin = u[5].x;
  let bmax = u[5].y;
  let density_factor = u[5].z;
  let color_factor = u[5].w;
  let bg_strength = u[6].x;
  let exposure = u[6].y;

  let ndc = vec2<f32>(v.pos.x / canvas.x * 2.0 - 1.0, 1.0 - v.pos.y / canvas.y * 2.0);
  let aspect = canvas.x / canvas.y;
  let rd = normalize(forward + right * ndc.x * aspect * tan_half_fov + up * ndc.y * tan_half_fov);

  let hit = intersect_box(ro, rd, vec3<f32>(bmin), vec3<f32>(bmax));
  let t0 = max(hit.x, 0.0);
  let t1 = hit.y;
  if (t1 <= t0) {
    return vec4<f32>(background(rd, bg_strength), 1.0);
  }

  var rgb_sum: vec3<f32> = vec3<f32>(0.0);
  var trans: f32 = 1.0;

  let safe_rd = select(rd, vec3<f32>(0.0001), abs(rd) < vec3<f32>(0.0001));
  let inv_rd = vec3<f32>(1.0) / safe_rd;
  let tDelta = abs(inv_rd) * ((bmax - bmin) / f32(S));
  
  var t_curr = t0;
  let p_entry = ro + rd * t_curr;
  let q_entry = grid_coord(p_entry, bmin, bmax);
  var vox = clamp(vec3<i32>(floor(q_entry)), vec3<i32>(0), vec3<i32>(i32(S) - 1));
  let step_dir = vec3<i32>(sign(rd));
  
  let next_vox = vec3<f32>(vox) + vec3<f32>(select(vec3<f32>(0.0), vec3<f32>(1.0), rd > vec3<f32>(0.0)));
  var tMax = (next_vox - q_entry) * inv_rd * ((bmax - bmin) / f32(S)) + vec3<f32>(t_curr);

  // DDA Voxel Traversal Loop
  for (var step_i = 0u; step_i < 256u; step_i++) {
    if (t_curr >= t1) { break; }

    let mask_val = textureLoad(mask_tex, vox, 0).r;
    let t_next = min(min(tMax.x, tMax.y), tMax.z);
    let t_exit = min(t_next, t1);

    if (!APPLY_LIVING || mask_val > 0.5) {
      let dist = t_exit - t_curr;
      if (dist > 0.0) {
        // Sample 3 times inside the macro-cell
        let num_sub_samples = 3u;
        let sub_dt = dist / f32(num_sub_samples);
        for (var sub_i = 0u; sub_i < num_sub_samples; sub_i++) {
          let t_sample = t_curr + (f32(sub_i) + 0.5) * sub_dt;
          let p_sample = ro + rd * t_sample;
          let features = sample_all_features(p_sample, bmin, bmax);
          let coord = relative_coord(p_sample, bmin, bmax);
          let raw = eval_siren(features, coord, {{FIRST_OMEGA}}, {{HIDDEN_OMEGA}});
          let rgb = max(raw.xyz * color_factor, vec3<f32>(0.0));
          var sigma = softplus(raw.w) * density_factor;
          if (APPLY_LIVING) {
            sigma *= mask_val;
          }
          let alpha = 1.0 - exp(-sigma * sub_dt);
          rgb_sum += trans * alpha * rgb;
          trans *= (1.0 - alpha);
          if (trans < 0.01) { break; }
        }
      }
    }

    if (trans < 0.01) { break; }

    t_curr = t_exit;
    if (tMax.x < tMax.y && tMax.x < tMax.z) {
      vox.x += step_dir.x;
      tMax.x += tDelta.x;
    } else if (tMax.y < tMax.z) {
      vox.y += step_dir.y;
      tMax.y += tDelta.y;
    } else {
      vox.z += step_dir.z;
      tMax.z += tDelta.z;
    }

    if (vox.x < 0 || vox.y < 0 || vox.z < 0 || vox.x >= i32(S) || vox.y >= i32(S) || vox.z >= i32(S)) {
      break;
    }
  }

  var col = rgb_sum + trans * background(rd, bg_strength);
  col = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * max(exposure, 0.001));
  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
  return vec4<f32>(col, 1.0);
}
