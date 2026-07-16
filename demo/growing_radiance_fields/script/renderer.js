import { createBuffer, createEmptyBuffer } from './gpu.js';

export class RadianceRenderer {
  constructor(device, format, model, shaderTemplate, maskShaderTemplate, packShaderTemplate) {
    this.device = device;
    this.format = format;
    this.model = model;
    this.samples = model.renderSamples;
    this.shaderTemplate = shaderTemplate;
    this.maskShaderTemplate = maskShaderTemplate;
    this.packShaderTemplate = packShaderTemplate;
    this._build();
  }

  _build() {
    const { device, model } = this;
    const r = model.weights.radiance;
    const hd = model.sirenHiddenDim;
    this.uniformBuf = createEmptyBuffer(device, 128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.l0w = createBuffer(device, pick(r, ['net.0.linear.weight', 'net.0.weight']).data, GPUBufferUsage.STORAGE);
    this.l1w = createBuffer(device, pick(r, ['net.1.linear.weight', 'net.1.weight']).data, GPUBufferUsage.STORAGE);
    this.l2w = createBuffer(device, pick(r, ['net.2.linear.weight', 'net.2.weight']).data, GPUBufferUsage.STORAGE);
    this.l3w = createBuffer(device, pick(r, ['net.3.weight', 'net.3.linear.weight']).data, GPUBufferUsage.STORAGE);

    const biases = new Float32Array(hd * 3 + 4);
    biases.set(pick(r, ['net.0.linear.bias', 'net.0.bias']).data, 0);
    biases.set(pick(r, ['net.1.linear.bias', 'net.1.bias']).data, hd);
    biases.set(pick(r, ['net.2.linear.bias', 'net.2.bias']).data, hd * 2);
    biases.set(pick(r, ['net.3.bias', 'net.3.linear.bias']).data, hd * 3);
    this.biases = createBuffer(device, biases, GPUBufferUsage.STORAGE);

    const S = model.coarseSize;
    
    // Allocate 3D textures for hardware sampling
    this.maskTex = device.createTexture({
      size: [S, S, S],
      dimension: '3d',
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    const numTex = model.channels / 4;
    this.stateTextures = [];
    for (let i = 0; i < numTex; i++) {
      this.stateTextures.push(device.createTexture({
        size: [S, S, S],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }));
    }

    this.texSampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    // Compile packing shaders
    this.packPipelines = [];
    this.packBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
      ],
    });

    const packPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.packBindGroupLayout] });

    for (let i = 0; i < numTex; i++) {
      const packCode = this.packShaderTemplate
        .replace(/{{S}}/g, S)
        .replace(/{{START_CH}}/g, i * 4);
      const packModule = device.createShaderModule({ code: packCode });
      const packPipeline = device.createComputePipeline({
        layout: packPipelineLayout,
        compute: { module: packModule, entryPoint: 'pack_main' },
      });
      this.packPipelines.push(packPipeline);
    }

    // Compile mask shader
    const maskCode = this.maskShaderTemplate
      .replace(/{{S}}/g, S)
      .replace(/{{C}}/g, model.channels)
      .replace(/{{LC}}/g, model.livingChannel)
      .replace(/{{LIVING_THRESHOLD}}/g, `${model.livingThreshold}`);
    
    const maskModule = device.createShaderModule({ code: maskCode });
    this.maskBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
      ],
    });

    this.maskPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.maskBindGroupLayout] }),
      compute: { module: maskModule, entryPoint: 'cs_main' },
    });

    // Generate dynamic texture bindings for radiance.wgsl
    let textureDecls = '';
    let sampleLogic = '';
    for (let i = 0; i < numTex; i++) {
      textureDecls += `@group(0) @binding(${9 + i}) var tex${i}: texture_3d<f32>;\n`;
      sampleLogic += `  let val${i} = textureSampleLevel(tex${i}, tex_sampler, uvw, 0.0);\n`;
      sampleLogic += `  features[${i * 4 + 0}] = val${i}.x; features[${i * 4 + 1}] = val${i}.y; features[${i * 4 + 2}] = val${i}.z; features[${i * 4 + 3}] = val${i}.w;\n`;
    }

    const sampleFuncReplace = `fn sample_all_features(p: vec3<f32>, bmin: f32, bmax: f32) -> array<f32, ${model.channels}> {
  let uvw = (p - vec3<f32>(bmin)) / (bmax - bmin);
  var features: array<f32, ${model.channels}>;
${sampleLogic}
  return features;
}`;

    const maskDeclSearch = '@group(0) @binding(7) var<storage, read> mask: array<f32>;';
    const maskDeclReplace = `@group(0) @binding(7) var mask_tex: texture_3d<f32>;\n@group(0) @binding(8) var tex_sampler: sampler;\n${textureDecls}`;
    
    const code = this.shaderTemplate
      .replace(maskDeclSearch, maskDeclReplace)
      .replace(/{{SAMPLE_ALL_FEATURES}}/g, sampleFuncReplace)
      .replace(/{{S}}/g, model.coarseSize)
      .replace(/{{C}}/g, model.channels)
      .replace(/{{HD}}/g, model.sirenHiddenDim)
      .replace(/{{INPUT_DIM}}/g, model.sirenInputDim)
      .replace(/{{NF}}/g, model.numFrequencies)
      .replace(/{{NUM_SAMPLES}}/g, this.samples)
      .replace(/{{LC}}/g, model.livingChannel)
      .replace(/{{LIVING_THRESHOLD}}/g, `${model.livingThreshold}`)
      .replace(/{{FIRST_OMEGA}}/g, `${model.firstOmega}`)
      .replace(/{{HIDDEN_OMEGA}}/g, `${model.hiddenOmega}`)
      .replace(/{{APPLY_LIVING}}/g, model.applyLivingMask ? 'true' : 'false');

    const module = device.createShaderModule({ code });
    
    const renderEntries = [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
      { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ];
    for (let i = 0; i < numTex; i++) {
      renderEntries.push({
        binding: 9 + i,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { viewDimension: '3d' }
      });
    }

    this.bindGroupLayout = device.createBindGroupLayout({ entries: renderEntries });
    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  createBindGroup(stateBuffer) {
    const { device } = this;
    const entries = [
      { binding: 0, resource: { buffer: stateBuffer } },
      { binding: 1, resource: { buffer: this.uniformBuf } },
      { binding: 2, resource: { buffer: this.l0w } },
      { binding: 3, resource: { buffer: this.l1w } },
      { binding: 4, resource: { buffer: this.l2w } },
      { binding: 5, resource: { buffer: this.l3w } },
      { binding: 6, resource: { buffer: this.biases } },
      { binding: 7, resource: this.maskTex.createView() },
      { binding: 8, resource: this.texSampler },
    ];
    for (let i = 0; i < this.stateTextures.length; i++) {
      entries.push({
        binding: 9 + i,
        resource: this.stateTextures[i].createView()
      });
    }
    return device.createBindGroup({
      layout: this.bindGroupLayout,
      entries
    });
  }

  createMaskBindGroup(stateBuffer) {
    return this.device.createBindGroup({
      layout: this.maskBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: stateBuffer } },
        { binding: 1, resource: this.maskTex.createView() },
      ],
    });
  }

  encode(encoder, textureView, stateBuffer, camera, canvas, controls, timestampWrites) {
    const frame = cameraFrame(camera);
    const bounds = this.model.voxelBounds;
    const u = new Float32Array(32);
    u[0] = canvas.width;
    u[1] = canvas.height;
    u[2] = Math.tan(camera.fov * Math.PI / 360);
    u[3] = performance.now() * 0.001;
    u.set(frame.origin, 4);
    u.set(frame.forward, 8);
    u.set(frame.right, 12);
    u.set(frame.up, 16);
    u[20] = bounds[0];
    u[21] = bounds[1];
    u[22] = this.model.densityFactor * controls.density;
    u[23] = this.model.colorFactor;
    u[24] = this.model.backgroundColor;
    u[25] = controls.exposure;
    this.device.queue.writeBuffer(this.uniformBuf, 0, u);

    // 1. Pack buffer state into 3D textures
    const packPassDesc = {};
    if (timestampWrites && timestampWrites.begin !== undefined) {
      packPassDesc.timestampWrites = {
        querySet: timestampWrites.querySet,
        beginningOfPassWriteIndex: timestampWrites.begin
      };
    }
    const packPass = encoder.beginComputePass(packPassDesc);
    const S = this.model.coarseSize;
    const dispatchG = Math.ceil(S / 4);

    for (let i = 0; i < this.stateTextures.length; i++) {
      const packBindGroup = this.device.createBindGroup({
        layout: this.packBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: this.stateTextures[i].createView() },
        ]
      });
      packPass.setPipeline(this.packPipelines[i]);
      packPass.setBindGroup(0, packBindGroup);
      packPass.dispatchWorkgroups(dispatchG, dispatchG, dispatchG);
    }
    packPass.end();

    // 2. Generate the 3D living mask texture
    const maskPass = encoder.beginComputePass();
    maskPass.setPipeline(this.maskPipeline);
    maskPass.setBindGroup(0, this.createMaskBindGroup(stateBuffer));
    maskPass.dispatchWorkgroups(dispatchG, dispatchG, dispatchG);
    maskPass.end();

    // 3. Render full-screen pass
    const passDesc = {
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.75, g: 0.82, b: 0.86, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };
    if (timestampWrites && timestampWrites.end !== undefined) {
      passDesc.timestampWrites = {
        querySet: timestampWrites.querySet,
        endOfPassWriteIndex: timestampWrites.end
      };
    }
    const pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.createBindGroup(stateBuffer));
    pass.draw(3, 1, 0, 0);
    pass.end();
  }
}

function pick(group, names) {
  for (const name of names) if (group[name]) return group[name];
  throw new Error(`Missing radiance tensor: ${names.join(' or ')}`);
}

function cameraFrame(camera) {
  const origin = camera.getPosition();
  const forward = normalize(sub(camera.target, origin));
  let right = normalize(cross(forward, [0, 1, 0]));
  if (length(right) < 1e-5) right = [1, 0, 0];
  const up = normalize(cross(right, forward));
  return { origin, forward, right, up };
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function length(v) { return Math.sqrt(dot(v, v)); }
function normalize(v) {
  const l = length(v);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 1];
}
