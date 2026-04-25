import loadShader from '../loadShader.js';

const defaultPreviewOrientation = {
  axisMap: new Uint32Array([1, 2, 0]),
  axisFlip: new Uint32Array([0, 1, 0]),
};
const axisPermutations = [
  [0, 1, 2],
  [0, 2, 1],
  [1, 0, 2],
  [1, 2, 0],
  [2, 0, 1],
  [2, 1, 0],
];
const quarterTurnTransforms = {
  x: {
    1: { axisMap: [0, 2, 1], axisFlip: [0, 1, 0] },
    '-1': { axisMap: [0, 2, 1], axisFlip: [0, 0, 1] },
  },
  y: {
    1: { axisMap: [2, 1, 0], axisFlip: [1, 0, 0] },
    '-1': { axisMap: [2, 1, 0], axisFlip: [0, 0, 1] },
  },
  z: {
    1: { axisMap: [1, 0, 2], axisFlip: [0, 1, 0] },
    '-1': { axisMap: [1, 0, 2], axisFlip: [1, 0, 0] },
  },
};

const getNetIndex = (name) => {
  const match = name.match(/^net\.(\d+)/);
  if (!match) {
    throw new Error(`Couldn't parse layer index from "${name}"`);
  }
  return Number(match[1]);
};

const parseManifestLayouts = (manifest, weights) => {
  const channels = manifest.meta.channels;
  const featureCount = channels * 4;
  const adaptHiddenSize = manifest.adaptation['adapt.0.bias'].count;
  const linearLayers = Object.entries(manifest.lppn)
    .filter(([name]) => name.endsWith('.linear.weight'))
    .sort((a, b) => getNetIndex(a[0]) - getNetIndex(b[0]));
  const linearBiasLayers = Object.entries(manifest.lppn)
    .filter(([name]) => name.endsWith('.linear.bias'))
    .sort((a, b) => getNetIndex(a[0]) - getNetIndex(b[0]));
  const headWeightEntry = Object.entries(manifest.lppn)
    .filter(([name]) => /\.weight$/.test(name) && !name.includes('.linear.'))
    .sort((a, b) => getNetIndex(a[0]) - getNetIndex(b[0]))
    .at(-1);
  const headBiasEntry = Object.entries(manifest.lppn)
    .filter(([name]) => /\.bias$/.test(name) && !name.includes('.linear.'))
    .sort((a, b) => getNetIndex(a[0]) - getNetIndex(b[0]))
    .at(-1);

  if (!linearLayers.length || linearLayers.length !== linearBiasLayers.length || !headWeightEntry || !headBiasEntry) {
    throw new Error('Unsupported LPPN manifest layout');
  }

  const lppnStart = linearLayers[0][1].offset;
  const lppnEnd = headBiasEntry[1].offset + (headBiasEntry[1].count * Float32Array.BYTES_PER_ELEMENT);
  const lppnParams = new Float32Array(
    weights,
    lppnStart,
    (lppnEnd - lppnStart) / Float32Array.BYTES_PER_ELEMENT
  );
  const hiddenSize = linearBiasLayers[0][1].count;
  const inputSize = linearLayers[0][1].shape[1];
  const harmonics = Math.max(1, Math.floor((inputSize - channels) / 6));

  return {
    adaptHiddenSize,
    featureCount,
    harmonics,
    hiddenSize,
    inputSize,
    lppnParams,
    lppnLayerCount: linearLayers.length,
    lppnLayerWeightOffsets: linearLayers.map(([, entry]) => (entry.offset - lppnStart) / Float32Array.BYTES_PER_ELEMENT),
    lppnLayerBiasOffsets: linearBiasLayers.map(([, entry]) => (entry.offset - lppnStart) / Float32Array.BYTES_PER_ELEMENT),
    headWeightOffset: (headWeightEntry[1].offset - lppnStart) / Float32Array.BYTES_PER_ELEMENT,
    headBiasOffset: (headBiasEntry[1].offset - lppnStart) / Float32Array.BYTES_PER_ELEMENT,
  };
};

const buildStepShaderConstants = (modelInfo) => [
  `const NCA_CHANNELS: u32 = ${modelInfo.channels}u;`,
  `const NCA_FEATURE_COUNT: u32 = ${modelInfo.featureCount}u;`,
  `const NCA_ADAPT_HIDDEN_SIZE: u32 = ${modelInfo.adaptHiddenSize}u;`,
  '',
].join('\n');

const buildLppnShaderConstants = (modelInfo) => [
  `const LPPN_CHANNELS: u32 = ${modelInfo.channels}u;`,
  `const LPPN_HARMONICS: u32 = ${modelInfo.harmonics}u;`,
  `const LPPN_HIDDEN_SIZE: u32 = ${modelInfo.hiddenSize}u;`,
  `const LPPN_COORD_FEATURES: u32 = ${(modelInfo.harmonics * 6)}u;`,
  `const LPPN_INPUT_SIZE: u32 = ${modelInfo.inputSize}u;`,
  `const LPPN_SINE_LAYERS: u32 = ${modelInfo.lppnLayerCount}u;`,
  `const LPPN_LAYER_WEIGHT_OFFSETS = array<u32, ${modelInfo.lppnLayerCount}>(${modelInfo.lppnLayerWeightOffsets.map((offset) => `${offset}u`).join(', ')});`,
  `const LPPN_LAYER_BIAS_OFFSETS = array<u32, ${modelInfo.lppnLayerCount}>(${modelInfo.lppnLayerBiasOffsets.map((offset) => `${offset}u`).join(', ')});`,
  `const LPPN_HEAD_WEIGHT_OFFSET: u32 = ${modelInfo.headWeightOffset}u;`,
  `const LPPN_HEAD_BIAS_OFFSET: u32 = ${modelInfo.headBiasOffset}u;`,
  '',
].join('\n');

const injectShaderConstants = (source, constants) => {
  return `${constants}${source}`;
};

class Volume {
  static async create(
    device,
    modelPath = '../../model/Globe',
    previewOrientation = defaultPreviewOrientation,
    scaleMultiplier
  ) {
    const [manifest, weights, stepProgram, finalizeProgram, rasterProgram] = await Promise.all([
      fetch(new URL(`${modelPath}/nca_manifest.json`, import.meta.url)).then((response) => response.json()),
      fetch(new URL(`${modelPath}/nca_weights.bin`, import.meta.url)).then((response) => response.arrayBuffer()),
      loadShader('./compute/nca_step.wgsl'),
      loadShader('./compute/nca_finalize.wgsl'),
      loadShader('./compute/nca_to_volume.wgsl'),
    ]);

    const { channels, coarse_size: coarseSize } = manifest.meta;
    const resolvedScaleMultiplier = scaleMultiplier ?? manifest.meta.scale ?? 1;
    const renderSize = Math.max(1, Math.round(coarseSize * resolvedScaleMultiplier));
    const packedX = Math.ceil(renderSize / 4);
    const stateLength = coarseSize * coarseSize * coarseSize * channels;
    const volumeLength = packedX * renderSize * renderSize;
    const alphaChannel = 0;
    const axisMap = Array.from(previewOrientation.axisMap ?? defaultPreviewOrientation.axisMap);
    const axisFlip = Array.from(previewOrientation.axisFlip ?? defaultPreviewOrientation.axisFlip);

    const createBuffer = (typedArray, usage) => {
      const buffer = device.createBuffer({
        size: typedArray.byteLength,
        usage,
        mappedAtCreation: true,
      });
      new typedArray.constructor(buffer.getMappedRange()).set(typedArray);
      buffer.unmap();
      return buffer;
    };

    const seedState = new Float32Array(stateLength);
    const center = Math.floor(coarseSize * 0.5);
    const centerIndex = (((center * coarseSize + center) * coarseSize + center) * channels) + alphaChannel;
    seedState[centerIndex] = 1;

    const stateBuffers = [
      createBuffer(seedState, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
      createBuffer(seedState, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
    ];
    const rawState = createBuffer(
      new Float32Array(stateLength),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );

    const data = device.createBuffer({
      size: volumeLength * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const configData = new Uint32Array([
        coarseSize,
        channels,
        alphaChannel,
        renderSize,
        packedX,
        axisMap[0],
        axisMap[1],
        axisMap[2],
        axisFlip[0],
        axisFlip[1],
        axisFlip[2],
        renderSize,
        renderSize,
        renderSize,
      ]);
    const config = createBuffer(
      configData,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const size = {
      cpu: new Uint32Array([renderSize, renderSize, renderSize, packedX]),
      gpu: createBuffer(
        new Uint32Array([renderSize, renderSize, renderSize, packedX]),
        GPUBufferUsage.UNIFORM
      ),
    };
    const stepState = createBuffer(
      new Uint32Array([0, 0, 0, 0]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const toFloatArray = (entry) => new Float32Array(weights, entry.offset, entry.count);
    const manifestLayouts = parseManifestLayouts(manifest, weights);
    const perceptionWeights = createBuffer(
      toFloatArray(manifest.perception['perceive.weight']),
      GPUBufferUsage.STORAGE
    );
    const adapt0Weights = createBuffer(
      toFloatArray(manifest.adaptation['adapt.0.weight']),
      GPUBufferUsage.STORAGE
    );
    const adapt0Bias = createBuffer(
      toFloatArray(manifest.adaptation['adapt.0.bias']),
      GPUBufferUsage.STORAGE
    );
    const adapt2Weights = createBuffer(
      toFloatArray(manifest.adaptation['adapt.2.weight']),
      GPUBufferUsage.STORAGE
    );
    const lppnParams = createBuffer(
      manifestLayouts.lppnParams,
      GPUBufferUsage.STORAGE
    );

    const modelInfo = {
      channels,
      coarseSize,
      adaptHiddenSize: manifestLayouts.adaptHiddenSize,
      featureCount: manifestLayouts.featureCount,
      harmonics: manifestLayouts.harmonics,
      hiddenSize: manifestLayouts.hiddenSize,
      inputSize: manifestLayouts.inputSize,
      lppnLayerCount: manifestLayouts.lppnLayerCount,
      lppnLayerWeightOffsets: manifestLayouts.lppnLayerWeightOffsets,
      lppnLayerBiasOffsets: manifestLayouts.lppnLayerBiasOffsets,
      headWeightOffset: manifestLayouts.headWeightOffset,
      headBiasOffset: manifestLayouts.headBiasOffset,
      renderSize,
      scaleMultiplier: resolvedScaleMultiplier,
    };
    const stepShaderConstants = buildStepShaderConstants(modelInfo);
    const lppnShaderConstants = buildLppnShaderConstants(modelInfo);

    const stepPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: injectShaderConstants(stepProgram, stepShaderConstants),
        }),
      },
    });

    const rasterPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: injectShaderConstants(rasterProgram, lppnShaderConstants),
        }),
      },
    });
    const finalizePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        entryPoint: 'main',
        module: device.createShaderModule({
          code: finalizeProgram,
        }),
      },
    });

    const stepBindings = [
      device.createBindGroup({
        layout: stepPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffers[0] } },
          { binding: 1, resource: { buffer: rawState } },
          { binding: 2, resource: { buffer: config } },
          { binding: 3, resource: { buffer: perceptionWeights } },
          { binding: 4, resource: { buffer: adapt0Weights } },
          { binding: 5, resource: { buffer: adapt0Bias } },
          { binding: 6, resource: { buffer: adapt2Weights } },
          { binding: 7, resource: { buffer: stepState } },
        ],
      }),
      device.createBindGroup({
        layout: stepPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffers[1] } },
          { binding: 1, resource: { buffer: rawState } },
          { binding: 2, resource: { buffer: config } },
          { binding: 3, resource: { buffer: perceptionWeights } },
          { binding: 4, resource: { buffer: adapt0Weights } },
          { binding: 5, resource: { buffer: adapt0Bias } },
          { binding: 6, resource: { buffer: adapt2Weights } },
          { binding: 7, resource: { buffer: stepState } },
        ],
      }),
    ];
    const finalizeBindings = [
      device.createBindGroup({
        layout: finalizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffers[0] } },
          { binding: 1, resource: { buffer: rawState } },
          { binding: 2, resource: { buffer: stateBuffers[1] } },
          { binding: 3, resource: { buffer: config } },
        ],
      }),
      device.createBindGroup({
        layout: finalizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffers[1] } },
          { binding: 1, resource: { buffer: rawState } },
          { binding: 2, resource: { buffer: stateBuffers[0] } },
          { binding: 3, resource: { buffer: config } },
        ],
      }),
    ];

    const rasterBindings = [
      device.createBindGroup({
        layout: rasterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffers[0] } },
          { binding: 1, resource: { buffer: data } },
          { binding: 2, resource: { buffer: config } },
          { binding: 3, resource: { buffer: lppnParams } },
        ],
      }),
      device.createBindGroup({
        layout: rasterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffers[1] } },
          { binding: 1, resource: { buffer: data } },
          { binding: 2, resource: { buffer: config } },
          { binding: 3, resource: { buffer: lppnParams } },
        ],
      }),
    ];

    return new Volume(
      device,
      data,
      volumeLength * Uint32Array.BYTES_PER_ELEMENT,
      size,
      seedState,
      stateBuffers,
      rawState,
      stepPipeline,
      finalizePipeline,
      rasterPipeline,
      stepBindings,
      finalizeBindings,
      rasterBindings,
      stepState,
      config,
      configData,
      lppnParams,
      manifest,
      modelInfo,
      {
        step: new Uint32Array([
          Math.ceil(coarseSize / 4),
          Math.ceil(coarseSize / 4),
          Math.ceil(coarseSize / 4),
        ]),
        raster: new Uint32Array([
          Math.ceil(packedX / 4),
          Math.ceil(renderSize / 4),
          Math.ceil(renderSize / 4),
        ]),
      }
    );
  }

  constructor(
    device,
    data,
    dataByteLength,
    size,
    seedState,
    stateBuffers,
    rawState,
    stepPipeline,
    finalizePipeline,
    rasterPipeline,
    stepBindings,
    finalizeBindings,
    rasterBindings,
    stepState,
    config,
    configData,
    lppnParams,
    manifest,
    modelInfo,
    workgroups
  ) {
    this.device = device;
    this.data = data;
    this.dataByteLength = dataByteLength;
    this.size = size;
    this.seedState = seedState;
    this.stateBuffers = stateBuffers;
    this.rawState = rawState;
    this.stepPipeline = stepPipeline;
    this.finalizePipeline = finalizePipeline;
    this.rasterPipeline = rasterPipeline;
    this.stepBindings = stepBindings;
    this.finalizeBindings = finalizeBindings;
    this.rasterBindings = rasterBindings;
    this.stepState = stepState;
    this.config = config;
    this.configData = configData;
    this.lppnParams = lppnParams;
    this.manifest = manifest;
    this.modelInfo = modelInfo;
    this.workgroups = workgroups;
    this.currentState = 0;
    this.tick = 0;
    this.orientationIndex = axisPermutations.findIndex(
      (permutation) =>
        permutation[0] === configData[5]
        && permutation[1] === configData[6]
        && permutation[2] === configData[7]
    );
    if (this.orientationIndex < 0) {
      this.orientationIndex = 0;
    }
  }

  generate(steps = 64) {
    this.reset();
    for (let i = 0; i < steps; i++) {
      const command = this.device.createCommandEncoder();
      this.encodeStep(command);
      this.device.queue.submit([command.finish()]);
    }
    const command = this.device.createCommandEncoder();
    this.encodeRaster(command);
    this.device.queue.submit([command.finish()]);
  }

  encodeStep(command) {
    const {
      device,
      finalizeBindings,
      finalizePipeline,
      stepBindings,
      stepPipeline,
      stepState,
      workgroups,
    } = this;

    device.queue.writeBuffer(stepState, 0, new Uint32Array([this.tick, 0, 0, 0]));
    this.tick += 1;

    const stepPass = command.beginComputePass();
    stepPass.setPipeline(stepPipeline);
    stepPass.setBindGroup(0, stepBindings[this.currentState]);
    stepPass.dispatchWorkgroups(workgroups.step[0], workgroups.step[1], workgroups.step[2]);
    stepPass.end();

    const finalizePass = command.beginComputePass();
    finalizePass.setPipeline(finalizePipeline);
    finalizePass.setBindGroup(0, finalizeBindings[this.currentState]);
    finalizePass.dispatchWorkgroups(workgroups.step[0], workgroups.step[1], workgroups.step[2]);
    finalizePass.end();

    this.currentState = 1 - this.currentState;
  }

  encodeRaster(command) {
    const { rasterBindings, rasterPipeline, workgroups } = this;
    const rasterPass = command.beginComputePass();
    rasterPass.setPipeline(rasterPipeline);
    rasterPass.setBindGroup(0, rasterBindings[this.currentState]);
    rasterPass.dispatchWorkgroups(workgroups.raster[0], workgroups.raster[1], workgroups.raster[2]);
    rasterPass.end();
  }

  step(command) {
    this.encodeStep(command);
    this.encodeRaster(command);
  }

  stepSimulationCount(command, count) {
    const total = Math.max(0, Math.floor(count));
    for (let index = 0; index < total; index += 1) {
      this.encodeStep(command);
    }
  }

  stepCount(command, count) {
    const total = Math.max(0, Math.floor(count));
    if (total === 0) {
      this.encodeRaster(command);
      return;
    }

    this.stepSimulationCount(command, total);
    this.encodeRaster(command);
  }

  rasterize(command) {
    this.encodeRaster(command);
  }

  render(command) {
    this.encodeRaster(command);
  }

  getData() {
    return this.data;
  }

  getSize() {
    return this.size.gpu;
  }

  getConfig() {
    return this.config;
  }

  getLppnParams() {
    return this.lppnParams;
  }

  getStateBuffer(index) {
    return this.stateBuffers[index];
  }

  getCurrentStateIndex() {
    return this.currentState;
  }

  getTick() {
    return this.tick;
  }

  getModelInfo() {
    return this.modelInfo;
  }

  getManifest() {
    return this.manifest;
  }

  getRenderInfo() {
    return {
      coarseSize: this.modelInfo.coarseSize,
      renderSize: this.modelInfo.renderSize,
      scaleMultiplier: this.modelInfo.scaleMultiplier,
    };
  }

  getDefaultDamageRadius() {
    return Math.max(1.5, this.modelInfo.coarseSize * 0.08);
  }

  getCrossSectionState() {
    return {
      x: this.configData[11],
      y: this.configData[12],
      z: this.configData[13],
    };
  }

  syncOrientationIndex() {
    this.orientationIndex = axisPermutations.findIndex(
      (permutation) =>
        permutation[0] === this.configData[5]
        && permutation[1] === this.configData[6]
        && permutation[2] === this.configData[7]
    );
    if (this.orientationIndex < 0) {
      this.orientationIndex = 0;
    }
  }

  reset() {
    const zeroState = new Float32Array(this.seedState.length);
    this.device.queue.writeBuffer(this.stateBuffers[0], 0, this.seedState);
    this.device.queue.writeBuffer(this.stateBuffers[1], 0, this.seedState);
    this.device.queue.writeBuffer(this.rawState, 0, zeroState);
    this.currentState = 0;
    this.tick = 0;
  }

  copySimulationStateFrom(source) {
    if (this.seedState.byteLength !== source.seedState.byteLength) {
      throw new Error('Cannot copy simulation state between incompatible volumes');
    }

    const command = this.device.createCommandEncoder();
    command.copyBufferToBuffer(source.stateBuffers[0], 0, this.stateBuffers[0], 0, this.seedState.byteLength);
    command.copyBufferToBuffer(source.stateBuffers[1], 0, this.stateBuffers[1], 0, this.seedState.byteLength);
    command.copyBufferToBuffer(source.rawState, 0, this.rawState, 0, this.seedState.byteLength);
    this.device.queue.submit([command.finish()]);

    this.currentState = source.currentState;
    this.tick = source.tick;
  }

  updateOrientation() {
    this.device.queue.writeBuffer(this.config, 0, this.configData);
  }

  setCrossSection(axis, value) {
    const renderSize = this.configData[3];
    const clamped = Math.max(0, Math.min(renderSize, Math.round(value)));
    const axisToIndex = { x: 11, y: 12, z: 13 };
    const configIndex = axisToIndex[axis];
    if (configIndex == null) {
      return;
    }
    this.configData[configIndex] = clamped;
    this.updateOrientation();
  }

  cycleOrientation() {
    this.orientationIndex = (this.orientationIndex + 1) % axisPermutations.length;
    const permutation = axisPermutations[this.orientationIndex];
    this.configData[5] = permutation[0];
    this.configData[6] = permutation[1];
    this.configData[7] = permutation[2];
    this.updateOrientation();
    return this.getOrientationLabel();
  }

  toggleFlip(axis) {
    const index = 8 + axis;
    this.configData[index] = this.configData[index] === 0 ? 1 : 0;
    this.updateOrientation();
    return this.getOrientationLabel();
  }

  rotateQuarterTurn(axis, direction) {
    const transform = quarterTurnTransforms[axis]?.[direction];
    if (!transform) {
      return this.getOrientationLabel();
    }

    const currentMap = [this.configData[5], this.configData[6], this.configData[7]];
    const currentFlip = [this.configData[8], this.configData[9], this.configData[10]];
    const nextMap = currentMap.map((mappedAxis) => transform.axisMap[mappedAxis]);
    const nextFlip = currentFlip.map((flip, index) => flip ^ transform.axisFlip[currentMap[index]]);

    this.configData[5] = nextMap[0];
    this.configData[6] = nextMap[1];
    this.configData[7] = nextMap[2];
    this.configData[8] = nextFlip[0];
    this.configData[9] = nextFlip[1];
    this.configData[10] = nextFlip[2];
    this.syncOrientationIndex();
    this.updateOrientation();
    return this.getOrientationLabel();
  }

  getOrientationLabel() {
    const axisNames = ['x', 'y', 'z'];
    const mapped = [5, 6, 7].map((index, axis) => {
      const name = axisNames[this.configData[index]];
      return this.configData[8 + axis] === 0 ? name : `-${name}`;
    });
    return `display(x,y,z)=model(${mapped.join(',')})`;
  }

  getOrientationState() {
    return {
      axisMap: [this.configData[5], this.configData[6], this.configData[7]],
      axisFlip: [this.configData[8], this.configData[9], this.configData[10]],
    };
  }

  displayToCoarseCoord(display) {
    const coarseSize = this.configData[0];
    const renderSize = this.configData[3];
    const axisMap = [this.configData[5], this.configData[6], this.configData[7]];
    const axisFlip = [this.configData[8] !== 0, this.configData[9] !== 0, this.configData[10] !== 0];

    const axisValue = (axis) => display[axis];
    const maybeFlip = (value, flip) => (flip ? (renderSize - 1) - value : value);
    const renderCoord = axisMap.map((axis, index) => maybeFlip(axisValue(axis), axisFlip[index]));
    const sourceCoord = (coord) => Math.min(
      coarseSize - 1,
      Math.max(0, (((coord + 0.5) * coarseSize) / renderSize) - 0.5)
    );

    return renderCoord.map((coord) => sourceCoord(coord));
  }

  damageAtWorldPosition(position, radius = this.getDefaultDamageRadius()) {
    const coarseSize = this.configData[0];
    const channels = this.configData[1];
    const renderSize = this.configData[3];
    const display = [
      Math.min(renderSize - 1, Math.max(0, position[0] + renderSize * 0.5)),
      Math.min(renderSize - 1, Math.max(0, position[1] + renderSize * 0.5)),
      Math.min(renderSize - 1, Math.max(0, position[2] + renderSize * 0.5)),
    ];
    const center = this.displayToCoarseCoord(display);
    const radiusSquared = radius * radius;
    const minZ = Math.max(0, Math.floor(center[2] - radius));
    const maxZ = Math.min(coarseSize - 1, Math.ceil(center[2] + radius));
    const bytesPerChannel = Float32Array.BYTES_PER_ELEMENT;
    const targets = [this.stateBuffers[0], this.stateBuffers[1], this.rawState];

    for (let z = minZ; z <= maxZ; z += 1) {
      const dzSquared = (z - center[2]) ** 2;
      if (dzSquared > radiusSquared) {
        continue;
      }
      const yzRadius = Math.sqrt(radiusSquared - dzSquared);
      const minY = Math.max(0, Math.floor(center[1] - yzRadius));
      const maxY = Math.min(coarseSize - 1, Math.ceil(center[1] + yzRadius));
      for (let y = minY; y <= maxY; y += 1) {
        const dySquared = (y - center[1]) ** 2;
        const remaining = radiusSquared - dzSquared - dySquared;
        if (remaining < 0) {
          continue;
        }
        const xRadius = Math.sqrt(remaining);
        const minX = Math.max(0, Math.floor(center[0] - xRadius));
        const maxX = Math.min(coarseSize - 1, Math.ceil(center[0] + xRadius));
        const voxelCount = maxX - minX + 1;
        const zeroLine = new Float32Array(voxelCount * channels);
        const offset = ((((z * coarseSize) + y) * coarseSize) + minX) * channels * bytesPerChannel;
        for (const target of targets) {
          this.device.queue.writeBuffer(target, offset, zeroLine);
        }
      }
    }
  }

  async countVisibleVoxels() {
    const staging = this.device.createBuffer({
      size: this.dataByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const command = this.device.createCommandEncoder();
    command.copyBufferToBuffer(this.data, 0, staging, 0, this.dataByteLength);
    this.device.queue.submit([command.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    let count = 0;
    for (let index = 0; index < bytes.length; index += 1) {
      if (bytes[index] > 0) {
        count += 1;
      }
    }
    return count;
  }
}

export default Volume;
