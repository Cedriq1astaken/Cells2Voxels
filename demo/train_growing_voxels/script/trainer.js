import { createBuffer, createEmptyBuffer } from './gpu.js';
import { float16ToFloat32, fromFloat16Array, toFloat16Array } from './f16.js';

const KERNEL_COUNT = 5;
const LIVING_CHANNEL = 3;
const LIVING_THRESHOLD = 0.1;
const LPPN_WIDTH = 64;
const FIRST_OMEGA = 10;
const HIDDEN_OMEGA = 10;
const COORD_FREQUENCIES = 1;
const TRAIN_SAMPLE_LIMIT = 8192;
const POOL_BUDGET_BYTES = 64 * 1024 * 1024;
const PARAM_STRIDE = 256;
const PARAM_CAPACITY = 4096;

export class VoxelTrainer {
  constructor(device, config, targetData, shaderSources, callbacks = {}) {
    if (!device.features.has('shader-f16')) {
      throw new Error('This GPU does not expose the WebGPU shader-f16 feature.');
    }
    this.device = device;
    this.queue = device.queue;
    this.config = { ...config };
    this.callbacks = callbacks;
    this.targetSize = config.targetSize;
    this.scale = config.scale;
    this.coarseSize = this.targetSize / this.scale;
    this.channels = config.channels;
    this.fcDim = config.fcDim;
    this.lppnWidth = LPPN_WIDTH;
    this.inputDim = this.channels + 6 * COORD_FREQUENCIES;
    this.baseLearningRate = config.learningRate;
    this.learningRate = config.learningRate;
    this.iteration = 0;
    this.lastReportIteration = 0;
    this.lastReportTime = 0;
    this.running = false;
    this.loopPromise = null;
    this.disposed = false;

    if (!Number.isInteger(this.coarseSize)) {
      throw new Error(`Target size ${this.targetSize} is not divisible by scale ${this.scale}.`);
    }
    if (this.channels < 4) throw new Error('At least four NCA channels are required.');
    if (this.config.stepMax > 96) throw new Error('The f16 trainer supports at most 96 rollout steps.');

    this.cells = this.coarseSize ** 3;
    this.stateElements = this.cells * this.channels;
    this.totalVoxels = this.targetSize ** 3;
    this.trainRows = Math.min(this.totalVoxels, TRAIN_SAMPLE_LIMIT);
    this.maxRows = this.trainRows;
    this.poolSize = Math.max(
      8,
      Math.min(64, Math.floor(POOL_BUDGET_BYTES / (this.stateElements * Uint16Array.BYTES_PER_ELEMENT))),
    );
    this.previewInterval = config.previewInterval ?? (this.targetSize <= 32 ? 8 : 16);
    this.shaderSources = shaderSources;
    this.resources = new Set();
    this.paramsOffset = 0;

    this.buildPipelines();
    this.buildModel();
    this.buildTrainingBuffers(targetData);
  }

  buildPipelines() {
    const replacements = {
      S: this.coarseSize,
      R: this.targetSize,
      C: this.channels,
      H: this.fcDim,
    };
    const shader = key => replaceTemplate(this.shaderSources[key], replacements);
    this.layouts = {
      ncaForward: this.createLayout([
        'storage', 'storage', 'read-only-storage', 'read-only-storage',
        'read-only-storage', 'read-only-storage', 'uniform',
      ]),
      dense: this.createLayout([
        'read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'uniform',
      ]),
      denseBackward: this.createLayout([
        'read-only-storage', 'read-only-storage', 'read-only-storage',
        'read-only-storage', 'storage', 'uniform',
      ]),
      loss: this.createLayout([
        'read-only-storage', 'read-only-storage', 'read-only-storage',
        'storage', 'storage', 'storage', 'uniform',
      ]),
      ncaBackward: this.createLayout([
        'read-only-storage', 'read-only-storage', 'read-only-storage', 'read-only-storage',
        'storage', 'storage', 'storage', 'uniform',
      ]),
      optimizer: this.createLayout(['storage', 'storage', 'storage', 'storage', 'uniform']),
    };

    const modules = {
      ncaForward: this.createModule('f16_nca_forward', shader('ncaForward')),
      dense: this.createModule('f16_dense', this.shaderSources.dense),
      denseBackward: this.createModule('f16_dense_backward', this.shaderSources.denseBackward),
      loss: this.createModule('f16_loss', shader('loss')),
      ncaBackward: this.createModule('f16_nca_backward', shader('ncaBackward')),
      optimizer: this.createModule('f16_optimizer', this.shaderSources.optimizer),
    };
    const pipeline = (family, entryPoint) => this.device.createComputePipeline({
      label: `${family}_${entryPoint}`,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layouts[family]] }),
      compute: { module: modules[family], entryPoint },
    });
    this.pipelines = {
      ncaHidden: pipeline('ncaForward', 'nca_hidden'),
      ncaStep: pipeline('ncaForward', 'nca_step'),
      denseForward: pipeline('dense', 'dense_forward'),
      activationDelta: pipeline('denseBackward', 'activation_delta'),
      denseDx: pipeline('denseBackward', 'dense_dx'),
      denseDw: pipeline('denseBackward', 'dense_dw'),
      denseDb: pipeline('denseBackward', 'dense_db'),
      buildInput: pipeline('loss', 'build_lppn_input'),
      lossGradient: pipeline('loss', 'morphology_loss_gradient'),
      reduceLoss: pipeline('loss', 'reduce_loss'),
      coarseGradient: pipeline('loss', 'coarse_state_gradient'),
      writePreview: pipeline('loss', 'write_preview'),
      ncaLocalGradient: pipeline('ncaBackward', 'nca_local_gradient'),
      ncaHiddenGradient: pipeline('ncaBackward', 'nca_hidden_gradient'),
      ncaPerceptionGradient: pipeline('ncaBackward', 'nca_perception_gradient'),
      ncaStateGradient: pipeline('ncaBackward', 'nca_state_gradient'),
      accumulateW2: pipeline('ncaBackward', 'accumulate_w2_gradient'),
      accumulateW1: pipeline('ncaBackward', 'accumulate_w1_gradient'),
      accumulateB1: pipeline('ncaBackward', 'accumulate_b1_gradient'),
      normalizeGradient: pipeline('optimizer', 'normalize_gradient'),
      adamUpdate: pipeline('optimizer', 'adam_update'),
    };
  }

  createLayout(types) {
    return this.device.createBindGroupLayout({
      entries: types.map((type, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: type === 'uniform'
          ? { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 }
          : { type },
      })),
    });
  }

  createModule(label, code) {
    const module = this.device.createShaderModule({ label, code });
    module.getCompilationInfo?.().then(info => {
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) console.error(`${label} WGSL errors:`, errors);
    });
    return module;
  }

  buildModel() {
    const perceptionDim = this.channels * KERNEL_COUNT;
    const hiddenBound = Math.sqrt(6 / LPPN_WIDTH) / HIDDEN_OMEGA;
    this.parameters = [];
    this.parameterMap = {};
    this.addParameter(
      'ncaW1',
      [perceptionDim, this.fcDim],
      randomNormal(perceptionDim * this.fcDim, 0.1 * Math.sqrt(2 / (perceptionDim + this.fcDim))),
      true,
    );
    this.addParameter('ncaB1', [this.fcDim], randomUniform(this.fcDim, 1 / Math.sqrt(perceptionDim)), true);
    this.addParameter(
      'ncaW2',
      [this.fcDim, this.channels],
      randomNormal(this.fcDim * this.channels, 0.1 * Math.sqrt(2 / (this.fcDim + this.channels))),
      true,
    );
    this.addParameter(
      'l0w',
      [this.inputDim, LPPN_WIDTH],
      randomUniform(this.inputDim * LPPN_WIDTH, 1 / this.inputDim),
    );
    this.addParameter('l0b', [LPPN_WIDTH], new Float32Array(LPPN_WIDTH));
    this.addParameter('l1w', [LPPN_WIDTH, LPPN_WIDTH], randomUniform(LPPN_WIDTH ** 2, hiddenBound));
    this.addParameter('l1b', [LPPN_WIDTH], new Float32Array(LPPN_WIDTH));
    this.addParameter('l2w', [LPPN_WIDTH, LPPN_WIDTH], randomUniform(LPPN_WIDTH ** 2, hiddenBound));
    this.addParameter('l2b', [LPPN_WIDTH], new Float32Array(LPPN_WIDTH));
    this.addParameter('l3w', [LPPN_WIDTH, 4], randomUniform(LPPN_WIDTH * 4, hiddenBound));
    this.addParameter('l3b', [4], new Float32Array(4));
  }

  addParameter(name, shape, values, normalize = false) {
    const count = shape.reduce((product, value) => product * value, 1);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const parameter = {
      name,
      shape,
      count,
      normalize,
      weight: this.track(createBuffer(this.device, toFloat16Array(values), usage)),
      gradient: this.track(createEmptyBuffer(this.device, alignedF16Bytes(count), usage)),
      firstMoment: this.track(createEmptyBuffer(this.device, alignedF16Bytes(count), usage)),
      secondMoment: this.track(createEmptyBuffer(this.device, alignedF16Bytes(count), usage)),
    };
    this.parameters.push(parameter);
    this.parameterMap[name] = parameter;
  }

  buildTrainingBuffers(targetData) {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const maxSteps = this.config.stepMax;
    const stateHistoryCount = (maxSteps + 1) * this.stateElements;
    const hiddenHistoryCount = maxSteps * this.cells * this.fcDim;
    this.paramsBuffer = this.track(this.device.createBuffer({
      size: PARAM_STRIDE * PARAM_CAPACITY,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
    this.dummyBuffers = Array.from({ length: 8 }, () => (
      this.track(createEmptyBuffer(this.device, 4, storage))
    ));
    this.dummyBuffer = this.dummyBuffers[0];
    this.targetBuffer = this.track(createBuffer(this.device, toFloat16Array(targetData), storage));
    this.kernelBuffer = this.track(createBuffer(
      this.device,
      toFloat16Array(buildPerceptionKernels()),
      storage,
    ));
    this.stateHistory = this.track(createEmptyBuffer(this.device, alignedF16Bytes(stateHistoryCount), storage));
    this.hiddenHistory = this.track(createEmptyBuffer(this.device, alignedF16Bytes(hiddenHistoryCount), storage));
    this.poolBuffer = this.track(createBuffer(this.device, this.makeInitialPool(), storage));
    this.seedBuffer = this.track(createBuffer(this.device, this.makeSeed(), storage));

    const rows = this.maxRows;
    this.lppnInput = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * this.inputDim), storage));
    this.h0 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.h1 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.h2 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.rawOutput = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * 4), storage));
    this.dOutput = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * 4), storage));
    this.dAlpha = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows), storage));
    this.lossValues = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows), storage));
    this.lossScalar = this.track(createEmptyBuffer(this.device, 4, storage));
    this.lossReadback = this.track(createEmptyBuffer(
      this.device,
      8,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    ));
    this.dH2 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.dH1 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.dH0 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.dZ2 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.dZ1 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.dZ0 = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * LPPN_WIDTH), storage));
    this.dInput = this.track(createEmptyBuffer(this.device, alignedF16Bytes(rows * this.inputDim), storage));

    this.gradStateA = this.track(createEmptyBuffer(this.device, alignedF16Bytes(this.stateElements), storage));
    this.gradStateB = this.track(createEmptyBuffer(this.device, alignedF16Bytes(this.stateElements), storage));
    this.baseStateGradient = this.track(createEmptyBuffer(this.device, alignedF16Bytes(this.stateElements), storage));
    this.dNcaOutput = this.track(createEmptyBuffer(this.device, alignedF16Bytes(this.stateElements), storage));
    this.dNcaHidden = this.track(createEmptyBuffer(this.device, alignedF16Bytes(this.cells * this.fcDim), storage));
    this.dPerception = this.track(createEmptyBuffer(
      this.device,
      alignedF16Bytes(this.cells * this.channels * KERNEL_COUNT),
      storage,
    ));
    this.previewBuffer = this.track(createEmptyBuffer(
      this.device,
      alignedF16Bytes(this.totalVoxels * 4),
      storage,
    ));
  }

  makeSeed() {
    const values = new Float32Array(this.stateElements);
    const center = Math.floor(this.coarseSize / 2);
    const radius = Math.max(0, Math.min(this.config.seedRadius - 1, center));
    for (let z = center - radius; z <= center + radius; z++) {
      for (let y = center - radius; y <= center + radius; y++) {
        for (let x = center - radius; x <= center + radius; x++) {
          const base = ((z * this.coarseSize + y) * this.coarseSize + x) * this.channels;
          for (let channel = LIVING_CHANNEL; channel < this.channels; channel++) {
            values[base + channel] = 1;
          }
        }
      }
    }
    return toFloat16Array(values);
  }

  makeInitialPool() {
    const seed = this.makeSeed();
    this.poolStateBytes = alignedF16Bytes(this.stateElements);
    const wordsPerState = this.poolStateBytes / Uint16Array.BYTES_PER_ELEMENT;
    const values = new Uint16Array(this.poolSize * wordsPerState);
    for (let index = 0; index < this.poolSize; index++) {
      values.set(seed, index * wordsPerState);
    }
    this.initialPoolData = values;
    return values;
  }

  track(resource) {
    this.resources.add(resource);
    return resource;
  }

  resetParameterArena() {
    this.paramsOffset = 0;
  }

  allocateParams(data) {
    if (this.paramsOffset >= PARAM_CAPACITY) {
      throw new Error('The f16 command parameter arena is exhausted.');
    }
    const offset = this.paramsOffset++ * PARAM_STRIDE;
    this.queue.writeBuffer(this.paramsBuffer, offset, data);
    return offset;
  }

  makeParams(uintValues = [], floatValues = []) {
    const buffer = new ArrayBuffer(64);
    const uints = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    for (const [index, value] of uintValues) uints[index] = value >>> 0;
    for (const [index, value] of floatValues) floats[index] = value;
    return buffer;
  }

  ncaParams(step, rolloutSteps, randomSeed) {
    return this.makeParams(
      [[0, step], [1, rolloutSteps], [2, randomSeed], [3, this.stateElements]],
      [[4, this.config.updateProbability]],
    );
  }

  denseParams(rows, inputDim, outputDim, omega) {
    return this.makeParams(
      [[0, rows], [1, inputDim], [2, outputDim]],
      [[4, omega]],
    );
  }

  lossParams(rows, sampleSeed, sampleOffset, sampleMode, finalStep) {
    return this.makeParams(
      [
        [0, rows], [1, sampleSeed], [2, sampleOffset], [3, sampleMode],
        [4, finalStep], [5, this.totalVoxels], [6, this.stateElements],
      ],
      [[8, this.scale], [9, this.config.overflowWeight]],
    );
  }

  optimizerParams(parameter, iteration) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    return this.makeParams(
      [[0, parameter.count], [1, iteration], [2, parameter.normalize ? 1 : 0]],
      [
        [4, this.learningRate],
        [5, beta1],
        [6, beta2],
        [7, 1 - beta1 ** iteration],
        [8, 1 - beta2 ** iteration],
        [9, 0.0001],
      ],
    );
  }

  bindGroup(cacheKey, layout, buffers) {
    this.bindGroups ??= new Map();
    if (this.bindGroups.has(cacheKey)) return this.bindGroups.get(cacheKey);
    const entries = buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer: buffer === this.dummyBuffer ? this.dummyBuffers[binding] : buffer },
    }));
    entries.push({
      binding: buffers.length,
      resource: { buffer: this.paramsBuffer, offset: 0, size: 64 },
    });
    const bindGroup = this.device.createBindGroup({ layout, entries });
    this.bindGroups.set(cacheKey, bindGroup);
    return bindGroup;
  }

  dispatch(pass, pipeline, bindGroup, workItems, params) {
    const offset = this.allocateParams(params);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup, [offset]);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(workItems / 64)));
  }

  ncaForwardBindGroup() {
    const p = this.parameterMap;
    return this.bindGroup('nca-forward', this.layouts.ncaForward, [
      this.stateHistory,
      this.hiddenHistory,
      this.kernelBuffer,
      p.ncaW1.weight,
      p.ncaB1.weight,
      p.ncaW2.weight,
    ]);
  }

  denseBindGroup(cacheKey, input, weight, bias, output) {
    return this.bindGroup(cacheKey, this.layouts.dense, [input, weight, bias, output]);
  }

  denseBackwardBindGroup(cacheKey, input0, input1, input2, input3, output) {
    return this.bindGroup(
      cacheKey,
      this.layouts.denseBackward,
      [input0, input1, input2, input3, output],
    );
  }

  lossBindGroup(cacheKey, input0, input1, input2, output0, output1, output2) {
    return this.bindGroup(
      cacheKey,
      this.layouts.loss,
      [input0, input1, input2, output0, output1, output2],
    );
  }

  ncaBackwardBindGroup(cacheKey, input0, input1, input2, input3, output0, output1, accumulator) {
    return this.bindGroup(
      cacheKey,
      this.layouts.ncaBackward,
      [input0, input1, input2, input3, output0, output1, accumulator],
    );
  }

  encodeDenseForward(pass, rows) {
    const p = this.parameterMap;
    const layers = [
      ['l0', this.lppnInput, p.l0w, p.l0b, this.h0, this.inputDim, LPPN_WIDTH, FIRST_OMEGA],
      ['l1', this.h0, p.l1w, p.l1b, this.h1, LPPN_WIDTH, LPPN_WIDTH, HIDDEN_OMEGA],
      ['l2', this.h1, p.l2w, p.l2b, this.h2, LPPN_WIDTH, LPPN_WIDTH, HIDDEN_OMEGA],
      ['l3', this.h2, p.l3w, p.l3b, this.rawOutput, LPPN_WIDTH, 4, 0],
    ];
    for (const [name, input, weight, bias, output, inputDim, outputDim, omega] of layers) {
      this.dispatch(
        pass,
        this.pipelines.denseForward,
        this.denseBindGroup(`dense-forward-${name}`, input, weight.weight, bias.weight, output),
        rows * outputDim,
        this.denseParams(rows, inputDim, outputDim, omega),
      );
    }
  }

  encodeDenseLayerBackward(pass, options) {
    const {
      name, rows, input, weight, bias, upstream, delta, inputGradient,
      inputDim, outputDim, omega, outputIsLinear = false,
    } = options;
    const activeDelta = outputIsLinear ? upstream : delta;
    const params = this.denseParams(rows, inputDim, outputDim, omega);
    if (!outputIsLinear) {
      this.dispatch(
        pass,
        this.pipelines.activationDelta,
        this.denseBackwardBindGroup(
          `activation-${name}`,
          input,
          weight.weight,
          bias.weight,
          upstream,
          delta,
        ),
        rows * outputDim,
        params,
      );
    }
    this.dispatch(
      pass,
      this.pipelines.denseDx,
      this.denseBackwardBindGroup(
        `dx-${name}`,
        activeDelta,
        weight.weight,
        this.dummyBuffer,
        this.dummyBuffer,
        inputGradient,
      ),
      rows * inputDim,
      params,
    );
    this.dispatch(
      pass,
      this.pipelines.denseDw,
      this.denseBackwardBindGroup(
        `dw-${name}`,
        input,
        this.dummyBuffer,
        this.dummyBuffer,
        activeDelta,
        weight.gradient,
      ),
      inputDim * outputDim,
      params,
    );
    this.dispatch(
      pass,
      this.pipelines.denseDb,
      this.denseBackwardBindGroup(
        `db-${name}`,
        this.dummyBuffer,
        this.dummyBuffer,
        this.dummyBuffer,
        activeDelta,
        bias.gradient,
      ),
      outputDim,
      params,
    );
  }

  encodeDenseBackward(pass, rows) {
    const p = this.parameterMap;
    this.encodeDenseLayerBackward(pass, {
      name: 'l3', rows, input: this.h2, weight: p.l3w, bias: p.l3b,
      upstream: this.dOutput, delta: this.dOutput, inputGradient: this.dH2,
      inputDim: LPPN_WIDTH, outputDim: 4, omega: 0, outputIsLinear: true,
    });
    this.encodeDenseLayerBackward(pass, {
      name: 'l2', rows, input: this.h1, weight: p.l2w, bias: p.l2b,
      upstream: this.dH2, delta: this.dZ2, inputGradient: this.dH1,
      inputDim: LPPN_WIDTH, outputDim: LPPN_WIDTH, omega: HIDDEN_OMEGA,
    });
    this.encodeDenseLayerBackward(pass, {
      name: 'l1', rows, input: this.h0, weight: p.l1w, bias: p.l1b,
      upstream: this.dH1, delta: this.dZ1, inputGradient: this.dH0,
      inputDim: LPPN_WIDTH, outputDim: LPPN_WIDTH, omega: HIDDEN_OMEGA,
    });
    this.encodeDenseLayerBackward(pass, {
      name: 'l0', rows, input: this.lppnInput, weight: p.l0w, bias: p.l0b,
      upstream: this.dH0, delta: this.dZ0, inputGradient: this.dInput,
      inputDim: this.inputDim, outputDim: LPPN_WIDTH, omega: FIRST_OMEGA,
    });
  }

  encodeNcaBackward(pass, rolloutSteps, randomSeed) {
    const p = this.parameterMap;
    let gradientNext = this.gradStateA;
    let gradientCurrent = this.gradStateB;
    for (let step = rolloutSteps - 1; step >= 0; step--) {
      const params = this.ncaParams(step, rolloutSteps, randomSeed);
      this.dispatch(
        pass,
        this.pipelines.ncaLocalGradient,
        this.ncaBackwardBindGroup(
          `nca-local-${gradientNext === this.gradStateA ? 'a' : 'b'}`,
          this.stateHistory,
          this.hiddenHistory,
          p.ncaW2.weight,
          gradientNext,
          this.dNcaOutput,
          this.baseStateGradient,
          this.dummyBuffer,
        ),
        this.stateElements,
        params,
      );
      this.dispatch(
        pass,
        this.pipelines.ncaHiddenGradient,
        this.ncaBackwardBindGroup(
          'nca-hidden-gradient',
          this.hiddenHistory,
          p.ncaW2.weight,
          this.dNcaOutput,
          this.dummyBuffer,
          this.dNcaHidden,
          this.dummyBuffer,
          this.dummyBuffer,
        ),
        this.cells * this.fcDim,
        params,
      );
      this.dispatch(
        pass,
        this.pipelines.accumulateW2,
        this.ncaBackwardBindGroup(
          'nca-w2-gradient',
          this.hiddenHistory,
          this.dNcaOutput,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          p.ncaW2.gradient,
        ),
        p.ncaW2.count,
        params,
      );
      this.dispatch(
        pass,
        this.pipelines.accumulateW1,
        this.ncaBackwardBindGroup(
          'nca-w1-gradient',
          this.stateHistory,
          this.dNcaHidden,
          this.kernelBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          p.ncaW1.gradient,
        ),
        p.ncaW1.count,
        params,
      );
      this.dispatch(
        pass,
        this.pipelines.accumulateB1,
        this.ncaBackwardBindGroup(
          'nca-b1-gradient',
          this.dNcaHidden,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
          p.ncaB1.gradient,
        ),
        p.ncaB1.count,
        params,
      );
      this.dispatch(
        pass,
        this.pipelines.ncaPerceptionGradient,
        this.ncaBackwardBindGroup(
          'nca-perception-gradient',
          this.dNcaHidden,
          p.ncaW1.weight,
          this.dummyBuffer,
          this.dummyBuffer,
          this.dPerception,
          this.dummyBuffer,
          this.dummyBuffer,
        ),
        this.cells * this.channels * KERNEL_COUNT,
        params,
      );
      this.dispatch(
        pass,
        this.pipelines.ncaStateGradient,
        this.ncaBackwardBindGroup(
          `nca-state-gradient-${gradientCurrent === this.gradStateA ? 'a' : 'b'}`,
          this.baseStateGradient,
          this.dPerception,
          this.kernelBuffer,
          this.dummyBuffer,
          gradientCurrent,
          this.dummyBuffer,
          this.dummyBuffer,
        ),
        this.stateElements,
        params,
      );
      [gradientNext, gradientCurrent] = [gradientCurrent, gradientNext];
    }
  }

  encodePreview(pass, finalStep) {
    for (let offset = 0; offset < this.totalVoxels; offset += this.maxRows) {
      const rows = Math.min(this.maxRows, this.totalVoxels - offset);
      const params = this.lossParams(rows, 0, offset, 1, finalStep);
      this.dispatch(
        pass,
        this.pipelines.buildInput,
        this.lossBindGroup(
          'build-input',
          this.stateHistory,
          this.dummyBuffer,
          this.dummyBuffer,
          this.lppnInput,
          this.dummyBuffer,
          this.dummyBuffer,
        ),
        rows * this.inputDim,
        params,
      );
      this.encodeDenseForward(pass, rows);
      this.dispatch(
        pass,
        this.pipelines.writePreview,
        this.lossBindGroup(
          'write-preview',
          this.lppnInput,
          this.rawOutput,
          this.dummyBuffer,
          this.previewBuffer,
          this.dummyBuffer,
          this.dummyBuffer,
        ),
        rows * 4,
        params,
      );
    }
  }

  encodeOptimizer(pass, iteration) {
    for (const parameter of this.parameters) {
      const params = this.optimizerParams(parameter, iteration);
      const bindGroup = this.bindGroup(
        `optimizer-${parameter.name}`,
        this.layouts.optimizer,
        [parameter.weight, parameter.gradient, parameter.firstMoment, parameter.secondMoment],
      );
      if (parameter.normalize) {
        this.dispatch(pass, this.pipelines.normalizeGradient, bindGroup, 1, params);
      }
      this.dispatch(pass, this.pipelines.adamUpdate, bindGroup, parameter.count, params);
    }
  }

  updateLearningRate() {
    const cycleStep = this.iteration % 4000;
    const decayCount = [1000, 2000, 3000].filter(step => cycleStep >= step).length;
    this.learningRate = this.baseLearningRate * 0.3 ** decayCount;
  }

  async trainIteration() {
    const nextIteration = this.iteration + 1;
    if (this.iteration > 0 && this.iteration % 4000 === 0) {
      this.queue.writeBuffer(this.poolBuffer, 0, this.initialPoolData);
    }
    this.updateLearningRate();
    this.resetParameterArena();

    const poolIndex = Math.floor(Math.random() * this.poolSize);
    const repetition = Math.floor(this.iteration / 4000) % 4;
    const injectInterval = 4 * 2 ** repetition;
    const injectSeed = nextIteration % injectInterval === 0;
    const rolloutRange = Math.max(1, this.config.stepMax - this.config.stepMin);
    const rolloutSteps = this.config.stepMin + Math.floor(Math.random() * rolloutRange);
    const keepPreview = nextIteration === 1 || nextIteration % this.previewInterval === 0;
    const randomSeed = Math.floor(Math.random() * 0xffffffff) >>> 0;
    const sampleSeed = Math.floor(Math.random() * 0xffffffff) >>> 0;
    const stateBytes = this.poolStateBytes;
    const encoder = this.device.createCommandEncoder({ label: 'f16_training_iteration' });

    if (injectSeed) {
      encoder.copyBufferToBuffer(this.seedBuffer, 0, this.stateHistory, 0, stateBytes);
    } else {
      encoder.copyBufferToBuffer(
        this.poolBuffer,
        poolIndex * stateBytes,
        this.stateHistory,
        0,
        stateBytes,
      );
    }
    for (const parameter of this.parameters) {
      encoder.clearBuffer(parameter.gradient);
      if (this.iteration > 0 && this.iteration % 4000 === 0) {
        encoder.clearBuffer(parameter.firstMoment);
        encoder.clearBuffer(parameter.secondMoment);
      }
    }

    const pass = encoder.beginComputePass({ label: 'f16_training' });
    const ncaBindGroup = this.ncaForwardBindGroup();
    for (let step = 0; step < rolloutSteps; step++) {
      const params = this.ncaParams(step, rolloutSteps, randomSeed);
      this.dispatch(
        pass,
        this.pipelines.ncaHidden,
        ncaBindGroup,
        this.cells * this.fcDim,
        params,
      );
      this.dispatch(pass, this.pipelines.ncaStep, ncaBindGroup, this.stateElements, params);
    }

    const lossParams = this.lossParams(this.trainRows, sampleSeed, 0, 0, rolloutSteps);
    this.dispatch(
      pass,
      this.pipelines.buildInput,
      this.lossBindGroup(
        'build-input',
        this.stateHistory,
        this.dummyBuffer,
        this.dummyBuffer,
        this.lppnInput,
        this.dummyBuffer,
        this.dummyBuffer,
      ),
      this.trainRows * this.inputDim,
      lossParams,
    );
    this.encodeDenseForward(pass, this.trainRows);
    this.dispatch(
      pass,
      this.pipelines.lossGradient,
      this.lossBindGroup(
        'loss-gradient',
        this.lppnInput,
        this.rawOutput,
        this.targetBuffer,
        this.dOutput,
        this.dAlpha,
        this.lossValues,
      ),
      this.trainRows,
      lossParams,
    );
    this.dispatch(
      pass,
      this.pipelines.reduceLoss,
      this.lossBindGroup(
        'reduce-loss',
        this.lossValues,
        this.stateHistory,
        this.dummyBuffer,
        this.lossScalar,
        this.dummyBuffer,
        this.dummyBuffer,
      ),
      1,
      lossParams,
    );
    this.encodeDenseBackward(pass, this.trainRows);
    this.dispatch(
      pass,
      this.pipelines.coarseGradient,
      this.lossBindGroup(
        'coarse-gradient',
        this.stateHistory,
        this.dInput,
        this.dAlpha,
        this.gradStateA,
        this.dummyBuffer,
        this.dummyBuffer,
      ),
      this.stateElements,
      lossParams,
    );
    this.encodeNcaBackward(pass, rolloutSteps, randomSeed);
    if (keepPreview) this.encodePreview(pass, rolloutSteps);
    this.encodeOptimizer(pass, nextIteration);
    pass.end();

    encoder.copyBufferToBuffer(
      this.stateHistory,
      rolloutSteps * stateBytes,
      this.poolBuffer,
      poolIndex * stateBytes,
      stateBytes,
    );
    if (keepPreview) encoder.copyBufferToBuffer(this.lossScalar, 0, this.lossReadback, 0, 4);
    this.queue.submit([encoder.finish()]);
    this.iteration = nextIteration;

    if (keepPreview) {
      await this.lossReadback.mapAsync(GPUMapMode.READ);
      const bits = new DataView(this.lossReadback.getMappedRange(0, 8)).getUint16(0, true);
      const loss = float16ToFloat32(bits);
      this.lossReadback.unmap();
      if (!Number.isFinite(loss)) throw new Error(`Loss became invalid after iteration ${this.iteration}.`);
      const reportTime = performance.now();
      const reportIterations = this.iteration - this.lastReportIteration;
      const iterationsPerSecond = reportIterations * 1000
        / Math.max(1, reportTime - this.lastReportTime);
      this.lastReportIteration = this.iteration;
      this.lastReportTime = reportTime;
      this.callbacks.onStep?.({
        iteration: this.iteration,
        loss,
        iterationsPerSecond,
        rolloutSteps,
        preview: {
          buffer: this.previewBuffer,
          byteLength: alignedF16Bytes(this.totalVoxels * 4),
        },
        poolSize: this.poolSize,
        learningRate: this.learningRate,
      });
    }
  }

  start() {
    if (this.running || this.disposed) return this.loopPromise;
    this.running = true;
    this.lastReportIteration = this.iteration;
    this.lastReportTime = performance.now();
    this.callbacks.onRunningChange?.(true);
    this.loopPromise = this.runLoop();
    return this.loopPromise;
  }

  async runLoop() {
    try {
      while (this.running && !this.disposed) {
        await this.trainIteration();
        if (this.iteration % 4 === 0) {
          await new Promise(resolve => requestAnimationFrame(resolve));
        }
      }
    } catch (error) {
      this.running = false;
      this.callbacks.onError?.(error);
      throw error;
    } finally {
      this.running = false;
      this.callbacks.onRunningChange?.(false);
    }
  }

  async stop() {
    this.running = false;
    try {
      await this.loopPromise;
    } catch {
      // The application callback already received the training error.
    }
    this.loopPromise = null;
  }

  async getExportInfo() {
    const readbacks = this.parameters.map(parameter => this.track(createEmptyBuffer(
      this.device,
      Math.max(8, Math.ceil(parameter.count * Uint16Array.BYTES_PER_ELEMENT / 8) * 8),
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    )));
    const encoder = this.device.createCommandEncoder();
    this.parameters.forEach((parameter, index) => {
      encoder.copyBufferToBuffer(
        parameter.weight,
        0,
        readbacks[index],
        0,
        Math.max(8, Math.ceil(parameter.count * Uint16Array.BYTES_PER_ELEMENT / 8) * 8),
      );
    });
    this.queue.submit([encoder.finish()]);
    const variables = {};
    await Promise.all(readbacks.map(async (buffer, index) => {
      await buffer.mapAsync(GPUMapMode.READ);
      const parameter = this.parameters[index];
      const values = new Uint16Array(buffer.getMappedRange(), 0, parameter.count);
      variables[parameter.name] = fromFloat16Array(values);
      buffer.unmap();
      buffer.destroy();
      this.resources.delete(buffer);
    }));
    return {
      iteration: this.iteration,
      targetSize: this.targetSize,
      coarseSize: this.coarseSize,
      scale: this.scale,
      channels: this.channels,
      fcDim: this.fcDim,
      lppnWidth: this.lppnWidth,
      seedRadius: this.config.seedRadius,
      poolSize: this.poolSize,
      variables,
    };
  }

  async dispose() {
    if (this.disposed) return;
    await this.stop();
    this.disposed = true;
    for (const resource of this.resources) resource.destroy?.();
    this.resources.clear();
    this.bindGroups?.clear();
  }
}

export function buildPerceptionKernels() {
  const smooth = [1, 2, 1];
  const derivative = [-1, 0, 1];
  const values = new Float32Array(KERNEL_COUNT * 27);
  let offset = 0;
  for (let kernel = 0; kernel < KERNEL_COUNT; kernel++) {
    for (let z = 0; z < 3; z++) {
      for (let y = 0; y < 3; y++) {
        for (let x = 0; x < 3; x++) {
          let value = 0;
          if (kernel === 0) {
            value = z === 1 && y === 1 && x === 1 ? 1 : 0;
          } else if (kernel === 1) {
            value = derivative[z] * smooth[y] * smooth[x] / 2;
          } else if (kernel === 2) {
            value = smooth[z] * derivative[y] * smooth[x] / 2;
          } else if (kernel === 3) {
            value = smooth[z] * smooth[y] * derivative[x] / 2;
          } else {
            const nonZeroAxes = [z - 1, y - 1, x - 1].filter(component => component !== 0).length;
            value = nonZeroAxes === 0 ? -88 : nonZeroAxes === 1 ? 6 : nonZeroAxes === 2 ? 3 : 2;
            value /= 8;
          }
          values[offset++] = value;
        }
      }
    }
  }
  return values;
}

export const MODEL_CONSTANTS = {
  kernelCount: KERNEL_COUNT,
  livingChannel: LIVING_CHANNEL,
  livingThreshold: LIVING_THRESHOLD,
  lppnWidth: LPPN_WIDTH,
  firstOmega: FIRST_OMEGA,
  hiddenOmega: HIDDEN_OMEGA,
  coordinateFrequencies: COORD_FREQUENCIES,
};

function alignedF16Bytes(count) {
  return Math.max(4, Math.ceil(count * Uint16Array.BYTES_PER_ELEMENT / 4) * 4);
}

function replaceTemplate(source, replacements) {
  if (!source) throw new Error('A required f16 training shader did not load.');
  return Object.entries(replacements).reduce(
    (code, [key, value]) => code.replaceAll(`{{${key}}}`, String(value)),
    source,
  );
}

function randomUniform(count, bound) {
  const values = new Float32Array(count);
  for (let index = 0; index < count; index++) values[index] = (Math.random() * 2 - 1) * bound;
  return values;
}

function randomNormal(count, standardDeviation) {
  const values = new Float32Array(count);
  for (let index = 0; index < count; index += 2) {
    const radius = Math.sqrt(-2 * Math.log(Math.max(Number.EPSILON, Math.random())));
    const angle = Math.PI * 2 * Math.random();
    values[index] = Math.cos(angle) * radius * standardDeviation;
    if (index + 1 < count) values[index + 1] = Math.sin(angle) * radius * standardDeviation;
  }
  return values;
}
