import loadShader from '../loadShader.js';

const injectShaderConstants = (source, constants) => {
  return `${constants}${source}`;
};

class Marcher {
  static async create(device, camera, volume) {
    const modelInfo = volume.getModelInfo();
    const shaderConstants = [
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
    const [intersection, program] = await Promise.all([
      loadShader('./compute/intersection.wgsl'),
      loadShader('./render/marcher.wgsl'),
    ]);
    const marcherProgram = injectShaderConstants(program, shaderConstants);
    const module = device.createShaderModule({
      code: marcherProgram + '\n' + intersection,
    });
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        entryPoint: 'vertexMain',
        module,
      },
      fragment: {
        entryPoint: 'fragmentMain',
        module,
        targets: [
          { format: 'rgba16float' },
          { format: 'rgba16float' },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
    const bindings = [0, 1].map((stateIndex) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: camera },
        },
        {
          binding: 1,
          resource: { buffer: volume.getData() },
        },
        {
          binding: 2,
          resource: { buffer: volume.getSize() },
        },
        {
          binding: 3,
          resource: { buffer: volume.getStateBuffer(stateIndex) },
        },
        {
          binding: 4,
          resource: { buffer: volume.getConfig() },
        },
        {
          binding: 5,
          resource: { buffer: volume.getLppnParams() },
        },
      ],
    }));
    return new Marcher(bindings, pipeline, volume);
  }

  constructor(bindings, pipeline, volume) {
    this.bindings = bindings;
    this.pipeline = pipeline;
    this.volume = volume;
  }

  render(pass) {
    const { bindings, pipeline, volume } = this;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings[volume.getCurrentStateIndex()]);
    pass.draw(6);
  }
}

export default Marcher;
