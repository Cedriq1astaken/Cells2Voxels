import loadShader from '../loadShader.js';

class Marcher {
  static async create(device, camera, volume) {
    const [intersection, program] = await Promise.all([
      loadShader('./compute/intersection.wgsl'),
      loadShader('./render/marcher.wgsl'),
    ]);
    const module = device.createShaderModule({
      code: program + '\n' + intersection,
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
    const bindings = [0, 1].map(() => device.createBindGroup({
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
          resource: { buffer: volume.getColorData() },
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
