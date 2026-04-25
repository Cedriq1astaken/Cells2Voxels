import loadShader from '../loadShader.js';

class Postprocessing {
  static async create(device, format) {
    const program = await loadShader('./render/postprocessing.wgsl');
    const module = device.createShaderModule({
      code: program,
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
        targets: [{ format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
    return new Postprocessing(device, pipeline);
  }

  constructor(device, pipeline) {
    this.descriptor = {
      colorAttachments: [{
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
        view: null,
      }],
    };
    this.device = device;
    this.pipeline = pipeline;
    this.bindings = undefined;
  }

  setTextures(color, data) {
    const { device, pipeline } = this;
    this.bindings = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: color,
        },
        {
          binding: 1,
          resource: data,
        },
      ],
    });
  }

  render(command, output) {
    const { bindings, descriptor, pipeline } = this;
    if (!bindings) {
      throw new Error('Trying to render postprocessing without textures');
    }
    descriptor.colorAttachments[0].view = output;
    const pass = command.beginRenderPass(descriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.draw(6);
    pass.end();
  }
}

export default Postprocessing;
