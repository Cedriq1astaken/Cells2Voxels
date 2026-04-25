import Camera from './camera.js';
import Postprocessing from './postprocessing.js';

const HALF_FLOAT_MIN_NORMAL = 2 ** -14;

const float16ToFloat32 = (value) => {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign * 0;
    }
    return sign * (fraction / 1024) * HALF_FLOAT_MIN_NORMAL;
  }

  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }

  return sign * (1 + fraction / 1024) * (2 ** (exponent - 15));
};

class Renderer {
  static async create(device) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const postprocessing = await Postprocessing.create(device, format);
    return new Renderer(device, format, postprocessing);
  }

  constructor(device, format, postprocessing) {
    this.camera = new Camera(device);
    this.canvas = document.createElement('canvas');
    const context = this.canvas.getContext('webgpu');
    if (!context) {
      throw new Error("Couldn't get GPUCanvasContext");
    }
    this.context = context;
    this.format = format;
    this.context.configure({ alphaMode: 'opaque', device, format: this.format });
    this.postprocessing = postprocessing;
    this.textures = [
      {
        attachment: {
          clearValue: { r: 0.28, g: 0.55, b: 0.9, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
          view: null,
        },
        texture: null,
      },
      {
        attachment: {
          clearValue: { r: 0, g: 0, b: 0, a: 10000 },
          loadOp: 'clear',
          storeOp: 'store',
          view: null,
        },
        texture: null,
      },
    ];
    this.descriptor = {
      colorAttachments: [
        this.textures[0].attachment,
        this.textures[1].attachment,
      ],
    };
    
    this.device = device;
    this.renderables = [];

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();

    this.animate = this.animate.bind(this);
    this.animation = {
      loop: () => {},
      request: requestAnimationFrame(this.animate),
      time: performance.now() / 1000,
    };
  }

  add(renderable) {
    this.renderables.push(renderable);
  }

  remove(renderable) {
    this.renderables = this.renderables.filter((item) => item !== renderable);
  }

  getCanvas() {
    return this.canvas;
  }

  getCamera() {
    return this.camera;
  }

  getDevice() {
    return this.device;
  }

  async readDataDepthAtCanvasPoint(clientX, clientY) {
    const { canvas, device, textures } = this;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }

    const normalizedX = (clientX - rect.left) / rect.width;
    const normalizedY = (clientY - rect.top) / rect.height;
    if (normalizedX < 0 || normalizedX > 1 || normalizedY < 0 || normalizedY > 1) {
      return null;
    }

    const pixelX = Math.min(canvas.width - 1, Math.max(0, Math.floor(normalizedX * canvas.width)));
    const pixelY = Math.min(canvas.height - 1, Math.max(0, Math.floor(normalizedY * canvas.height)));
    const bytesPerRow = 256;
    const staging = device.createBuffer({
      size: bytesPerRow,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const command = device.createCommandEncoder();
    command.copyTextureToBuffer(
      {
        texture: textures[1].texture,
        origin: { x: pixelX, y: pixelY, z: 0 },
      },
      {
        buffer: staging,
        bytesPerRow,
        rowsPerImage: 1,
      },
      {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
      }
    );
    device.queue.submit([command.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const bytes = staging.getMappedRange();
    const channels = new Uint16Array(bytes, 0, 4);
    const depth = float16ToFloat32(channels[3]);
    staging.unmap();
    staging.destroy();

    if (!Number.isFinite(depth) || depth >= 9999) {
      return null;
    }
    return depth;
  }

  setAnimationLoop(loop) {
    this.animation.loop = loop;
  }

  animate() {
    const { animation, device } = this;
    const time = performance.now() / 1000;
    const delta = Math.min(time - animation.time, 0.1);
    animation.time = time;
    animation.request = requestAnimationFrame(this.animate);

    const command = device.createCommandEncoder();
    animation.loop(command, delta, time);
    this.render(command);
    device.queue.submit([command.finish()]);
  }

  render(command) {
    const { context, descriptor, postprocessing, renderables } = this;
    const pass = command.beginRenderPass(descriptor);
    renderables.forEach((renderable) => renderable.render(pass));
    pass.end();
    postprocessing.render(command, context.getCurrentTexture().createView());
  }

  resize() {
    const { camera, canvas, device, postprocessing, textures } = this;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const pixelRatio = window.devicePixelRatio || 1;
    const size = [Math.floor(width * pixelRatio), Math.floor(height * pixelRatio)];
    canvas.width = size[0];
    canvas.height = size[1];
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    camera.setAspect(width / height);
    const views = textures.map((texture) => {
      if (texture.texture) {
        texture.texture.destroy();
      }
      texture.texture = device.createTexture({
        format: 'rgba16float',
        size,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
      });
      texture.attachment.view = texture.texture.createView();
      return texture.attachment.view;
    });
    postprocessing.setTextures(views[0], views[1]);
  }
}

export default Renderer;
