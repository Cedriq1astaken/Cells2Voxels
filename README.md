# Cells2Voxels

A browser-based set of 3D morphogenesis demos inspired by [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v2).

## Demos

- `demo/growing_voxels/` - evolve and interact with pretrained voxel NCAs.
- `demo/growing_radiance_fields/` - experimental radiance-field morphogenesis demo.
- `demo/train_growing_voxels/` - train a voxel NCA + LPPN in the browser and export a model for the pretrained viewer.

The training demo uses f16 WebGPU forward/backward passes with f32 Adam master weights, so it needs a browser and GPU with `shader-f16` support.

## Run locally

Serve the repository with any local static-file server, then open one of the demo folders in a WebGPU-capable browser such as Chrome or Edge.

For example:

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000/demo/growing_voxels/`.

## Notebooks

`notebook/` contains the accompanying exploratory notebooks, including the growing-voxel workflow.

## Status

This is an active demo project. The voxel demo and browser trainer are the main working paths; the radiance-field demo is still experimental.