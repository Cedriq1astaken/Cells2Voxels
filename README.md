# Cells2Voxels

A browser-based set of 3D morphogenesis unofficial demos inspired by [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v2) (Pajouheshgar et al., 2025).


[Launch the live demo](https://cedriq1astaken.github.io/Cells2Voxels/demo/growing_voxels/) · [Open the training demo](https://cedriq1astaken.github.io/Cells2Voxels/demo/train_growing_voxels/)

![Demo gif](/misc/demo.gif)

## Demos

- `demo/growing_voxels/` - evolve and interact with pretrained voxel NCAs.
- `demo/train_growing_voxels/` - train a voxel NCA + LPPN in the browser and export a model for the pretrained viewer.

The training demo uses f16 WebGPU forward/backward passes with f32 Adam master weights, so it needs a browser and GPU with `shader-f16` support.

### Growing Voxels

The demo runs entirely on webgpu, from the model inference to the visualization. The NCA grows on a small 3D grid (usually 4 or 8 times smaller than the original model). At every step, each cell looks at its neighbors and updates its own values. The LPPN then turns this small grid into a larger colored voxel model by using both the cell values and the position being drawn inside each cell. A living mask decides which parts of the model should appear.

The viewer supports real-time orbit controls, per-axis model rotation, adjustable LPPN resolution (default by 4 times the NCA grid but can go up to 8), cross sections (some models have interior data), and click-to-damage regeneration. The full decoded volume is retained for damage and sectioning, while surface compaction avoids drawing completely enclosed voxels. Simulation and rendering are paced separately so the interface can remain responsive while larger models grow.

The graphics use a rasterized WebGPU pipeline. Before drawing, a GPU shader removes empty voxels and voxels hidden inside the model. The renderer keeps one cube mesh and draws a copy of it at every visible voxel position, using the voxel's stored color and opacity. Depth testing, hidden-face removal, transparency, and simple lighting produce the final image. Moving the camera only redraws the existing cubes. Growth, damage, cross sections, and resolution changes rebuild the visible voxel list.

### Training Growing Voxels

The training demo trains the NCA and LPPN directly in the browser. It accepts MagicaVoxel `.vox` models or `.obj` meshes with matching Cells2Pixels `.vol` textures and converts them into colored voxel targets. Training starts from a small seed, grows it for a random number of steps, compares the result with the target, and adjusts the model. A pool of earlier growth states helps it learn to continue growing and recover from imperfect states.

Most training calculations use f16 to reduce GPU memory and improve speed. Adam keeps its main copy of the weights in f32 for stability. The page estimates memory use before training, shows the model as it learns, plots the loss, and exports files that can be opened directly in the Growing Voxels viewer.

The GPU handles the growth steps, voxel decoding, loss calculation, backpropagation, and weight updates. The training data stays on the GPU instead of being copied through JavaScript after every step.

The training preview uses the same rasterized approach as the pretrained viewer. It removes hidden voxels, then draws copies of one shared cube for the visible voxels. Its voxel data uses f16 to save memory.

A quick warning: this is still an experimental browser demo, not a replacement for full-scale training. Larger models can use a lot of GPU memory and may not run well on an average computer

## Notebooks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cedriq1astaken/Cells2Voxels/blob/master/notebook/Growing_Voxel.ipynb)

The notebooks contain the longer-form training workflows behind the demos. They are mainly for experimenting with targets, training settings, and exports rather than simply running the finished viewer.

### [Growing Voxel](notebook/Growing%20Voxel.ipynb)

The main 3D voxel training notebook: it loads `.vox`, `.npy`, or `.obj` + `.vol` targets, trains a coarse 3D NCA with an LPPN decoder, and includes previews, slices, loss plots, recovery tests, and selectable training modes. It can export a browser-compatible model and create a verified Google Drive backup, and is intended to run in Google Colab with a CUDA GPU.

## TODO (These are ideas and may never be completed)

### Growing Radiance Fields

- [ ] Explore a version where the 3D NCA grows a radiance field instead of a voxel model.
