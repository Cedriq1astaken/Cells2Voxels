# Cells2Voxels

A browser-based set of 3D morphogenesis demos inspired by [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v2).
![DEMO front page](/misc/demo.png)

## Demos

- `demo/growing_voxels/` - evolve and interact with pretrained voxel NCAs.
- `demo/train_growing_voxels/` - train a voxel NCA + LPPN in the browser and export a model for the pretrained viewer.

The training demo uses f16 WebGPU forward/backward passes with f32 Adam master weights, so it needs a browser and GPU with `shader-f16` support.

### Growing Voxels

The pretrained viewer runs entirely with WebGPU. The NCA grows on a small 3D grid. At every step, each cell looks at its neighbors and updates its own values. The LPPN then turns this small grid into a larger colored voxel model by using both the cell values and the position being drawn inside each cell. A living mask decides which parts of the model should appear.

The viewer supports real-time orbit controls, per-axis model rotation, adjustable LPPN resolution, cross sections, and click-to-damage regeneration. The full decoded volume is retained for damage and sectioning, while surface compaction avoids drawing completely enclosed voxels. Simulation and rendering are paced separately so the interface can remain responsive while larger models grow.

The graphics use a rasterized WebGPU pipeline. Before drawing, a GPU shader removes empty voxels and voxels hidden inside the model. The renderer keeps one cube mesh and draws a copy of it at every visible voxel position, using the voxel's stored color and opacity. Depth testing, hidden-face removal, transparency, and simple lighting produce the final image. Moving the camera only redraws the existing cubes. Growth, damage, cross sections, and resolution changes rebuild the visible voxel list.

### Training Growing Voxels

The training demo trains the NCA and LPPN directly in the browser. It accepts MagicaVoxel `.vox` models or `.obj` meshes with matching Cells2Pixels `.vol` textures and converts them into colored voxel targets. Training starts from a small seed, grows it for a random number of steps, compares the result with the target, and adjusts the model. A pool of earlier growth states helps it learn to continue growing and recover from imperfect states.

Most training calculations use f16 to reduce GPU memory and improve speed. Adam keeps its main copy of the weights in f32 for stability. The page estimates memory use before training, shows the model as it learns, plots the loss, and exports files that can be opened directly in the Growing Voxels viewer.

The GPU handles the growth steps, voxel decoding, loss calculation, backpropagation, and weight updates. The training data stays on the GPU instead of being copied through JavaScript after every step.

The training preview uses the same rasterized approach as the pretrained viewer. It removes hidden voxels, then draws copies of one shared cube for the visible voxels. Its voxel data uses f16 to save memory.

## TODO

### Growing Radiance Fields

Explore a version where the 3D NCA grows a radiance field instead of a voxel model. A neural renderer would turn the grown state into color and density and render it from the camera's view. The current prototype is experimental. Future work includes finishing training, improving rendering speed, adding damage and regeneration, and exporting models in a format that the browser demo can load.

## Notebooks

`notebook/` contains the accompanying exploratory notebooks, including the growing-voxel workflow.
