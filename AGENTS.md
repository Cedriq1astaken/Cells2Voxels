# Cells2Voxels Workspace Context

This file stores durable context for future agent work. Update it when project structure, goals, or important assumptions change.

## Project purpose

- This workspace is a demo-oriented extension of **Neural Cellular Automata: From Cells to Pixels**: <https://arxiv.org/html/2506.22899v2>.
- Its inherited core idea is a coarse-lattice NCA decoded by a coordinate-conditioned Local Pattern Producing Network (LPPN).
- Active development focuses on browser-based 3D morphogenesis demos.

## Authoritative layout

- `Cells2Pixels/` was previously copied in as official reference code, but has been removed from this workspace.
- The official 2D demo was referenced as `Cells2Pixels/Cells2Pixels.github.io/`, but is not currently present locally.
- `demo/growing_voxels/`: demo 1, pretrained voxel morphogenesis.
- `demo/growing_radiance_fields/`: demo 2, incomplete radiance-field morphogenesis.
- `demo/train_growing_voxels/`: demo 3, in-browser voxel-morphogenesis training.
- `notebook/Growing Voxel.ipynb` and `notebook/Growing Radiance Field.ipynb` are currently present.
- `notebook/Generate Model Images.ipynb` is a training-free utility that loads NPY/VOX targets or solid-voxelizes OBJ meshes with their Cells2Pixels VOL textures, displays four Matplotlib voxel views, and saves `[Model name].png` plus alternate views for model info panels.
- The user described notebooks for demos 1 and 3, while the observed filenames naturally map to demos 1 and 2. Verify this mapping if it becomes relevant.

## Demo architecture

- The demos are browser applications built with WebGPU JavaScript and WGSL.
- `growing_voxels` loads exported NCA/LPPN weights, evolves a coarse 3D NCA in ping-pong GPU buffers, decodes a higher-resolution RGBA volume, compacts living voxels, and renders them interactively. It supports damage and regeneration.
  - Every model initially loads at a global 4x LPPN display scale; the scale control can still change it at runtime, independently of the scale recorded in the model manifest.
  - Rendering compaction keeps only occupied voxels with at least one exposed six-connected face. The full decoded volume remains available, and cross-section clipping exposes the newly cut interior surface before compaction.
  - Simulation pacing is time-based with a 30 Hz baseline (`1x`), independent of render FPS; long-frame catch-up is clamped to avoid GPU submission bursts.
  - Frog and Tomato show a GPU-performance warning. Any decoded size of 200 or greater loads paused and requires the user to press Start.
  - LPPN scale rebuilds stop frame submission and await the WebGPU queue before destroying decoder/renderer resources; the scale control stays disabled during rebuilding.
  - LPPN decoding remains dense in storage but sparse in neural evaluation: a conservative fine-alpha scan compacts only 4x4x4 blocks whose one-voxel-expanded neighborhood can pass the existing living test, then an indirect dispatch evaluates those blocks. The output is cleared before each sparse decode so damage and disappearing regions cannot leave stale voxels.
  - On devices with `shader-f16`, the fine-resolution decoded RGBA and interpolated alpha buffers use f16 storage (with an f32 fallback); LPPN neural calculations remain f32.
  - Damage picking ray-marches the decoded RGBA alpha field on the GPU using the same alpha threshold as rendering, then clears full state vectors with a GPU brush pass. Evolution pauses while the pick readback is pending and shows one wound frame before resuming.
  - Render instances use direct f32 XYZ/RGBA records in a dynamically growing GPU buffer, favoring vertex throughput over packed-memory savings. Voxel-count readback is throttled, stable bind groups are cached, and model/scale replacement explicitly destroys owned GPU buffers.
- `growing_radiance_fields` evolves a coarse 3D NCA and uses neural radiance decoding plus ray rendering. Treat it as experimental and incomplete.
- `train_growing_voxels` imports VOX or OBJ+VOL targets, trains a 3D NCA+LPPN in-browser using f16 WebGPU forward/backward buffers plus f32 Adam master weights and moments, previews the result, and exports a package compatible with the pretrained voxel demo.

## Paper concepts relevant here

- Identical NCA cells iteratively apply learned local perception and update rules on a coarse grid.
- LPPN combines interpolated local cell state with intra-primitive coordinates to render higher-resolution appearance.
- Periodic local-coordinate encoding provides continuity across cube-cell boundaries.
- Morphogenesis training supervises rendered appearance and the living mask and uses stochastic updates, varied rollouts, and state pools for robustness.
- The trainer's living gate is a zero-padded `3x3x3` max pool over the circularly interpolated fine alpha field. The pretrained viewer computes the same fine alpha field in a separate GPU pass before LPPN decoding.
- In `notebook/Growing Voxel.ipynb`, sparse-target training uses separately normalized foreground/background decoded-RGBA losses, decoded-alpha Tversky, and a low-weight continuous NCA-alpha scaffold; its training-only straight-through living gate preserves the hard forward/rendered mask and exported model format.
- Voxel and radiance-field morphogenesis in this workspace extend beyond the paper's main 2D morphology experiment.

## Working guidance

- Prioritize `demo/` and `notebook/` for active work. Do not assume the official `Cells2Pixels/` reference code is available locally.
- Read a demo's app orchestration, GPU wrapper, shaders, loader/exporter, renderer, and UI together before changing behavior.
- Keep the three demos conceptually separate while preserving compatible model formats where appropriate.
- `growing_voxels` and `train_growing_voxels` intentionally share the same named UI palette tokens; mirror shared visual-color changes in both `style.css` files while keeping each layout local.
- Deferred trainer-performance roadmap: preserve the NCA/LPPN architecture and model format, first bound GPU work in flight and separate/throttle previews, then optimize perception reuse and tiled gradient reductions. Validate fixed-seed small targets against the existing path before replacing it.
- Preserve unrelated user changes and generated/model assets.
- Do not edit notebooks mechanically unless the task specifically involves them.
- `notebook/Growing Voxel.ipynb` begins with a Google Drive authorization cell and ends with a timestamped Drive backup cell that creates and verifies a portable checkpoint plus fresh web export; its following shutdown cell flushes Drive before calling Colab runtime unassignment.
- Validate changes with the narrowest relevant check first and report checks that could not be run.
- Never store secrets, credentials, access tokens, or private chat content here.

## Agent access limitations

- Agents can inspect files in this workspace and messages in the current conversation.
- Agents cannot see separate previous chats unless their contents are supplied or made accessible.
