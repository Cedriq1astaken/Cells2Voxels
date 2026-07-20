# Bunny

## Source and license

This target comes from the mesh dataset distributed with the Cells2Pixels project.

- Original dataset: Stanford Bunny
- Original source: [The Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
- Dataset documentation: [Cells2Pixels project](https://cells2pixels.github.io/)
- Original target file: `stanford_bunny.obj`

The Cells2Pixels dataset contains a lower-resolution manifold version of the Stanford Bunny and a higher-resolution version closer to the original. The model was remeshed with PyMeshLab by the MeshNCA authors.

The Stanford repository permits research use, free mirroring and redistribution, and publication of rendered images in scholarly articles or books when the Stanford Computer Graphics Laboratory is credited. Commercial use requires permission. Refer to the repository's ["Please acknowledge" terms](https://graphics.stanford.edu/data/3Dscanrep/) for the complete requirements.

## Processing for Cells2Voxels

The provided mesh was converted into a dense RGBA voxel target and padded to a training resolution of `116 x 116 x 116`.

The exported Cells2Voxels model uses a `29 x 29 x 29` NCA grid, a `4x` LPPN scale, and 64 channels.