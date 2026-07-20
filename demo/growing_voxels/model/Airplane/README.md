# Airplane

## Source and license

This target comes from the mesh dataset distributed with the Cells2Pixels project.

- Original creator: YahooJAPAN
- Original source: [Plane on Thingiverse](https://www.thingiverse.com/thing:182252)
- Dataset documentation: [Cells2Pixels project](https://cells2pixels.github.io/)
- License: [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/)
- Original target file: `airplane.obj`

The dataset contains the original airplane mesh and a remeshed version with exaggerated windows cut out of the mesh. The remeshed version was created with PyMeshLab by the MeshNCA authors and is also distributed under CC BY 3.0, with attribution to the original creator.

## Processing for Cells2Voxels

The provided mesh was converted into a dense RGBA voxel target and padded to a training resolution of `136 x 136 x 136`.

The exported Cells2Voxels model uses a `34 x 34 x 34` NCA grid, a `4x` LPPN scale, 48 channels, and one coordinate frequency.