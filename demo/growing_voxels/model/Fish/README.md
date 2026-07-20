# Fish

## Source and license

This target comes from the mesh dataset distributed with the Cells2Pixels project.

- Original model: Blub the Fish
- Original creator: Keenan Crane
- Creator's model repository: [Keenan Crane's Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/)
- Dataset documentation: [Cells2Pixels project](https://cells2pixels.github.io/)
- License: Public domain

The dataset contains the original quad control mesh and textures, along with triangle meshes at two resolutions. The remeshed versions were created with PyMeshLab by the MeshNCA authors and are also released into the public domain.

## Processing for Cells2Voxels

The provided mesh and texture data were converted into a dense RGBA voxel target and padded to a training resolution of `144 x 144 x 144`.

The exported Cells2Voxels model uses a `36 x 36 x 36` NCA grid, a `4x` LPPN scale, and 48 channels.