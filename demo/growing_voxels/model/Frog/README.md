# Frog

## Source and license

- Original creator or dataset: 	Lawrence Berkeley Laboratory, USA
- Source page: http://klacansky.com/open-scivis-datasets/
- Direct download: [frog_256x256x44_uint8.raw](http://klacansky.com/open-scivis-datasets/frog/frog_256x256x44_uint8.raw)
- License or usage terms: Not specified on the dataset page
- Original filename: `frog.npy`

## Processing

The original volumetric data was segmented into distinct anatomical components, with each component assigned a unique color for easier differentiation. The resulting model was converted into a dense RGBA voxel target and padded to a training resolution of 264³.