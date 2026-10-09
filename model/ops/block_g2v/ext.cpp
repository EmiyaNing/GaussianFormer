#include <torch/extension.h>

#include "block_g2v.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &BlockG2VForward,
        "S2GO-style 4x4x4 block dense Gaussian-to-voxel forward (CUDA)");
  m.def("backward", &BlockG2VBackward,
        "S2GO-style 4x4x4 block dense Gaussian-to-voxel backward (CUDA)");
  m.def("tile_edge", &BlockG2VTileEdge,
        "Compile-time tile edge in voxels (verifies a tile-size sweep build)");
}
