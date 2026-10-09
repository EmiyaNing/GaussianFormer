#pragma once

#include <torch/extension.h>
#include <vector>

// Must match model/head/localagg/src/config.h (NUM_CHANNELS 18): the shared
// dense G2V path materialises a fixed channel count.
#define BLOCK_G2V_CHANNELS 18

// The compile-time tile edge (voxels) this translation unit was built with.
// Exposed so a tile-size sweep can VERIFY the build instead of trusting the
// label it passed to the compiler: if the macro plumbing ever breaks, every
// row of the sweep would silently re-measure the default tile.
int BlockG2VTileEdge();

std::vector<torch::Tensor> BlockG2VForward(const torch::Tensor& pts, const torch::Tensor& points_int,
                                           const torch::Tensor& means3D, const torch::Tensor& means3D_int,
                                           const torch::Tensor& opacities, const torch::Tensor& semantics,
                                           const torch::Tensor& radii, const torch::Tensor& cov3D,
                                           int64_t grid_x, int64_t grid_y, int64_t grid_z);

std::vector<torch::Tensor> BlockG2VBackward(
    const torch::Tensor& grad_out, const torch::Tensor& pts,
    const torch::Tensor& points_int, const torch::Tensor& means3D,
    const torch::Tensor& means3D_int, const torch::Tensor& opacities,
    const torch::Tensor& semantics, const torch::Tensor& radii,
    const torch::Tensor& cov3D, const torch::Tensor& sorted_keys,
    const torch::Tensor& sorted_ids, int64_t grid_x, int64_t grid_y,
    int64_t grid_z);
