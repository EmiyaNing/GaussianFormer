#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> SemanticGaussianLocalAggForward(
    const torch::Tensor& points, const torch::Tensor& scales,
    const torch::Tensor& opacities, const torch::Tensor& semantic_seed,
    const torch::Tensor& valid_mask, const torch::Tensor& pc_min,
    int grid_x, int grid_y, int grid_z, float cell_size, float support_sigma,
    float scale_eps, float denom_eps, int neighbour_cells, bool include_self);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SemanticGaussianLocalAggBackward(
    const torch::Tensor& grad_residual, const torch::Tensor& points,
    const torch::Tensor& scales, const torch::Tensor& opacities,
    const torch::Tensor& semantic_seed, const torch::Tensor& valid_mask,
    const torch::Tensor& pc_min, const torch::Tensor& residual,
    const torch::Tensor& denominator, int grid_x, int grid_y, int grid_z,
    float cell_size, float support_sigma, float scale_eps, float denom_eps,
    int neighbour_cells, bool include_self);
