/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "local_aggregate.h"

// 声明逆渲染函数
torch::Tensor LocalAggregateInverseCUDA(
    const torch::Tensor& pts,
    const torch::Tensor& points_int,
    const torch::Tensor& means3D,
    const torch::Tensor& means3D_int,
    const torch::Tensor& opacity,
    const torch::Tensor& occupancy_gt,
    const torch::Tensor& cov3D,
    const torch::Tensor& radii,
    const int H, int W, int D);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("local_aggregate", &LocalAggregateCUDA);
  m.def("local_aggregate_backward", &LocalAggregateBackwardCUDA);
  m.def("local_aggregate_inverse", &LocalAggregateInverseCUDA);
}