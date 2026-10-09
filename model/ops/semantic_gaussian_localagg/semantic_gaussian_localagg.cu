// Sparse CUDA local aggregation for point-anchored semantic Gaussians.
//
// The implementation deliberately favours an explicit, readable execution
// path: sort Gaussian ids by a finite world-space cell key, then let one CUDA
// thread process one query point and its nearby cell ranges.  It never creates
// a dense GxG distance matrix or a Python-side [G,K,*] gather tensor.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include <cmath>
#include <tuple>

#include "semantic_gaussian_localagg.h"

namespace {

struct BinningState {
  torch::Tensor sorted_keys;
  torch::Tensor sorted_ids;
  torch::Tensor cell_starts;
  torch::Tensor cell_ends;
};

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
  return max(low, min(value, high));
}

__device__ __forceinline__ int cell_key(const float* point, const float* pc_min,
                                        int grid_x, int grid_y, int grid_z,
                                        float cell_size) {
  const int x = clamp_int(static_cast<int>(floorf((point[0] - pc_min[0]) / cell_size)),
                          0, grid_x - 1);
  const int y = clamp_int(static_cast<int>(floorf((point[1] - pc_min[1]) / cell_size)),
                          0, grid_y - 1);
  const int z = clamp_int(static_cast<int>(floorf((point[2] - pc_min[2]) / cell_size)),
                          0, grid_z - 1);
  return (x * grid_y + y) * grid_z + z;
}

__global__ void MakeKeysKernel(const float* points, const bool* valid_mask,
                               int batch, int point_count, const float* pc_min,
                               int grid_x, int grid_y, int grid_z, float cell_size,
                               int total_cells, int* keys, int* ids) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * point_count;
  if (index >= total) return;
  ids[index] = index;
  if (!valid_mask[index]) {
    keys[index] = total_cells;  // Sentinel sorts after every valid cell key.
    return;
  }
  const int scene = index / point_count;
  const int local_key = cell_key(points + index * 3, pc_min, grid_x, grid_y, grid_z, cell_size);
  keys[index] = scene * (grid_x * grid_y * grid_z) + local_key;
}

__global__ void MarkCellRangesKernel(const int* sorted_keys, int item_count,
                                     int total_cells, int* cell_starts, int* cell_ends) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= item_count) return;
  const int key = sorted_keys[index];
  if (key >= total_cells) return;
  if (index == 0 || sorted_keys[index - 1] != key) cell_starts[key] = index;
  if (index == item_count - 1 || sorted_keys[index + 1] != key) cell_ends[key] = index + 1;
}

// Accumulate one candidate neighbour into a query's residual.
//
// Returns true when the neighbour passed both the validity and the per-axis
// box support test, i.e. when it actually contributed work.  The density-cap
// path counts these returns, so the cap bounds *accepted* neighbours rather
// than merely scanned candidates.
__device__ __forceinline__ bool AccumulateForwardNeighbour(
    const float* points, const float* scales, const float* opacities,
    const float* seeds, const bool* valid_mask, int query, int gaussian,
    int classes, float query_x, float query_y, float query_z,
    float support_sigma, float scale_eps, bool include_self,
    float* residual_row, float& weight_sum) {
  if ((!include_self && gaussian == query) || !valid_mask[gaussian]) return false;
  const float* mean = points + gaussian * 3;
  const float* scale = scales + gaussian * 3;
  const float sx = fmaxf(scale[0], scale_eps);
  const float sy = fmaxf(scale[1], scale_eps);
  const float sz = fmaxf(scale[2], scale_eps);
  const float rx = query_x - mean[0];
  const float ry = query_y - mean[1];
  const float rz = query_z - mean[2];
  if (fabsf(rx) > support_sigma * sx || fabsf(ry) > support_sigma * sy ||
      fabsf(rz) > support_sigma * sz) {
    return false;
  }
  const float exponent = (rx / sx) * (rx / sx) + (ry / sy) * (ry / sy) +
                         (rz / sz) * (rz / sz);
  const float weight = opacities[gaussian] * expf(-0.5f * exponent);
  weight_sum += weight;
  const float* seed = seeds + gaussian * classes;
  for (int c = 0; c < classes; ++c) residual_row[c] += weight * seed[c];
  return true;
}

// Backward counterpart of AccumulateForwardNeighbour.  Selection must match
// the forward pass exactly, otherwise the gradient would describe a different
// neighbour set than the one that produced `residual`.
__device__ __forceinline__ bool AccumulateBackwardNeighbour(
    const float* grad_row, const float* points, const float* scales,
    const float* opacities, const float* seeds, const bool* valid_mask,
    int query, int gaussian, int classes, float query_x, float query_y,
    float query_z, float support_sigma, float scale_eps, float normalizer,
    const float* residual_row, bool include_self, float* grad_scales,
    float* grad_opacities, float* grad_seeds) {
  if ((!include_self && gaussian == query) || !valid_mask[gaussian]) return false;
  const float* mean = points + gaussian * 3;
  const float* scale = scales + gaussian * 3;
  const float sx = fmaxf(scale[0], scale_eps);
  const float sy = fmaxf(scale[1], scale_eps);
  const float sz = fmaxf(scale[2], scale_eps);
  const float rx = query_x - mean[0];
  const float ry = query_y - mean[1];
  const float rz = query_z - mean[2];
  if (fabsf(rx) > support_sigma * sx || fabsf(ry) > support_sigma * sy ||
      fabsf(rz) > support_sigma * sz) {
    return false;
  }
  const float exponent = (rx / sx) * (rx / sx) + (ry / sy) * (ry / sy) +
                         (rz / sz) * (rz / sz);
  const float kernel = expf(-0.5f * exponent);
  const float weight = opacities[gaussian] * kernel;
  const float* seed = seeds + gaussian * classes;
  float grad_weight = 0.0f;
  for (int c = 0; c < classes; ++c) {
    atomicAdd(grad_seeds + gaussian * classes + c,
              (weight / normalizer) * grad_row[c]);
    grad_weight += grad_row[c] * (seed[c] - residual_row[c]) / normalizer;
  }
  atomicAdd(grad_opacities + gaussian, grad_weight * kernel);
  // d w / d scale_d = w * residual_d^2 / scale_d^3.  Respect the
  // clamp_min derivative by stopping the gradient at/below scale_eps.
  if (scale[0] > scale_eps)
    atomicAdd(grad_scales + gaussian * 3, grad_weight * weight * rx * rx / (sx * sx * sx));
  if (scale[1] > scale_eps)
    atomicAdd(grad_scales + gaussian * 3 + 1, grad_weight * weight * ry * ry / (sy * sy * sy));
  if (scale[2] > scale_eps)
    atomicAdd(grad_scales + gaussian * 3 + 2, grad_weight * weight * rz * rz / (sz * sz * sz));
  return true;
}

// True when the (dx, dy, dz) offset lies on the Chebyshev shell of radius
// `shell`.  Iterating shells in increasing order visits cells from the query's
// own cell outwards, which makes a neighbour cap keep the nearest cells first
// instead of an arbitrary lexicographic slice.
__device__ __forceinline__ bool OnShell(int dx, int dy, int dz, int shell) {
  const int ax = abs(dx), ay = abs(dy), az = abs(dz);
  const int outer = max(ax, max(ay, az));
  return outer == shell;
}

__global__ void ForwardKernel(
    const float* points, const float* scales, const float* opacities,
    const float* seeds, const bool* valid_mask, const float* pc_min,
    const int* sorted_ids, const int* cell_starts, const int* cell_ends,
    int batch, int point_count, int classes, int grid_x, int grid_y, int grid_z,
    float cell_size, float support_sigma, float scale_eps, float denom_eps,
    int neighbour_cells, bool include_self, int max_neighbours,
    float* residual, float* denominator) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * point_count;
  if (query >= total) return;
  const int out_offset = query * classes;
  if (!valid_mask[query]) {
    denominator[query] = 1.0f;
    for (int c = 0; c < classes; ++c) residual[out_offset + c] = 0.0f;
    return;
  }
  const int scene = query / point_count;
  const float* query_point = points + query * 3;
  const int local_key = cell_key(query_point, pc_min, grid_x, grid_y, grid_z, cell_size);
  const int base_x = local_key / (grid_y * grid_z);
  const int remainder = local_key % (grid_y * grid_z);
  const int base_y = remainder / grid_z;
  const int base_z = remainder % grid_z;
  const float qx = query_point[0], qy = query_point[1], qz = query_point[2];
  float* residual_row = residual + out_offset;

  float weight_sum = 0.0f;
  // C is small (17 occupancy classes); writing the output first avoids a
  // dynamic shared-memory buffer and keeps this reference CUDA kernel clear.
  for (int c = 0; c < classes; ++c) residual_row[c] = 0.0f;

  if (max_neighbours <= 0) {
    // Uncapped path: unchanged from the original release, including its
    // accumulation order, so existing numerics stay bit-comparable.
    for (int dx = -neighbour_cells; dx <= neighbour_cells; ++dx) {
      const int x = base_x + dx;
      if (x < 0 || x >= grid_x) continue;
      for (int dy = -neighbour_cells; dy <= neighbour_cells; ++dy) {
        const int y = base_y + dy;
        if (y < 0 || y >= grid_y) continue;
        for (int dz = -neighbour_cells; dz <= neighbour_cells; ++dz) {
          const int z = base_z + dz;
          if (z < 0 || z >= grid_z) continue;
          const int cell = scene * (grid_x * grid_y * grid_z) + (x * grid_y + y) * grid_z + z;
          const int begin = cell_starts[cell];
          if (begin < 0) continue;
          const int end = cell_ends[cell];
          for (int slot = begin; slot < end; ++slot) {
            AccumulateForwardNeighbour(
                points, scales, opacities, seeds, valid_mask, query,
                sorted_ids[slot], classes, qx, qy, qz, support_sigma, scale_eps,
                include_self, residual_row, weight_sum);
          }
        }
      }
    }
  } else {
    // Density-capped path: visit Chebyshev shells from the query cell outwards
    // and stop once `max_neighbours` neighbours have been accepted.
    int accepted = 0;
    for (int shell = 0; shell <= neighbour_cells && accepted < max_neighbours; ++shell) {
      for (int dx = -shell; dx <= shell && accepted < max_neighbours; ++dx) {
        for (int dy = -shell; dy <= shell && accepted < max_neighbours; ++dy) {
          for (int dz = -shell; dz <= shell && accepted < max_neighbours; ++dz) {
            if (!OnShell(dx, dy, dz, shell)) continue;
            const int x = base_x + dx;
            const int y = base_y + dy;
            const int z = base_z + dz;
            if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) {
              continue;
            }
            const int cell = scene * (grid_x * grid_y * grid_z) + (x * grid_y + y) * grid_z + z;
            const int begin = cell_starts[cell];
            if (begin < 0) continue;
            const int end = cell_ends[cell];
            for (int slot = begin; slot < end && accepted < max_neighbours; ++slot) {
              if (AccumulateForwardNeighbour(
                      points, scales, opacities, seeds, valid_mask, query,
                      sorted_ids[slot], classes, qx, qy, qz, support_sigma,
                      scale_eps, include_self, residual_row, weight_sum)) {
                ++accepted;
              }
            }
          }
        }
      }
    }
  }

  const float normalizer = weight_sum + denom_eps;
  denominator[query] = normalizer;
  for (int c = 0; c < classes; ++c) residual_row[c] /= normalizer;
}

__global__ void BackwardKernel(
    const float* grad_residual, const float* points, const float* scales,
    const float* opacities, const float* seeds, const bool* valid_mask,
    const float* pc_min, const float* residual, const float* denominator,
    const int* sorted_ids, const int* cell_starts, const int* cell_ends,
    int batch, int point_count, int classes, int grid_x, int grid_y, int grid_z,
    float cell_size, float support_sigma, float scale_eps, int neighbour_cells,
    bool include_self, int max_neighbours,
    float* grad_scales, float* grad_opacities, float* grad_seeds) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * point_count;
  if (query >= total || !valid_mask[query]) return;
  const int out_offset = query * classes;
  const int scene = query / point_count;
  const float* query_point = points + query * 3;
  const int local_key = cell_key(query_point, pc_min, grid_x, grid_y, grid_z, cell_size);
  const int base_x = local_key / (grid_y * grid_z);
  const int remainder = local_key % (grid_y * grid_z);
  const int base_y = remainder / grid_z;
  const int base_z = remainder % grid_z;
  const float qx = query_point[0], qy = query_point[1], qz = query_point[2];
  const float normalizer = denominator[query];
  const float* grad_row = grad_residual + out_offset;
  const float* residual_row = residual + out_offset;

  if (max_neighbours <= 0) {
    for (int dx = -neighbour_cells; dx <= neighbour_cells; ++dx) {
      const int x = base_x + dx;
      if (x < 0 || x >= grid_x) continue;
      for (int dy = -neighbour_cells; dy <= neighbour_cells; ++dy) {
        const int y = base_y + dy;
        if (y < 0 || y >= grid_y) continue;
        for (int dz = -neighbour_cells; dz <= neighbour_cells; ++dz) {
          const int z = base_z + dz;
          if (z < 0 || z >= grid_z) continue;
          const int cell = scene * (grid_x * grid_y * grid_z) + (x * grid_y + y) * grid_z + z;
          const int begin = cell_starts[cell];
          if (begin < 0) continue;
          const int end = cell_ends[cell];
          for (int slot = begin; slot < end; ++slot) {
            AccumulateBackwardNeighbour(
                grad_row, points, scales, opacities, seeds, valid_mask, query,
                sorted_ids[slot], classes, qx, qy, qz, support_sigma, scale_eps,
                normalizer, residual_row, include_self, grad_scales,
                grad_opacities, grad_seeds);
          }
        }
      }
    }
  } else {
    int accepted = 0;
    for (int shell = 0; shell <= neighbour_cells && accepted < max_neighbours; ++shell) {
      for (int dx = -shell; dx <= shell && accepted < max_neighbours; ++dx) {
        for (int dy = -shell; dy <= shell && accepted < max_neighbours; ++dy) {
          for (int dz = -shell; dz <= shell && accepted < max_neighbours; ++dz) {
            if (!OnShell(dx, dy, dz, shell)) continue;
            const int x = base_x + dx;
            const int y = base_y + dy;
            const int z = base_z + dz;
            if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) {
              continue;
            }
            const int cell = scene * (grid_x * grid_y * grid_z) + (x * grid_y + y) * grid_z + z;
            const int begin = cell_starts[cell];
            if (begin < 0) continue;
            const int end = cell_ends[cell];
            for (int slot = begin; slot < end && accepted < max_neighbours; ++slot) {
              if (AccumulateBackwardNeighbour(
                      grad_row, points, scales, opacities, seeds, valid_mask,
                      query, sorted_ids[slot], classes, qx, qy, qz,
                      support_sigma, scale_eps, normalizer, residual_row,
                      include_self, grad_scales, grad_opacities, grad_seeds)) {
                ++accepted;
              }
            }
          }
        }
      }
    }
  }
}

BinningState BuildBinning(const torch::Tensor& points, const torch::Tensor& valid_mask,
                          const torch::Tensor& pc_min, int batch, int point_count,
                          int grid_x, int grid_y, int grid_z, float cell_size) {
  const int items = batch * point_count;
  const int cells_per_scene = grid_x * grid_y * grid_z;
  const int total_cells = batch * cells_per_scene;
  auto int_options = points.options().dtype(torch::kInt32);
  auto keys = torch::empty({items}, int_options);
  auto ids = torch::empty({items}, int_options);
  const int threads = 256;
  MakeKeysKernel<<<(items + threads - 1) / threads, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      points.data_ptr<float>(), valid_mask.data_ptr<bool>(), batch, point_count,
      pc_min.data_ptr<float>(), grid_x, grid_y, grid_z, cell_size, total_cells,
      keys.data_ptr<int>(), ids.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto sorted_keys = torch::empty_like(keys);
  auto sorted_ids = torch::empty_like(ids);
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys.data_ptr<int>(), sorted_keys.data_ptr<int>(),
                                  ids.data_ptr<int>(), sorted_ids.data_ptr<int>(), items,
                                  0, sizeof(int) * 8, at::cuda::getDefaultCUDAStream());
  auto temp = torch::empty({static_cast<long long>(temp_bytes)}, points.options().dtype(torch::kUInt8));
  cub::DeviceRadixSort::SortPairs(temp.data_ptr(), temp_bytes, keys.data_ptr<int>(), sorted_keys.data_ptr<int>(),
                                  ids.data_ptr<int>(), sorted_ids.data_ptr<int>(), items,
                                  0, sizeof(int) * 8, at::cuda::getDefaultCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto starts = torch::full({total_cells}, -1, int_options);
  auto ends = torch::full({total_cells}, -1, int_options);
  MarkCellRangesKernel<<<(items + threads - 1) / threads, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      sorted_keys.data_ptr<int>(), items, total_cells, starts.data_ptr<int>(), ends.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted_keys, sorted_ids, starts, ends};
}

void CheckInputs(const torch::Tensor& points, const torch::Tensor& scales,
                 const torch::Tensor& opacities, const torch::Tensor& seeds,
                 const torch::Tensor& valid_mask, const torch::Tensor& pc_min) {
  TORCH_CHECK(points.is_cuda() && scales.is_cuda() && opacities.is_cuda() && seeds.is_cuda() &&
              valid_mask.is_cuda() && pc_min.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(points.scalar_type() == torch::kFloat32 && scales.scalar_type() == torch::kFloat32 &&
              opacities.scalar_type() == torch::kFloat32 && seeds.scalar_type() == torch::kFloat32,
              "semantic Gaussian localagg supports float32 attributes only");
  TORCH_CHECK(points.is_contiguous() && scales.is_contiguous() && opacities.is_contiguous() &&
              seeds.is_contiguous() && valid_mask.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(points.dim() == 3 && points.size(2) == 3 && scales.sizes() == points.sizes(),
              "points/scales must be [B,G,3]");
  TORCH_CHECK(opacities.sizes() == points.sizes().slice(0, 2), "opacities must be [B,G]");
  TORCH_CHECK(seeds.dim() == 3 && seeds.size(0) == points.size(0) && seeds.size(1) == points.size(1),
              "semantic_seed must be [B,G,C]");
  TORCH_CHECK(valid_mask.sizes() == points.sizes().slice(0, 2), "valid_mask must be [B,G]");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> SemanticGaussianLocalAggForward(
    const torch::Tensor& points, const torch::Tensor& scales, const torch::Tensor& opacities,
    const torch::Tensor& semantic_seed, const torch::Tensor& valid_mask,
    const torch::Tensor& pc_min, int grid_x, int grid_y, int grid_z, float cell_size,
    float support_sigma, float scale_eps, float denom_eps, int neighbour_cells,
    bool include_self, int max_neighbours) {
  CheckInputs(points, scales, opacities, semantic_seed, valid_mask, pc_min);
  const int batch = points.size(0), point_count = points.size(1), classes = semantic_seed.size(2);
  auto bins = BuildBinning(points, valid_mask, pc_min, batch, point_count,
                           grid_x, grid_y, grid_z, cell_size);
  auto residual = torch::zeros_like(semantic_seed);
  auto denominator = torch::zeros({batch, point_count}, points.options());
  const int total = batch * point_count, threads = 128;
  ForwardKernel<<<(total + threads - 1) / threads, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      points.data_ptr<float>(), scales.data_ptr<float>(), opacities.data_ptr<float>(),
      semantic_seed.data_ptr<float>(), valid_mask.data_ptr<bool>(), pc_min.data_ptr<float>(),
      bins.sorted_ids.data_ptr<int>(), bins.cell_starts.data_ptr<int>(), bins.cell_ends.data_ptr<int>(),
      batch, point_count, classes, grid_x, grid_y, grid_z, cell_size, support_sigma,
      scale_eps, denom_eps, neighbour_cells, include_self, max_neighbours,
      residual.data_ptr<float>(), denominator.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(residual, denominator);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SemanticGaussianLocalAggBackward(
    const torch::Tensor& grad_residual, const torch::Tensor& points, const torch::Tensor& scales,
    const torch::Tensor& opacities, const torch::Tensor& semantic_seed,
    const torch::Tensor& valid_mask, const torch::Tensor& pc_min,
    const torch::Tensor& residual, const torch::Tensor& denominator, int grid_x, int grid_y,
    int grid_z, float cell_size, float support_sigma, float scale_eps, float /*denom_eps*/,
    int neighbour_cells, bool include_self, int max_neighbours) {
  CheckInputs(points, scales, opacities, semantic_seed, valid_mask, pc_min);
  TORCH_CHECK(grad_residual.sizes() == semantic_seed.sizes(), "grad_residual shape mismatch");
  const int batch = points.size(0), point_count = points.size(1), classes = semantic_seed.size(2);
  auto bins = BuildBinning(points, valid_mask, pc_min, batch, point_count,
                           grid_x, grid_y, grid_z, cell_size);
  auto grad_scales = torch::zeros_like(scales);
  auto grad_opacities = torch::zeros_like(opacities);
  auto grad_seed = torch::zeros_like(semantic_seed);
  const int total = batch * point_count, threads = 128;
  BackwardKernel<<<(total + threads - 1) / threads, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      grad_residual.contiguous().data_ptr<float>(), points.data_ptr<float>(), scales.data_ptr<float>(),
      opacities.data_ptr<float>(), semantic_seed.data_ptr<float>(), valid_mask.data_ptr<bool>(),
      pc_min.data_ptr<float>(), residual.data_ptr<float>(), denominator.data_ptr<float>(),
      bins.sorted_ids.data_ptr<int>(), bins.cell_starts.data_ptr<int>(), bins.cell_ends.data_ptr<int>(),
      batch, point_count, classes, grid_x, grid_y, grid_z, cell_size, support_sigma, scale_eps,
      neighbour_cells, include_self, max_neighbours,
      grad_scales.data_ptr<float>(), grad_opacities.data_ptr<float>(),
      grad_seed.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(grad_scales, grad_opacities, grad_seed);
}
