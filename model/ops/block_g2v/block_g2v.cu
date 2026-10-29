// S2GO-style block Gaussian-to-voxel rasteriser (forward only).
//
// The repository's dense `localagg` (`model/head/localagg/src/forward.cu`) is a
// naive per-voxel scan: one thread per voxel walks its own Gaussian list
// straight from global memory, with no block-level cooperation.  Measured at
// G=76,800 it runs at ~48% of peak DRAM bandwidth with traffic matching
// ``pairs x 112 B``, i.e. cache absorbs little of the reuse (see
// analysis/sparse_render/results/P4-2_block_g2v_bound.md).
//
// This kernel mirrors what S2GO describes for its efficient G2V:
//
//   1. partition the voxel grid into 4x4x4 voxel tiles;
//   2. bin each Gaussian into every *tile* its voxel AABB overlaps, and sort
//      the (tile, gaussian) pairs, so a tile owns the UNION of its voxels'
//      Gaussian lists;
//   3. one CTA per tile, cooperating to stream that union through shared
//      memory in chunks, with each of the 64 threads rendering one voxel.
//
// The bound computed in P4-2 is that step 2-3 cut the number of Gaussian loads
// from ``L = sum_g |voxels(g)|`` to ``U = sum_g |tiles(g)|``, a factor of ~22
// on uniform, clustered and real point distributions alike.  This kernel exists
// to measure how much of that bound is realisable.
//
// FORWARD ONLY.  No backward is implemented, so this kernel can be used to
// profile inference but must not be dropped into a training path.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include "block_g2v.h"

namespace {

// Tile edge in voxels.  Overridable at build time (``-DBLOCK_G2V_TILE_EDGE=8``)
// so the tile-size sweep can find how much of the analytic load-count bound is
// reachable; the default build is unchanged at 4.
#ifndef BLOCK_G2V_TILE_EDGE
#define BLOCK_G2V_TILE_EDGE 4
#endif

constexpr int TILE_EDGE = BLOCK_G2V_TILE_EDGE;                // 4x4x4 voxels
constexpr int TILE_VOXELS = TILE_EDGE * TILE_EDGE * TILE_EDGE; // 64
// One thread per voxel.  The render kernel derives a tile voxel from ``tid``
// with no stride loop, so a whole tile must fit in one CTA.
static_assert(TILE_VOXELS <= 1024, "tile too large for a single CTA");
constexpr int THREADS = TILE_VOXELS;                          // one per voxel
constexpr int LOAD_CHUNK = 64;                                // Gaussians staged at once

__device__ __forceinline__ int clamp_int(int value, int low, int high) {
  return max(low, min(value, high));
}

// Same voxel AABB as localagg's getRect: [p - r, p + r + 1) clipped to the grid.
__device__ __forceinline__ void voxelRect(const int* p, int radius, int3 grid,
                                          int3& lo, int3& hi) {
  lo = make_int3(clamp_int(p[0] - radius, 0, grid.x),
                 clamp_int(p[1] - radius, 0, grid.y),
                 clamp_int(p[2] - radius, 0, grid.z));
  hi = make_int3(clamp_int(p[0] + radius + 1, 0, grid.x),
                 clamp_int(p[1] + radius + 1, 0, grid.y),
                 clamp_int(p[2] + radius + 1, 0, grid.z));
}

// The tile AABB covering a Gaussian's voxel AABB.  Kept separate from
// ``voxelRect`` on purpose: mixing the two coordinate spaces is exactly the bug
// that made the first version of this kernel disagree with localagg.
__device__ __forceinline__ void tileRect(const int* p, int radius, int3 grid,
                                         int3& tile_lo, int3& tile_hi) {
  int3 lo, hi;
  voxelRect(p, radius, grid, lo, hi);
  tile_lo = make_int3(lo.x / TILE_EDGE, lo.y / TILE_EDGE, lo.z / TILE_EDGE);
  tile_hi = make_int3((hi.x + TILE_EDGE - 1) / TILE_EDGE,
                      (hi.y + TILE_EDGE - 1) / TILE_EDGE,
                      (hi.z + TILE_EDGE - 1) / TILE_EDGE);
}

}  // namespace

int BlockG2VTileEdge() { return TILE_EDGE; }

namespace {

__global__ void CountTilesKernel(const int* __restrict__ means_int,
                                 const int* __restrict__ radii, int gaussian_count,
                                 int3 grid, int3 tile_grid,
                                 unsigned int* __restrict__ counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= gaussian_count) return;
  int3 lo, hi;
  tileRect(means_int + 3 * index, radii[index], grid, lo, hi);
  if (hi.x <= lo.x || hi.y <= lo.y || hi.z <= lo.z) {
    counts[index] = 0u;
    return;
  }
  counts[index] = static_cast<unsigned int>((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z));
}

__global__ void EmitPairsKernel(const int* __restrict__ means_int,
                                const int* __restrict__ radii, int gaussian_count,
                                int3 grid, int3 tile_grid,
                                const unsigned int* __restrict__ offsets,
                                int* __restrict__ tile_keys,
                                int* __restrict__ gaussian_ids) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= gaussian_count) return;
  int3 lo, hi;
  tileRect(means_int + 3 * index, radii[index], grid, lo, hi);
  int cursor = static_cast<int>(offsets[index]);
  for (int bx = lo.x; bx < hi.x; ++bx) {
    for (int by = lo.y; by < hi.y; ++by) {
      for (int bz = lo.z; bz < hi.z; ++bz) {
        tile_keys[cursor] = (bx * tile_grid.y + by) * tile_grid.z + bz;
        gaussian_ids[cursor] = index;
        ++cursor;
      }
    }
  }
}

__global__ void MarkTileRangesKernel(const int* __restrict__ sorted_keys,
                                     int item_count, int total_tiles,
                                     int* __restrict__ tile_starts,
                                     int* __restrict__ tile_ends) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= item_count) return;
  const int key = sorted_keys[index];
  if (key < 0 || key >= total_tiles) return;
  if (index == 0 || sorted_keys[index - 1] != key) tile_starts[key] = index;
  if (index == item_count - 1 || sorted_keys[index + 1] != key) tile_ends[key] = index + 1;
}

// One CTA per voxel tile.  Threads stream the tile's Gaussian union through
// shared memory in chunks; each of the 64 threads renders one voxel.
__global__ void RenderTileKernel(const float* __restrict__ pts,
                                 const int* __restrict__ points_int,
                                 int point_count, int3 grid, int3 tile_grid,
                                 const int* __restrict__ tile_starts,
                                 const int* __restrict__ tile_ends,
                                 const int* __restrict__ tile_list,
                                 const float* __restrict__ means3D,
                                 const int* __restrict__ means3D_int,
                                 const int* __restrict__ radii,
                                 const float* __restrict__ cov3D,
                                 const float* __restrict__ opacity,
                                 const float* __restrict__ semantic,
                                 int total_tiles, float* __restrict__ out) {
  const int tile = blockIdx.x;
  if (tile >= total_tiles) return;
  const int begin = tile_starts[tile];
  const int tid = threadIdx.x;

  __shared__ float sh_mean[LOAD_CHUNK * 3];
  __shared__ float sh_cov[LOAD_CHUNK * 6];
  __shared__ float sh_op[LOAD_CHUNK];
  __shared__ float sh_sem[LOAD_CHUNK * BLOCK_G2V_CHANNELS];
  // The tile owns the UNION of its voxels' Gaussian lists, so each voxel must
  // still reapply the per-Gaussian voxel AABB test that the naive operator gets
  // for free from its per-voxel binning.
  __shared__ int sh_lo[LOAD_CHUNK * 3];
  __shared__ int sh_hi[LOAD_CHUNK * 3];

  // Which voxel of the tile does this thread render?
  const int bz = tile % tile_grid.z;
  const int rest = tile / tile_grid.z;
  const int by = rest % tile_grid.y;
  const int bx = rest / tile_grid.y;
  const int vz = tid % TILE_EDGE;
  const int vy = (tid / TILE_EDGE) % TILE_EDGE;
  const int vx = tid / (TILE_EDGE * TILE_EDGE);
  const int gx = bx * TILE_EDGE + vx;
  const int gy = by * TILE_EDGE + vy;
  const int gz = bz * TILE_EDGE + vz;
  const bool active = (gx < grid.x) && (gy < grid.y) && (gz < grid.z);
  const int point_index = (gx * grid.y + gy) * grid.z + gz;

  float accumulator[BLOCK_G2V_CHANNELS];
  for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) accumulator[c] = 0.0f;

  if (begin >= 0) {
    const int end = tile_ends[tile];
    const float px = active ? pts[3 * point_index + 0] : 0.0f;
    const float py = active ? pts[3 * point_index + 1] : 0.0f;
    const float pz = active ? pts[3 * point_index + 2] : 0.0f;

    for (int base = begin; base < end; base += LOAD_CHUNK) {
      const int chunk = min(LOAD_CHUNK, end - base);
      // Cooperative load of this chunk of the tile's union list.
      for (int i = tid; i < chunk; i += THREADS) {
        const int gaussian = tile_list[base + i];
        sh_mean[i * 3 + 0] = means3D[gaussian * 3 + 0];
        sh_mean[i * 3 + 1] = means3D[gaussian * 3 + 1];
        sh_mean[i * 3 + 2] = means3D[gaussian * 3 + 2];
        for (int k = 0; k < 6; ++k) sh_cov[i * 6 + k] = cov3D[gaussian * 6 + k];
        sh_op[i] = opacity[gaussian];
        for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) {
          sh_sem[i * BLOCK_G2V_CHANNELS + c] = semantic[gaussian * BLOCK_G2V_CHANNELS + c];
        }
        int3 lo, hi;
        voxelRect(means3D_int + gaussian * 3, radii[gaussian], grid, lo, hi);
        sh_lo[i * 3 + 0] = lo.x; sh_lo[i * 3 + 1] = lo.y; sh_lo[i * 3 + 2] = lo.z;
        sh_hi[i * 3 + 0] = hi.x; sh_hi[i * 3 + 1] = hi.y; sh_hi[i * 3 + 2] = hi.z;
      }
      __syncthreads();
      if (active) {
        for (int i = 0; i < chunk; ++i) {
          // Reapply the voxel AABB test: the shared list is a tile union, not
          // this voxel's list.
          if (gx < sh_lo[i * 3 + 0] || gx >= sh_hi[i * 3 + 0] ||
              gy < sh_lo[i * 3 + 1] || gy >= sh_hi[i * 3 + 1] ||
              gz < sh_lo[i * 3 + 2] || gz >= sh_hi[i * 3 + 2]) {
            continue;
          }
          const float dx = sh_mean[i * 3 + 0] - px;
          const float dy = sh_mean[i * 3 + 1] - py;
          const float dz = sh_mean[i * 3 + 2] - pz;
          float power = sh_cov[i * 6 + 0] * dx * dx + sh_cov[i * 6 + 1] * dy * dy +
                        sh_cov[i * 6 + 2] * dz * dz;
          power = -0.5f * power -
                  (sh_cov[i * 6 + 3] * dx * dy + sh_cov[i * 6 + 4] * dy * dz +
                   sh_cov[i * 6 + 5] * dx * dz);
          const float weight = sh_op[i] * __expf(power);
          for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) {
            accumulator[c] += sh_sem[i * BLOCK_G2V_CHANNELS + c] * weight;
          }
        }
      }
      __syncthreads();
    }
  }

  if (active) {
    for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) {
      out[point_index * BLOCK_G2V_CHANNELS + c] = accumulator[c];
    }
  }
}

}  // namespace

// Backward: one thread per (gaussian, tile) pair from the forward binning.
//
// Every thread accumulates over the tile's 64 voxels locally and issues ONE
// atomicAdd per parameter, instead of one per (gaussian, voxel) pair.  With
// ~22 voxels per tile that cuts gradient atomics by the same factor the
// forward cuts loads -- this is the "organise the backward by Gaussian" idea.
__global__ void BlockBackwardKernel(
    const float* __restrict__ grad_out, const float* __restrict__ pts,
    int point_count, int3 grid, int3 tile_grid,
    const int* __restrict__ sorted_keys, const int* __restrict__ sorted_ids,
    int total_pairs, const float* __restrict__ means3D,
    const int* __restrict__ means3D_int, const int* __restrict__ radii,
    const float* __restrict__ cov3D, const float* __restrict__ opacity,
    const float* __restrict__ semantic, float* __restrict__ grad_means,
    float* __restrict__ grad_opacity, float* __restrict__ grad_semantic,
    float* __restrict__ grad_cov) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= total_pairs) return;
  const int gaussian = sorted_ids[pair];
  const int tile = sorted_keys[pair];
  if (tile < 0 || tile >= tile_grid.x * tile_grid.y * tile_grid.z) return;

  const int bz = tile % tile_grid.z;
  const int rest = tile / tile_grid.z;
  const int by = rest % tile_grid.y;
  const int bx = rest / tile_grid.y;

  int3 lo, hi;
  voxelRect(means3D_int + gaussian * 3, radii[gaussian], grid, lo, hi);

  const float mx = means3D[gaussian * 3 + 0];
  const float my = means3D[gaussian * 3 + 1];
  const float mz = means3D[gaussian * 3 + 2];
  const float c0 = cov3D[gaussian * 6 + 0], c1 = cov3D[gaussian * 6 + 1],
              c2 = cov3D[gaussian * 6 + 2], c3 = cov3D[gaussian * 6 + 3],
              c4 = cov3D[gaussian * 6 + 4], c5 = cov3D[gaussian * 6 + 5];
  const float op = opacity[gaussian];

  float acc_mean[3] = {0.f, 0.f, 0.f};
  float acc_cov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float acc_op = 0.f;
  float acc_sem[BLOCK_G2V_CHANNELS];
  for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) acc_sem[c] = 0.f;

  for (int vz = 0; vz < TILE_EDGE; ++vz) {
    const int gz = bz * TILE_EDGE + vz;
    if (gz < lo.z || gz >= hi.z || gz >= grid.z) continue;
    for (int vy = 0; vy < TILE_EDGE; ++vy) {
      const int gy = by * TILE_EDGE + vy;
      if (gy < lo.y || gy >= hi.y || gy >= grid.y) continue;
      for (int vx = 0; vx < TILE_EDGE; ++vx) {
        const int gx = bx * TILE_EDGE + vx;
        if (gx < lo.x || gx >= hi.x || gx >= grid.x) continue;
        const int point_index = (gx * grid.y + gy) * grid.z + gz;
        const float dx = mx - pts[3 * point_index + 0];
        const float dy = my - pts[3 * point_index + 1];
        const float dz = mz - pts[3 * point_index + 2];
        float power = c0 * dx * dx + c1 * dy * dy + c2 * dz * dz;
        power = -0.5f * power - (c3 * dx * dy + c4 * dy * dz + c5 * dx * dz);
        const float kernel = __expf(power);
        const float weight = op * kernel;

        const float* grad_row = grad_out + point_index * BLOCK_G2V_CHANNELS;
        const float* sem_row = semantic + gaussian * BLOCK_G2V_CHANNELS;
        float grad_weight = 0.f;
        for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) {
          acc_sem[c] += grad_row[c] * weight;
          grad_weight += grad_row[c] * sem_row[c];
        }
        // d w / d opacity = kernel (avoids dividing by opacity, which may be 0)
        acc_op += grad_weight * kernel;
        const float grad_power = grad_weight * weight;
        acc_mean[0] += grad_power * (-c0 * dx - c3 * dy - c5 * dz);
        acc_mean[1] += grad_power * (-c1 * dy - c3 * dx - c4 * dz);
        acc_mean[2] += grad_power * (-c2 * dz - c4 * dy - c5 * dx);
        acc_cov[0] += grad_power * (-0.5f * dx * dx);
        acc_cov[1] += grad_power * (-0.5f * dy * dy);
        acc_cov[2] += grad_power * (-0.5f * dz * dz);
        acc_cov[3] += grad_power * (-dx * dy);
        acc_cov[4] += grad_power * (-dy * dz);
        acc_cov[5] += grad_power * (-dx * dz);
      }
    }
  }

  atomicAdd(grad_opacity + gaussian, acc_op);
  for (int c = 0; c < BLOCK_G2V_CHANNELS; ++c) {
    atomicAdd(grad_semantic + gaussian * BLOCK_G2V_CHANNELS + c, acc_sem[c]);
  }
  for (int k = 0; k < 3; ++k) atomicAdd(grad_means + gaussian * 3 + k, acc_mean[k]);
  for (int k = 0; k < 6; ++k) atomicAdd(grad_cov + gaussian * 6 + k, acc_cov[k]);
}

std::vector<torch::Tensor> BlockG2VForward(const torch::Tensor& pts, const torch::Tensor& points_int,
                              const torch::Tensor& means3D, const torch::Tensor& means3D_int,
                              const torch::Tensor& opacities, const torch::Tensor& semantics,
                              const torch::Tensor& radii, const torch::Tensor& cov3D,
                              int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  TORCH_CHECK(pts.is_cuda() && means3D.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(pts.is_contiguous() && means3D.is_contiguous() &&
              opacities.is_contiguous() && semantics.is_contiguous() &&
              radii.is_contiguous() && cov3D.is_contiguous() &&
              points_int.is_contiguous() && means3D_int.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(semantics.size(1) == BLOCK_G2V_CHANNELS,
              "semantics must have BLOCK_G2V_CHANNELS columns");

  const int point_count = static_cast<int>(pts.size(0));
  const int gaussian_count = static_cast<int>(means3D.size(0));
  const int3 grid = make_int3(static_cast<int>(grid_x), static_cast<int>(grid_y),
                              static_cast<int>(grid_z));
  const int3 tile_grid = make_int3((grid.x + TILE_EDGE - 1) / TILE_EDGE,
                                   (grid.y + TILE_EDGE - 1) / TILE_EDGE,
                                   (grid.z + TILE_EDGE - 1) / TILE_EDGE);
  const int total_tiles = tile_grid.x * tile_grid.y * tile_grid.z;

  auto stream = at::cuda::getDefaultCUDAStream();
  auto float_options = pts.options().dtype(torch::kFloat32);
  auto int_options = pts.options().dtype(torch::kInt32);

  // No Gaussians: the field is identically zero.  Handle it before any kernel
  // so the scan never indexes an empty offsets tensor.
  if (gaussian_count == 0) {
    return {torch::zeros({point_count, BLOCK_G2V_CHANNELS}, float_options),
            torch::empty({0}, int_options), torch::empty({0}, int_options)};
  }

  // 1. How many tiles does each Gaussian touch?
  auto counts = torch::zeros({gaussian_count}, int_options);
  {
    const int threads = 256;
    CountTilesKernel<<<(gaussian_count + threads - 1) / threads, threads, 0, stream>>>(
        means3D_int.data_ptr<int>(), radii.data_ptr<int>(), gaussian_count, grid,
        tile_grid, reinterpret_cast<unsigned int*>(counts.data_ptr<int>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 2. Exclusive prefix sum -> per-Gaussian offsets, and the total pair count.
  auto offsets = torch::zeros({gaussian_count}, int_options);
  int total_pairs = 0;
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts.data_ptr<int>(),
                                  offsets.data_ptr<int>(), gaussian_count, stream);
    auto temp = torch::empty({static_cast<long long>(temp_bytes)},
                             pts.options().dtype(torch::kUInt8));
    cub::DeviceScan::ExclusiveSum(temp.data_ptr(), temp_bytes, counts.data_ptr<int>(),
                                  offsets.data_ptr<int>(), gaussian_count, stream);
    // Total pairs = offsets.back() + counts.back().
    auto counts_cpu = counts.to(torch::kCPU);
    auto offsets_cpu = offsets.to(torch::kCPU);
    total_pairs = offsets_cpu[gaussian_count - 1].item<int>() +
                  counts_cpu[gaussian_count - 1].item<int>();
  }

  auto out = torch::zeros({point_count, BLOCK_G2V_CHANNELS}, float_options);
  if (total_pairs == 0) {
    return {out, torch::empty({0}, int_options), torch::empty({0}, int_options)};
  }

  // 3. Emit (tile, gaussian) pairs and sort them by tile.
  auto tile_keys = torch::empty({total_pairs}, int_options);
  auto gaussian_ids = torch::empty({total_pairs}, int_options);
  {
    const int threads = 256;
    EmitPairsKernel<<<(gaussian_count + threads - 1) / threads, threads, 0, stream>>>(
        means3D_int.data_ptr<int>(), radii.data_ptr<int>(), gaussian_count, grid,
        tile_grid, reinterpret_cast<const unsigned int*>(offsets.data_ptr<int>()),
        tile_keys.data_ptr<int>(), gaussian_ids.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto sorted_keys = torch::empty_like(tile_keys);
  auto sorted_ids = torch::empty_like(gaussian_ids);
  {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, tile_keys.data_ptr<int>(),
                                    sorted_keys.data_ptr<int>(), gaussian_ids.data_ptr<int>(),
                                    sorted_ids.data_ptr<int>(), total_pairs, 0,
                                    sizeof(int) * 8, stream);
    auto temp = torch::empty({static_cast<long long>(temp_bytes)},
                             pts.options().dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortPairs(temp.data_ptr(), temp_bytes, tile_keys.data_ptr<int>(),
                                    sorted_keys.data_ptr<int>(), gaussian_ids.data_ptr<int>(),
                                    sorted_ids.data_ptr<int>(), total_pairs, 0,
                                    sizeof(int) * 8, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 4. Tile ranges.
  auto tile_starts = torch::full({total_tiles}, -1, int_options);
  auto tile_ends = torch::full({total_tiles}, -1, int_options);
  {
    const int threads = 256;
    MarkTileRangesKernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
        sorted_keys.data_ptr<int>(), total_pairs, total_tiles,
        tile_starts.data_ptr<int>(), tile_ends.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 5. Render, one CTA per tile.
  RenderTileKernel<<<total_tiles, THREADS, 0, stream>>>(
      pts.data_ptr<float>(), points_int.data_ptr<int>(), point_count, grid, tile_grid,
      tile_starts.data_ptr<int>(), tile_ends.data_ptr<int>(), sorted_ids.data_ptr<int>(),
      means3D.data_ptr<float>(), means3D_int.data_ptr<int>(), radii.data_ptr<int>(),
      cov3D.data_ptr<float>(), opacities.data_ptr<float>(),
      semantics.data_ptr<float>(), total_tiles, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out, sorted_keys, sorted_ids};
}

std::vector<torch::Tensor> BlockG2VBackward(
    const torch::Tensor& grad_out, const torch::Tensor& pts,
    const torch::Tensor& points_int, const torch::Tensor& means3D,
    const torch::Tensor& means3D_int, const torch::Tensor& opacities,
    const torch::Tensor& semantics, const torch::Tensor& radii,
    const torch::Tensor& cov3D, const torch::Tensor& sorted_keys,
    const torch::Tensor& sorted_ids, int64_t grid_x, int64_t grid_y,
    int64_t grid_z) {
  const int point_count = static_cast<int>(pts.size(0));
  const int gaussian_count = static_cast<int>(means3D.size(0));
  const int3 grid = make_int3(static_cast<int>(grid_x), static_cast<int>(grid_y),
                              static_cast<int>(grid_z));
  const int3 tile_grid = make_int3((grid.x + TILE_EDGE - 1) / TILE_EDGE,
                                   (grid.y + TILE_EDGE - 1) / TILE_EDGE,
                                   (grid.z + TILE_EDGE - 1) / TILE_EDGE);

  auto grad_means = torch::zeros_like(means3D);
  auto grad_opacity = torch::zeros_like(opacities);
  auto grad_semantic = torch::zeros_like(semantics);
  auto grad_cov = torch::zeros_like(cov3D);
  const int total_pairs = static_cast<int>(sorted_ids.numel());
  if (total_pairs == 0) {
    return {grad_means, grad_opacity, grad_semantic, grad_cov};
  }

  auto stream = at::cuda::getDefaultCUDAStream();
  const int threads = 128;
  BlockBackwardKernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
      grad_out.contiguous().data_ptr<float>(), pts.data_ptr<float>(), point_count,
      grid, tile_grid, sorted_keys.data_ptr<int>(), sorted_ids.data_ptr<int>(),
      total_pairs, means3D.data_ptr<float>(), means3D_int.data_ptr<int>(),
      radii.data_ptr<int>(), cov3D.data_ptr<float>(), opacities.data_ptr<float>(),
      semantics.data_ptr<float>(), grad_means.data_ptr<float>(),
      grad_opacity.data_ptr<float>(), grad_semantic.data_ptr<float>(),
      grad_cov.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_means, grad_opacity, grad_semantic, grad_cov};
}
