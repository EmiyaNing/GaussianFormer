// Additive Gaussian evaluation follows localagg_react's AABB and quadratic
// form. This implementation builds coarse CSR bins and re-enumerates the
// exact support in backward, without storing per-voxel Gaussian duplicates.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_scan.cuh>
#include <vector>

using at::Tensor;
constexpr int C = 18;

__device__ __forceinline__ void bounds(const int* center, const int* radius,
    int h, int w, int d, int3& lo, int3& hi) {
    lo = make_int3(min(h, max(0, center[0] - radius[0])),
                   min(w, max(0, center[1] - radius[1])),
                   min(d, max(0, center[2] - radius[2])));
    hi = make_int3(min(h, max(0, center[0] + radius[0] + 1)),
                   min(w, max(0, center[1] + radius[1] + 1)),
                   min(d, max(0, center[2] + radius[2] + 1)));
}

__global__ void query_map_kernel(int n, const int* cells, int* heads,
                                  int* next, int h, int w, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = cells[3*i], y = cells[3*i+1], z = cells[3*i+2];
    next[i] = -1;
    if (x >= 0 && x < h && y >= 0 && y < w && z >= 0 && z < d)
        next[i] = atomicExch(heads + (x*w + y)*d + z, i);
}

// Two passes over coarse AABBs: count, then scatter. No CPU count readback,
// global pair sort, or list of per-voxel Gaussian records is needed.
template<bool Scatter>
__global__ void bins_kernel(int p, const int* centers, const int* radii,
    int h, int w, int d, int tile, int* counts, const int64_t* offsets,
    int* ids, int capacity) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= p) return;
    int3 lo, hi;
    bounds(centers + 3*g, radii + 3*g, h, w, d, lo, hi);
    if (hi.x <= lo.x || hi.y <= lo.y || hi.z <= lo.z) return;
    lo.x /= tile; lo.y /= tile; lo.z /= tile;
    hi.x = (hi.x - 1) / tile + 1;
    hi.y = (hi.y - 1) / tile + 1;
    hi.z = (hi.z - 1) / tile + 1;
    const int tw = (w + tile - 1) / tile, td = (d + tile - 1) / tile;
    for (int x = lo.x; x < hi.x; ++x)
      for (int y = lo.y; y < hi.y; ++y)
        for (int z = lo.z; z < hi.z; ++z) {
            const int bin = (x*tw + y)*td + z;
            const int position = atomicAdd(counts + bin, 1);
            if (Scatter) {
                const int64_t index = offsets[bin] + position;
                if (index < capacity) ids[index] = g;
            }
        }
}

__global__ void render_kernel(int n, int p, const float* points, const int* cells,
    const float* means, const int* centers, const int* radii, const float* opacity,
    const float* semantics, const float* inverse, const int64_t* offsets, const int* ids,
    int capacity, int h, int w, int d, int tile, float* output) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    const int x = cells[3*q], y = cells[3*q+1], z = cells[3*q+2];
    float sum[C] = {0};
    if (x >= 0 && x < h && y >= 0 && y < w && z >= 0 && z < d) {
        const int bin = ((x/tile)*((w+tile-1)/tile) + y/tile)*((d+tile-1)/tile) + z/tile;
        // Capacity is a performance hint, never a truncation threshold. If
        // this bin overflows, enumerate all Gaussians with the same exact
        // support test. Earlier fully stored bins can still use their CSR.
        const bool fallback = offsets[bin+1] > capacity;
        const int begin = fallback ? 0 : offsets[bin];
        const int end = fallback ? p : offsets[bin+1];
        for (int j = begin; j < end; ++j) {
            const int g = fallback ? j : ids[j];
            const int* ctr = centers + 3*g;
            const int* rad = radii + 3*g;
            if (x < ctr[0]-rad[0] || x > ctr[0]+rad[0] ||
                y < ctr[1]-rad[1] || y > ctr[1]+rad[1] ||
                z < ctr[2]-rad[2] || z > ctr[2]+rad[2]) continue;
            const float dx = means[3*g] - points[3*q];
            const float dy = means[3*g+1] - points[3*q+1];
            const float dz = means[3*g+2] - points[3*q+2];
            const float* cov = inverse + 6*g;
            float power = cov[0]*dx*dx + cov[1]*dy*dy + cov[2]*dz*dz;
            power = -.5f * power - (cov[3]*dx*dy + cov[4]*dy*dz + cov[5]*dx*dz);
            const float weight = opacity[g] * expf(power);
            #pragma unroll
            for (int ch = 0; ch < C; ++ch)
                sum[ch] += semantics[g*C+ch] * weight;
        }
    }
    #pragma unroll
    for (int ch = 0; ch < C; ++ch) output[q*C+ch] = sum[ch];
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta /= 2)
        value += __shfl_down_sync(0xffffffff, value, delta);
    return value;
}

// One warp owns one Gaussian. Its lanes process different supported voxels;
// the geometry derivative uses sum_c semantic[c] * output_gradient[c] once.
__global__ void backward_kernel(int p, const float* points, const float* means,
    const int* centers, const int* radii, const float* opacity, const float* semantics,
    const float* inverse, const int* heads, const int* next, const float* grad,
    int h, int w, int d, float* grad_means, float* grad_opacity,
    float* grad_semantics, float* grad_inverse) {
    const int lane = threadIdx.x % 32;
    const int g = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (g >= p) return;
    int3 lo, hi;
    bounds(centers + 3*g, radii + 3*g, h, w, d, lo, hi);
    const int ny = max(hi.y-lo.y, 0), nz = max(hi.z-lo.z, 0);
    const int volume = max(hi.x-lo.x, 0) * ny * nz;
    const float* cov = inverse + 6*g;
    float sem[C], ds[C] = {0};
    #pragma unroll
    for (int ch = 0; ch < C; ++ch) sem[ch] = semantics[g*C+ch];
    float dm[3] = {0}, dc[6] = {0}, dop = 0;
    for (int index = lane; index < volume; index += 32) {
        const int x = lo.x + index / (ny*nz);
        const int y = lo.y + (index / nz) % ny;
        const int z = lo.z + index % nz;
        for (int q = heads[(x*w+y)*d+z]; q >= 0; q = next[q]) {
            const float dx = means[3*g] - points[3*q];
            const float dy = means[3*g+1] - points[3*q+1];
            const float dz = means[3*g+2] - points[3*q+2];
            float power = cov[0]*dx*dx + cov[1]*dy*dy + cov[2]*dz*dz;
            power = -.5f*power - (cov[3]*dx*dy + cov[4]*dy*dz + cov[5]*dx*dz);
            const float weight = expf(power);
            float dot = 0;
            #pragma unroll
            for (int ch = 0; ch < C; ++ch) {
                const float value = grad[q*C+ch];
                dot += sem[ch] * value;
                ds[ch] += opacity[g] * (weight * value);
            }
            dop += weight * dot;
            const float coefficient = opacity[g] * weight * dot;
            dm[0] -= coefficient * (cov[0]*dx + cov[3]*dy + cov[5]*dz);
            dm[1] -= coefficient * (cov[1]*dy + cov[3]*dx + cov[4]*dz);
            dm[2] -= coefficient * (cov[2]*dz + cov[4]*dy + cov[5]*dx);
            dc[0] -= .5f * coefficient * dx * dx;
            dc[1] -= .5f * coefficient * dy * dy;
            dc[2] -= .5f * coefficient * dz * dz;
            dc[3] -= coefficient * dx * dy;
            dc[4] -= coefficient * dy * dz;
            dc[5] -= coefficient * dx * dz;
        }
    }
    #pragma unroll
    for (int ch = 0; ch < C; ++ch) {
        const float value = warp_sum(ds[ch]);
        if (lane == 0) grad_semantics[g*C+ch] = value;
    }
    #pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        const float value = warp_sum(dm[ch]);
        if (lane == 0) grad_means[g*3+ch] = value;
    }
    #pragma unroll
    for (int ch = 0; ch < 6; ++ch) {
        const float value = warp_sum(dc[ch]);
        if (lane == 0) grad_inverse[g*6+ch] = value;
    }
    dop = warp_sum(dop);
    if (lane == 0) grad_opacity[g] = dop;
}

std::vector<Tensor> query_map_cuda(const Tensor& cells, int h, int w, int d) {
    auto heads = at::full({h*w*d}, -1, cells.options());
    auto next = at::empty({cells.size(0)}, cells.options());
    if (cells.size(0)) query_map_kernel<<<(cells.size(0)+255)/256, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(cells.size(0), cells.data_ptr<int>(),
            heads.data_ptr<int>(), next.data_ptr<int>(), h, w, d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {heads, next};
}

std::vector<Tensor> tile_bins_cuda(const Tensor& centers, const Tensor& radii,
    int h, int w, int d, int tile, int capacity) {
    const int count = ((h+tile-1)/tile)*((w+tile-1)/tile)*((d+tile-1)/tile);
    auto counts = at::zeros({count}, centers.options());
    auto offsets = at::zeros({count+1}, centers.options().dtype(at::kLong));
    auto ids = at::empty({capacity}, centers.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int p = centers.size(0);
    if (p) bins_kernel<false><<<(p+255)/256, 256, 0, stream>>>(p,
        centers.data_ptr<int>(), radii.data_ptr<int>(), h, w, d, tile,
        counts.data_ptr<int>(), nullptr, nullptr, 0);
    // Prefix sums may exceed int32 even though each per-bin count fits. Keep
    // them wide so exceptionally large supports still take the exact fallback.
    auto wide_counts = counts.to(at::kLong);
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, wide_counts.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>()+1, count, stream));
    auto scratch = at::empty({static_cast<int64_t>(bytes)}, centers.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(scratch.data_ptr(), bytes,
        wide_counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>()+1, count, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr(), 0, count*sizeof(int), stream));
    if (p) bins_kernel<true><<<(p+255)/256, 256, 0, stream>>>(p,
        centers.data_ptr<int>(), radii.data_ptr<int>(), h, w, d, tile,
        counts.data_ptr<int>(), offsets.data_ptr<int64_t>(), ids.data_ptr<int>(), capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {offsets, ids};
}

Tensor render_cuda(const Tensor& points, const Tensor& cells, const Tensor& means,
    const Tensor& centers, const Tensor& radii, const Tensor& opacity,
    const Tensor& semantics, const Tensor& inverse, const Tensor& offsets,
    const Tensor& ids, int h, int w, int d, int tile) {
    auto output = at::empty({points.size(0), C}, points.options());
    if (points.size(0)) render_kernel<<<(points.size(0)+127)/128, 128, 0,
        at::cuda::getCurrentCUDAStream()>>>(points.size(0), means.size(0),
        points.data_ptr<float>(), cells.data_ptr<int>(), means.data_ptr<float>(),
        centers.data_ptr<int>(), radii.data_ptr<int>(), opacity.data_ptr<float>(),
        semantics.data_ptr<float>(), inverse.data_ptr<float>(), offsets.data_ptr<int64_t>(),
        ids.data_ptr<int>(), ids.numel(), h, w, d, tile, output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<Tensor> backward_cuda(const Tensor& points, const Tensor& means,
    const Tensor& centers, const Tensor& radii, const Tensor& opacity,
    const Tensor& semantics, const Tensor& inverse, const Tensor& heads,
    const Tensor& next, const Tensor& grad, int h, int w, int d) {
    auto dm = at::empty_like(means), dop = at::empty_like(opacity);
    auto ds = at::empty_like(semantics), dc = at::empty_like(inverse);
    if (means.size(0)) backward_kernel<<<(means.size(0)+3)/4, 128, 0,
        at::cuda::getCurrentCUDAStream()>>>(means.size(0), points.data_ptr<float>(),
        means.data_ptr<float>(), centers.data_ptr<int>(), radii.data_ptr<int>(),
        opacity.data_ptr<float>(), semantics.data_ptr<float>(), inverse.data_ptr<float>(),
        heads.data_ptr<int>(), next.data_ptr<int>(), grad.data_ptr<float>(), h, w, d,
        dm.data_ptr<float>(), dop.data_ptr<float>(), ds.data_ptr<float>(), dc.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dm, dop, ds, dc};
}
