// One copy from strided per-level [B,V,C,H,W] into [B,V,sum(HW),C].
// The transpose has no saved feature values, and supports channels-last FPNs.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kMaxLevels = 8;
struct Pyramid {
  const float* input[kMaxLevels];
  float* output[kMaxLevels];
  int64_t stride[kMaxLevels][5];
  int height[kMaxLevels], width[kMaxLevels], start[kMaxLevels];
  int levels, cameras, channels, pixels;
};

__global__ void pack_kernel(Pyramid p, float* packed, const float* gradient,
                             int64_t count, bool backward) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += int64_t(blockDim.x) * gridDim.x) {
    const int c = index % p.channels;
    const int pixel = (index / p.channels) % p.pixels;
    const int v = (index / (int64_t(p.channels) * p.pixels)) % p.cameras;
    const int b = index / (int64_t(p.channels) * p.pixels * p.cameras);
    int level = 0;
    while (level + 1 < p.levels && pixel >= p.start[level + 1]) ++level;
    const int local = pixel - p.start[level];
    const auto* s = p.stride[level];
    const int64_t offset = b*s[0] + v*s[1] + c*s[2] +
                          (local/p.width[level])*s[3] + (local%p.width[level])*s[4];
    if (backward) p.output[level][offset] = gradient[index];
    else packed[index] = p.input[level][offset];
  }
}
}  // namespace

at::Tensor pack_pyramid_forward(const std::vector<at::Tensor>& levels) {
  TORCH_CHECK(!levels.empty() && levels.size() <= kMaxLevels, "Expected 1..8 pyramid levels");
  const auto& first = levels[0];
  TORCH_CHECK(first.is_cuda() && first.scalar_type()==at::kFloat && first.dim()==5,
              "Pyramid levels must be CUDA float32 [B,V,C,H,W]");
  const at::cuda::OptionalCUDAGuard guard(device_of(first));
  Pyramid p{};
  p.levels=levels.size(); p.cameras=first.size(1); p.channels=first.size(2);
  for (int l=0; l<p.levels; ++l) {
    const auto& f=levels[l];
    TORCH_CHECK(f.device()==first.device() && f.scalar_type()==at::kFloat && f.dim()==5 &&
                f.size(0)==first.size(0) && f.size(1)==p.cameras && f.size(2)==p.channels &&
                f.is_non_overlapping_and_dense(), "Pyramid levels must be dense with matching B,V,C");
    p.input[l]=f.data_ptr<float>(); p.height[l]=f.size(3); p.width[l]=f.size(4);
    p.start[l]=p.pixels; p.pixels+=p.height[l]*p.width[l];
    for(int d=0;d<5;++d) p.stride[l][d]=f.stride(d);
  }
  auto output=at::empty({first.size(0),p.cameras,p.pixels,p.channels},first.options());
  if(output.numel()) {
    const int blocks=std::min<int64_t>((output.numel()+255)/256,4096);
    pack_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
        p,output.data_ptr<float>(),nullptr,output.numel(),false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::vector<at::Tensor> pack_pyramid_backward(
    const at::Tensor& gradient, const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<std::vector<int64_t>>& strides) {
  TORCH_CHECK(gradient.is_cuda() && gradient.scalar_type()==at::kFloat && gradient.is_contiguous(),
              "Packed gradient must be contiguous CUDA float32");
  TORCH_CHECK(!sizes.empty() && sizes.size()<=kMaxLevels && sizes.size()==strides.size(),
              "Invalid pyramid gradient metadata");
  const at::cuda::OptionalCUDAGuard guard(device_of(gradient));
  Pyramid p{};
  p.levels=sizes.size(); p.cameras=gradient.size(1); p.channels=gradient.size(3);
  std::vector<at::Tensor> result;
  for(int l=0;l<p.levels;++l) {
    TORCH_CHECK(sizes[l].size()==5 && strides[l].size()==5,"Expected five pyramid dimensions");
    result.push_back(at::empty_strided(sizes[l],strides[l],gradient.options()));
    p.output[l]=result.back().data_ptr<float>();
    p.height[l]=sizes[l][3]; p.width[l]=sizes[l][4];
    p.start[l]=p.pixels; p.pixels+=p.height[l]*p.width[l];
    for(int d=0;d<5;++d) p.stride[l][d]=strides[l][d];
  }
  TORCH_CHECK(gradient.size(2)==p.pixels,"Packed gradient size mismatch");
  if(gradient.numel()) {
    const int blocks=std::min<int64_t>((gradient.numel()+255)/256,4096);
    pack_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
        p,nullptr,gradient.data_ptr<float>(),gradient.numel(),true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return result;
}
