#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

template<typename T, bool Backward>
__global__ void image_affine_kernel(const T* x, const float* scale, const float* bias,
                                    T* y, int64_t count, int channels, int spatial, bool nhwc) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int c = nhwc ? i % channels : (i / spatial) % channels;
  // Match BN eval's float arithmetic, with one rounding on BF16 output.
  float value = (float)x[i] * scale[c];
  if (!Backward) value += bias[c];
  y[i] = (T)value;
}

at::Tensor image_affine(const at::Tensor& x, const at::Tensor& scale,
                        const at::Tensor& bias, bool nhwc, bool backward) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4 &&
              x.is_contiguous(nhwc ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous),
              "Image affine expects dense CUDA NCHW or channels_last");
  TORCH_CHECK(scale.device() == x.device() && scale.scalar_type() == at::kFloat &&
              scale.is_contiguous() && scale.numel() == x.size(1), "Invalid image affine scale");
  if (!backward) {
    TORCH_CHECK(bias.device() == x.device() && bias.scalar_type() == at::kFloat &&
                bias.is_contiguous() && bias.numel() == x.size(1), "Invalid image affine bias");
  }
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto y = at::empty_like(x);
  if (!x.numel()) return y;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x.scalar_type(), "image_affine", [&] {
    if (backward) {
      image_affine_kernel<scalar_t, true><<<(x.numel()+255)/256, 256, 0,
          at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<scalar_t>(), scale.data_ptr<float>(), nullptr,
          y.data_ptr<scalar_t>(), x.numel(), x.size(1), x.size(2)*x.size(3), nhwc);
    } else {
      image_affine_kernel<scalar_t, false><<<(x.numel()+255)/256, 256, 0,
          at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<scalar_t>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
          y.data_ptr<scalar_t>(), x.numel(), x.size(1), x.size(2)*x.size(3), nhwc);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor image_affine_forward(const at::Tensor& x, const at::Tensor& scale,
                                const at::Tensor& bias, bool nhwc) {
  return image_affine(x, scale, bias, nhwc, false);
}
at::Tensor image_affine_backward(const at::Tensor& grad, const at::Tensor& scale, bool nhwc) {
  return image_affine(grad, scale, at::Tensor(), nhwc, true);
}
