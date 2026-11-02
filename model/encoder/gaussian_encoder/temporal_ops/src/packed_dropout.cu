// Match torch.rand(float32) > probability, including its generator advance.
// Warp ballots write one bit per draw; there is no full-size float/bool array.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <curand_kernel.h>
#include <functional>
#include <mutex>

namespace {
__global__ void mask_kernel(int64_t count, float probability, at::PhiloxCudaState philox,
                             uint32_t* bits, int64_t bit_offset, int64_t total_count,
                             bool merge_words) {
  const auto seeds=at::cuda::philox::unpack(philox);
  const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t threads=int64_t(blockDim.x)*gridDim.x;
  const int64_t rounded=((count-1)/(threads*4)+1)*threads*4;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),index,std::get<1>(seeds),&state);
  for(int64_t begin=index;begin<rounded;begin+=threads*4) {
    const float4 values=curand_uniform4(&state);
    #pragma unroll
    for(int component=0;component<4;++component) {
      const int64_t position=begin+threads*component;
      const float raw=(&values.x)[component];
      const float uniform=raw==1.f?0.f:raw;
      const uint32_t ballot=__ballot_sync(0xffffffff,position<count && uniform>probability);
      if((threadIdx.x&31)==0 && position<count) {
        const int64_t global=bit_offset+position;
        const int shift=global&31;
        if(!merge_words) bits[global>>5]=ballot;
        else {
          atomicOr(bits+(global>>5),ballot<<shift);
          if(shift && (global>>5)+1<(total_count+31)/32)
            atomicOr(bits+(global>>5)+1,ballot>>(32-shift));
        }
      }
    }
    __syncthreads();
  }
}
}  // namespace

at::Tensor packed_dropout(const at::Tensor& reference, int64_t count, double probability) {
  TORCH_CHECK(reference.is_cuda() && reference.scalar_type()==at::kFloat,
              "Packed dropout expects a CUDA float32 reference");
  TORCH_CHECK(count>=0 && count<=INT32_MAX && probability>=0. && probability<=1.,
              "Invalid dropout draw count or probability");
  const at::cuda::OptionalCUDAGuard guard(device_of(reference));
  if(!count || probability==0.) return at::empty({0},reference.options().dtype(at::kInt));
  // TensorIterator's signed 32-bit limit applies to BYTE offsets, not just
  // element counts. Large float rand outputs split into contiguous halves.
  constexpr int64_t max_float_elements=(int64_t(INT32_MAX)-1)/sizeof(float)+1;
  const bool split=count>max_float_elements;
  auto bits=split?at::zeros({(count+31)/32},reference.options().dtype(at::kInt)):
                  at::empty({(count+31)/32},reference.options().dtype(at::kInt));
  // This is TensorIterator float-uniform's execution policy, not a new RNG.
  const auto* props=at::cuda::getCurrentDeviceProperties();
  const int max_blocks=props->multiProcessorCount*(props->maxThreadsPerMultiProcessor/256);
  auto generator=at::cuda::detail::getDefaultCUDAGenerator(reference.get_device());
  auto* implementation=at::check_generator<at::CUDAGeneratorImpl>(generator);
  auto reserve=[&](int64_t elements) {
    const int blocks=std::min<int64_t>((elements+255)/256,max_blocks);
    const uint64_t increment=((elements-1)/(int64_t(256)*blocks*4)+1)*4;
    std::lock_guard<std::mutex> lock(implementation->mutex_);
    return implementation->philox_cuda_state(increment);
  };
  // PyTorch reserves the original iterator's RNG range before inspecting
  // can_use_32bit_indexing; that first range is unused for split iterators.
  auto first_state=reserve(count);
  std::function<void(int64_t,int64_t)> launch;
  launch=[&](int64_t elements,int64_t offset) {
    if(elements>max_float_elements) {
      const int64_t left=elements/2;
      launch(left,offset);
      launch(elements-left,offset+left);
      return;
    }
    const int blocks=std::min<int64_t>((elements+255)/256,max_blocks);
    auto state=split?reserve(elements):first_state;
    mask_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
        elements,float(probability),state,reinterpret_cast<uint32_t*>(bits.data_ptr<int>()),
        offset,count,split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  launch(count,0);
  return bits;
}
