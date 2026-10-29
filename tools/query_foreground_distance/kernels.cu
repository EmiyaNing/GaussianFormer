#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Exact exhaustive minima, one warp per Gaussian. Foreground is a union of
// equal axis-aligned voxel cubes, not an interpolated distance-field proxy.
__global__ void distance_kernel(int ng,int nt,float cell,const float* means,
    const float* gt,float* result) {
    const int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(g>=ng)return;
    const float x=means[3*g],y=means[3*g+1],z=means[3*g+2],half=.5f*cell;
    float nearest_center=CUDART_INF_F,nearest_volume=CUDART_INF_F;
    for(int t=lane;t<nt;t+=32) {
        const float dx=x-gt[3*t],dy=y-gt[3*t+1],dz=z-gt[3*t+2];
        const float center=dx*dx+dy*dy+dz*dz;
        const float bx=fmaxf(fabsf(dx)-half,0.f),by=fmaxf(fabsf(dy)-half,0.f),bz=fmaxf(fabsf(dz)-half,0.f);
        const float volume=bx*bx+by*by+bz*bz;
        nearest_center=fminf(nearest_center,center);
        nearest_volume=fminf(nearest_volume,volume);
    }
    for(int offset=16;offset>0;offset/=2) {
        nearest_center=fminf(nearest_center,__shfl_down_sync(0xffffffff,nearest_center,offset));
        nearest_volume=fminf(nearest_volume,__shfl_down_sync(0xffffffff,nearest_volume,offset));
    }
    if(lane==0) {
        result[2*g]=sqrtf(nearest_center)/cell;
        result[2*g+1]=sqrtf(nearest_volume)/cell;
    }
}

torch::Tensor distances(torch::Tensor means,torch::Tensor gt,double cell) {
    const c10::cuda::CUDAGuard guard(means.device());
    TORCH_CHECK(means.is_cuda()&&gt.is_cuda()&&means.device()==gt.device()&&means.is_contiguous()&&gt.is_contiguous(),"Contiguous same-device CUDA inputs required");
    TORCH_CHECK(means.scalar_type()==torch::kFloat32&&gt.scalar_type()==torch::kFloat32,"FP32 inputs required");
    TORCH_CHECK(means.dim()==2&&means.size(1)==3&&gt.dim()==2&&gt.size(1)==3&&gt.size(0)>0&&cell>0,"Invalid shape or grid size");
    auto result=torch::empty({means.size(0),2},means.options());
    distance_kernel<<<(means.size(0)*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(means.size(0),gt.size(0),cell,
        means.data_ptr<float>(),gt.data_ptr<float>(),result.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("distances",&distances);}
