// Frozen-output diagnostic; native integer centers/radii are supplied by PyTorch.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void coverage_kernel(int ng,int H,int W,int D,const float* means,
    const float* inverse,const float* alpha,const float* semantics,
    const int* centers,const int* radii,const float* xyz,const int* lookup,
    const int* winner,const float* maximum,int* counts,float* mass,float* nearest,
    unsigned long long* owners) {
    const int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(g>=ng || !(alpha[g]>0))return;
    const int* p=centers+g*3;const int r=radii[g];
    const int x0=max(0,p[0]-r),x1=min(H,p[0]+r+1);
    const int y0=max(0,p[1]-r),y1=min(W,p[1]+r+1);
    const int z0=max(0,p[2]-r),z1=min(D,p[2]+r+1),ny=y1-y0,nz=z1-z0;
    const float* inv=inverse+g*6;
    for(int k=lane;k<(x1-x0)*ny*nz;k+=32) {
        const int v=((x0+k/(ny*nz))*W+y0+(k/nz)%ny)*D+z0+k%nz;
        const int t=lookup[v];if(t<0)continue;
        const float dx=means[g*3]-xyz[v*3],dy=means[g*3+1]-xyz[v*3+1],dz=means[g*3+2]-xyz[v*3+2];
        float power=inv[0]*dx*dx+inv[1]*dy*dy+inv[2]*dz*dz;
        power=-.5f*power-(inv[3]*dx*dy+inv[4]*dy*dz+inv[5]*dx*dz);
        const float kernel=expf(power),weight=alpha[g]*kernel;
        if(!(weight>0))continue;
        const float d=sqrtf(fmaxf(-2.f*power,0.f)),e=sqrtf(dx*dx+dy*dy+dz*dz);
        int* n=counts+t*7;float* a=mass+t*11;
        atomicAdd(n,1);
        if(d<=1.f)atomicAdd(n+1,1);
        if(d<=2.f)atomicAdd(n+2,1);
        if(d<=3.f)atomicAdd(n+3,1);
        // All distances are nonnegative; IEEE unsigned ordering is monotone.
        atomicMin(reinterpret_cast<unsigned int*>(nearest+t*2),__float_as_uint(d));
        atomicMin(reinterpret_cast<unsigned int*>(nearest+t*2+1),__float_as_uint(e));
        atomicAdd(a,kernel);atomicAdd(a+1,weight);atomicAdd(a+4,kernel*d);
        const float w=weight*semantics[g*17+winner[t]];
        if(!(w>0))continue;
        atomicAdd(n+4,1);
        const float q=maximum[t]>0.f ? w/maximum[t] : 0.f;
        if(q>=.01f)atomicAdd(n+5,1);
        if(q>=.10f)atomicAdd(n+6,1);
        atomicAdd(a+2,w);atomicAdd(a+3,q*q);
        atomicAdd(a+5,w*d);atomicAdd(a+6,w*e);
        const int band=d<=1.f ? 0 : (d<=2.f ? 1 : (d<=3.f ? 2 : 3));
        atomicAdd(a+7+band,w);
        const unsigned long long packed=(static_cast<unsigned long long>(__float_as_uint(w))<<32)
            | static_cast<unsigned int>(0xffffffffu-static_cast<unsigned int>(g));
        atomicMax(owners+t,packed);
    }
}

std::vector<torch::Tensor> coverage(torch::Tensor means,torch::Tensor inverse,
    torch::Tensor alpha,torch::Tensor semantics,torch::Tensor centers,torch::Tensor radii,
    torch::Tensor xyz,torch::Tensor lookup,torch::Tensor winner,torch::Tensor maximum,
    int H,int W,int D) {
    const c10::cuda::CUDAGuard guard(means.device());
    const int ng=means.size(0),nt=winner.numel(),nv=H*W*D;
    for(const auto& t:{means,inverse,alpha,semantics,centers,radii,xyz,lookup,winner,maximum})
        TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Contiguous CUDA tensors on one device required");
    for(const auto& t:{means,inverse,alpha,semantics,xyz,maximum})
        TORCH_CHECK(t.scalar_type()==torch::kFloat32,"FP32 required");
    for(const auto& t:{centers,radii,lookup,winner})
        TORCH_CHECK(t.scalar_type()==torch::kInt32,"Int32 indices required");
    TORCH_CHECK(means.numel()==ng*3&&inverse.numel()==ng*6&&alpha.numel()==ng&&semantics.numel()==ng*17
        &&centers.numel()==ng*3&&radii.numel()==ng&&xyz.numel()==nv*3&&lookup.numel()==nv&&maximum.numel()==nt,"Invalid shapes");
    auto counts=torch::zeros({nt,7},means.options().dtype(torch::kInt32));
    auto mass=torch::zeros({nt,11},means.options());
    auto nearest=torch::full({nt,2},INFINITY,means.options());
    auto owners=torch::zeros({nt},means.options().dtype(torch::kInt64));
    coverage_kernel<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        ng,H,W,D,means.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),
        semantics.data_ptr<float>(),centers.data_ptr<int>(),radii.data_ptr<int>(),xyz.data_ptr<float>(),
        lookup.data_ptr<int>(),winner.data_ptr<int>(),maximum.data_ptr<float>(),counts.data_ptr<int>(),
        mass.data_ptr<float>(),nearest.data_ptr<float>(),reinterpret_cast<unsigned long long*>(owners.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {counts,mass,nearest,owners};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("coverage",&coverage);}
