#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Same integer candidate cube, FP32 exponent and covariance ordering as census.
// Three deduplicated query bitsets: 1sigma, strict 3sigma, native positive support.
__global__ void support_kernel(int ng,int nt,int words,int children,int H,int W,int D,
    const float* means,const float* inverse,const float* alpha,const int* centers,
    const int* radii,const float* xyz,const int* lookup,unsigned long long* bits) {
    const int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(g>=ng || !(alpha[g]>0)) return;
    const int q=g/children,word=q/64;
    const unsigned long long bit=1ull<<(q%64);
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
        if(!(alpha[g]*expf(power)>0))continue;
        const float d=sqrtf(fmaxf(-2.f*power,0.f));
        atomicOr(bits+(2*nt+t)*words+word,bit);
        if(d<=3.f)atomicOr(bits+(nt+t)*words+word,bit);
        if(d<=1.f)atomicOr(bits+t*words+word,bit);
    }
}

__global__ void reduce_kernel(int nt,int nq,int words,const unsigned long long* bits,
    int* query_coverage,int* distinct_queries) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=3*nt)return;
    const int s=i/nt;const unsigned long long* b=bits+i*words;
    int total=0;
    for(int w=0;w<words;++w) {
        unsigned long long value=b[w];total+=__popcll(value);
        while(value) {
            const int q=w*64+__ffsll(value)-1;value&=value-1;
            if(q<nq)atomicAdd(query_coverage+s*nq+q,1);
        }
    }
    distinct_queries[i]=total;
}

std::vector<torch::Tensor> support(torch::Tensor means,torch::Tensor inverse,
    torch::Tensor alpha,torch::Tensor centers,torch::Tensor radii,torch::Tensor xyz,
    torch::Tensor lookup,int nt,int nq,int children,int H,int W,int D) {
    const c10::cuda::CUDAGuard guard(means.device());
    const int ng=means.size(0),words=(nq+63)/64,nv=H*W*D;
    for(const auto& t:{means,inverse,alpha,centers,radii,xyz,lookup})
        TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Contiguous same-device CUDA inputs required");
    for(const auto& t:{means,inverse,alpha,xyz})TORCH_CHECK(t.scalar_type()==torch::kFloat32,"FP32 required");
    for(const auto& t:{centers,radii,lookup})TORCH_CHECK(t.scalar_type()==torch::kInt32,"Int32 required");
    TORCH_CHECK(nt>=0&&ng==nq*children&&means.numel()==ng*3&&inverse.numel()==ng*6&&alpha.numel()==ng
        &&centers.numel()==ng*3&&radii.numel()==ng&&xyz.numel()==nv*3&&lookup.numel()==nv,"Shape mismatch");
    auto opts=means.options().dtype(torch::kInt32);
    auto bits=torch::zeros({3,nt,words},opts.dtype(torch::kInt64));
    auto coverage=torch::zeros({3,nq},opts),counts=torch::zeros({3,nt},opts);
    support_kernel<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ng,nt,words,children,H,W,D,
        means.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),centers.data_ptr<int>(),
        radii.data_ptr<int>(),xyz.data_ptr<float>(),lookup.data_ptr<int>(),reinterpret_cast<unsigned long long*>(bits.data_ptr<int64_t>()));
    if(nt)reduce_kernel<<<(3*nt+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(nt,nq,words,
        reinterpret_cast<unsigned long long*>(bits.data_ptr<int64_t>()),coverage.data_ptr<int>(),counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {bits,coverage,counts};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("support",&support);}
