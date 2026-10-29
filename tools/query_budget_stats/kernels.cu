#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// One warp enumerates exactly the production candidate cube of one Gaussian.
__global__ void census_kernel(int ng,int H,int W,int D,int children,int words,
    const float* means,const float* inverse,const float* alpha,const float* semantics,
    const int* centers,const int* radii,const float* xyz,const int* labels,
    const bool* camera,const int* lookup,const int* winner,int* counts,
    unsigned long long* bits,unsigned long long* owners,int* gs) {
    const int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(g>=ng || !(alpha[g]>0))return;
    const int q=g/children,word=q/64;const unsigned long long bit=1ull<<(q%64);
    const int* p=centers+g*3;const int r=radii[g];
    const int x0=max(0,p[0]-r),x1=min(H,p[0]+r+1);
    const int y0=max(0,p[1]-r),y1=min(W,p[1]+r+1);
    const int z0=max(0,p[2]-r),z1=min(D,p[2]+r+1),ny=y1-y0,nz=z1-z0;
    const float* inv=inverse+g*6;
    int nc=0,nf=0,cf=0,cc=0,f3=0;
    for(int k=lane;k<(x1-x0)*ny*nz;k+=32) {
        const int v=((x0+k/(ny*nz))*W+y0+(k/nz)%ny)*D+z0+k%nz;
        if(!camera[v] || labels[v]<0 || labels[v]>17)continue;
        const float dx=means[g*3]-xyz[v*3],dy=means[g*3+1]-xyz[v*3+1],dz=means[g*3+2]-xyz[v*3+2];
        float power=inv[0]*dx*dx+inv[1]*dy*dy+inv[2]*dz*dz;
        power=-.5f*power-(inv[3]*dx*dy+inv[4]*dy*dz+inv[5]*dx*dz);
        const float weight=alpha[g]*expf(power);if(!(weight>0))continue;
        const float d=sqrtf(fmaxf(-2.f*power,0.f));
        ++nc;cc+=(d<=1.f);
        const int t=lookup[v];
        if(t>=0) {
            ++nf;cf+=(d<=1.f);f3+=(d<=3.f);
            atomicAdd(counts+t*3,1);
            if(d<=1.f) {
                atomicAdd(counts+t*3+1,1);
                atomicOr(bits+t*words+word,bit);
            }
            if(d<=3.f)atomicAdd(counts+t*3+2,1);
        }
        const float w=weight*semantics[g*17+winner[v]];
        if(w>0) {
            const unsigned long long packed=(static_cast<unsigned long long>(__float_as_uint(w))<<32)
                | static_cast<unsigned int>(0xffffffffu-static_cast<unsigned int>(g));
            atomicMax(owners+v,packed);
        }
    }
    for(int off=16;off>0;off/=2) {
        nc+=__shfl_down_sync(0xffffffff,nc,off);nf+=__shfl_down_sync(0xffffffff,nf,off);
        cf+=__shfl_down_sync(0xffffffff,cf,off);cc+=__shfl_down_sync(0xffffffff,cc,off);
        f3+=__shfl_down_sync(0xffffffff,f3,off);
    }
    if(lane==0) {gs[g*5]=nc;gs[g*5+1]=nf;gs[g*5+2]=cf;gs[g*5+3]=cc;gs[g*5+4]=f3;}
}

__global__ void reduce_bits(int nt,int nq,int words,const unsigned long long* bits,
    const int* assigned,int* own,int* query_counts,int* query_coverage) {
    const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nt)return;
    const unsigned long long* b=bits+t*words;int total=0;
    for(int w=0;w<words;++w)total+=__popcll(b[w]);
    query_counts[t]=total;const int q0=assigned[t];own[t]=(b[q0/64]>>(q0%64))&1ull;
    for(int w=0;w<words;++w) {
        unsigned long long value=b[w];
        while(value) {
            const int q=w*64+__ffsll(value)-1;value&=value-1;
            if(q>=nq)continue;
            atomicAdd(query_coverage+q*3,1);
            atomicAdd(query_coverage+q*3+(total>1?1:2),1);
        }
    }
}

std::vector<torch::Tensor> census(torch::Tensor means,torch::Tensor inverse,
    torch::Tensor alpha,torch::Tensor semantics,torch::Tensor centers,torch::Tensor radii,
    torch::Tensor xyz,torch::Tensor labels,torch::Tensor camera,torch::Tensor lookup,
    torch::Tensor winner,torch::Tensor assigned,int nq,int children,int H,int W,int D) {
    const c10::cuda::CUDAGuard guard(means.device());
    const int ng=means.size(0),nt=assigned.numel(),nv=H*W*D,words=(nq+63)/64;
    for(const auto& t:{means,inverse,alpha,semantics,centers,radii,xyz,labels,camera,lookup,winner,assigned})
        TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Contiguous CUDA tensors on one device required");
    for(const auto& t:{means,inverse,alpha,semantics,xyz})TORCH_CHECK(t.scalar_type()==torch::kFloat32,"FP32 required");
    for(const auto& t:{centers,radii,labels,lookup,winner,assigned})TORCH_CHECK(t.scalar_type()==torch::kInt32,"Int32 required");
    TORCH_CHECK(camera.scalar_type()==torch::kBool,"Boolean camera required");
    TORCH_CHECK(ng==nq*children&&inverse.numel()==ng*6&&semantics.numel()==ng*17&&xyz.numel()==nv*3
        &&labels.numel()==nv&&camera.numel()==nv&&lookup.numel()==nv&&winner.numel()==nv,"Invalid shapes");
    auto opts=means.options().dtype(torch::kInt32);
    auto counts=torch::zeros({nt,3},opts),gs=torch::zeros({ng,5},opts);
    auto bits=torch::zeros({nt,words},opts.dtype(torch::kInt64));
    auto owners=torch::zeros({nv},opts.dtype(torch::kInt64));
    auto own=torch::zeros({nt},opts),query_counts=torch::zeros({nt},opts);
    auto query_coverage=torch::zeros({nq,3},opts);
    census_kernel<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ng,H,W,D,children,words,
        means.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),semantics.data_ptr<float>(),
        centers.data_ptr<int>(),radii.data_ptr<int>(),xyz.data_ptr<float>(),labels.data_ptr<int>(),camera.data_ptr<bool>(),
        lookup.data_ptr<int>(),winner.data_ptr<int>(),counts.data_ptr<int>(),reinterpret_cast<unsigned long long*>(bits.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(owners.data_ptr<int64_t>()),gs.data_ptr<int>());
    if(nt)reduce_bits<<<(nt+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(nt,nq,words,
        reinterpret_cast<unsigned long long*>(bits.data_ptr<int64_t>()),assigned.data_ptr<int>(),own.data_ptr<int>(),
        query_counts.data_ptr<int>(),query_coverage.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {counts,bits,owners,gs,own,query_counts,query_coverage};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("census",&census);}
