#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void classify_sigma(int ng,int H,int W,int D,const float* means,
    const float* invs,const float* alpha,const int* centers,const int* radii,
    const float* xyz,const int* labels,const bool* camera,int* output) {
    int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(g>=ng)return;
    unsigned int bits[6]={0,0,0,0,0,0};
    if(alpha[g]>0) {
        const int* p=centers+g*3;int r=radii[g];
        int x0=max(0,p[0]-r),x1=min(H,p[0]+r+1),y0=max(0,p[1]-r),y1=min(W,p[1]+r+1),z0=max(0,p[2]-r),z1=min(D,p[2]+r+1);
        int ny=y1-y0,nz=z1-z0;const float* inv=invs+g*6;
        for(int k=lane;k<(x1-x0)*ny*nz;k+=32) {
            int v=((x0+k/(ny*nz))*W+y0+(k/nz)%ny)*D+z0+k%nz;
            int y=labels[v];if(y<0||y>=18)continue;
            float dx=means[g*3]-xyz[v*3],dy=means[g*3+1]-xyz[v*3+1],dz=means[g*3+2]-xyz[v*3+2];
            float power=inv[0]*dx*dx+inv[1]*dy*dy+inv[2]*dz*dz;
            power=-.5f*power-(inv[3]*dx*dy+inv[4]*dy*dz+inv[5]*dx*dz);
            if(!(alpha[g]*expf(power)>0)||power< -4.5f)continue;
            unsigned int bit=1u<<y;
            bits[4]|=bit;if(camera[v])bits[5]|=bit;
            if(power>= -2.f) {bits[2]|=bit;if(camera[v])bits[3]|=bit;}
            if(power>= -.5f) {bits[0]|=bit;if(camera[v])bits[1]|=bit;}
        }
    }
    for(int offset=16;offset>0;offset/=2)
        for(int s=0;s<6;++s)bits[s]|=__shfl_down_sync(0xffffffff,bits[s],offset);
    if(lane==0)for(int s=0;s<6;++s)output[s*ng+g]=bits[s];
}

torch::Tensor membership(torch::Tensor means,torch::Tensor inverse,torch::Tensor alpha,
    torch::Tensor centers,torch::Tensor radii,torch::Tensor xyz,torch::Tensor labels,
    torch::Tensor camera,int H,int W,int D) {
    const c10::cuda::CUDAGuard guard(means.device());
    int ng=means.size(0),nv=H*W*D;
    for(const auto& t:{means,inverse,alpha,centers,radii,xyz,labels,camera})
        TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Contiguous CUDA inputs on one device required");
    for(const auto& t:{means,inverse,alpha,xyz})TORCH_CHECK(t.scalar_type()==torch::kFloat32,"FP32 inputs required");
    for(const auto& t:{centers,radii,labels})TORCH_CHECK(t.scalar_type()==torch::kInt32,"Int32 inputs required");
    TORCH_CHECK(camera.scalar_type()==torch::kBool,"Boolean camera mask required");
    TORCH_CHECK(means.numel()==ng*3&&inverse.numel()==ng*6&&alpha.numel()==ng&&centers.numel()==ng*3&&radii.numel()==ng&&xyz.numel()==nv*3&&labels.numel()==nv&&camera.numel()==nv,"Invalid input shape");
    auto out=torch::zeros({3,2,ng},means.options().dtype(torch::kInt32));
    classify_sigma<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ng,H,W,D,
        means.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),centers.data_ptr<int>(),radii.data_ptr<int>(),xyz.data_ptr<float>(),labels.data_ptr<int>(),camera.data_ptr<bool>(),out.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("membership",&membership);}
