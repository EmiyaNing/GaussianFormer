#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
// Each GS enumerates every GT center in its circumscribed cube, then tests
// the exact rotated Mahalanobis ellipsoid. GTs and classes are camera-valid.
__global__ void stats(int N,int H,int W,int D,float h,float x0,float y0,float z0,
 const float* means,const float* scales,const float* rotation,const int64_t* classes,
 const float* xyz,const int64_t* labels,const int* lookup,
 float* values,float* direction,int* counts,int* core,int* same_core,int* core_per_gs){
 const int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=N)return;
 const float* m=means+3*g;const float* s=scales+3*g;const float* R=rotation+9*g;
 const float radius=3.f*fmaxf(s[0],fmaxf(s[1],s[2]));
 const int ax=max(0,(int)floorf((m[0]-radius-x0)/h)),bx=min(H-1,(int)floorf((m[0]+radius-x0)/h));
 const int ay=max(0,(int)floorf((m[1]-radius-y0)/h)),by=min(W-1,(int)floorf((m[1]+radius-y0)/h));
 const int az=max(0,(int)floorf((m[2]-radius-z0)/h)),bz=min(D-1,(int)floorf((m[2]+radius-z0)/h));
 float total=0.f,dx=0.f,dy=0.f,dz=0.f;int n=0,ncore=0;
 for(int a=ax;a<=bx;++a)for(int b=ay;b<=by;++b)for(int c=az;c<=bz;++c){
  const int v=(a*W+b)*D+c,t=lookup[v];if(t<0)continue;
  const float d[3]={m[0]-xyz[3*v],m[1]-xyz[3*v+1],m[2]-xyz[3*v+2]};
  float maha=0.f;
  for(int k=0;k<3;++k){const float u=(R[3*k]*d[0]+R[3*k+1]*d[1]+R[3*k+2]*d[2])/s[k];maha+=u*u;}
  if(maha>9.f)continue;
  const bool same=classes[g]==labels[v];
  if(maha<=1.f){atomicAdd(core+t,1);++ncore;if(same)atomicAdd(same_core+t,1);}
  if(!same)continue;
  const float distance=sqrtf(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
  total+=distance;++n;
  if(distance>1e-12f){dx+=d[0]/distance;dy+=d[1]/distance;dz+=d[2]/distance;}
 }
 counts[g]=n;core_per_gs[g]=ncore;const float div=n?1.f/n:0.f;
 values[g]=total*div;direction[3*g]=dx*div;direction[3*g+1]=dy*div;direction[3*g+2]=dz*div;
}
std::vector<torch::Tensor> statistics(torch::Tensor means,torch::Tensor scales,torch::Tensor rotation,
 torch::Tensor classes,torch::Tensor xyz,torch::Tensor labels,torch::Tensor lookup,int nt,
 std::vector<int64_t> shape,std::vector<double> lower,double h){
 const c10::cuda::CUDAGuard guard(means.device());
 TORCH_CHECK(shape.size()==3&&lower.size()==3&&h>0,"Invalid grid");
 int N=means.size(0),nv=shape[0]*shape[1]*shape[2];
 for(const auto& t:{means,scales,rotation,classes,xyz,labels,lookup})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Same-device contiguous CUDA tensors required");
 for(const auto& t:{means,scales,rotation,xyz})TORCH_CHECK(t.scalar_type()==torch::kFloat32,"FP32 required");
 for(const auto& t:{classes,labels})TORCH_CHECK(t.scalar_type()==torch::kInt64,"Int64 labels required");
 TORCH_CHECK(lookup.scalar_type()==torch::kInt32,"Int32 lookup required");
 TORCH_CHECK(means.numel()==3*N&&scales.numel()==3*N&&rotation.numel()==9*N&&classes.numel()==N&&xyz.numel()==3*nv&&labels.numel()==nv&&lookup.numel()==nv,"Shape mismatch");
 auto f=means.options(),i=f.dtype(torch::kInt32);
 auto values=torch::zeros({N},f),direction=torch::zeros({N,3},f),counts=torch::zeros({N},i),
  core=torch::zeros({nt},i),same_core=torch::zeros({nt},i),per_gs=torch::zeros({N},i);
 stats<<<(N+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(N,shape[0],shape[1],shape[2],h,lower[0],lower[1],lower[2],
  means.data_ptr<float>(),scales.data_ptr<float>(),rotation.data_ptr<float>(),classes.data_ptr<int64_t>(),
  xyz.data_ptr<float>(),labels.data_ptr<int64_t>(),lookup.data_ptr<int>(),values.data_ptr<float>(),direction.data_ptr<float>(),
  counts.data_ptr<int>(),core.data_ptr<int>(),same_core.data_ptr<int>(),per_gs.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {values,direction,counts,core,same_core,per_gs};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("statistics",&statistics);}
