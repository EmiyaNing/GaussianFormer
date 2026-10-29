#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Four query bitsets: geometric 1sigma, strict 3sigma, native, same-class 1sigma.
// GS counts retain distinct voxel contacts for each individual GS; query and
// group coverage are unions, never sums of overlapping GS contacts.
__global__ void census(int ng,int nt,int words,int children,int H,int W,int D,
 const float* means,const float* inverse,const float* alpha,const int* centers,
 const int* radii,const float* xyz,const int* lookup,const int* labels,
 const int* classes,unsigned long long* bits,int* gs_counts) {
 const int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
 if(g>=ng || !(alpha[g]>0))return;
 const int word=(g/children)/64;const unsigned long long bit=1ull<<((g/children)%64);
 const int* p=centers+g*3;const int r=radii[g];
 const int x0=max(0,p[0]-r),x1=min(H,p[0]+r+1);
 const int y0=max(0,p[1]-r),y1=min(W,p[1]+r+1);
 const int z0=max(0,p[2]-r),z1=min(D,p[2]+r+1),ny=y1-y0,nz=z1-z0;
 const float* inv=inverse+g*6;
 int core=0,same=0;
 for(int k=lane;k<(x1-x0)*ny*nz;k+=32){
  const int v=((x0+k/(ny*nz))*W+y0+(k/nz)%ny)*D+z0+k%nz;
  const int t=lookup[v];if(t<0)continue;
  const float dx=means[g*3]-xyz[v*3],dy=means[g*3+1]-xyz[v*3+1],dz=means[g*3+2]-xyz[v*3+2];
  float power=-.5f*(inv[0]*dx*dx+inv[1]*dy*dy+inv[2]*dz*dz)-(inv[3]*dx*dy+inv[4]*dy*dz+inv[5]*dx*dz);
  if(!(alpha[g]*expf(power)>0))continue;
  const float d=sqrtf(fmaxf(-2.f*power,0.f));
  atomicOr(bits+(2*nt+t)*words+word,bit);
  if(d<=3.f)atomicOr(bits+(nt+t)*words+word,bit);
  if(d<=1.f){
   ++core;atomicOr(bits+t*words+word,bit);
   if(labels[v]==classes[g]){++same;atomicOr(bits+(3*nt+t)*words+word,bit);}
  }
 }
 for(int offset=16;offset>0;offset/=2){
  core+=__shfl_down_sync(0xffffffff,core,offset);
  same+=__shfl_down_sync(0xffffffff,same,offset);
 }
 if(lane==0){gs_counts[g*2]=core;gs_counts[g*2+1]=same;}
}

__global__ void reduce(int nt,int nq,int words,const unsigned long long* bits,
 const int* groups,int* coverage,int* unions){
 const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4*nt)return;
 const int s=i/nt;int seen=0;
 for(int w=0;w<words;++w){
  unsigned long long v=bits[i*words+w];
  while(v){const int q=w*64+__ffsll(v)-1;v&=v-1;
   if(q<nq){atomicAdd(coverage+s*nq+q,1);seen|=1<<groups[q];}
  }
 }
 if(seen){atomicAdd(unions+s,1);
  for(int b=0;b<5;++b)if(seen&(1<<b))atomicAdd(unions+(b+1)*4+s,1);
 }
}

std::vector<torch::Tensor> support(torch::Tensor means,torch::Tensor inverse,
 torch::Tensor alpha,torch::Tensor centers,torch::Tensor radii,torch::Tensor xyz,
 torch::Tensor lookup,torch::Tensor labels,torch::Tensor classes,torch::Tensor groups,
 int nt,int nq,int children,int H,int W,int D){
 const c10::cuda::CUDAGuard guard(means.device());
 const int ng=means.size(0),words=(nq+63)/64,nv=H*W*D;
 for(const auto& t:{means,inverse,alpha,centers,radii,xyz,lookup,labels,classes,groups})
  TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Contiguous same-device CUDA inputs required");
 for(const auto& t:{means,inverse,alpha,xyz})TORCH_CHECK(t.scalar_type()==torch::kFloat32,"FP32 required");
 for(const auto& t:{centers,radii,lookup,labels,classes,groups})TORCH_CHECK(t.scalar_type()==torch::kInt32,"Int32 required");
 TORCH_CHECK(nt>=0&&nq>0&&children>0&&H>0&&W>0&&D>0&&ng==nq*children&&means.numel()==ng*3
  &&inverse.numel()==ng*6&&alpha.numel()==ng&&centers.numel()==ng*3&&radii.numel()==ng
  &&xyz.numel()==nv*3&&lookup.numel()==nv&&labels.numel()==nv&&classes.numel()==ng&&groups.numel()==nq,"Shape mismatch");
 TORCH_CHECK(groups.min().item<int>()>=0&&groups.max().item<int>()<5,"Group IDs must be 0..4");
 auto opts=means.options().dtype(torch::kInt32);
 auto bits=torch::zeros({4,nt,words},opts.dtype(torch::kInt64));
 auto counts=torch::zeros({ng,2},opts),coverage=torch::zeros({4,nq},opts),unions=torch::zeros({6,4},opts);
 census<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ng,nt,words,children,H,W,D,
  means.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),centers.data_ptr<int>(),radii.data_ptr<int>(),
  xyz.data_ptr<float>(),lookup.data_ptr<int>(),labels.data_ptr<int>(),classes.data_ptr<int>(),
  reinterpret_cast<unsigned long long*>(bits.data_ptr<int64_t>()),counts.data_ptr<int>());
 if(nt)reduce<<<(4*nt+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(nt,nq,words,
  reinterpret_cast<unsigned long long*>(bits.data_ptr<int64_t>()),groups.data_ptr<int>(),coverage.data_ptr<int>(),unions.data_ptr<int>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {bits,coverage,unions,counts};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("support",&support);}
