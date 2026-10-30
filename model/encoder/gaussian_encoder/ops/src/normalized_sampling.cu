// Factored logits -> masked softmax -> bilinear sample -> point/camera/level sum.
// Probability rows live only in block shared memory. Backward reconstructs them
// from the small query/camera logits and per-row max/reciprocal denominator.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>

namespace {
__device__ float warp_sum(float x) {
  for (int delta=16; delta; delta>>=1) x += __shfl_down_sync(0xffffffff, x, delta);
  return __shfl_sync(0xffffffff, x, 0);
}
__device__ float warp_max(float x) {
  for (int delta=16; delta; delta>>=1) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, delta));
  return __shfl_sync(0xffffffff, x, 0);
}

__device__ float sample(const float* f, float* df, int base, int channels,
                         int h, int w, float x, float y, float grad,
                         float& dx, float& dy) {
  const float xx=x*w-.5f, yy=y*h-.5f;
  const int x0=floorf(xx), y0=floorf(yy);
  const float u=xx-x0, v=yy-y0;
  float values[4]={0,0,0,0};
  #pragma unroll
  for (int j=0;j<2;++j) {
    #pragma unroll
    for (int i=0;i<2;++i) {
      const int xi=x0+i, yi=y0+j;
      if (xi>=0 && xi<w && yi>=0 && yi<h) {
        const int offset=base+(yi*w+xi)*channels;
        values[j*2+i]=f[offset];
        if (df) atomicAdd(df+offset, grad*(i?u:1-u)*(j?v:1-v));
      }
    }
  }
  dx=w*((1-v)*(values[1]-values[0])+v*(values[3]-values[2]));
  dy=h*((1-u)*(values[2]-values[0])+u*(values[3]-values[1]));
  return (1-v)*((1-u)*values[0]+u*values[1])+v*((1-u)*values[2]+u*values[3]);
}

__global__ void forward_kernel(const float* f, const int* shape, const int* starts,
    const float* loc, const bool* visible, const bool* drop,
    const float* q, const float* camera, float* out, float* stats,
    int N,int K,int V,int L,int G,int C,int S) {
  extern __shared__ float probabilities[];
  const int query=blockIdx.x, n=query%N, b=query/N;
  const int g=threadIdx.x/32, lane=threadIdx.x%32;
  const int slots=K*V*L, width=C/G, lkg=L*K*G;
  float maximum=-CUDART_INF_F;
  for(int s=lane;s<slots;s+=32) {
    const int l=s%L, v=(s/L)%V, k=s/(L*V), qi=(l*K+k)*G+g;
    const bool valid=visible[(query*K+k)*V+v] &&
        (!drop || drop[(query*V+v)*lkg+qi]);
    const float value=valid?q[query*lkg+qi]+camera[(b*V+v)*lkg+qi]:-CUDART_INF_F;
    probabilities[g*slots+s]=value;
    maximum=fmaxf(maximum,value);
  }
  maximum=warp_max(maximum);
  float denom=0;
  for(int s=lane;s<slots;s+=32) {
    const float value=probabilities[g*slots+s];
    const float p=maximum==-CUDART_INF_F?0:expf(value-maximum);
    probabilities[g*slots+s]=p;
    denom+=p;
  }
  denom=warp_sum(denom);
  const float inv=denom>0?1.f/denom:0;
  if(lane==0) {stats[(query*G+g)*2]=maximum;stats[(query*G+g)*2+1]=inv;}
  for(int s=lane;s<slots;s+=32) probabilities[g*slots+s]*=inv;
  __syncwarp();
  if(lane>=width) return;
  const int c=g*width+lane;
  float result=0;
  for(int s=0;s<slots;++s) {
    const float p=probabilities[g*slots+s];
    if(p==0) continue;
    const int l=s%L,v=(s/L)%V,k=s/(L*V);
    const int li=((query*K+k)*V+v)*2;
    const float x=loc[li],y=loc[li+1];
    if(!(x>0 && x<1 && y>0 && y<1)) continue;
    float dx,dy;
    result+=p*sample(f,nullptr,((b*V+v)*S+starts[l])*C+c,C,
                     shape[l*2],shape[l*2+1],x,y,0,dx,dy);
  }
  out[query*C+c]=result;
}

__global__ void backward_kernel(const float* f,const int* shape,const int* starts,
    const float* loc,const bool* visible,const bool* drop,const float* q,const float* camera,
    const float* out,const float* stats,const float* grad,
    float* df,float* dloc,float* dq,float* dcam,
    int N,int K,int V,int L,int G,int C,int S,int buckets) {
  const int query=blockIdx.x,n=query%N,b=query/N;
  const int g=threadIdx.x/32,lane=threadIdx.x%32,width=C/G,lkg=L*K*G;
  const int c=g*width+lane;
  const float go=lane<width?grad[query*C+c]:0;
  const float delta=warp_sum(lane<width?go*out[query*C+c]:0);
  const float maximum=stats[(query*G+g)*2],inv=stats[(query*G+g)*2+1];
  if(inv==0) return;
  for(int s=0;s<K*V*L;++s) {
    const int l=s%L,v=(s/L)%V,k=s/(L*V),qi=(l*K+k)*G+g;
    if(!visible[(query*K+k)*V+v] || (drop && !drop[(query*V+v)*lkg+qi])) continue;
    const float p=expf(q[query*lkg+qi]+camera[(b*V+v)*lkg+qi]-maximum)*inv;
    const int li=((query*K+k)*V+v)*2;
    const float x=loc[li],y=loc[li+1];
    float value=0,dx=0,dy=0;
    if(lane<width && x>0 && x<1 && y>0 && y<1) {
      value=sample(f,df,((b*V+v)*S+starts[l])*C+c,C,shape[l*2],shape[l*2+1],
                   x,y,go*p,dx,dy);
    }
    const float dp=warp_sum(go*value), gx=warp_sum(go*p*dx), gy=warp_sum(go*p*dy);
    if(lane==0) {
      const float dlogit=p*(dp-delta);
      // One warp owns each (query,group): camera contributions add locally.
      dq[query*lkg+qi]+=dlogit;
      // Partial reductions bound camera atomic contention to 64 queries.
      atomicAdd(dcam+((b*buckets+n/64)*V+v)*lkg+qi,dlogit);
      atomicAdd(dloc+li,gx);atomicAdd(dloc+li+1,gy);
    }
  }
}

void check_inputs(const at::Tensor& f,const at::Tensor& shape,const at::Tensor& starts,
    const at::Tensor& loc,const at::Tensor& visible,const at::Tensor& drop,
    const at::Tensor& q,const at::Tensor& camera) {
  TORCH_CHECK(f.is_cuda() && f.dim()==4 && f.scalar_type()==at::kFloat && f.is_contiguous(),
              "Fused DAF expects CUDA float32 features [B,V,S,C]");
  for(const auto& t:{shape,starts,loc,visible,drop,q,camera}) {
    TORCH_CHECK(t.device()==f.device() && t.is_contiguous(),"Fused DAF tensors must be contiguous on one device");
  }
  TORCH_CHECK(q.scalar_type()==at::kFloat && q.dim()==5 && q.size(0)==f.size(0),"Invalid query logits [B,N,L,K,G]");
  const auto B=q.size(0),N=q.size(1),L=q.size(2),K=q.size(3),G=q.size(4),V=f.size(1);
  TORCH_CHECK(K>0 && V>0 && L>0 && G>0 && G<=16 && f.size(3)%G==0 && f.size(3)/G<=32,
              "Fused DAF supports up to 16 groups and up to 32 channels/group");
  TORCH_CHECK(K*V*L*G*sizeof(float)<=48*1024,"Fused DAF shared-memory row exceeds 48KiB");
  TORCH_CHECK(shape.scalar_type()==at::kInt && shape.sizes()==at::IntArrayRef({L,2}) &&
              starts.scalar_type()==at::kInt && starts.sizes()==at::IntArrayRef({L}),"Invalid pyramid metadata");
  TORCH_CHECK(loc.scalar_type()==at::kFloat && loc.sizes()==at::IntArrayRef({B,N*K,V,2}),"Invalid projected coordinates");
  TORCH_CHECK(visible.scalar_type()==at::kBool && visible.sizes()==at::IntArrayRef({B,N,K,V}),"Invalid visibility mask");
  TORCH_CHECK(drop.scalar_type()==at::kBool && (drop.numel()==0 || drop.sizes()==at::IntArrayRef({B,N,V,L,K,G})),"Invalid dropout mask");
  TORCH_CHECK(camera.scalar_type()==at::kFloat && camera.sizes()==at::IntArrayRef({B,V,L,K,G}),"Invalid camera logits");
}
} // namespace

std::vector<at::Tensor> normalized_sampling_forward(const at::Tensor& f,const at::Tensor& shape,
    const at::Tensor& starts,const at::Tensor& loc,const at::Tensor& visible,const at::Tensor& drop,
    const at::Tensor& q,const at::Tensor& camera) {
  check_inputs(f,shape,starts,loc,visible,drop,q,camera);
  const at::cuda::OptionalCUDAGuard guard(device_of(f));
  const int B=q.size(0),N=q.size(1),L=q.size(2),K=q.size(3),G=q.size(4),V=f.size(1),C=f.size(3),S=f.size(2);
  auto out=at::empty({B,N,C},f.options()),stats=at::empty({B,N,G,2},f.options());
  if(B*N) {
    forward_kernel<<<B*N,G*32,K*V*L*G*sizeof(float),at::cuda::getCurrentCUDAStream()>>>(
        f.data_ptr<float>(),shape.data_ptr<int>(),starts.data_ptr<int>(),loc.data_ptr<float>(),
        visible.data_ptr<bool>(),drop.numel()?drop.data_ptr<bool>():nullptr,q.data_ptr<float>(),
        camera.data_ptr<float>(),out.data_ptr<float>(),stats.data_ptr<float>(),N,K,V,L,G,C,S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out,stats};
}

std::vector<at::Tensor> normalized_sampling_backward(const at::Tensor& f,const at::Tensor& shape,
    const at::Tensor& starts,const at::Tensor& loc,const at::Tensor& visible,const at::Tensor& drop,
    const at::Tensor& q,const at::Tensor& camera,const at::Tensor& out,const at::Tensor& stats,
    const at::Tensor& grad) {
  const at::cuda::OptionalCUDAGuard guard(device_of(f));
  const int B=q.size(0),N=q.size(1),L=q.size(2),K=q.size(3),G=q.size(4),V=f.size(1),C=f.size(3),S=f.size(2),buckets=(N+63)/64;
  auto df=at::zeros_like(f),dloc=at::zeros_like(loc),dq=at::zeros_like(q);
  auto dcam=at::zeros({B,buckets,V,L,K,G},f.options());
  if(B*N) {
    backward_kernel<<<B*N,G*32,0,at::cuda::getCurrentCUDAStream()>>>(f.data_ptr<float>(),shape.data_ptr<int>(),
        starts.data_ptr<int>(),loc.data_ptr<float>(),visible.data_ptr<bool>(),drop.numel()?drop.data_ptr<bool>():nullptr,
        q.data_ptr<float>(),camera.data_ptr<float>(),out.data_ptr<float>(),stats.data_ptr<float>(),grad.data_ptr<float>(),
        df.data_ptr<float>(),dloc.data_ptr<float>(),dq.data_ptr<float>(),dcam.data_ptr<float>(),N,K,V,L,G,C,S,buckets);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {df,dloc,dq,dcam.sum(1)};
}
