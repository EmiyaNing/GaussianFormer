// Project -> visibility/dropout -> joint K*V*L softmax -> bilinear aggregation.
// Geometry and probabilities stay in shared memory. Backward reconstructs
// them and only allocates/writes gradients for frames that require them.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>

namespace {
constexpr int kMaxFrames=16;
struct FeatureMemory {
  const float* value[kMaxFrames];
  float* gradient[kMaxFrames];
  int cameras, pixels, channels;
};

__device__ float warp_sum(float value) {
  for(int offset=16;offset;offset>>=1) value+=__shfl_down_sync(0xffffffff,value,offset);
  return __shfl_sync(0xffffffff,value,0);
}
__device__ float warp_max(float value) {
  for(int offset=16;offset;offset>>=1) value=fmaxf(value,__shfl_down_sync(0xffffffff,value,offset));
  return __shfl_sync(0xffffffff,value,0);
}
__device__ bool keep(const int* dropout,int64_t index) {
  return !dropout || (uint32_t(dropout[index>>5])&(uint32_t(1)<<(index&31)));
}

// One [u_normalized,v_normalized,1/depth,visible] record for each (keypoint,view).
__device__ void project_query(const float* points,const float* projection,const float* wh,
    const bool* camera_valid,float* geometry,int query,int N,int K,int V) {
  const int b=query/N;
  for(int index=threadIdx.x;index<K*V;index+=blockDim.x) {
    const int k=index/V,v=index%V;
    const float* xyz=points+(int64_t(query)*K+k)*3;
    const float* p=projection+(b*V+v)*16;
    const float px=((p[0]*xyz[0]+p[1]*xyz[1])+p[2]*xyz[2])+p[3];
    const float py=((p[4]*xyz[0]+p[5]*xyz[1])+p[6]*xyz[2])+p[7];
    const float depth=((p[8]*xyz[0]+p[9]*xyz[1])+p[10]*xyz[2])+p[11];
    const float denominator=fmaxf(depth,1e-5f);
    const float x=(px/denominator)/wh[(b*V+v)*2];
    const float y=(py/denominator)/wh[(b*V+v)*2+1];
    geometry[index*4]=x; geometry[index*4+1]=y;
    geometry[index*4+2]=1.f/denominator;
    geometry[index*4+3]=camera_valid[b*V+v] && depth>1e-5f &&
                          x>0.f && x<1.f && y>0.f && y<1.f;
  }
  __syncthreads();
}

__device__ float bilinear(const float* values,float* gradients,int64_t base,int channels,
    int height,int width,float x,float y,float grad,float& dx,float& dy) {
  const float xx=x*width-.5f, yy=y*height-.5f;
  const int x0=floorf(xx),y0=floorf(yy);
  const float u=xx-x0,v=yy-y0;
  float pixels[4]={0,0,0,0};
  #pragma unroll
  for(int j=0;j<2;++j) {
    #pragma unroll
    for(int i=0;i<2;++i) {
      const int xi=x0+i,yi=y0+j;
      if(xi>=0 && xi<width && yi>=0 && yi<height) {
        const int64_t offset=base+(yi*width+xi)*int64_t(channels);
        pixels[j*2+i]=values[offset];
        if(gradients) atomicAdd(gradients+offset,grad*(i?u:1-u)*(j?v:1-v));
      }
    }
  }
  dx=width*((1-v)*(pixels[1]-pixels[0])+v*(pixels[3]-pixels[2]));
  dy=height*((1-u)*(pixels[2]-pixels[0])+u*(pixels[3]-pixels[1]));
  return (1-v)*((1-u)*pixels[0]+u*pixels[1])+v*((1-u)*pixels[2]+u*pixels[3]);
}

__global__ void forward_kernel(FeatureMemory memory,const int* shape,const int* starts,
    const float* points,const float* projection,const float* wh,const bool* camera_valid,
    const int* dropout,const float* q,const float* camera,float* out,float* statistics,
    int N,int K,int V,int L,int G) {
  extern __shared__ float shared[];
  const int query=blockIdx.x,b=query/N,g=threadIdx.x/32,lane=threadIdx.x%32;
  const int slots=K*V*L,lkg=L*K*G,width=memory.channels/G;
  float* geometry=shared;
  float* probabilities=shared+K*V*4+g*slots;
  project_query(points,projection,wh,camera_valid,geometry,query,N,K,V);

  float maximum=-CUDART_INF_F;
  for(int s=lane;s<slots;s+=32) {
    const int l=s%L,v=(s/L)%V,k=s/(L*V),qi=(l*K+k)*G+g;
    const int64_t dropout_index=(int64_t(query)*V+v)*lkg+qi;
    const bool valid=geometry[(k*V+v)*4+3]!=0.f && keep(dropout,dropout_index);
    const float value=valid?q[int64_t(query)*lkg+qi]+camera[(b*V+v)*lkg+qi]:-CUDART_INF_F;
    probabilities[s]=value; maximum=fmaxf(maximum,value);
  }
  maximum=warp_max(maximum);
  float denominator=0.f;
  for(int s=lane;s<slots;s+=32) {
    const float probability=maximum==-CUDART_INF_F?0.f:expf(probabilities[s]-maximum);
    probabilities[s]=probability; denominator+=probability;
  }
  denominator=warp_sum(denominator);
  const float inverse=denominator>0.f?1.f/denominator:0.f;
  if(lane==0) {
    statistics[(int64_t(query)*G+g)*2]=maximum;
    statistics[(int64_t(query)*G+g)*2+1]=inverse;
  }
  for(int s=lane;s<slots;s+=32) probabilities[s]*=inverse;
  __syncwarp();
  if(lane>=width) return;
  const int c=g*width+lane;
  float result=0.f;
  // Reject an invisible view once, before visiting its pyramid levels.
  for(int k=0;k<K;++k) for(int v=0;v<V;++v) {
    const float* geo=geometry+(k*V+v)*4;
    if(geo[3]==0.f) continue;
    const int frame=v/memory.cameras,local_camera=v%memory.cameras;
    const int64_t base=(int64_t(b)*memory.cameras+local_camera)*memory.pixels*memory.channels+c;
    for(int l=0;l<L;++l) {
      const float probability=probabilities[(k*V+v)*L+l];
      if(probability==0.f) continue;
      float dx,dy;
      result+=probability*bilinear(memory.value[frame],nullptr,
          base+int64_t(starts[l])*memory.channels,memory.channels,
          shape[l*2],shape[l*2+1],geo[0],geo[1],0.f,dx,dy);
    }
  }
  out[int64_t(query)*memory.channels+c]=result;
}

__global__ void backward_kernel(FeatureMemory memory,const int* shape,const int* starts,
    const float* points,const float* projection,const float* wh,const bool* camera_valid,
    const int* dropout,const float* q,const float* camera,const float* out,const float* statistics,
    const float* grad,float* dpoints,float* dq,float* dcamera,
    int N,int K,int V,int L,int G,int buckets) {
  extern __shared__ float geometry[];
  const int query=blockIdx.x,b=query/N,n=query%N,g=threadIdx.x/32,lane=threadIdx.x%32;
  const int lkg=L*K*G,width=memory.channels/G,c=g*width+lane;
  project_query(points,projection,wh,camera_valid,geometry,query,N,K,V);
  const float go=lane<width?grad[int64_t(query)*memory.channels+c]:0.f;
  const float delta=warp_sum(lane<width?go*out[int64_t(query)*memory.channels+c]:0.f);
  const float maximum=statistics[(int64_t(query)*G+g)*2];
  const float inverse=statistics[(int64_t(query)*G+g)*2+1];
  if(inverse==0.f) return;
  for(int k=0;k<K;++k) {
    float dxyz[3]={0.f,0.f,0.f};
    for(int v=0;v<V;++v) {
      const float* geo=geometry+(k*V+v)*4;
      if(geo[3]==0.f) continue;
      const int frame=v/memory.cameras,local_camera=v%memory.cameras;
      const int64_t base=(int64_t(b)*memory.cameras+local_camera)*memory.pixels*memory.channels+c;
      float gx=0.f,gy=0.f;
      for(int l=0;l<L;++l) {
        const int qi=(l*K+k)*G+g;
        if(!keep(dropout,(int64_t(query)*V+v)*lkg+qi)) continue;
        const float probability=expf(q[int64_t(query)*lkg+qi]+camera[(b*V+v)*lkg+qi]-maximum)*inverse;
        float value=0.f,dx=0.f,dy=0.f;
        if(lane<width) value=bilinear(memory.value[frame],memory.gradient[frame],
            base+int64_t(starts[l])*memory.channels,memory.channels,
            shape[l*2],shape[l*2+1],geo[0],geo[1],go*probability,dx,dy);
        const float dp=warp_sum(go*value);
        gx+=warp_sum(go*probability*dx); gy+=warp_sum(go*probability*dy);
        if(lane==0) {
          const float dlogit=probability*(dp-delta);
          if(dq) dq[int64_t(query)*lkg+qi]+=dlogit;
          if(dcamera) atomicAdd(dcamera+((int64_t(b)*buckets+n/64)*V+v)*lkg+qi,dlogit);
        }
      }
      if(lane==0 && dpoints) {
        const float* p=projection+(b*V+v)*16;
        const float w=wh[(b*V+v)*2],h=wh[(b*V+v)*2+1];
        const float dx=gx/w*geo[2],dy=gy/h*geo[2];
        const float dz=-(gx/w*(geo[0]*w)+gy/h*(geo[1]*h))*geo[2];
        #pragma unroll
        for(int j=0;j<3;++j) dxyz[j]+=dx*p[j]+dy*p[4+j]+dz*p[8+j];
      }
    }
    if(lane==0 && dpoints) {
      #pragma unroll
      for(int j=0;j<3;++j) atomicAdd(dpoints+(int64_t(query)*K+k)*3+j,dxyz[j]);
    }
  }
}

FeatureMemory check_inputs(const at::Tensor& points,const at::Tensor& projection,
    const at::Tensor& wh,const at::Tensor& valid,const at::Tensor& shape,const at::Tensor& starts,
    const at::Tensor& dropout,const at::Tensor& q,const at::Tensor& camera,
    const std::vector<at::Tensor>& frames) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kFloat && q.is_contiguous() && q.dim()==5,
              "Expected CUDA float32 query logits [B,N,L,K,G]");
  TORCH_CHECK(!frames.empty() && frames.size()<=kMaxFrames,"Expected 1..16 temporal frames");
  const int B=q.size(0),N=q.size(1),L=q.size(2),K=q.size(3),G=q.size(4);
  FeatureMemory memory{};
  for(int t=0;t<frames.size();++t) {
    const auto& f=frames[t];
    TORCH_CHECK(f.device()==q.device() && f.scalar_type()==at::kFloat && f.is_contiguous() &&
                f.dim()==4 && f.size(0)==B,"Expected contiguous packed frame [B,V,S,C]");
    if(t==0) {memory.cameras=f.size(1);memory.pixels=f.size(2);memory.channels=f.size(3);}
    TORCH_CHECK(f.sizes()==frames[0].sizes(),"All temporal frames need matching feature shapes");
    memory.value[t]=f.data_ptr<float>();
  }
  const int V=frames.size()*memory.cameras;
  TORCH_CHECK(K>0 && L>0 && G>0 && G<=16 && memory.channels%G==0 && memory.channels/G<=32,
              "Temporal sampling supports up to 16 groups and 32 channels/group");
  TORCH_CHECK((K*V*4+G*K*V*L)*sizeof(float)<=48*1024,"Temporal shared-memory row exceeds 48 KiB");
  for(const auto& tensor:{points,projection,wh,valid,shape,starts,dropout,camera})
    TORCH_CHECK(tensor.device()==q.device() && tensor.is_contiguous(),"Temporal inputs must be contiguous on one CUDA device");
  TORCH_CHECK(points.scalar_type()==at::kFloat && points.sizes()==at::IntArrayRef({B,N,K,3}),"Invalid query keypoints");
  TORCH_CHECK(projection.scalar_type()==at::kFloat && projection.sizes()==at::IntArrayRef({B,V,4,4}) &&
              !projection.requires_grad(),"Expected fixed float32 camera projections [B,V,4,4]");
  TORCH_CHECK(wh.scalar_type()==at::kFloat && wh.sizes()==at::IntArrayRef({B,V,2}) &&
              !wh.requires_grad(),"Expected fixed float32 image sizes [B,V,2]");
  TORCH_CHECK(valid.scalar_type()==at::kBool && valid.sizes()==at::IntArrayRef({B,V}),"Invalid camera validity mask");
  TORCH_CHECK(shape.scalar_type()==at::kInt && shape.sizes()==at::IntArrayRef({L,2}) &&
              starts.scalar_type()==at::kInt && starts.sizes()==at::IntArrayRef({L}),"Invalid FPN metadata");
  TORCH_CHECK(dropout.scalar_type()==at::kInt && (dropout.numel()==0 ||
              dropout.numel()==(int64_t(B)*N*V*L*K*G+31)/32),"Invalid packed dropout bits");
  TORCH_CHECK(camera.scalar_type()==at::kFloat && camera.sizes()==at::IntArrayRef({B,V,L,K,G}),"Invalid camera logits");
  return memory;
}
}  // namespace

std::vector<at::Tensor> temporal_sampling_forward(
    const at::Tensor& points,const at::Tensor& projection,const at::Tensor& wh,
    const at::Tensor& valid,const at::Tensor& shape,const at::Tensor& starts,
    const at::Tensor& dropout,const at::Tensor& q,const at::Tensor& camera,
    const std::vector<at::Tensor>& frames) {
  auto memory=check_inputs(points,projection,wh,valid,shape,starts,dropout,q,camera,frames);
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  const int B=q.size(0),N=q.size(1),L=q.size(2),K=q.size(3),G=q.size(4),V=frames.size()*memory.cameras;
  auto output=at::empty({B,N,memory.channels},q.options());
  auto statistics=at::empty({B,N,G,2},q.options());
  if(B*N) {
    const int shared=(K*V*4+G*K*V*L)*sizeof(float);
    forward_kernel<<<B*N,G*32,shared,at::cuda::getCurrentCUDAStream()>>>(
        memory,shape.data_ptr<int>(),starts.data_ptr<int>(),points.data_ptr<float>(),
        projection.data_ptr<float>(),wh.data_ptr<float>(),valid.data_ptr<bool>(),
        dropout.numel()?dropout.data_ptr<int>():nullptr,q.data_ptr<float>(),camera.data_ptr<float>(),
        output.data_ptr<float>(),statistics.data_ptr<float>(),N,K,V,L,G);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output,statistics};
}

std::vector<at::Tensor> temporal_sampling_backward(
    const at::Tensor& points,const at::Tensor& projection,const at::Tensor& wh,
    const at::Tensor& valid,const at::Tensor& shape,const at::Tensor& starts,
    const at::Tensor& dropout,const at::Tensor& q,const at::Tensor& camera,
    const std::vector<at::Tensor>& frames,const at::Tensor& output,const at::Tensor& statistics,
    const at::Tensor& grad,bool need_points,bool need_query,bool need_camera,
    const std::vector<bool>& need_frames) {
  auto memory=check_inputs(points,projection,wh,valid,shape,starts,dropout,q,camera,frames);
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  TORCH_CHECK(need_frames.size()==frames.size(),"Invalid temporal feature gradient mask");
  TORCH_CHECK(grad.device()==q.device() && grad.scalar_type()==at::kFloat && grad.is_contiguous() &&
              grad.sizes()==output.sizes(),"Invalid temporal output gradient");
  const int B=q.size(0),N=q.size(1),L=q.size(2),K=q.size(3),G=q.size(4),V=frames.size()*memory.cameras;
  const int buckets=(N+63)/64;
  auto empty=at::empty({0},q.options());
  auto dpoints=need_points?at::zeros_like(points):empty;
  auto dq=need_query?at::zeros_like(q):empty;
  auto partial=need_camera?at::zeros({B,buckets,V,L,K,G},q.options()):empty;
  std::vector<at::Tensor> gradients;
  for(int t=0;t<frames.size();++t) {
    gradients.push_back(need_frames[t]?at::zeros_like(frames[t]):empty);
    memory.gradient[t]=need_frames[t]?gradients.back().data_ptr<float>():nullptr;
  }
  if(B*N) {
    backward_kernel<<<B*N,G*32,K*V*4*sizeof(float),at::cuda::getCurrentCUDAStream()>>>(
        memory,shape.data_ptr<int>(),starts.data_ptr<int>(),points.data_ptr<float>(),
        projection.data_ptr<float>(),wh.data_ptr<float>(),valid.data_ptr<bool>(),
        dropout.numel()?dropout.data_ptr<int>():nullptr,q.data_ptr<float>(),camera.data_ptr<float>(),
        output.data_ptr<float>(),statistics.data_ptr<float>(),grad.data_ptr<float>(),
        need_points?dpoints.data_ptr<float>():nullptr,need_query?dq.data_ptr<float>():nullptr,
        need_camera?partial.data_ptr<float>():nullptr,N,K,V,L,G,buckets);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  std::vector<at::Tensor> result={dpoints,dq,need_camera?partial.sum(1):empty};
  result.insert(result.end(),gradients.begin(),gradients.end());
  return result;
}
