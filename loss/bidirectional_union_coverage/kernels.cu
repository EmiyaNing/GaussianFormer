#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ float rho(float x){return x<1.f?.5f*x*x:x-.5f;}
__device__ float project(const float* x,const float* m,const float* s,const float* R,float* u){
    float v[3],d[3]={x[0]-m[0],x[1]-m[1],x[2]-m[2]},maha=0.f;
    for(int a=0;a<3;++a){v[a]=R[3*a]*d[0]+R[3*a+1]*d[1]+R[3*a+2]*d[2];u[a]=v[a]/s[a];maha+=u[a]*u[a];}
    if(maha<=1.f)return 0.f;
    float lo=0.f,hi=fmaxf(s[0],fmaxf(s[1],s[2]))*sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    for(int t=0;t<32;++t){
        float lambda=.5f*(lo+hi),norm=0.f;
        for(int a=0;a<3;++a){float z=s[a]*v[a]/(s[a]*s[a]+lambda);norm+=z*z;}
        if(norm>1.f)lo=lambda;else hi=lambda;
    }
    float distance2=0.f;
    for(int a=0;a<3;++a){u[a]=s[a]*v[a]/(s[a]*s[a]+.5f*(lo+hi));float z=v[a]-s[a]*u[a];distance2+=z*z;}
    return sqrtf(distance2);
}

// If a center is r away, its solid-ellipsoid distance is at least
// max(0,r-largest_axis). Nonnegative semantic costs and a feasible seed
// therefore give an exhaustive finite search radius, without a fixed K.
__global__ void query_kernel(int n,const float* points,const int* classes,
    const float* means,const float* scales,const float* rotation,const float* neglogp,
    const float* semantic_lower,const int* heads,const int* links,const int64_t* seeds,
    int H,int W,int D,float x0,float y0,float z0,float h,float maxscale,
    int64_t* ids,float* projection,float* distances,float* costs,int* visited,float* radii){
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const float* x=points+3*j;int c=classes[j],best=seeds[2*j],count=1;
    float best_u[3];float best_d=project(x,means+3*best,scales+3*best,rotation+9*best,best_u);
    float value=rho(best_d/h)+neglogp[17*best+c];
    int second=seeds[2*j+1];
    if(second!=best){
        float u[3];float d=project(x,means+3*second,scales+3*second,rotation+9*second,u);++count;
        float v=rho(d/h)+neglogp[17*second+c];
        if(v<value||(v==value&&second<best)){value=v;best=second;best_d=d;for(int a=0;a<3;++a)best_u[a]=u[a];}
    }
    float budget=fmaxf(0.f,value-semantic_lower[c]);
    float bound=h*(budget<.5f?sqrtf(2.f*budget):budget+.5f)+maxscale+4e-5f;
    radii[j]=bound;
    int ax=max(0,(int)floorf((x[0]-bound-x0)/h)),bx=min(H-1,(int)floorf((x[0]+bound-x0)/h));
    int ay=max(0,(int)floorf((x[1]-bound-y0)/h)),by=min(W-1,(int)floorf((x[1]+bound-y0)/h));
    int az=max(0,(int)floorf((x[2]-bound-z0)/h)),bz=min(D-1,(int)floorf((x[2]+bound-z0)/h));
    for(int a=ax;a<=bx;++a)for(int b=ay;b<=by;++b)for(int k=az;k<=bz;++k){
        for(int g=heads[(a*W+b)*D+k];g>=0;g=links[g]){
            float semantic=neglogp[17*g+c];if(semantic>value+2e-5f)continue;
            const float* m=means+3*g;const float* s=scales+3*g;
            float dx=x[0]-m[0],dy=x[1]-m[1],dz=x[2]-m[2];
            float lower=fmaxf(0.f,sqrtf(dx*dx+dy*dy+dz*dz)-fmaxf(s[0],fmaxf(s[1],s[2])));
            if(rho(lower/h)+semantic>value+2e-5f)continue;
            float u[3];float distance=project(x,m,s,rotation+9*g,u);++count;
            float candidate=rho(distance/h)+semantic;
            if(candidate<value||(candidate==value&&g<best)){
                value=candidate;best=g;best_d=distance;for(int t=0;t<3;++t)best_u[t]=u[t];
            }
        }
    }
    ids[j]=best;distances[j]=best_d;costs[j]=value;visited[j]=count;
    for(int a=0;a<3;++a)projection[3*j+a]=best_u[a];
}

std::vector<torch::Tensor> query(torch::Tensor points,torch::Tensor classes,
    torch::Tensor means,torch::Tensor scales,torch::Tensor rotation,torch::Tensor neglogp,
    torch::Tensor semantic_lower,torch::Tensor heads,torch::Tensor links,torch::Tensor seeds,
    std::vector<int64_t> shape,std::vector<double> lower,double h,double maxscale){
    const c10::cuda::CUDAGuard guard(points.device());
    for(auto t:{points,classes,means,scales,rotation,neglogp,semantic_lower,heads,links,seeds})
        TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==points.device(),"Contiguous same-device CUDA tensors required");
    TORCH_CHECK(points.scalar_type()==torch::kFloat32&&classes.scalar_type()==torch::kInt32&&seeds.scalar_type()==torch::kInt64,"Wrong query dtype");
    TORCH_CHECK(points.dim()==2&&points.size(1)==3&&classes.numel()==points.size(0),"Wrong query shapes");
    TORCH_CHECK(means.size(0)>0&&neglogp.size(0)==means.size(0)&&neglogp.size(1)==17,"Expected positive GS count and 17 foreground classes");
    TORCH_CHECK(seeds.size(0)==points.size(0)&&seeds.size(1)==2&&shape.size()==3&&lower.size()==3,"Invalid seeds/grid");
    int n=points.size(0);auto ids=torch::empty({n},seeds.options());
    auto u=torch::empty({n,3},points.options()),d=torch::empty({n},points.options()),cost=torch::empty({n},points.options());
    auto visited=torch::empty({n},classes.options()),radius=torch::empty({n},points.options());
    if(n)query_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(n,
        points.data_ptr<float>(),classes.data_ptr<int>(),means.data_ptr<float>(),scales.data_ptr<float>(),rotation.data_ptr<float>(),
        neglogp.data_ptr<float>(),semantic_lower.data_ptr<float>(),heads.data_ptr<int>(),links.data_ptr<int>(),seeds.data_ptr<int64_t>(),
        shape[0],shape[1],shape[2],lower[0],lower[1],lower[2],h,maxscale,
        ids.data_ptr<int64_t>(),u.data_ptr<float>(),d.data_ptr<float>(),cost.data_ptr<float>(),visited.data_ptr<int>(),radius.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,u,d,cost,visited,radius};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("query",&query);}
