#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>

struct Grid {
    int H,W,D; float h,x0,y0,z0;
    __device__ int index(int x,int y,int z) const {
        return x>=0&&x<H&&y>=0&&y<W&&z>=0&&z<D ? (x*W+y)*D+z : -1;
    }
    __device__ int cell(const float* p) const {
        return index(floorf((p[0]-x0)/h),floorf((p[1]-y0)/h),floorf((p[2]-z0)/h));
    }
};

__device__ float projection(const float* x,const float* m,const float* s,
                           const float* R,float* u) {
    float v[3],d[3]={x[0]-m[0],x[1]-m[1],x[2]-m[2]},maha=0.f;
    for(int a=0;a<3;++a){v[a]=R[3*a]*d[0]+R[3*a+1]*d[1]+R[3*a+2]*d[2];u[a]=v[a]/s[a];maha+=u[a]*u[a];}
    if(maha<=1.f)return 0.f;
    float lo=0.f,hi=fmaxf(s[0],fmaxf(s[1],s[2]))*sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    for(int it=0;it<32;++it){
        const float t=.5f*(lo+hi);float norm=0.f;
        for(int a=0;a<3;++a){const float z=s[a]*v[a]/(s[a]*s[a]+t);norm+=z*z;}
        if(norm>1.f)lo=t;else hi=t;
    }
    float distance2=0.f;
    for(int a=0;a<3;++a){u[a]=s[a]*v[a]/(s[a]*s[a]+.5f*(lo+hi));const float z=v[a]-s[a]*u[a];distance2+=z*z;}
    return sqrtf(distance2);
}

__global__ void hash_kernel(int n,const float* means,Grid grid,int* heads,int* links){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int cell=grid.cell(means+3*i);
    if(cell>=0)links[i]=atomicExch(heads+cell,i);
}

__global__ void correspondence_kernel(int n,const float* means,const int64_t* seed,
    const int* fg_ids,const int64_t* labels,const int* components,Grid grid,float radius,
    int* assigned,float* purity,int* counts){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int id=fg_ids[seed[i]],c=labels[id];assigned[2*i]=c;assigned[2*i+1]=components[id];
    const float* m=means+3*i;
    const int x0=max(0,(int)floorf((m[0]-radius-grid.x0)/grid.h));
    const int x1=min(grid.H-1,(int)floorf((m[0]+radius-grid.x0)/grid.h));
    const int y0=max(0,(int)floorf((m[1]-radius-grid.y0)/grid.h));
    const int y1=min(grid.W-1,(int)floorf((m[1]+radius-grid.y0)/grid.h));
    const int z0=max(0,(int)floorf((m[2]-radius-grid.z0)/grid.h));
    const int z1=min(grid.D-1,(int)floorf((m[2]+radius-grid.z0)/grid.h));
    int total=0,same=0;
    for(int x=x0;x<=x1;++x)for(int y=y0;y<=y1;++y)for(int z=z0;z<=z1;++z){
        const int v=(x*grid.W+y)*grid.D+z;if(components[v]<=0)continue;
        const float dx=grid.x0+(x+.5f)*grid.h-m[0],dy=grid.y0+(y+.5f)*grid.h-m[1],dz=grid.z0+(z+.5f)*grid.h-m[2];
        if(dx*dx+dy*dy+dz*dz>radius*radius)continue;
        ++total;if(labels[v]==c)++same;
    }
    counts[i]=total;purity[i]=total?float(same)/total:0.f;
}

// Geometric branch-and-bound is exhaustive: any improving ellipsoid center
// lies within best_distance + global_max_scale. No fixed-K truncation.
__global__ void cover_kernel(int n,const float* points,const float* means,const float* scales,
    const float* rotation,const int* heads,const int* links,const int64_t* seed,
    const int* point_component,const int* assigned,const bool* eligible,Grid grid,
    float maximum_scale,float local_radius,bool semantic,int64_t* ids,float* us,float* distances,bool* fallback){
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const float* x=points+3*j;float best_u[3];int best=seed[j];
    float distance=projection(x,means+3*best,scales+3*best,rotation+9*best,best_u);
    const float radius=distance+maximum_scale+2e-5f;
    const int ax=max(0,(int)floorf((x[0]-radius-grid.x0)/grid.h));
    const int bx=min(grid.H-1,(int)floorf((x[0]+radius-grid.x0)/grid.h));
    const int ay=max(0,(int)floorf((x[1]-radius-grid.y0)/grid.h));
    const int by=min(grid.W-1,(int)floorf((x[1]+radius-grid.y0)/grid.h));
    const int az=max(0,(int)floorf((x[2]-radius-grid.z0)/grid.h));
    const int bz=min(grid.D-1,(int)floorf((x[2]+radius-grid.z0)/grid.h));
    for(int a=ax;a<=bx && distance>0.f;++a)for(int b=ay;b<=by && distance>0.f;++b)for(int c=az;c<=bz && distance>0.f;++c){
        for(int g=heads[(a*grid.W+b)*grid.D+c];g>=0;g=links[g]){
            const float* m=means+3*g;const float* s=scales+3*g;
            const float dx=x[0]-m[0],dy=x[1]-m[1],dz=x[2]-m[2];
            if(sqrtf(dx*dx+dy*dy+dz*dz)-fmaxf(s[0],fmaxf(s[1],s[2]))>distance+2e-5f)continue;
            float u[3];const float d=projection(x,m,s,rotation+9*g,u);
            if(d<distance){distance=d;best=g;for(int k=0;k<3;++k)best_u[k]=u[k];}
            if(distance==0.f)break;
        }
    }
    ids[2*j]=best;distances[2*j]=distance;
    for(int k=0;k<3;++k)us[6*j+k]=best_u[k];
    bool found=false;float same=CUDART_INF_F;int same_id=best;float same_u[3];
    if(semantic){
        const int x0=max(0,(int)floorf((x[0]-local_radius-grid.x0)/grid.h));
        const int x1=min(grid.H-1,(int)floorf((x[0]+local_radius-grid.x0)/grid.h));
        const int y0=max(0,(int)floorf((x[1]-local_radius-grid.y0)/grid.h));
        const int y1=min(grid.W-1,(int)floorf((x[1]+local_radius-grid.y0)/grid.h));
        const int z0=max(0,(int)floorf((x[2]-local_radius-grid.z0)/grid.h));
        const int z1=min(grid.D-1,(int)floorf((x[2]+local_radius-grid.z0)/grid.h));
        for(int a=x0;a<=x1 && same>0.f;++a)for(int b=y0;b<=y1 && same>0.f;++b)for(int c=z0;c<=z1 && same>0.f;++c){
            for(int g=heads[(a*grid.W+b)*grid.D+c];g>=0;g=links[g]){
                if(!eligible[g] || assigned[2*g+1]!=point_component[j])continue;
                const float* m=means+3*g;float dx=x[0]-m[0],dy=x[1]-m[1],dz=x[2]-m[2];
                if(dx*dx+dy*dy+dz*dz>local_radius*local_radius)continue;
                const float* s=scales+3*g;
                if(sqrtf(dx*dx+dy*dy+dz*dz)-fmaxf(s[0],fmaxf(s[1],s[2]))>same+2e-5f)continue;
                found=true;float u[3];const float d=projection(x,m,s,rotation+9*g,u);
                if(d<same){same=d;same_id=g;for(int k=0;k<3;++k)same_u[k]=u[k];}
                if(same==0.f)break;
            }
        }
    }
    fallback[j]=!found;ids[2*j+1]=found?same_id:best;distances[2*j+1]=found?same:distance;
    for(int k=0;k<3;++k)us[6*j+3+k]=found?same_u[k]:best_u[k];
}

__global__ void fit_kernel(int count,int samples,const float* positions,const float* means,
    const int* assigned,const int* components,const bool* active,Grid grid,float radius,
    float* closest,bool* found){
    const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=count)return;
    const float* p=positions+3*t;for(int a=0;a<3;++a)closest[3*t+a]=p[a];
    if(!active[t])return;
    const int g=t/samples,component=assigned[2*g+1];const float* m=means+3*g;
    const float half=.5f*grid.h;float best=CUDART_INF_F;
    const int own=grid.cell(p);
    if(own>=0&&components[own]==component){
        const int z=own%grid.D,y=(own/grid.D)%grid.W,x=own/(grid.W*grid.D);
        const float dx=grid.x0+(x+.5f)*grid.h-m[0],dy=grid.y0+(y+.5f)*grid.h-m[1],dz=grid.z0+(z+.5f)*grid.h-m[2];
        if(dx*dx+dy*dy+dz*dz<=radius*radius){found[t]=true;return;}
    }
    const int x0=max(0,(int)floorf((m[0]-radius-grid.x0)/grid.h));
    const int x1=min(grid.H-1,(int)floorf((m[0]+radius-grid.x0)/grid.h));
    const int y0=max(0,(int)floorf((m[1]-radius-grid.y0)/grid.h));
    const int y1=min(grid.W-1,(int)floorf((m[1]+radius-grid.y0)/grid.h));
    const int z0=max(0,(int)floorf((m[2]-radius-grid.z0)/grid.h));
    const int z1=min(grid.D-1,(int)floorf((m[2]+radius-grid.z0)/grid.h));
    // Bounds prune individual cubes, but all local same-component cubes remain eligible.
    for(int x=x0;x<=x1;++x)for(int y=y0;y<=y1;++y)for(int z=z0;z<=z1;++z){
        const int id=(x*grid.W+y)*grid.D+z;if(components[id]!=component)continue;
        const float center[3]={grid.x0+(x+.5f)*grid.h,grid.y0+(y+.5f)*grid.h,grid.z0+(z+.5f)*grid.h};
        float radius2=0.f,distance2=0.f,q[3];
        for(int a=0;a<3;++a){float d=center[a]-m[a];radius2+=d*d;q[a]=fminf(center[a]+half,fmaxf(center[a]-half,p[a]));d=p[a]-q[a];distance2+=d*d;}
        if(radius2>radius*radius || distance2>=best)continue;
        best=distance2;for(int a=0;a<3;++a)closest[3*t+a]=q[a];
        if(best==0.f){found[t]=true;return;}
    }
    found[t]=isfinite(best);
}

Grid make_grid(std::vector<int64_t> shape,std::vector<double> lower,double h){
    TORCH_CHECK(shape.size()==3&&lower.size()==3&&h>0,"Invalid grid");
    return Grid{int(shape[0]),int(shape[1]),int(shape[2]),float(h),float(lower[0]),float(lower[1]),float(lower[2])};
}
void check_cuda(const std::vector<torch::Tensor>& tensors){
    for(const auto& t:tensors)TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==tensors[0].device(),"Contiguous same-device CUDA tensors required");
}
std::vector<torch::Tensor> build_hash(torch::Tensor means,std::vector<int64_t> shape,std::vector<double> lower,double h){
    const c10::cuda::CUDAGuard guard(means.device());check_cuda({means});
    TORCH_CHECK(means.scalar_type()==torch::kFloat32&&means.dim()==2&&means.size(1)==3,"FP32 [N,3] means required");
    Grid grid=make_grid(shape,lower,h);auto opt=means.options().dtype(torch::kInt32);
    auto heads=torch::full({grid.H*grid.W*grid.D},-1,opt),links=torch::full({means.size(0)},-1,opt);
    hash_kernel<<<(means.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(means.size(0),means.data_ptr<float>(),grid,heads.data_ptr<int>(),links.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {heads,links};
}
std::vector<torch::Tensor> correspondence(torch::Tensor means,torch::Tensor seed,torch::Tensor fg_ids,
    torch::Tensor labels,torch::Tensor components,std::vector<int64_t> shape,std::vector<double> lower,double h,double radius){
    const c10::cuda::CUDAGuard guard(means.device());check_cuda({means,seed,fg_ids,labels,components});
    auto assigned=torch::empty({means.size(0),2},components.options());auto purity=torch::zeros({means.size(0)},means.options());
    auto counts=torch::zeros({means.size(0)},components.options());
    correspondence_kernel<<<(means.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(means.size(0),means.data_ptr<float>(),seed.data_ptr<int64_t>(),fg_ids.data_ptr<int>(),labels.data_ptr<int64_t>(),components.data_ptr<int>(),make_grid(shape,lower,h),radius,assigned.data_ptr<int>(),purity.data_ptr<float>(),counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {assigned,purity,counts};
}
std::vector<torch::Tensor> cover(torch::Tensor points,torch::Tensor means,torch::Tensor scales,torch::Tensor rotation,
    torch::Tensor heads,torch::Tensor links,torch::Tensor seed,torch::Tensor point_component,torch::Tensor assigned,
    torch::Tensor eligible,std::vector<int64_t> shape,std::vector<double> lower,double h,double maximum_scale,double local_radius,bool semantic){
    const c10::cuda::CUDAGuard guard(points.device());check_cuda({points,means,scales,rotation,heads,links,seed,point_component,assigned,eligible});
    const int n=points.size(0);auto ids=torch::empty({n,2},seed.options());auto u=torch::empty({n,2,3},points.options());
    auto distance=torch::empty({n,2},points.options());auto fallback=torch::empty({n},eligible.options());
    cover_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(n,points.data_ptr<float>(),means.data_ptr<float>(),scales.data_ptr<float>(),rotation.data_ptr<float>(),heads.data_ptr<int>(),links.data_ptr<int>(),seed.data_ptr<int64_t>(),point_component.data_ptr<int>(),assigned.data_ptr<int>(),eligible.data_ptr<bool>(),make_grid(shape,lower,h),maximum_scale,local_radius,semantic,ids.data_ptr<int64_t>(),u.data_ptr<float>(),distance.data_ptr<float>(),fallback.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {ids,u,distance,fallback};
}
std::vector<torch::Tensor> fit(torch::Tensor positions,torch::Tensor means,torch::Tensor assigned,
    torch::Tensor components,torch::Tensor active,int samples,std::vector<int64_t> shape,std::vector<double> lower,double h,double radius){
    const c10::cuda::CUDAGuard guard(positions.device());check_cuda({positions,means,assigned,components,active});
    TORCH_CHECK(positions.size(0)==means.size(0)*samples,"Sample/GS shape mismatch");
    auto closest=torch::empty_like(positions);auto found=torch::zeros({positions.size(0)},active.options());
    fit_kernel<<<(positions.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(positions.size(0),samples,positions.data_ptr<float>(),means.data_ptr<float>(),assigned.data_ptr<int>(),components.data_ptr<int>(),active.data_ptr<bool>(),make_grid(shape,lower,h),radius,closest.data_ptr<float>(),found.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {closest,found};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("build_hash",&build_hash);m.def("correspondence",&correspondence);m.def("cover",&cover);m.def("fit",&fit);}
