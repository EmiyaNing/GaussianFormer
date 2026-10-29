// Diagnostics only. Integer boxes are computed by the production PyTorch code.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void probe_kernel(int ng,int H,int W,int D,const float* means,const float* scales,
    const float* inverse,const float* rotation,const float* alpha,const float* semantic,
    const int* centers,const int* radii,const float* xyz,const int* labels,const bool* camera,
    const int* prediction,const int* winner,const int* assigned,const float* target_sigmoid,
    const float* ce,const float* lovasz,int* bits,float* stats,unsigned long long* owners) {
    int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(g>=ng)return;
    float acc[37]={0};unsigned int membership[2]={0,0};
    if(alpha[g]>0) {
        const int* center=centers+g*3;int r=radii[g];
        int x0=max(0,center[0]-r),x1=min(H,center[0]+r+1),y0=max(0,center[1]-r),y1=min(W,center[1]+r+1),z0=max(0,center[2]-r),z1=min(D,center[2]+r+1);
        int ny=y1-y0,nz=z1-z0;const float* inv=inverse+g*6;const float* rot=rotation+g*9;
        for(int k=lane;k<(x1-x0)*ny*nz;k+=32) {
            int v=((x0+k/(ny*nz))*W+y0+(k/nz)%ny)*D+z0+k%nz;
            int y=labels[v];if(y<0||y>=18)continue;
            float dx=means[g*3]-xyz[v*3],dy=means[g*3+1]-xyz[v*3+1],dz=means[g*3+2]-xyz[v*3+2];
            float power=inv[0]*dx*dx+inv[1]*dy*dy+inv[2]*dz*dz;
            power=-.5f*power-(inv[3]*dx*dy+inv[4]*dy*dz+inv[5]*dx*dz);
            float kernel=expf(power),weight=alpha[g]*kernel;if(!(weight>0))continue;
            membership[0]|=1u<<y;acc[0]+=1;acc[5]+=kernel;
            if(!camera[v])continue;
            membership[1]|=1u<<y;acc[1]+=1;acc[6]+=kernel;
            const bool fg=y<17,fp=y==17&&prediction[v]<17;
            acc[fg?2:3]+=1;acc[fg?7:8]+=kernel;
            if(fp) {
                acc[4]+=1;acc[9]+=kernel;
                float contribution=weight*semantic[g*17+winner[v]];
                if(contribution>0) {
                    unsigned long long packed=(static_cast<unsigned long long>(__float_as_uint(contribution))<<32)
                        | static_cast<unsigned int>(0xffffffffu-static_cast<unsigned int>(g));
                    atomicMax(owners+v,packed);
                }
            }
            float dots[2]={0,0};
            for(int c=0;c<17;c++) {
                dots[0]+=semantic[g*17+c]*ce[v*17+c];
                if(fg)dots[1]+=semantic[g*17+c]*lovasz[v*17+c];
            }
            float local[3]={rot[0]*dx+rot[1]*dy+rot[2]*dz,rot[3]*dx+rot[4]*dy+rot[5]*dz,rot[6]*dx+rot[7]*dy+rot[8]*dz};
            float md[3]={-(inv[0]*dx+inv[3]*dy+inv[5]*dz),-(inv[3]*dx+inv[1]*dy+inv[4]*dz),-(inv[5]*dx+inv[4]*dy+inv[2]*dz)};
            for(int src=0;src<(fg?2:1);src++) {
                int group=src==1?2:(fg?1:0),base=10+group*9;
                float z=weight*dots[src];acc[base]+=kernel*dots[src];acc[base+8]+=z;
                for(int j=0;j<3;j++) {
                    float scale=scales[g*3+j];acc[base+1+j]+=z*local[j]*local[j]/(scale*scale*scale);
                    acc[base+4+j]+=z*md[j];
                }
                int target=assigned[g];
                if(target>=0)acc[base+7]+=weight*(src==0?ce[v*17+target]:lovasz[v*17+target])*target_sigmoid[g];
            }
        }
    }
    for(int offset=16;offset>0;offset/=2) {
        for(int j=0;j<37;j++)acc[j]+=__shfl_down_sync(0xffffffff,acc[j],offset);
        for(int j=0;j<2;j++)membership[j]|=__shfl_down_sync(0xffffffff,membership[j],offset);
    }
    if(lane==0) {
        for(int j=0;j<37;j++)stats[g*37+j]=acc[j];
        bits[g]=membership[0];bits[ng+g]=membership[1];
    }
}

__global__ void mass_kernel(int ng,int H,int W,int D,const float* means,const float* inverse,
    const float* alpha,const int* centers,const int* radii,const float* xyz,const bool* camera,float* result) {
    int g=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(g>=ng)return;
    float sum=0;
    if(alpha[g]>0) {
        const int* p=centers+g*3;int r=radii[g];
        int x0=max(0,p[0]-r),x1=min(H,p[0]+r+1),y0=max(0,p[1]-r),y1=min(W,p[1]+r+1),z0=max(0,p[2]-r),z1=min(D,p[2]+r+1),ny=y1-y0,nz=z1-z0;
        const float* inv=inverse+g*6;
        for(int k=lane;k<(x1-x0)*ny*nz;k+=32) {
            int v=((x0+k/(ny*nz))*W+y0+(k/nz)%ny)*D+z0+k%nz;if(!camera[v])continue;
            float dx=means[g*3]-xyz[v*3],dy=means[g*3+1]-xyz[v*3+1],dz=means[g*3+2]-xyz[v*3+2];
            float power=inv[0]*dx*dx+inv[1]*dy*dy+inv[2]*dz*dz;
            power=-.5f*power-(inv[3]*dx*dy+inv[4]*dy*dz+inv[5]*dx*dz);
            float kernel=expf(power);if(alpha[g]*kernel>0)sum+=kernel;
        }
    }
    for(int offset=16;offset>0;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0)result[g]=sum;
}

std::vector<torch::Tensor> probe(torch::Tensor means,torch::Tensor scales,torch::Tensor inverse,
    torch::Tensor rotation,torch::Tensor alpha,torch::Tensor semantic,torch::Tensor centers,
    torch::Tensor radii,torch::Tensor xyz,torch::Tensor labels,torch::Tensor camera,
    torch::Tensor prediction,torch::Tensor winner,torch::Tensor assigned,torch::Tensor target_sigmoid,
    torch::Tensor ce,torch::Tensor lovasz,int H,int W,int D) {
    const c10::cuda::CUDAGuard guard(means.device());int ng=means.size(0),nv=H*W*D;
    for(const auto& t:{means,scales,inverse,rotation,alpha,semantic,centers,radii,xyz,labels,camera,prediction,winner,assigned,target_sigmoid,ce,lovasz})
        TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==means.device(),"Contiguous tensors on one CUDA device required");
    TORCH_CHECK(means.scalar_type()==torch::kFloat32 && scales.scalar_type()==torch::kFloat32 && alpha.scalar_type()==torch::kFloat32 && ce.scalar_type()==torch::kFloat32,"FP32 required");
    TORCH_CHECK(labels.scalar_type()==torch::kInt32 && camera.scalar_type()==torch::kBool && centers.scalar_type()==torch::kInt32,"Invalid integer/mask types");
    TORCH_CHECK(means.numel()==ng*3&&scales.numel()==ng*3&&inverse.numel()==ng*6&&rotation.numel()==ng*9&&semantic.numel()==ng*17&&xyz.numel()==nv*3&&ce.numel()==nv*17&&lovasz.numel()==nv*17,"Invalid shapes");
    auto bits=torch::zeros({2,ng},means.options().dtype(torch::kInt32));
    auto stats=torch::zeros({ng,37},means.options());auto owners=torch::zeros({nv},means.options().dtype(torch::kInt64));
    probe_kernel<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ng,H,W,D,means.data_ptr<float>(),scales.data_ptr<float>(),inverse.data_ptr<float>(),rotation.data_ptr<float>(),alpha.data_ptr<float>(),semantic.data_ptr<float>(),centers.data_ptr<int>(),radii.data_ptr<int>(),xyz.data_ptr<float>(),labels.data_ptr<int>(),camera.data_ptr<bool>(),prediction.data_ptr<int>(),winner.data_ptr<int>(),assigned.data_ptr<int>(),target_sigmoid.data_ptr<float>(),ce.data_ptr<float>(),lovasz.data_ptr<float>(),bits.data_ptr<int>(),stats.data_ptr<float>(),reinterpret_cast<unsigned long long*>(owners.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {bits,stats,owners};
}

torch::Tensor mass(torch::Tensor means,torch::Tensor inverse,torch::Tensor alpha,torch::Tensor centers,
    torch::Tensor radii,torch::Tensor xyz,torch::Tensor camera,int H,int W,int D) {
    const c10::cuda::CUDAGuard guard(means.device());int ng=means.size(0);
    auto result=torch::zeros({ng},means.options());
    mass_kernel<<<(ng*32+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ng,H,W,D,means.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),centers.data_ptr<int>(),radii.data_ptr<int>(),xyz.data_ptr<float>(),camera.data_ptr<bool>(),result.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("probe",&probe);m.def("mass",&mass);}
