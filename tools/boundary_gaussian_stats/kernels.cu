// Read-only diagnostics for the additive localagg renderer. No model mutation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ float influence(int g, int v, const float* means, const float* xyz,
                           const float* inverse, const float* alpha, float& power) {
    const float dx = means[g*3] - xyz[v*3];
    const float dy = means[g*3+1] - xyz[v*3+1];
    const float dz = means[g*3+2] - xyz[v*3+2];
    const float* cov = inverse + g*6;
    power = cov[0]*dx*dx + cov[1]*dy*dy + cov[2]*dz*dz;
    power = -0.5f*power - (cov[3]*dx*dy + cov[4]*dy*dz + cov[5]*dx*dz);
    return alpha[g] * expf(power);
}

__device__ int boundary_flags(unsigned int labels) {
    const unsigned int foreground = labels & ((1u<<17)-1);
    return (__popc(labels) >= 2) | ((__popc(foreground) >= 2)<<1)
        | (((foreground != 0) && ((labels & (1u<<17)) != 0))<<2);
}

// One warp/Gaussian, parallel enumeration of the EXACT inclusive integer box.
__global__ void classify(int ng, int H, int W, int D, float step, float multiplier,
                         float px, float py, float pz, const float* means,
                         const float* scales, const float* inverse, const float* alpha,
                         const float* xyz, const int* labels, const bool* camera,
                         int* class_bits, int* flags) {
    const int g = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const int lane = threadIdx.x%32;
    if (g >= ng) return;
    const int cx = int((means[g*3]-px)/step);
    const int cy = int((means[g*3+1]-py)/step);
    const int cz = int((means[g*3+2]-pz)/step);
    const float scale = fmaxf(scales[g*3], fmaxf(scales[g*3+1], scales[g*3+2]));
    const int r = int(ceilf(scale*multiplier/step));
    const int x0=max(0,cx-r), x1=min(H,cx+r+1);
    const int y0=max(0,cy-r), y1=min(W,cy+r+1);
    const int z0=max(0,cz-r), z1=min(D,cz+r+1);
    const int ny=y1-y0, nz=z1-z0;
    unsigned int bits[4] = {0,0,0,0};
    for (int k=lane; k<(x1-x0)*ny*nz; k+=32) {
        const int x=x0+k/(ny*nz), y=y0+(k/nz)%ny, z=z0+k%nz;
        const int v=(x*W+y)*D+z, label=labels[v];
        if (label<0 || label>=18) continue;
        float power;
        const float weight=influence(g,v,means,xyz,inverse,alpha,power);
        if (!(weight>0)) continue; // Numerical zero cannot affect rendered logits.
        const unsigned int bit=1u<<label;
        bits[0] |= bit;
        if (camera[v]) bits[1] |= bit;
        if (power>=-4.5f) {
            bits[2] |= bit;
            if (camera[v]) bits[3] |= bit;
        }
    }
    for (int offset=16; offset>0; offset/=2)
        for (int s=0; s<4; ++s) bits[s] |= __shfl_down_sync(0xffffffff,bits[s],offset);
    if (lane==0) {
        int value=0;
        for (int s=0; s<4; ++s) {
            class_bits[s*ng+g]=bits[s];
            value |= boundary_flags(bits[s]) << (s*3);
        }
        flags[g]=value;
    }
}

__global__ void attribute(int ng, int H, int W, int D, float step, float multiplier,
                          float px, float py, float pz, const float* means,
                          const float* scales, const float* inverse, const float* alpha,
                          const float* semantics, const float* xyz, const int* labels,
                          const int* winner, const int* flags, int* unions,
                          unsigned long long* owners) {
    const int g=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if (g>=ng) return;
    const int cx=int((means[g*3]-px)/step), cy=int((means[g*3+1]-py)/step), cz=int((means[g*3+2]-pz)/step);
    const float scale=fmaxf(scales[g*3], fmaxf(scales[g*3+1],scales[g*3+2]));
    const int r=int(ceilf(scale*multiplier/step));
    const int x0=max(0,cx-r), x1=min(H,cx+r+1), y0=max(0,cy-r), y1=min(W,cy+r+1), z0=max(0,cz-r), z1=min(D,cz+r+1);
    const int ny=y1-y0, nz=z1-z0, nv=H*W*D;
    for (int k=lane; k<(x1-x0)*ny*nz; k+=32) {
        const int x=x0+k/(ny*nz), y=y0+(k/nz)%ny, z=z0+k%nz, v=(x*W+y)*D+z;
        if (labels[v]<0 || labels[v]>=18) continue;
        float power;
        const float weight=influence(g,v,means,xyz,inverse,alpha,power);
        if (!(weight>0)) continue;
        const bool core=power>=-4.5f;
        const int selected=flags[g] & (core ? 4095 : 63);
        if (selected) atomicOr(unions+v, selected);
        const float contribution=weight*semantics[g*17+winner[v]];
        if (!(contribution>0)) continue;
        // Positive IEEE floats have monotone bit order; ties choose smaller G id.
        const unsigned long long packed=(static_cast<unsigned long long>(__float_as_uint(contribution))<<32)
            | static_cast<unsigned int>(0xffffffffu-static_cast<unsigned int>(g));
        atomicMax(owners+v, packed);
        if (core) atomicMax(owners+nv+v, packed);
    }
}

// Five mutually exclusive outcomes: correct foreground, wrong class, miss, FP, TN.
// Each scope has 9 cohorts: all, union x3, dominant x3, nonboundary dominant, unowned.
__global__ void count_errors(int nv, const int* labels, const int* prediction,
                             const bool* camera, const int* flags, const int* unions,
                             const unsigned long long* owners, long long* counts) {
    __shared__ unsigned int local[4*9*5];
    for (int j=threadIdx.x; j<180; j+=blockDim.x) local[j]=0;
    __syncthreads();
    const int v=blockIdx.x*blockDim.x+threadIdx.x;
    if (v<nv && labels[v]>=0 && labels[v]<18) {
        const int y=labels[v], pred=prediction[v];
        const int code=y==17 ? (pred==17 ? 4 : 3) : (pred==17 ? 2 : (pred==y ? 0 : 1));
        for (int s=0; s<4; ++s) {
            if ((s%2) && !camera[v]) continue;
            const auto packed=owners[(s/2)*nv+v];
            const int owner_flags=packed ? flags[0xffffffffu-static_cast<unsigned int>(packed)] : 0;
            const int owned=(owner_flags>>(s*3))&7, covered=(unions[v]>>(s*3))&7;
            const int base=s*45+code;
            atomicAdd(local+base, 1u);
            for (int t=0;t<3;++t) {
                if (covered&(1<<t)) atomicAdd(local+base+(t+1)*5,1u);
                if (owned&(1<<t)) atomicAdd(local+base+(t+4)*5,1u);
            }
            if (packed && !(owned&1)) atomicAdd(local+base+7*5,1u);
            if (!packed) atomicAdd(local+base+8*5,1u);
        }
    }
    __syncthreads();
    for (int j=threadIdx.x; j<180; j+=blockDim.x)
        atomicAdd(reinterpret_cast<unsigned long long*>(counts+j),static_cast<unsigned long long>(local[j]));
}

std::vector<torch::Tensor> diagnose(torch::Tensor means, torch::Tensor scales,
        torch::Tensor inverse, torch::Tensor alpha, torch::Tensor semantics,
        torch::Tensor xyz, torch::Tensor labels, torch::Tensor camera,
        torch::Tensor winner, torch::Tensor prediction,
        int H, int W, int D, double step, double multiplier,
        double px, double py, double pz) {
    const c10::cuda::CUDAGuard guard(means.device());
    const int ng=means.size(0), nv=H*W*D;
    for (const auto& t : {means,scales,inverse,alpha,semantics,xyz,labels,camera,winner,prediction}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==means.device(), "CUDA contiguous inputs on one GPU required");
    }
    TORCH_CHECK(means.scalar_type()==torch::kFloat32 && scales.scalar_type()==torch::kFloat32 && inverse.scalar_type()==torch::kFloat32 && alpha.scalar_type()==torch::kFloat32 && semantics.scalar_type()==torch::kFloat32 && xyz.scalar_type()==torch::kFloat32,"FP32 inputs required");
    TORCH_CHECK(labels.scalar_type()==torch::kInt32 && winner.scalar_type()==torch::kInt32 && prediction.scalar_type()==torch::kInt32 && camera.scalar_type()==torch::kBool,"label int32/camera bool required");
    TORCH_CHECK(means.numel()==ng*3 && scales.numel()==ng*3 && inverse.numel()==ng*6 && alpha.numel()==ng && semantics.numel()==ng*17 && xyz.numel()==nv*3 && labels.numel()==nv && camera.numel()==nv && winner.numel()==nv && prediction.numel()==nv,"Invalid shapes");
    auto integers=means.options().dtype(torch::kInt32), longs=means.options().dtype(torch::kInt64);
    auto bits=torch::zeros({4,ng},integers), flags=torch::zeros({ng},integers);
    auto unions=torch::zeros({nv},integers), owners=torch::zeros({2,nv},longs);
    auto counts=torch::zeros({4,9,5},longs);
    const auto stream=at::cuda::getCurrentCUDAStream();
    classify<<<(ng*32+255)/256,256,0,stream>>>(ng,H,W,D,step,multiplier,px,py,pz,means.data_ptr<float>(),scales.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),xyz.data_ptr<float>(),labels.data_ptr<int>(),camera.data_ptr<bool>(),bits.data_ptr<int>(),flags.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attribute<<<(ng*32+255)/256,256,0,stream>>>(ng,H,W,D,step,multiplier,px,py,pz,means.data_ptr<float>(),scales.data_ptr<float>(),inverse.data_ptr<float>(),alpha.data_ptr<float>(),semantics.data_ptr<float>(),xyz.data_ptr<float>(),labels.data_ptr<int>(),winner.data_ptr<int>(),flags.data_ptr<int>(),unions.data_ptr<int>(),reinterpret_cast<unsigned long long*>(owners.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    count_errors<<<(nv+255)/256,256,0,stream>>>(nv,labels.data_ptr<int>(),prediction.data_ptr<int>(),camera.data_ptr<bool>(),flags.data_ptr<int>(),unions.data_ptr<int>(),reinterpret_cast<unsigned long long*>(owners.data_ptr<int64_t>()),reinterpret_cast<long long*>(counts.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {bits,flags,unions,owners,counts};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("diagnose", &diagnose); }
