#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <THC/THCAtomics.cuh>


__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}


__global__ void deformable_aggregation_kernel(
    const int num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    float *output_ptr = output + idx;
    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride
        + groups_index
    );

    float result = 0;
    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        const float loc_w = sample_location[loc_offset];
        const float loc_h = sample_location[loc_offset + 1];
        
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;

                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const float *weights_ptr = (
                    weights + weight_offset + scale_index * num_groups
                    + weight_cam_stride * cam_index
                );
                result += bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_ptr) * *weights_ptr;
            }
        }
    }
    *output_ptr = result;
}


// Read the query gradient directly rather than materializing sum-backward's
// expanded [B, N*K, C] tensor. Point/channel parallelism stays unchanged.
template <bool SumPoints>
__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups,
    int points_per_query
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;
    const int grad_index = SumPoints
        ? (idx / num_embeds / points_per_query) * num_embeds + idx % num_embeds
        : idx;
    const float grad = grad_output[grad_index];

    const int channel_index = idx % num_embeds;
    const int groups_index = channel_index / (num_embeds / num_groups);
    idx /= num_embeds;
    const int pts_index = idx % num_pts;
    idx /= num_pts;
    const int batch_index = idx;

    const int value_cam_stride = num_feat * num_embeds;
    const int weight_cam_stride = num_scale * num_groups;
    int loc_offset = (batch_index * num_pts + pts_index) * num_cams << 1;
    int value_offset = batch_index * num_cams * value_cam_stride + channel_index;
    int weight_offset = (
        (batch_index * num_pts + pts_index) * num_cams * weight_cam_stride
        + groups_index
    );

    for (int cam_index = 0; cam_index < num_cams; ++cam_index, loc_offset += 2) {
        const float loc_w = sample_location[loc_offset];
        const float loc_h = sample_location[loc_offset + 1];
        
        if (loc_w > 0 && loc_w < 1 && loc_h > 0 && loc_h < 1) {
            for (int scale_index = 0; scale_index < num_scale; ++scale_index) {
                const int scale_offset = scale_start_index[scale_index] * num_embeds;

                const int spatial_shape_ptr = scale_index << 1;
                const int h = spatial_shape[spatial_shape_ptr];
                const int w = spatial_shape[spatial_shape_ptr + 1];

                const float h_im = loc_h * h - 0.5;
                const float w_im = loc_w * w - 0.5;

                const int value_ptr = value_offset + scale_offset + value_cam_stride * cam_index;
                const int weights_ptr = weight_offset + scale_index * num_groups + weight_cam_stride * cam_index;
                const float weight = weights[weights_ptr];

                float *grad_location_ptr = grad_sampling_location + loc_offset;
                float *grad_weights_ptr = grad_weights + weights_ptr;
                bilinear_sampling_grad(
                    mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
                    value_ptr,
                    grad,
                    grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
                );
            }
        }
    }
}


void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_pts,
    int num_groups
) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    if (num_kernels == 0) return;
    deformable_aggregation_kernel
        <<<(num_kernels + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_pts,
  int num_groups,
  int points_per_query
) {
    const int num_kernels = batch_size * num_pts * num_embeds;
    if (num_kernels == 0) return;
    if (points_per_query == 1) {
      deformable_aggregation_grad_kernel<false>
        <<<(num_kernels + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups, 1);
    } else {
      deformable_aggregation_grad_kernel<true>
        <<<(num_kernels + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups,
        points_per_query);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void deformable_aggregation_sum_kernel(
    int count, float* output, const float* feature, const int* shape,
    const int* starts, const float* location, const float* weights,
    int cameras, int feature_count, int channels, int levels, int points,
    int groups, int points_per_query) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int channel = index % channels;
    const int queries = points / points_per_query;
    const int query = (index / channels) % queries;
    const int batch = index / channels / queries;
    const int camera_stride = feature_count * channels;
    const int weight_camera_stride = levels * groups;
    const int group = channel / (channels / groups);
    float total = 0.f;
    for (int k = 0; k < points_per_query; ++k) {
        const int point = batch * points + query * points_per_query + k;
        float result = 0.f;
        for (int camera = 0; camera < cameras; ++camera) {
            const int offset = point * cameras + camera;
            const float x = location[offset * 2];
            const float y = location[offset * 2 + 1];
            if (x > 0 && x < 1 && y > 0 && y < 1) {
                for (int level = 0; level < levels; ++level) {
                    const int h = shape[level * 2];
                    const int w = shape[level * 2 + 1];
                    const int base = (batch * cameras + camera) * camera_stride
                                     + starts[level] * channels + channel;
                    const float weight = weights[offset * weight_camera_stride
                                                 + level * groups + group];
                    const float yy = y * h - .5f;
                    const float xx = x * w - .5f;
                    result += bilinear_sampling(feature, h, w, channels, yy, xx, base) * weight;
                }
            }
        }
        total += result;
    }
    output[index] = total;
}


void deformable_aggregation_sum(
    float* output, const float* feature, const int* shape, const int* starts,
    const float* location, const float* weights, int batches, int cameras,
    int feature_count, int channels, int levels, int points, int groups,
    int points_per_query) {
    const int count = batches * (points / points_per_query) * channels;
    if (count == 0) return;
    deformable_aggregation_sum_kernel<<<(count + 255) / 256, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        count, output, feature, shape, starts, location, weights, cameras,
        feature_count, channels, levels, points, groups, points_per_query);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__device__ __forceinline__ float weight_warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ float weight_warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return __shfl_sync(0xffffffff, value, 0);
}

// Each warp normalizes one query/group over K*V*L. The common 216-element
// case fits eight values per lane, so logits are read just once.
template<bool Small>
__global__ void masked_weight_softmax_kernel(int rows, int length, int groups,
    const float* logits, const bool* mask, float* output) {
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int base = (row / groups) * length * groups + row % groups;
    float values[8];
    float maximum = -CUDART_INF_F;
    bool any = false;
    if (Small) {
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int item = lane + slot * 32;
            const bool valid = item < length && mask[base + item * groups];
            values[slot] = valid ? logits[base + item * groups] : -CUDART_INF_F;
            maximum = fmaxf(maximum, values[slot]);
            any |= valid;
        }
    } else {
        for (int item = lane; item < length; item += 32) {
            const bool valid = mask[base + item * groups];
            maximum = fmaxf(maximum, valid ? logits[base + item * groups] : -CUDART_INF_F);
            any |= valid;
        }
    }
    const bool valid_row = __any_sync(0xffffffff, any);
    if (!valid_row) {
        for (int item = lane; item < length; item += 32)
            output[base + item * groups] = 0.f;
        return;
    }
    maximum = weight_warp_max(maximum);
    float total = 0.f;
    if (Small) {
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            values[slot] = lane + slot * 32 < length ? expf(values[slot] - maximum) : 0.f;
            total += values[slot];
        }
    } else {
        for (int item = lane; item < length; item += 32)
            total += expf((mask[base + item * groups] ? logits[base + item * groups]
                          : -CUDART_INF_F) - maximum);
    }
    const float reciprocal = 1.f / weight_warp_sum(total);
    if (Small) {
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int item = lane + slot * 32;
            if (item < length) output[base + item * groups] = values[slot] * reciprocal;
        }
    } else {
        for (int item = lane; item < length; item += 32)
            output[base + item * groups] = expf((mask[base + item * groups]
                ? logits[base + item * groups] : -CUDART_INF_F) - maximum) * reciprocal;
    }
}

__global__ void masked_weight_softmax_grad_kernel(int rows, int length, int groups,
    const float* probabilities, const float* grad, float* grad_logits) {
    const int lane = threadIdx.x % 32;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= rows) return;
    const int base = (row / groups) * length * groups + row % groups;
    float dot = 0.f;
    for (int item = lane; item < length; item += 32) {
        const int index = base + item * groups;
        dot += probabilities[index] * grad[index];
    }
    dot = weight_warp_sum(dot);
    for (int item = lane; item < length; item += 32) {
        const int index = base + item * groups;
        grad_logits[index] = probabilities[index] * (grad[index] - dot);
    }
}

at::Tensor masked_weight_softmax_forward_cuda(const at::Tensor& logits, const at::Tensor& mask) {
    auto output = at::empty_like(logits);
    const int rows = logits.size(0) * logits.size(1) * logits.size(5);
    const int length = logits.size(2) * logits.size(3) * logits.size(4);
    if (rows) {
        if (length <= 256)
            masked_weight_softmax_kernel<true><<<(rows+3)/4, 128, 0,
                at::cuda::getCurrentCUDAStream()>>>(rows, length, logits.size(5),
                logits.data_ptr<float>(), mask.data_ptr<bool>(), output.data_ptr<float>());
        else
            masked_weight_softmax_kernel<false><<<(rows+3)/4, 128, 0,
                at::cuda::getCurrentCUDAStream()>>>(rows, length, logits.size(5),
                logits.data_ptr<float>(), mask.data_ptr<bool>(), output.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

at::Tensor masked_weight_softmax_backward_cuda(const at::Tensor& probabilities, const at::Tensor& grad) {
    auto output = at::empty_like(probabilities);
    const int rows = probabilities.size(0) * probabilities.size(1) * probabilities.size(5);
    const int length = probabilities.size(2) * probabilities.size(3) * probabilities.size(4);
    if (rows) {
        masked_weight_softmax_grad_kernel<<<(rows+3)/4, 128, 0,
            at::cuda::getCurrentCUDAStream()>>>(rows, length, probabilities.size(5),
            probabilities.data_ptr<float>(), grad.data_ptr<float>(), output.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
