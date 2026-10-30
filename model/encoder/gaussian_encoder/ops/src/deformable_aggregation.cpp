#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

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
);
  

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
);


at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_cams = _mc_ms_feat.size(1);
  int num_feat = _mc_ms_feat.size(2);
  int num_embeds = _mc_ms_feat.size(3);
  int num_scale = _spatial_shape.size(0);
  int num_pts = _sampling_location.size(1);
  int num_groups = _weights.size(4);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output = at::zeros({batch_size, num_pts, num_embeds}, _mc_ms_feat.options());
  deformable_aggregation(
    output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups
  );
  return output;
}

void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_cams = _mc_ms_feat.size(1);
  int num_feat = _mc_ms_feat.size(2);
  int num_embeds = _mc_ms_feat.size(3);
  int num_scale = _spatial_shape.size(0);
  int num_pts = _sampling_location.size(1);
  int num_groups = _weights.size(4);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>();

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_pts, num_groups, 1
  );
}


void deformable_aggregation_sum(
    float*, const float*, const int*, const int*, const float*, const float*,
    int, int, int, int, int, int, int, int);

void check_sum_inputs(const at::Tensor& feature, const at::Tensor& shape,
                      const at::Tensor& starts, const at::Tensor& location,
                      const at::Tensor& weights, int64_t points_per_query) {
  TORCH_CHECK(feature.is_cuda() && feature.scalar_type() == at::kFloat &&
              feature.dim() == 4 && feature.is_contiguous(), "DAF features must be contiguous CUDA float32 [B,V,S,C]");
  for (const auto& tensor : {shape, starts, location, weights}) {
    TORCH_CHECK(tensor.device() == feature.device() && tensor.is_contiguous(),
                "DAF tensors must be contiguous and on the same CUDA device");
  }
  TORCH_CHECK(shape.scalar_type() == at::kInt && starts.scalar_type() == at::kInt &&
              shape.dim() == 2 && shape.size(1) == 2 && starts.dim() == 1 &&
              starts.size(0) == shape.size(0), "Invalid DAF pyramid metadata");
  TORCH_CHECK(location.scalar_type() == at::kFloat && location.dim() == 4 &&
              location.size(0) == feature.size(0) && location.size(2) == feature.size(1) &&
              location.size(3) == 2, "Invalid DAF sampling locations");
  TORCH_CHECK(points_per_query > 0 && location.size(1) % points_per_query == 0,
              "DAF point count must be divisible by points_per_query");
  TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.dim() == 5 &&
              weights.size(0) == feature.size(0) && weights.size(1) == location.size(1) &&
              weights.size(2) == feature.size(1) && weights.size(3) == shape.size(0) &&
              weights.size(4) > 0 && feature.size(3) % weights.size(4) == 0,
              "Invalid DAF weights or channel groups");
}

at::Tensor deformable_aggregation_sum_forward(
    const at::Tensor& feature, const at::Tensor& shape, const at::Tensor& starts,
    const at::Tensor& location, const at::Tensor& weights, int64_t points_per_query) {
  check_sum_inputs(feature, shape, starts, location, weights, points_per_query);
  const at::cuda::OptionalCUDAGuard guard(device_of(feature));
  auto output = at::empty({feature.size(0), location.size(1) / points_per_query,
                           feature.size(3)}, feature.options());
  deformable_aggregation_sum(
      output.data_ptr<float>(), feature.data_ptr<float>(), shape.data_ptr<int>(),
      starts.data_ptr<int>(), location.data_ptr<float>(), weights.data_ptr<float>(),
      feature.size(0), feature.size(1), feature.size(2), feature.size(3),
      shape.size(0), location.size(1), weights.size(4), points_per_query);
  return output;
}

void deformable_aggregation_sum_backward(
    const at::Tensor& feature, const at::Tensor& shape, const at::Tensor& starts,
    const at::Tensor& location, const at::Tensor& weights, const at::Tensor& grad,
    at::Tensor& grad_feature, at::Tensor& grad_location, at::Tensor& grad_weights,
    int64_t points_per_query) {
  const at::cuda::OptionalCUDAGuard guard(device_of(feature));
  deformable_aggregation_grad(
      feature.data_ptr<float>(), shape.data_ptr<int>(), starts.data_ptr<int>(),
      location.data_ptr<float>(), weights.data_ptr<float>(), grad.data_ptr<float>(),
      grad_feature.data_ptr<float>(), grad_location.data_ptr<float>(),
      grad_weights.data_ptr<float>(), feature.size(0), feature.size(1), feature.size(2),
      feature.size(3), shape.size(0), location.size(1), weights.size(4), points_per_query);
}


at::Tensor masked_weight_softmax_forward_cuda(const at::Tensor&, const at::Tensor&);
at::Tensor masked_weight_softmax_backward_cuda(const at::Tensor&, const at::Tensor&);

at::Tensor masked_weight_softmax_forward(const at::Tensor& logits, const at::Tensor& mask) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat &&
              logits.is_contiguous() && logits.dim() == 6 && logits.size(5) > 0 &&
              logits.size(2) * logits.size(3) * logits.size(4) > 0,
              "Masked DAF softmax expects contiguous CUDA float32 [B,N,K,V,L,G]");
  TORCH_CHECK(mask.device() == logits.device() && mask.scalar_type() == at::kBool &&
              mask.is_contiguous() && mask.sizes() == logits.sizes(),
              "DAF softmax mask must be bool and match logits");
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  return masked_weight_softmax_forward_cuda(logits, mask);
}

at::Tensor masked_weight_softmax_backward(const at::Tensor& probabilities, const at::Tensor& grad) {
  TORCH_CHECK(grad.device() == probabilities.device() && grad.scalar_type() == at::kFloat &&
              grad.is_contiguous() && grad.sizes() == probabilities.sizes(),
              "DAF softmax gradient must be contiguous float32 and match probabilities");
  const at::cuda::OptionalCUDAGuard guard(device_of(probabilities));
  return masked_weight_softmax_backward_cuda(probabilities, grad);
}

at::Tensor image_affine_forward(const at::Tensor&, const at::Tensor&, const at::Tensor&, bool);
at::Tensor image_affine_backward(const at::Tensor&, const at::Tensor&, bool);
std::vector<at::Tensor> normalized_sampling_forward(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> normalized_sampling_backward(const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("image_affine_forward", &image_affine_forward);
  m.def("image_affine_backward", &image_affine_backward);
  m.def("normalized_sampling_forward", &normalized_sampling_forward);
  m.def("normalized_sampling_backward", &normalized_sampling_backward);
  m.def("masked_weight_softmax_forward", &masked_weight_softmax_forward);
  m.def("masked_weight_softmax_backward", &masked_weight_softmax_backward);
  m.def("deformable_aggregation_sum_forward", &deformable_aggregation_sum_forward);
  m.def("deformable_aggregation_sum_backward", &deformable_aggregation_sum_backward);
  m.def(
    "deformable_aggregation_forward",
    &deformable_aggregation_forward,
    "deformable_aggregation_forward"
  );
  m.def(
    "deformable_aggregation_backward",
    &deformable_aggregation_backward,
    "deformable_aggregation_backward"
  );
}
