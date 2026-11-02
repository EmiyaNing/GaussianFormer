#include <torch/extension.h>

at::Tensor pack_pyramid_forward(const std::vector<at::Tensor>&);
std::vector<at::Tensor> pack_pyramid_backward(
    const at::Tensor&, const std::vector<std::vector<int64_t>>&,
    const std::vector<std::vector<int64_t>>&);
at::Tensor packed_dropout(const at::Tensor&, int64_t, double);
std::vector<at::Tensor> temporal_sampling_forward(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const std::vector<at::Tensor>&);
std::vector<at::Tensor> temporal_sampling_backward(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const std::vector<at::Tensor>&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, bool, bool, bool,
    const std::vector<bool>&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_pyramid_forward", &pack_pyramid_forward);
  m.def("pack_pyramid_backward", &pack_pyramid_backward);
  m.def("packed_dropout", &packed_dropout);
  m.def("temporal_sampling_forward", &temporal_sampling_forward);
  m.def("temporal_sampling_backward", &temporal_sampling_backward);
}
