// Reuses GaussianFormer's local aggregation math; saves only backward indices.
// Retains the license terms in the vendored GraphDeco/Inria source headers.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "aggregator.h"
#include "aggregator_impl.h"
#include "backward.h"
#include "config.h"

using torch::Tensor;

static std::function<char*(size_t)> resize_buffer(Tensor& tensor) {
    return [&tensor](size_t count) {
        tensor.resize_({static_cast<int64_t>(count)});
        return reinterpret_cast<char*>(tensor.data_ptr<uint8_t>());
    };
}

std::vector<Tensor> compact_forward(
    Tensor pts, Tensor points_int, Tensor means, Tensor means_int,
    Tensor opacity, Tensor semantics, Tensor radii, Tensor covariance,
    int H, int W, int D, bool save_backward) {
    const c10::cuda::CUDAGuard guard(means.device());
    const int P = means.size(0), N = pts.size(0);
    auto logits = torch::zeros({N, NUM_CHANNELS}, means.options());
    auto indices = torch::empty({0}, means.options().dtype(torch::kInt32));
    if (P == 0 || N == 0) return {logits, indices, indices};
    auto geometry = torch::empty({0}, means.options().dtype(torch::kUInt8));
    auto binning = torch::empty_like(geometry);
    auto image = torch::empty_like(geometry);
    const int R = LocalAggregator::Aggregator::forward(
        resize_buffer(geometry), resize_buffer(binning), resize_buffer(image),
        P, N, pts.data_ptr<float>(), points_int.data_ptr<int>(),
        means.data_ptr<float>(), means_int.data_ptr<int>(),
        opacity.data_ptr<float>(), semantics.data_ptr<float>(),
        covariance.data_ptr<float>(), radii.data_ptr<int>(), H, W, D,
        logits.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!save_backward) return {logits, indices, indices};
    char* geom_ptr = reinterpret_cast<char*>(geometry.data_ptr<uint8_t>());
    char* bin_ptr = reinterpret_cast<char*>(binning.data_ptr<uint8_t>());
    auto geom = LocalAggregator::GeometryState::fromChunk(geom_ptr, P);
    auto bins = LocalAggregator::BinningState::fromChunk(bin_ptr, R);
    // Clone into independent storage: a view would retain the full scratch.
    auto offsets = torch::from_blob(geom.point_offsets, {P}, means.options().dtype(torch::kInt32)).clone();
    auto keys = torch::from_blob(bins.point_list_keys_unsorted, {R}, means.options().dtype(torch::kInt32)).clone();
    return {logits, offsets, keys};
}

std::vector<Tensor> compact_backward(
    Tensor offsets, Tensor keys, Tensor pts, Tensor points_int,
    Tensor means, Tensor opacity, Tensor semantics, Tensor covariance,
    Tensor out_grad, int H, int W, int D) {
    const c10::cuda::CUDAGuard guard(means.device());
    const int P = means.size(0), N = pts.size(0);
    auto means_grad = torch::zeros_like(means);
    auto opacity_grad = torch::zeros_like(opacity);
    auto semantics_grad = torch::zeros_like(semantics);
    auto covariance_grad = torch::zeros_like(covariance);
    if (P == 0 || N == 0) return {means_grad, opacity_grad, semantics_grad, covariance_grad};
    auto voxel2pts = torch::full({H * W * D}, -1, points_int.options());
    BACKWARD::preprocess(N, points_int.data_ptr<int>(), dim3(H, W, D), voxel2pts.data_ptr<int>());
    BACKWARD::render(
        P, reinterpret_cast<uint32_t*>(offsets.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(keys.data_ptr<int>()), voxel2pts.data_ptr<int>(),
        pts.data_ptr<float>(), means.data_ptr<float>(), covariance.data_ptr<float>(),
        opacity.data_ptr<float>(), semantics.data_ptr<float>(), out_grad.data_ptr<float>(),
        means_grad.data_ptr<float>(), opacity_grad.data_ptr<float>(),
        semantics_grad.data_ptr<float>(), covariance_grad.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {means_grad, opacity_grad, semantics_grad, covariance_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &compact_forward);
    module.def("backward", &compact_backward);
}
