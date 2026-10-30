#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

using at::Tensor;
std::vector<Tensor> query_map_cuda(const Tensor&, int, int, int);
std::vector<Tensor> tile_bins_cuda(const Tensor&, const Tensor&, int, int, int, int, int);
Tensor render_cuda(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                   const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                   const Tensor&, const Tensor&, int, int, int, int);
std::vector<Tensor> backward_cuda(const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, int, int, int);

void check(const Tensor& tensor, at::ScalarType type, const at::Device& device) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                tensor.scalar_type() == type && tensor.is_contiguous(),
                "React render tensors must be contiguous CUDA tensors with matching dtype/device");
}

std::vector<Tensor> query_map(const Tensor& cells, int h, int w, int d) {
    check(cells, at::kInt, cells.device());
    TORCH_CHECK(cells.dim() == 2 && cells.size(1) == 3 && h > 0 && w > 0 && d > 0,
                "Invalid React query grid");
    c10::cuda::CUDAGuard guard(cells.device());
    return query_map_cuda(cells, h, w, d);
}

std::vector<Tensor> tile_bins(const Tensor& centers, const Tensor& radii,
                             int h, int w, int d, int tile, int capacity) {
    check(centers, at::kInt, centers.device());
    check(radii, at::kInt, centers.device());
    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3 &&
                radii.sizes() == centers.sizes() && h > 0 && w > 0 && d > 0 &&
                tile > 0 && capacity >= 0, "Invalid React tile geometry");
    c10::cuda::CUDAGuard guard(centers.device());
    return tile_bins_cuda(centers, radii, h, w, d, tile, capacity);
}

Tensor render(const Tensor& points, const Tensor& cells, const Tensor& means,
              const Tensor& centers, const Tensor& radii, const Tensor& opacity,
              const Tensor& semantics, const Tensor& inverse, const Tensor& offsets,
              const Tensor& ids, int h, int w, int d, int tile) {
    for (const auto& value : {points, means, opacity, semantics, inverse})
        check(value, at::kFloat, points.device());
    for (const auto& value : {cells, centers, radii, ids})
        check(value, at::kInt, points.device());
    check(offsets, at::kLong, points.device());
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3 && cells.sizes() == points.sizes(),
                "Invalid React queries");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3 && centers.sizes() == means.sizes() &&
                radii.sizes() == means.sizes() && opacity.numel() == means.size(0) &&
                semantics.dim() == 2 && semantics.size(0) == means.size(0) &&
                semantics.size(1) == 18 && inverse.dim() == 2 &&
                inverse.size(0) == means.size(0) && inverse.size(1) == 6,
                "React tiled renderer requires 18 semantic channels and packed inverse [G,6]");
    TORCH_CHECK(h > 0 && w > 0 && d > 0 && tile > 0 &&
                offsets.numel() == ((h + tile - 1) / tile) *
                                  ((w + tile - 1) / tile) * ((d + tile - 1) / tile) + 1,
                "Invalid React tile offsets");
    c10::cuda::CUDAGuard guard(points.device());
    return render_cuda(points, cells, means, centers, radii, opacity, semantics,
                       inverse, offsets, ids, h, w, d, tile);
}

std::vector<Tensor> backward(const Tensor& points, const Tensor& means,
    const Tensor& centers, const Tensor& radii, const Tensor& opacity,
    const Tensor& semantics, const Tensor& inverse, const Tensor& heads,
    const Tensor& next, const Tensor& grad, int h, int w, int d) {
    c10::cuda::CUDAGuard guard(points.device());
    return backward_cuda(points, means, centers, radii, opacity, semantics,
                          inverse, heads, next, grad, h, w, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query_map", &query_map);
    m.def("tile_bins", &tile_bins);
    m.def("render", &render);
    m.def("backward", &backward);
}
