#include <torch/extension.h>

#include "semantic_gaussian_localagg.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &SemanticGaussianLocalAggForward,
        "Semantic Gaussian local aggregation forward (CUDA)");
  m.def("backward", &SemanticGaussianLocalAggBackward,
        "Semantic Gaussian local aggregation backward (CUDA)");
}
