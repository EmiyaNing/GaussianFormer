"""Phase-B fused CUDA local aggregation head for semantic Gaussians.

This module is intentionally separate from the Phase-A reference head.  Both
heads predict identical point-anchored Gaussian attributes, but this module
dispatches their local semantic aggregation to the custom sparse CUDA operator
instead of Phase A's Python KNN/gather implementation.
"""
from mmengine.registry import MODELS

from .opus_gaussian_residual_head import OPUSGaussianResidualHead
from .point_anchored_semantic_gaussian import PointAnchoredSemanticGaussianCUDA


_PointAnchoredGaussianLocalAggCUDA = PointAnchoredSemanticGaussianCUDA


@MODELS.register_module()
class OPUSSemanticGaussianLocalAggCUDAHead(OPUSGaussianResidualHead):
    """Phase-B head using custom sparse CUDA local aggregation.

    The parent head still owns the OPUS output contract, Gaussian auxiliary
    loss inputs, and the three evaluation modes.  This class replaces only
    the local residual aggregator, keeping Phase-A model semantics intact.
    """
    def __init__(self, gaussian_scale_range=((.1, .1, .1), (.6, .6, .6)),
                 gaussian_initial_scale=.35, gaussian_initial_opacity=.02,
                 gaussian_num_neighbors=8, gaussian_include_self=True,
                 gaussian_query_chunk_size=256, gaussian_cell_size=1.8,
                 gaussian_support_sigma=3.0, gaussian_scale_eps=1e-4,
                 gaussian_denom_eps=1e-6, ensemble_gamma=1.,
                 eval_mode='ensemble', **kwargs):
        super().__init__(
            gaussian_scale_range=gaussian_scale_range,
            gaussian_initial_scale=gaussian_initial_scale,
            gaussian_initial_opacity=gaussian_initial_opacity,
            gaussian_num_neighbors=gaussian_num_neighbors,
            gaussian_include_self=gaussian_include_self,
            gaussian_query_chunk_size=gaussian_query_chunk_size,
            ensemble_gamma=ensemble_gamma, eval_mode=eval_mode, **kwargs)
        # ``gaussian_num_neighbors`` remains accepted for config compatibility
        # with Phase A, but localagg uses scale-bounded spatial support rather
        # than a global fixed-K search.
        self.gaussian = _PointAnchoredGaussianLocalAggCUDA(
            self.embed_dims, self.point_multipliers[-1], self.num_classes,
            self.pc_range, gaussian_scale_range, gaussian_initial_scale,
            gaussian_initial_opacity, gaussian_num_neighbors,
            gaussian_include_self, gaussian_query_chunk_size,
            cell_size=gaussian_cell_size,
            support_sigma=gaussian_support_sigma, scale_eps=gaussian_scale_eps,
            denom_eps=gaussian_denom_eps)
