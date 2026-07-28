"""Phase-B fused CUDA local aggregation head for semantic Gaussians.

This module is intentionally separate from the Phase-A reference head.  Both
heads predict identical point-anchored Gaussian attributes, but this module
dispatches their local semantic aggregation to the custom sparse CUDA operator
instead of Phase A's Python KNN/gather implementation.
"""
import torch
from mmengine.registry import MODELS

from model.ops.semantic_gaussian_localagg import (
    CUDA_LOCALAGG_AVAILABLE, semantic_gaussian_localagg)
from .opus_gaussian_residual_head import (
    OPUSGaussianResidualHead, _PointAnchoredGaussianResidual)


class _PointAnchoredGaussianLocalAggCUDA(_PointAnchoredGaussianResidual):
    """Phase-A attribute heads coupled to the fused CUDA localagg operator.

    Inherited learnable attributes:
        slot_embedding: Distinguishes final OPUS child slots.
        position_encoder: Encodes each final child point position.
        trunk: Produces shared Gaussian attribute features.
        scale_head: Predicts axis-aligned Gaussian scale.
        opacity_head: Predicts Gaussian opacity.
        residual_head: Predicts per-Gaussian semantic residual seed.

    This class deliberately has no center or rotation head: every Gaussian
    center is exactly its OPUS final point and its covariance is diagonal.
    """
    def __init__(self, *args, pc_range, cell_size, support_sigma=3.0,
                 scale_eps=1e-4, denom_eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) < 5:
            raise ValueError('scale_range must be provided as the fifth Gaussian constructor argument')
        scale_range = args[4]
        self.pc_range = tuple(pc_range)
        self.cell_size = float(cell_size)
        self.support_sigma = float(support_sigma)
        self.scale_eps = float(scale_eps)
        self.denom_eps = float(denom_eps)
        self.max_scale = float(max(scale_range[1]))

    def forward(self, token, points, valid_mask):
        """Predict attributes and aggregate residuals with custom CUDA localagg.

        Args:
            token: Final OPUS query feature, float tensor ``[B,Q,D]``.
            points: Final OPUS child point in metres, float tensor ``[B,Q,R,3]``.
            valid_mask: Valid query indicator, bool tensor ``[B,Q]``.

        Returns:
            A dictionary matching Phase A: flattened ``points [B,G,3]``,
            ``scales [B,G,3]``, ``opacities [B,G,1]``, ``residual [B,G,C]``
            and ``valid_mask [B,G]`` where ``G=Q*R``.
        """
        if not points.is_cuda or not CUDA_LOCALAGG_AVAILABLE:
            raise RuntimeError(
                'OPUSSemanticGaussianLocalAggCUDAHead requires the compiled CUDA '
                'extension; use OPUSGaussianResidualHead for Phase A.')
        batch, queries, children, _ = points.shape
        if children != self.num_children:
            raise ValueError('final OPUS child count does not match Gaussian slot count')
        slot = self.slot_embedding.weight[None, None]
        normalized = ((points - self.pc_lower.to(points)) /
                      (self.pc_upper.to(points) - self.pc_lower.to(points))).clamp(0., 1.)
        child_feature = token[:, :, None] + slot + self.position_encoder(normalized)
        hidden = self.trunk(child_feature)
        scales = self.scale_min.to(points) + (self.scale_max.to(points) - self.scale_min.to(points)) * \
            torch.sigmoid(self.scale_head(hidden))
        opacities = torch.sigmoid(self.opacity_head(hidden))
        seed = self.residual_head(hidden)
        flat_points = points.flatten(1, 2)
        flat_scales = scales.flatten(1, 2)
        flat_opacities = opacities.flatten(1, 2)
        flat_seed = seed.flatten(1, 2)
        flat_valid = valid_mask[:, :, None].expand(-1, -1, children).flatten(1)
        # The kernel is intentionally float32 in V1.  These casts remain part
        # of autograd, so mixed-precision attribute heads still receive grads.
        residual = semantic_gaussian_localagg(
            flat_points.float(), flat_scales.float(), flat_opacities.float(),
            flat_seed.float(), flat_valid, self.pc_range, self.cell_size,
            self.support_sigma, self.scale_eps, self.denom_eps, self.include_self,
            self.max_scale)
        return dict(points=flat_points, scales=flat_scales, opacities=flat_opacities,
                    residual=residual, valid_mask=flat_valid)


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
            pc_range=self.pc_range, cell_size=gaussian_cell_size,
            support_sigma=gaussian_support_sigma, scale_eps=gaussian_scale_eps,
            denom_eps=gaussian_denom_eps)
