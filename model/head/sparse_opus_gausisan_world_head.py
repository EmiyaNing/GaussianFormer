"""Sparse OPUS perception head and the learned empty-bias Gaussian renderer."""

import torch
from mmengine.registry import MODELS

from .gaussian_opus_head import _flatten_grouped
from .opus_v1_gaussian_trainable_bias_head import OPUSV1GaussianTrainableBiasHead
from .sparseworld_strict_head import SparseWorldStrictHead
from model.encoder.sparse_opus_gausisan_world_encoder import (  # registry imports
    SparseOpusGaussianEncoder,
    SparseOpusGaussianTransformer,
)


@MODELS.register_module()
class SparseOpusGaussianRenderer(OPUSV1GaussianTrainableBiasHead):
    """Render grouped Gaussians into 18 logits, then subtract the empty bias.

    Inputs contain means/scales/rotations/opacity and 17 positive semantic
    scores per child. The analytic empty field avoids a global Gaussian in the
    local CUDA backward. Both paths preserve the learned scalar and bias.
    """

    def __init__(self, analytic_empty=False, compact_aggregation=False, **kwargs):
        super().__init__(**kwargs)
        self.analytic_empty = bool(analytic_empty)
        self.compact_aggregation = bool(compact_aggregation)
        if self.compact_aggregation:
            from model.ops.sparse_localagg import CompactLocalAggregator
            self.aggregator = CompactLocalAggregator(**kwargs['cuda_kwargs'])

    def _render_batched(self, sampled_xyz, means, opacity, semantics, scales, cov_inv):
        if not self.analytic_empty:
            return super()._render_batched(sampled_xyz, means, opacity, semantics, scales, cov_inv)
        foreground = super()._render_batched(
            sampled_xyz, means[:, :-1], opacity[:, :-1], semantics[:, :-1],
            scales[:, :-1], cov_inv[:, :-1])
        # Empty geometry and opacity are fixed buffers. Only its semantic
        # scalar is learned. Match the CUDA exponent and clipped box support,
        # including configurations where the empty Gaussian is not global.
        with torch.no_grad():
            delta = sampled_xyz.float() - means[:, -1:].detach()
            dx, dy, dz = delta.unbind(-1)
            cov = cov_inv[:, -1].detach()
            power = -0.5 * (cov[:, 0, 0, None] * dx.square() +
                            cov[:, 1, 1, None] * dy.square() +
                            cov[:, 2, 2, None] * dz.square())
            power -= (cov[:, 0, 1, None] * dx * dy +
                      cov[:, 1, 2, None] * dy * dz +
                      cov[:, 0, 2, None] * dx * dz)
            lower = self.aggregator.pc_min
            points_int = ((sampled_xyz - lower) / self.aggregator.grid_size).int()
            center = ((means[:, -1:].detach() - lower) / self.aggregator.grid_size).int()
            radius = torch.ceil(scales[:, -1:].detach().amax(-1) *
                                self.aggregator.scale_multiplier / self.aggregator.grid_size).int()
            supported = ((points_int >= center - radius[..., None]) &
                         (points_int < center + radius[..., None] + 1)).all(-1)
            weight = power.exp() * opacity[:, -1].detach().reshape(-1, 1) * supported
        return foreground + weight[:, None] * semantics[:, -1, :, None]

    def render_state(self, gaussian, sampled_xyz):
        flattened = _flatten_grouped(gaussian)
        if flattened.scales.detach().amax() > self.max_render_scale:
            raise ValueError("Gaussian scale exceeds bounded local aggregation range")
        arguments = self.prepare_gaussian_args(flattened)
        logits = self._render_batched(sampled_xyz, *arguments)
        return torch.cat(
            (
                logits[:, : self.empty_label],
                logits[:, self.empty_label :] - self.empty_bias,
            ),
            dim=1,
        )


@MODELS.register_module()
class SparseOpusGaussianPerceptionHead(SparseWorldStrictHead):
    """Return OPUS proposals together with their grouped Gaussian states."""

    def forward(self, mlvl_feats, metas, points_scale=None):
        batch_size = mlvl_feats[0].shape[0]
        initial = self.init_points.weight[None, :, None].expand(batch_size, -1, -1, -1)
        scale = self.points_scale if points_scale is None else points_scale
        if scale is not None:
            initial = initial * scale.unsqueeze(1)
        queries = initial.new_zeros(
            batch_size, len(self.init_points.weight), self.embed_dims
        )
        self.update_stamp_assignment(force=self.pretrain)
        features, scores, points, states = self.transformer(
            initial,
            queries,
            mlvl_feats,
            metas["projection_mat"].to(initial),
            metas["image_wh"].to(initial),
        )
        return dict(
            init_points=initial,
            all_cls_scores=scores,
            all_refine_pts=points,
            query_feat=features,
            representation=states,
        )
