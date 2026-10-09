"""P3 control: dense Gaussian-to-voxel rendering, everything else identical.

Plan reference: ``plans/sparse_render_plans/README.md`` section 5.3 ("faithful
S2GO-style G2V") and the premise under test in
``analysis/sparse_render_result/P3-1_whole_model_profile.md`` /
``P3-2_training_cost_probe.md``: *does materialising a dense ``V x C`` field
actually dominate cost at the model level?*

The P0 matrix answered that at the **operator** level (dense G2V 663 MiB vs
sparse G2Q 31 MiB) and P3-1/P3-2 showed the operator-level gap does **not**
appear at the model level for the sparse variants.  What was still missing is a
model-level *dense* reference.  This module supplies it with a controlled
swap:

* it subclasses :class:`PointAnchoredSemanticGaussian`, so attribute prediction
  (scale / opacity / semantic seed) and therefore the parameter count are
  **identical** to the sparse Gaussian adapter;
* only the operator changes: instead of the sparse G2Q CUDA op, the field is
  materialised over the full voxel grid with the repository's dense
  Gaussian-to-voxel operator (``LocalAggregator``, the same one GaussianFormer
  uses) and then sampled back at the point locations.

Same backbone, same data, same schedule, same interface -> any model-level
difference is attributable to the operator, which is exactly what P3 needs.

Note this is a *dense-rendering* control, not a faithful S2GO reimplementation:
S2GO's specific ``4x4x4`` block kernel is still not implemented, so numbers from
this module must be labelled ``ours/dense-G2V`` and never compared to S2GO's
paper numbers.
"""
from __future__ import annotations

import torch
from torch import nn
from mmengine.registry import MODELS

from ..point_anchored_semantic_gaussian import PointAnchoredSemanticGaussian


@MODELS.register_module()
class DenseG2VResidualAggregator(PointAnchoredSemanticGaussian):
    """Dense G2V rendering sampled back at the predicted points."""

    def __init__(self, *args, grid_shape=(200, 200, 16), grid_size=0.4,
                 scale_multiplier=3.0, render_channels=18, dense_op='naive',
                 **kwargs):
        if dense_op not in ('naive', 'block'):
            raise ValueError("dense_op must be 'naive' or 'block'")
        self.dense_op = dense_op
        super().__init__(*args, **kwargs)
        from model.head.localagg.local_aggregate import LocalAggregator

        # The dense operator's channel count is a compile-time constant
        # (``#define NUM_CHANNELS 18`` in model/head/localagg/src/config.h), so
        # the seed is zero-padded up to it and the output is sliced back.
        self.render_channels = int(render_channels)
        if self.render_channels < self.num_classes:
            raise ValueError(
                'render_channels must be >= num_classes; the dense operator '
                'always materialises render_channels columns')
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.grid_size = float(grid_size)
        self.scale_multiplier = float(scale_multiplier)
        self.localagg = LocalAggregator(
            scale_multiplier=self.scale_multiplier,
            H=self.grid_shape[0], W=self.grid_shape[1], D=self.grid_shape[2],
            pc_min=(float(self.pc_lower[0]), float(self.pc_lower[1]),
                    float(self.pc_lower[2])),
            grid_size=self.grid_size)
        # Voxel centres are a fixed function of the grid configuration; cache
        # them as a non-persistent buffer so they follow .to(device)/.cuda()
        # without polluting checkpoints.
        h, w, d = self.grid_shape
        xs = (torch.arange(h, dtype=torch.float32) + 0.5) * self.grid_size + self.pc_lower[0]
        ys = (torch.arange(w, dtype=torch.float32) + 0.5) * self.grid_size + self.pc_lower[1]
        zs = (torch.arange(d, dtype=torch.float32) + 0.5) * self.grid_size + self.pc_lower[2]
        centres = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'),
                              dim=-1).reshape(1, -1, 3)
        self.register_buffer('voxel_centres', centres, persistent=False)
        # Integer voxel coordinates of every centre, needed by the block kernel.
        lower = torch.tensor(
            [float(self.pc_lower[0]), float(self.pc_lower[1]), float(self.pc_lower[2])])
        self.register_buffer(
            'voxel_ints',
            ((centres[0] - lower) / self.grid_size).to(torch.int32).contiguous(),
            persistent=False)

    def _voxel_indices(self, points):
        """Flat voxel index per point, clamped into the grid."""
        h, w, d = self.grid_shape
        lower = self.pc_lower.to(points)
        coords = ((points - lower) / self.grid_size).floor().long()
        shape = coords.new_tensor(self.grid_shape)
        coords = torch.maximum(torch.minimum(coords, shape - 1),
                               torch.zeros_like(coords))
        return (coords[..., 0] * w + coords[..., 1]) * d + coords[..., 2]

    def forward(self, token, points, query_valid_mask, point_valid_mask=None):
        values = self._predict_attributes(
            token, points, query_valid_mask, point_valid_mask)
        flat_points = values['points']                  # [B, G, 3]
        scales = values['scales']                       # [B, G, 3]
        opacities = values['opacities']                 # [B, G, 1]
        seed = values['seed']                           # [B, G, C]

        cv = self.voxel_centres.to(flat_points)
        batch, count, _ = flat_points.shape
        classes = seed.shape[-1]
        if classes < self.render_channels:
            seed = torch.cat(
                [seed, seed.new_zeros(batch, count,
                                      self.render_channels - classes)], dim=-1)

        # Clamp Gaussian centres into the grid: LocalAggregator asserts they are
        # inside, and SparseWorld can emit points marginally outside pc_range.
        lower = self.pc_lower.to(flat_points)
        upper = self.pc_upper.to(flat_points)
        inside_points = flat_points.clamp(lower + 1e-3, upper - 1e-3)

        residual = flat_points.new_zeros(batch, count, classes)
        for index in range(batch):
            # The dense operators take the INVERSE covariance: GaussianHead
            # computes ``CovInv = Cov.inverse()`` and passes that as cov3D, and
            # the kernel evaluates ``d^T cov3D d``.  Using scales**2 here would
            # change the weights (though not the radius, hence not the cost).
            inverse_covariance = torch.zeros(
                count, 3, 3, device=flat_points.device, dtype=flat_points.dtype)
            inverse_covariance[:, 0, 0] = 1.0 / scales[index, :, 0] ** 2
            inverse_covariance[:, 1, 1] = 1.0 / scales[index, :, 1] ** 2
            inverse_covariance[:, 2, 2] = 1.0 / scales[index, :, 2] ** 2

            if self.dense_op == 'block':
                from model.ops.block_g2v import block_g2v_forward
                lower = self.pc_lower.to(flat_points)
                means_int = ((inside_points[index] - lower) / self.grid_size) \
                    .to(torch.int32).contiguous()
                radii = torch.ceil(
                    scales[index].max(dim=-1).values * self.scale_multiplier
                    / self.grid_size).to(torch.int32).clamp_min(1).contiguous()
                cov6 = inverse_covariance.flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous()
                dense = block_g2v_forward(
                    cv.squeeze(0), self.voxel_ints, inside_points[index], means_int,
                    opacities[index, :, 0], seed[index], radii, cov6, self.grid_shape)
            else:
                # LocalAggregator expects opacities as [B, G] (GaussianHead
                # passes origi_opa.reshape(bs, g)); the CUDA backward returns a
                # flat [G] gradient and autograd rejects a [G, 1] input.
                dense = self.localagg(
                    cv,
                    inside_points[index:index + 1],
                    opacities[index:index + 1, :, 0],
                    seed[index:index + 1],
                    scales[index:index + 1],
                    inverse_covariance.unsqueeze(0))
            voxel_index = self._voxel_indices(flat_points[index])
            residual[index] = dense[voxel_index][:, :classes]

        values['residual'] = residual
        values.pop('seed')
        return values
