"""Reusable point-anchored Semantic Gaussian residual modules.

The modules in this file own only Gaussian attribute prediction and local
semantic aggregation.  Model-specific logit fusion and rasterization stay in
the OPUS/SparseWorld adapters that call them.
"""
from __future__ import annotations

import torch
from torch import nn
from mmengine.registry import MODELS

from model.ops.semantic_gaussian_localagg import (
    CUDA_LOCALAGG_AVAILABLE, semantic_gaussian_localagg)

try:
    from mmcv.ops import knn as mmcv_knn
except (ImportError, ModuleNotFoundError):
    mmcv_knn = None


@MODELS.register_module()
class PointAnchoredSemanticGaussian(nn.Module):
    """Predict Gaussian attributes and aggregate residuals on a point KNN graph.

    ``point_valid_mask`` is deliberately separate from ``query_valid_mask``.
    SparseWorld can emit individual child points outside its configured range,
    while all children still belong to an otherwise valid query.
    """

    def __init__(self, embed_dims, num_children, num_classes, pc_range,
                 scale_range=((.1, .1, .1), (.6, .6, .6)),
                 initial_scale=.35, initial_opacity=.02, num_neighbors=8,
                 include_self=True, query_chunk_size=256):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_children = int(num_children)
        self.num_classes = int(num_classes)
        self.num_neighbors = int(num_neighbors)
        self.include_self = bool(include_self)
        self.query_chunk_size = int(query_chunk_size)
        self.register_buffer('pc_lower', torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('pc_upper', torch.tensor(pc_range[3:], dtype=torch.float32))
        self.register_buffer('scale_min', torch.tensor(scale_range[0], dtype=torch.float32))
        self.register_buffer('scale_max', torch.tensor(scale_range[1], dtype=torch.float32))
        self.slot_embedding = nn.Embedding(self.num_children, self.embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))
        self.trunk = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims // 2), nn.ReLU(inplace=True))
        hidden_dims = self.embed_dims // 2
        self.scale_head = nn.Linear(hidden_dims, 3)
        self.opacity_head = nn.Linear(hidden_dims, 1)
        self.residual_head = nn.Linear(hidden_dims, self.num_classes)
        scale_prob = ((torch.tensor(initial_scale) - self.scale_min) /
                      (self.scale_max - self.scale_min)).clamp(.001, .999)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, torch.logit(scale_prob).mean().item())
        nn.init.zeros_(self.opacity_head.weight)
        nn.init.constant_(
            self.opacity_head.bias, torch.logit(torch.tensor(initial_opacity)).item())
        # A zero residual makes both downstream fusion modes exactly equal to
        # the base classifier at warm start.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _predict_attributes(self, token, points, query_valid_mask,
                            point_valid_mask=None):
        if token.ndim != 3:
            raise ValueError('token must have shape [B,Q,D]')
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError('points must have shape [B,Q,R,3]')
        batch, queries, children, _ = points.shape
        if token.shape[:2] != (batch, queries) or token.shape[-1] != self.embed_dims:
            raise ValueError('token and point query dimensions do not match')
        if children != self.num_children:
            raise ValueError('final OPUS child count does not match Gaussian slot count')
        if query_valid_mask.shape != (batch, queries):
            raise ValueError('query_valid_mask must have shape [B,Q]')
        if point_valid_mask is not None and point_valid_mask.shape != (batch, queries, children):
            raise ValueError('point_valid_mask must have shape [B,Q,R]')

        slot = self.slot_embedding.weight[None, None]
        normalized = ((points - self.pc_lower.to(points)) /
                      (self.pc_upper.to(points) - self.pc_lower.to(points))).clamp(0., 1.)
        child_feature = token[:, :, None] + slot + self.position_encoder(normalized)
        hidden = self.trunk(child_feature)
        scales = self.scale_min.to(points) + (
            self.scale_max.to(points) - self.scale_min.to(points)) * \
            torch.sigmoid(self.scale_head(hidden))
        opacities = torch.sigmoid(self.opacity_head(hidden))
        seed = self.residual_head(hidden)
        flat_valid = query_valid_mask.bool()[:, :, None].expand(
            -1, -1, children)
        if point_valid_mask is not None:
            flat_valid = flat_valid & point_valid_mask.bool()
        return dict(
            points=points.flatten(1, 2),
            scales=scales.flatten(1, 2),
            opacities=opacities.flatten(1, 2),
            seed=seed.flatten(1, 2),
            valid_mask=flat_valid.flatten(1))

    def _knn(self, points, neighbours):
        count = points.shape[0]
        neighbours = min(neighbours, count)
        if points.is_cuda and mmcv_knn is not None:
            return mmcv_knn(
                neighbours, points[None].contiguous(),
                points[None].contiguous())[0].transpose(0, 1)
        indices = []
        for start in range(0, count, self.query_chunk_size):
            distance = torch.cdist(
                points[start:start + self.query_chunk_size].float(), points.float())
            indices.append(distance.topk(
                neighbours, dim=-1, largest=False).indices)
        return torch.cat(indices, dim=0)

    def forward(self, token, points, query_valid_mask, point_valid_mask=None):
        values = self._predict_attributes(
            token, points, query_valid_mask, point_valid_mask)
        residual = torch.zeros_like(values['seed'])
        for batch_index in range(points.shape[0]):
            active = values['valid_mask'][batch_index]
            if not active.any():
                continue
            active_points = values['points'][batch_index, active]
            active_scales = values['scales'][batch_index, active]
            active_opacity = values['opacities'][batch_index, active]
            active_seed = values['seed'][batch_index, active]
            neighbour_count = (self.num_neighbors if self.include_self
                               else self.num_neighbors + 1)
            neighbour_index = self._knn(active_points, neighbour_count)
            if not self.include_self and neighbour_index.shape[1] > 1:
                neighbour_index = neighbour_index[:, 1:]
            neighbour_points = active_points[neighbour_index]
            neighbour_scales = active_scales[neighbour_index].clamp_min(1e-4)
            neighbour_opacity = active_opacity[neighbour_index]
            neighbour_seed = active_seed[neighbour_index]
            delta = (active_points[:, None] - neighbour_points) / neighbour_scales
            kernel = torch.exp(-.5 * delta.square().sum(dim=-1))
            weights = neighbour_opacity.squeeze(-1) * kernel
            message = (weights[..., None] * neighbour_seed).sum(dim=1) / \
                weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            residual[batch_index, active] = message
        values['residual'] = residual
        values.pop('seed')
        return values


@MODELS.register_module()
class PointAnchoredSemanticGaussianCUDA(PointAnchoredSemanticGaussian):
    """Point-anchored attributes with fused sparse CUDA local aggregation."""

    def __init__(self, *args, cell_size=1.8, support_sigma=3., scale_eps=1e-4,
                 denom_eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = tuple(
            torch.cat((self.pc_lower, self.pc_upper)).tolist())
        self.cell_size = float(cell_size)
        self.support_sigma = float(support_sigma)
        self.scale_eps = float(scale_eps)
        self.denom_eps = float(denom_eps)
        self.max_scale = float(self.scale_max.max())

    def forward(self, token, points, query_valid_mask, point_valid_mask=None):
        if not points.is_cuda or not CUDA_LOCALAGG_AVAILABLE:
            raise RuntimeError(
                'PointAnchoredSemanticGaussianCUDA requires the compiled CUDA extension')
        values = self._predict_attributes(
            token, points, query_valid_mask, point_valid_mask)
        values['residual'] = semantic_gaussian_localagg(
            values['points'].float(), values['scales'].float(),
            values['opacities'].float(), values['seed'].float(),
            values['valid_mask'], self.pc_range, self.cell_size,
            self.support_sigma, self.scale_eps, self.denom_eps,
            self.include_self, self.max_scale)
        values.pop('seed')
        return values
