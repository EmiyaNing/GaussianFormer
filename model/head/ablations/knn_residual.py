"""P1 control adapters: neighbourhood smoothing without learned Gaussians.

Plan reference: ``plans/sparse_render_plans/P1_frozen_backbone_semantic_validation.md``,
adapter A2 ("KNN-8 mean/max residual").

A2 exists to rule out the cheapest alternative explanation for any gain from
the Gaussian adapter: *ordinary neighbourhood smoothing is already enough*.
It therefore keeps the same interface and the same residual-seed mechanism as
:class:`PointAnchoredSemanticGaussianCUDA` but removes the learned Gaussian
attributes entirely:

* no per-axis scale head, no opacity head;
* neighbours are selected by a fixed ``k`` nearest-neighbour rule rather than
  by a learned support box;
* the message is a plain mean or max over neighbour seeds rather than an
  opacity-weighted Gaussian kernel.

``scales`` and ``opacities`` are returned as inert constants purely because
``SparseGaussianWorld`` reads those keys for logging; they carry no gradient
and are not a learned part of this variant.
"""
from __future__ import annotations

import torch
from torch import nn
from mmengine.registry import MODELS

try:
    from mmcv.ops import knn as mmcv_knn
except (ImportError, ModuleNotFoundError):
    mmcv_knn = None


@MODELS.register_module()
class KNNResidualAggregator(nn.Module):
    """Fixed-k neighbour aggregation of a learned semantic residual seed."""

    def __init__(self, embed_dims, num_children, num_classes, pc_range,
                 num_neighbors=8, include_self=True, mode='mean',
                 query_chunk_size=256, gaussian_scale=0.35):
        super().__init__()
        if mode not in ('mean', 'max', 'self', 'attention', 'gaussian'):
            raise ValueError(
                "mode must be 'mean', 'max', 'self', 'attention' or 'gaussian'")
        # ``mode='gaussian'`` exists to separate the two things a matched-k
        # comparison of A5 against A2 confounds: the **weight profile** and the
        # **neighbour selection**.  It uses A2's selection verbatim (the k
        # Euclidean nearest points) and only replaces the uniform 1/k weights
        # with a fixed isotropic Gaussian of the displacement, so A2-vs-gaussian
        # isolates the weighting and A5-cap8-vs-gaussian isolates the selection.
        # The scale is fixed rather than learned so the parameter count stays
        # identical to A2's (plan T1.2).
        self.gaussian_scale = float(gaussian_scale)
        if self.gaussian_scale <= 0:
            raise ValueError('gaussian_scale must be positive')
        self.embed_dims = int(embed_dims)
        self.num_children = int(num_children)
        self.num_classes = int(num_classes)
        self.num_neighbors = int(num_neighbors)
        self.include_self = bool(include_self)
        self.mode = mode
        self.query_chunk_size = int(query_chunk_size)
        self.register_buffer('pc_lower',
                             torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('pc_upper',
                             torch.tensor(pc_range[3:], dtype=torch.float32))
        self.slot_embedding = nn.Embedding(self.num_children, self.embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))
        self.trunk = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims // 2), nn.ReLU(inplace=True))
        self.residual_head = nn.Linear(self.embed_dims // 2, self.num_classes)
        # Zero residual makes the warm start exactly equal to the base
        # classifier, matching the Gaussian adapter's initialisation.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.mode == 'attention':
            # A3: GraphGSOcc-style graph attention over the same k-NN graph.
            # ``attn_value`` is zero-initialised so the warm start is still
            # exactly the base classifier.
            hidden = self.embed_dims // 2
            self.attn_query = nn.Linear(hidden, hidden)
            self.attn_key = nn.Linear(hidden, hidden)
            self.attn_value = nn.Linear(hidden, self.num_classes)
            nn.init.zeros_(self.attn_value.weight)
            nn.init.zeros_(self.attn_value.bias)

    def _knn(self, points, neighbours):
        count = points.shape[0]
        neighbours = min(neighbours, count)
        if points.is_cuda and mmcv_knn is not None:
            return mmcv_knn(neighbours, points[None].contiguous(),
                            points[None].contiguous())[0].transpose(0, 1)
        indices = []
        for start in range(0, count, self.query_chunk_size):
            distance = torch.cdist(
                points[start:start + self.query_chunk_size].float(),
                points.float())
            indices.append(distance.topk(neighbours, dim=-1,
                                         largest=False).indices)
        return torch.cat(indices, dim=0)

    def forward(self, token, points, query_valid_mask, point_valid_mask=None):
        batch, queries, children, _ = points.shape
        slot = self.slot_embedding.weight[None, None]
        normalized = ((points - self.pc_lower.to(points)) /
                      (self.pc_upper.to(points) - self.pc_lower.to(points))
                      ).clamp(0., 1.)
        child_feature = token[:, :, None] + slot + self.position_encoder(normalized)
        hidden = self.trunk(child_feature)
        seed = self.residual_head(hidden)

        flat_points = points.flatten(1, 2)
        valid = query_valid_mask.bool()[:, :, None].expand(-1, -1, children)
        if point_valid_mask is not None:
            valid = valid & point_valid_mask.bool()
        valid = valid.flatten(1)

        if self.mode == 'self':
            # A1: a purely point-wise residual with no neighbourhood at all.
            # Any gain over A0 is then attributable to extra capacity, not to
            # spatial aggregation.
            seed = seed.flatten(1, 2)
            return dict(
                points=flat_points,
                residual=torch.where(valid.unsqueeze(-1), seed,
                                     torch.zeros_like(seed)),
                valid_mask=valid,
                scales=torch.ones_like(flat_points),
                opacities=torch.ones_like(flat_points[..., 0]))

        if self.mode == 'attention':
            flat_hidden = hidden.flatten(1, 2)
            query = self.attn_query(flat_hidden)
            key = self.attn_key(flat_hidden)
        seed = seed.flatten(1, 2)

        residual = torch.zeros_like(seed)
        neighbour_count = self.num_neighbors if self.include_self \
            else self.num_neighbors + 1
        for index in range(batch):
            active = valid[index]
            if not active.any():
                continue
            active_points = flat_points[index][active]
            neighbour_index = self._knn(active_points, neighbour_count)
            if not self.include_self and neighbour_index.shape[1] > 1:
                neighbour_index = neighbour_index[:, 1:]
            if self.mode == 'attention':
                query_active = query[index][active]
                key_neighbours = key[index][neighbour_index]
                scale = query_active.shape[-1] ** 0.5
                # Scaled dot-product attention over the k-NN graph.
                logits = (query_active[:, None, :] * key_neighbours).sum(-1) / scale
                weights = torch.softmax(logits, dim=1)
                values = self.attn_value(flat_hidden[index])[neighbour_index]
                residual[index][active] = (weights[..., None] * values).sum(dim=1)
                continue
            neighbours = seed[index][neighbour_index]
            if self.mode == 'gaussian':
                # Fixed isotropic Gaussian over the query-to-neighbour
                # displacement; normalised, so the operator stays a weighted
                # mean and only the *profile* differs from A2's uniform 1/k.
                neighbour_points = flat_points[index][neighbour_index]
                delta = active_points[:, None, :] - neighbour_points
                power = -0.5 * (delta / self.gaussian_scale).square().sum(dim=-1)
                weight = torch.exp(power)
                weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
                message = (weight[..., None] * neighbours).sum(dim=1)
            elif self.mode == 'mean':
                message = neighbours.mean(dim=1)
            else:
                message = neighbours.max(dim=1).values
            residual[index][active] = message

        # Inert placeholders: SparseGaussianWorld reads these keys for logging
        # only, and this variant deliberately has no learned Gaussian shape.
        return dict(
            points=flat_points,
            residual=residual,
            valid_mask=valid,
            scales=torch.ones_like(flat_points),
            opacities=torch.ones_like(flat_points[..., 0]))
