"""OPUS-V1 point decoder with low-dimensional Gaussian output attributes.

The high-dimensional state remains one feature vector per OPUS query.  A
Gaussian is assembled only after each decoder layer has produced its child
points and 17 non-empty class logits.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from mmengine.registry import MODELS

from .gaussian_encoder.utils import GaussianPrediction
from .opus_encoder import StrictOPUSV1Encoder


class _GaussianAttributeBranch(nn.Module):
    def __init__(self, embed_dims, children, scale_min, scale_max,
                 initial_scale, initial_opacity):
        super().__init__()
        if not (0 < scale_min < initial_scale < scale_max):
            raise ValueError('initial_scale must lie strictly inside positive scale bounds')
        if not (0 < initial_opacity < 1):
            raise ValueError('initial_opacity must lie strictly between zero and one')
        self.num_children = children
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, children * 8))
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        scale_fraction = (initial_scale - scale_min) / (scale_max - scale_min)
        with torch.no_grad():
            bias = last.bias.view(children, 8)
            # A small anisotropy gives the rotation branch a gradient from
            # its first occupancy update; an isotropic Gaussian cannot reveal
            # its orientation to the renderer.
            bias[:, :3] = (math.log(scale_fraction / (1 - scale_fraction)) +
                           bias.new_tensor([-0.5, 0.0, 0.5]))
            bias[:, 7] = math.log(initial_opacity / (1 - initial_opacity))

    def forward(self, query_features):
        batch, queries = query_features.shape[:2]
        raw = self.net(query_features).view(batch, queries, self.num_children, 8)
        scales = self.scale_min + (self.scale_max - self.scale_min) * raw[..., :3].sigmoid()
        identity = torch.zeros_like(raw[..., 3:7])
        identity[..., 0] = 1.0
        rotations = F.normalize(raw[..., 3:7] + identity, dim=-1, eps=1e-6)
        opacities = raw[..., 7:8].sigmoid()
        return scales, rotations, opacities


@MODELS.register_module()
class OPUSV1GaussianEncoder(StrictOPUSV1Encoder):
    """Strict OPUS-V1 decoder plus child Gaussian attributes.

    ``attribute_stages`` defaults to the final stage, the one rendered by the
    initial experiment.  Other stages use fixed valid attributes so they can
    be rendered for diagnostics without owning permanently unused parameters.
    """

    def __init__(self, scale_min=0.08, scale_max=0.40, initial_scale=0.20,
                 initial_opacity=0.10, attribute_stages=None,
                 render_eps=1e-4, **kwargs):
        super().__init__(**kwargs)
        if attribute_stages is None:
            attribute_stages = (len(self.layers) - 1,)
        self.attribute_stages = tuple(attribute_stages)
        if len(set(self.attribute_stages)) != len(self.attribute_stages):
            raise ValueError('attribute_stages must be unique')
        if any(index < 0 or index >= len(self.layers) for index in self.attribute_stages):
            raise ValueError('attribute_stages contains an invalid layer')
        self.initial_scale = float(initial_scale)
        self.initial_opacity = float(initial_opacity)
        self.render_eps = float(render_eps)
        self.pc_range = tuple(self.layers[0].pc_range)
        self.attribute_branches = nn.ModuleDict({
            str(index): _GaussianAttributeBranch(
                self.layers[index].cls_branch[-1].in_features,
                self.layers[index].num_refine,
                scale_min, scale_max, initial_scale, initial_opacity)
            for index in self.attribute_stages
        })

    def _make_gaussian(self, features, points, logits, valid, stage):
        batch, queries, children = points.shape[:3]
        bounded = points.clamp(self.render_eps, 1.0 - self.render_eps)
        lower = points.new_tensor(self.pc_range[:3])
        extent = points.new_tensor(self.pc_range[3:]) - lower
        means = lower + bounded * extent
        if str(stage) in self.attribute_branches:
            scales, rotations, opacity = self.attribute_branches[str(stage)](features)
        else:
            scales = means.new_full((batch, queries, children, 3), self.initial_scale)
            rotations = means.new_zeros((batch, queries, children, 4))
            rotations[..., 0] = 1.0
            opacity = means.new_full((batch, queries, children, 1), self.initial_opacity)
        opacity = opacity * valid[:, :, None, None].to(opacity.dtype)
        return GaussianPrediction(means, scales, rotations, opacity, logits)

    def forward(self, query_features, query_points, ms_img_feats, metas,
                query_valid_mask=None, **kwargs):
        if ms_img_feats[0].shape[1] != self.num_frames * self.num_views:
            raise ValueError('OPUS feature camera dimension must equal num_frames * num_views')
        if query_valid_mask is None:
            query_valid_mask = torch.ones(query_features.shape[:2], device=query_features.device,
                                          dtype=torch.bool)
        batch = query_features.shape[0]
        grouped_features = []
        for feature in ms_img_feats:
            _, cameras, channels, height, width = feature.shape
            if cameras != self.num_frames * self.num_views:
                raise ValueError('all OPUS feature levels must have T * N images')
            if channels % self.num_groups:
                raise ValueError('OPUS embed channels must be divisible by sampling groups')
            grouped_features.append(feature.reshape(
                batch, self.num_frames, self.num_views, self.num_groups,
                channels // self.num_groups, height, width
            ).permute(0, 1, 3, 2, 5, 6, 4).reshape(
                batch * self.num_frames * self.num_groups, self.num_views, height, width,
                channels // self.num_groups
            ).contiguous())

        points = query_points.unsqueeze(2)
        representation = []
        for stage, layer in enumerate(self.layers):
            query_features, logits, points, visible = layer(
                points, query_features, grouped_features, metas)
            gaussian = self._make_gaussian(
                query_features, points, logits, query_valid_mask, stage)
            representation.append({
                'query_features': query_features,
                'query_points': points,
                'opus_logits': logits,
                'gaussian': gaussian,
                'query_valid_mask': query_valid_mask,
                'visible': visible,
            })
            # Preserve the original OPUS-V1 stage-to-stage point proposal rule.
            points = points.detach()
        return {'representation': representation}
