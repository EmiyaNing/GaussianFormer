"""Isolated privileged-image probe; production SparseWorld is unchanged."""
import copy

import torch
from torch import nn


class FrozenOPUSImageReader(nn.Module):
    """Read ONE camera timestamp using pretrained OPUS sampling and mixing.

    The original mixer expects five temporal slots. Repeating the sampled
    single-frame tensor preserves its pretrained shape; no extra timestamps
    or RGB images are introduced. Invalid samples are explicitly zeroed.
    """

    def __init__(self, layer):
        super().__init__()
        self.sampling = copy.deepcopy(layer.sampling)
        self.original_frames = self.sampling.num_frames
        self.sampling.num_frames = 1
        self.mixing = copy.deepcopy(layer.mixing)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, points, query, image_features, metas):
        groups = self.sampling.num_groups
        grouped = []
        for feature in image_features:
            b, views, channels, height, width = feature.shape
            assert views == self.sampling.num_views
            grouped.append(feature.reshape(
                b, views, groups, channels // groups, height, width
            ).permute(0, 2, 1, 4, 5, 3).reshape(
                b * groups, views, height, width, channels // groups
            ).contiguous())
        sampled, valid = self.sampling(points, query, grouped, metas)
        # valid: B,T,Q,G,P; sampled: B,Q,G,T*P,C.
        valid_points = valid.permute(0, 2, 3, 1, 4).flatten(3, 4).bool()
        sampled = sampled * valid_points.unsqueeze(-1)
        visible = valid_points.any(dim=(-1, -2))
        sampled = sampled.repeat(1, 1, 1, self.original_frames, 1)
        extra = self.mixing(sampled, query) - query
        return extra * visible.unsqueeze(-1), visible


class AdaptiveOPUSImageReader(nn.Module):
    """Trainable, genuinely single-timestamp OPUS-style image reader.

    Sampling is initialized from a pretrained five-frame SparseWorld layer.
    Its temporal point-mixing weights are folded by summing the five
    coefficients that used to multiply repeated copies of the same point.
    This preserves the old operation at initialization while giving the new
    one-frame reader its own trainable sampling and mixing parameters.
    """

    def __init__(self, layer):
        super().__init__()
        self.sampling = copy.deepcopy(layer.sampling)
        self.sampling.num_frames = 1
        old = layer.mixing
        self.mixing = type(old)(
            embed_dims=old.out_proj.out_features,
            in_points=self.sampling.num_points,
            n_groups=old.n_groups,
            out_points=old.out_points,
        )
        self._fold_mixing(old)
        # deepcopy preserves requires_grad=False on a frozen source layer.
        # This reader is explicitly adaptive, including offsets and scales.
        self.requires_grad_(True)

    @torch.no_grad()
    def _fold_mixing(self, old):
        target = self.mixing
        dims = target.group_dim
        channel_size = dims * dims
        source_frames = old.in_points // self.sampling.num_points
        if old.in_points != source_frames * self.sampling.num_points:
            raise ValueError('Cannot fold OPUS temporal mixing: point count mismatch')
        if source_frames < 1:
            raise ValueError('Source OPUS mixing has no temporal slots')
        source_gen = old.generator
        target_gen = target.generator
        target_gen.weight.zero_()
        target_gen.bias.zero_()
        for group in range(target.n_groups):
            src = group * (channel_size + old.out_points * old.in_points)
            dst = group * (channel_size + target.out_points * target.in_points)
            target_gen.weight[dst:dst + channel_size].copy_(
                source_gen.weight[src:src + channel_size])
            target_gen.bias[dst:dst + channel_size].copy_(
                source_gen.bias[src:src + channel_size])
            src_point = src + channel_size
            dst_point = dst + channel_size
            point_weight = source_gen.weight[
                src_point:src_point + old.out_points * old.in_points
            ].reshape(old.out_points, source_frames,
                      self.sampling.num_points, source_gen.in_features)
            point_bias = source_gen.bias[
                src_point:src_point + old.out_points * old.in_points
            ].reshape(old.out_points, source_frames, self.sampling.num_points)
            target_gen.weight[dst_point:dst_point + target.out_points *
                              target.in_points].copy_(
                                  point_weight.sum(dim=1).reshape(
                                      -1, target_gen.in_features))
            target_gen.bias[dst_point:dst_point + target.out_points *
                            target.in_points].copy_(point_bias.sum(dim=1).reshape(-1))
        target.out_proj.load_state_dict(old.out_proj.state_dict())

    def forward(self, points, query, image_features, metas):
        grouped = []
        groups = self.sampling.num_groups
        for feature in image_features:
            b, views, channels, height, width = feature.shape
            if views != self.sampling.num_views:
                raise ValueError(f'Expected {self.sampling.num_views} views, got {views}')
            grouped.append(feature.reshape(
                b, views, groups, channels // groups, height, width
            ).permute(0, 2, 1, 4, 5, 3).reshape(
                b * groups, views, height, width, channels // groups
            ).contiguous())
        sampled, valid = self.sampling(points, query, grouped, metas)
        valid_points = valid.permute(0, 2, 3, 1, 4).flatten(3, 4).bool()
        sampled = sampled * valid_points.unsqueeze(-1)
        visible = valid_points.any(dim=(-1, -2))
        extra = self.mixing(sampled, query) - query
        return extra * visible.unsqueeze(-1), visible


class FutureImageFusionProbe(nn.Module):
    """A shared zero-output residual adapter, initially exact baseline."""

    def __init__(self, dims=256):
        super().__init__()
        self.query_norm = nn.LayerNorm(dims)
        self.image_norm = nn.LayerNorm(dims)
        self.fusion = nn.Sequential(
            nn.Linear(2 * dims, dims), nn.GELU(), nn.Linear(dims, dims))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, query, extra):
        delta = self.fusion(torch.cat(
            [self.query_norm(query), self.image_norm(extra)], dim=-1))
        return query + delta, delta


def decoded_offset(student, features, logits):
    b, q = features.shape[:2]
    offset = student.reg(features).view(b, q, student.num_refines, 3) * .5
    velocity = student.velocity(features).view(b, q, student.num_refines, 2)
    moving = ((logits.argmax(-1) >= 2) & (logits.argmax(-1) <= 10))
    return torch.cat([offset[..., :2] + velocity * moving.unsqueeze(-1),
                      offset[..., 2:]], dim=-1)


def probe_prediction(student, probe, state, extra, mode):
    """Independent per-horizon correction; never changes baseline recurrence."""
    features, delta = probe(state['query'], extra)
    logits = student.semantic(features).view(
        features.shape[0], features.shape[1], student.num_refines, 17)
    points = state['points']
    if mode == 'joint':
        # Same pre-refinement centre as baseline: only change decoded offset.
        points = points + decoded_offset(student, features, logits) - \
            state['offset']
    elif mode != 'semantic':
        raise ValueError(mode)
    return points.flatten(1, 2), logits.flatten(1, 2), delta
