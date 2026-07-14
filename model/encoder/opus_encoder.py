import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


def _sample_image_features(feature_maps, points, metas):
    """Project ego-frame points and average valid multi-view FPN samples."""
    projection = metas['projection_mat'].to(dtype=points.dtype)
    image_wh = metas['image_wh'].to(dtype=points.dtype)
    batch_size, num_queries = points.shape[:2]
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    camera_points = torch.einsum('bnij,bqj->bnqi', projection, homogeneous)
    depth = camera_points[..., 2:3]
    # Do not feed enormous or non-finite grids to grid_sample. Invalid points
    # are masked later, but CUDA grid_sample still differentiates its grid.
    valid_depth = depth[..., 0] > 1e-3
    safe_depth = torch.where(valid_depth.unsqueeze(-1), depth,
                             torch.ones_like(depth))
    uv = camera_points[..., :2] / safe_depth
    valid = valid_depth
    valid = valid & (uv[..., 0] >= 0) & (uv[..., 0] < image_wh[:, :, None, 0])
    valid = valid & (uv[..., 1] >= 0) & (uv[..., 1] < image_wh[:, :, None, 1])

    fused = None
    valid_count = None
    for feature in feature_maps:
        _, num_cams, channels, height, width = feature.shape
        grid = uv / image_wh[:, :, None, :] * 2.0 - 1.0
        # A finite location outside the image has zero sampled value and a
        # well-defined gradient. It replaces every invalid projection.
        grid = torch.where(valid[..., None], grid, torch.full_like(grid, 2.0))
        grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2.0, 2.0)
        sampled = F.grid_sample(
            feature.reshape(batch_size * num_cams, channels, height, width),
            grid.reshape(batch_size * num_cams, num_queries, 1, 2),
            mode='bilinear', padding_mode='zeros', align_corners=False,
        ).squeeze(-1).reshape(batch_size, num_cams, channels, num_queries)
        masked = sampled * valid[:, :, None, :].to(sampled.dtype)
        level_sum = masked.sum(dim=1).transpose(1, 2)
        level_count = valid.sum(dim=1).unsqueeze(-1).to(sampled.dtype)
        fused = level_sum if fused is None else fused + level_sum
        valid_count = level_count if valid_count is None else valid_count + level_count
    return fused / valid_count.clamp_min(1.0), valid_count[..., 0] > 0


class OPUSDecoderLayer(BaseModule):
    def __init__(self, embed_dims=128, num_heads=8, feedforward_channels=512,
                 dropout=0.1, point_step=0.08):
        super().__init__()
        self.cross_proj = nn.Linear(embed_dims, embed_dims)
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims),
        )
        self.point_delta = nn.Linear(embed_dims, 3)
        self.point_step = point_step

    def forward(self, query, points, feature_maps, metas):
        image_feature, visible = _sample_image_features(feature_maps, points, metas)
        query = self.norm1(query + self.cross_proj(image_feature))
        attended, _ = self.self_attn(query, query, query, need_weights=False)
        query = self.norm2(query + attended)
        query = self.norm3(query + self.ffn(query))
        # Keep invisible queries valid: their learned prior still receives self-attention.
        delta = torch.tanh(self.point_delta(query)) * self.point_step
        points = (points + delta).clamp_(0.0, 1.0)
        return query, points, visible


@MODELS.register_module()
class OPUSEncoder(BaseModule):
    """Sparse query decoder. Point multiplication is deferred to OPUSHead."""

    def __init__(self, embed_dims=128, num_decoder=6, num_heads=8,
                 feedforward_channels=512, dropout=0.1, point_step=0.08,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.layers = nn.ModuleList([
            OPUSDecoderLayer(embed_dims, num_heads, feedforward_channels, dropout,
                             point_step)
            for _ in range(num_decoder)
        ])

    def forward(self, query_features, query_points, ms_img_feats, metas, **kwargs):
        representation = []
        visible = None
        for layer in self.layers:
            query_features, query_points, visible = layer(
                query_features, query_points, ms_img_feats, metas)
            representation.append({
                'query_features': query_features,
                'query_points': query_points,
                'visible': visible,
            })
        return {'representation': representation}
