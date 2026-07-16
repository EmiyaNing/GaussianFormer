import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


def _sample_image_features(feature_maps, points, metas,
                           pc_range=(-40., -40., -1., 40., 40., 5.4)):
    """Project ego-frame points and average valid multi-view FPN samples."""
    projection = metas['projection_mat'].to(dtype=points.dtype)
    image_wh = metas['image_wh'].to(dtype=points.dtype)
    batch_size, num_queries = points.shape[:2]
    lower = points.new_tensor(pc_range[:3])
    extent = points.new_tensor(pc_range[3:]) - lower
    ego_points = points * extent + lower
    homogeneous = torch.cat([ego_points, torch.ones_like(ego_points[..., :1])], dim=-1)
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
                 dropout=0.1, point_step=0.08, lidar_ball_radius=2.4,
                 lidar_ball_k=16, pc_range=(-40., -40., -1., 40., 40., 5.4)):
        super().__init__()
        self.cross_proj = nn.Linear(embed_dims, embed_dims)
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.norm4 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims),
        )
        self.point_delta = nn.Linear(embed_dims, 3)
        self.point_step = point_step
        self.lidar_ball_radius = lidar_ball_radius
        self.lidar_ball_k = lidar_ball_k
        self.pc_range = tuple(pc_range)
        self.lidar_proj = nn.Linear(embed_dims, embed_dims)

    def _aggregate_lidar(self, query_points, memory_features, memory_points,
                         query_valid_mask):
        """Fixed-K ball-query pooling from the 8x sparse LiDAR memory."""
        if memory_features is None or memory_points is None:
            return query_points.new_zeros((*query_points.shape[:2], self.lidar_proj.out_features))
        lower = query_points.new_tensor(self.pc_range[:3])
        extent = query_points.new_tensor(self.pc_range[3:]) - lower
        output = query_points.new_zeros((*query_points.shape[:2], self.lidar_proj.out_features))
        radius_squared = self.lidar_ball_radius ** 2
        for batch_idx, (features, points) in enumerate(zip(memory_features, memory_points)):
            valid_queries = query_valid_mask[batch_idx]
            if not valid_queries.any() or points.numel() == 0:
                continue
            current_query = query_points[batch_idx, valid_queries]
            # Distance is evaluated in metres, while coordinates stored by OPUS are normalized.
            delta = (current_query[:, None, :] - points[None, :, :]) * extent
            distance_squared = delta.square().sum(dim=-1)
            neighbours = min(self.lidar_ball_k, points.shape[0])
            nearest_distance, nearest_index = distance_squared.topk(
                neighbours, dim=1, largest=False)
            within_ball = nearest_distance <= radius_squared
            weights = torch.exp(-nearest_distance.clamp_min(1e-12).sqrt() /
                                max(self.lidar_ball_radius, 1e-6))
            weights = weights * within_ball.to(weights.dtype)
            gathered = features[nearest_index]
            pooled = (gathered * weights.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            output[batch_idx, valid_queries] = self.lidar_proj(pooled)
        return output

    def forward(self, query, points, feature_maps, metas, query_valid_mask=None,
                lidar_memory_features=None, lidar_memory_points=None):
        if query_valid_mask is None:
            query_valid_mask = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        image_feature, visible = _sample_image_features(
            feature_maps, points, metas, self.pc_range)
        query = self.norm1(query + self.cross_proj(image_feature))
        lidar_feature = self._aggregate_lidar(
            points, lidar_memory_features, lidar_memory_points, query_valid_mask)
        query = self.norm4(query + lidar_feature)
        attended, _ = self.self_attn(
            query, query, query, key_padding_mask=~query_valid_mask, need_weights=False)
        query = self.norm2(query + attended)
        query = self.norm3(query + self.ffn(query))
        # Keep invisible queries valid: their learned prior still receives self-attention.
        delta = torch.tanh(self.point_delta(query)) * self.point_step
        points = (points + delta).clamp_(0.0, 1.0)
        query = query * query_valid_mask.unsqueeze(-1).to(query.dtype)
        points = points * query_valid_mask.unsqueeze(-1).to(points.dtype)
        return query, points, visible


@MODELS.register_module()
class OPUSEncoder(BaseModule):
    """Sparse query decoder. Point multiplication is deferred to OPUSHead."""

    def __init__(self, embed_dims=128, num_decoder=6, num_heads=8,
                 feedforward_channels=512, dropout=0.1, point_step=0.08,
                 lidar_ball_radius=2.4, lidar_ball_k=16,
                 pc_range=(-40., -40., -1., 40., 40., 5.4),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.layers = nn.ModuleList([
            OPUSDecoderLayer(embed_dims, num_heads, feedforward_channels, dropout,
                             point_step, lidar_ball_radius, lidar_ball_k, pc_range)
            for _ in range(num_decoder)
        ])

    def forward(self, query_features, query_points, ms_img_feats, metas,
                query_valid_mask=None, lidar_memory_features=None,
                lidar_memory_points=None, **kwargs):
        if query_valid_mask is None:
            query_valid_mask = torch.ones(
                query_features.shape[:2], device=query_features.device, dtype=torch.bool)
        representation = []
        visible = None
        for layer in self.layers:
            query_features, query_points, visible = layer(
                query_features, query_points, ms_img_feats, metas, query_valid_mask,
                lidar_memory_features, lidar_memory_points)
            representation.append({
                'query_features': query_features,
                'query_points': query_points,
                'query_valid_mask': query_valid_mask,
                'visible': visible,
            })
        return {'representation': representation}


class _OfficialOPUSDecoderLayer(BaseModule):
    """Coarse-to-fine OPUS-V1 layer with one feature token per query group."""

    def __init__(self, embed_dims, last_refine, num_refine, num_heads,
                 feedforward_channels, dropout, pc_range):
        super().__init__()
        self.last_refine = last_refine
        self.num_refine = num_refine
        self.pc_range = tuple(pc_range)
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * last_refine, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.cross_proj = nn.Linear(embed_dims, embed_dims)
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims))
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3 * num_refine))

    def _sample_group_features(self, feature_maps, points, metas):
        batch_size, queries, group_size, _ = points.shape
        sampled, visible = _sample_image_features(
            feature_maps, points.reshape(batch_size, queries * group_size, 3), metas, self.pc_range)
        sampled = sampled.view(batch_size, queries, group_size, -1).mean(dim=2)
        visible = visible.view(batch_size, queries, group_size).any(dim=2)
        return sampled, visible

    def forward(self, query, points, feature_maps, metas, query_valid_mask):
        # The full parent point set, rather than its centroid alone, is part
        # of the next-stage query representation in OPUS-V1.
        query = query + self.position_encoder(points.flatten(2))
        image_feature, visible = self._sample_group_features(feature_maps, points, metas)
        query = self.norm1(query + self.cross_proj(image_feature))
        attended, _ = self.self_attn(query, query, query,
                                     key_padding_mask=~query_valid_mask,
                                     need_weights=False)
        query = self.norm2(query + attended)
        query = self.norm3(query + self.ffn(query))
        parent_center = points.mean(dim=2, keepdim=True)
        # Keep the official residual parameterisation unconstrained.  Points
        # outside the scene are rejected only by the common rasterizer.
        child_points = parent_center + self.reg_branch(query).view(
            *query.shape[:2], self.num_refine, 3)
        valid = query_valid_mask[:, :, None, None].to(child_points.dtype)
        child_points = child_points * valid
        query = query * query_valid_mask.unsqueeze(-1).to(query.dtype)
        return query, child_points, visible


@MODELS.register_module()
class OfficialOPUSV1Encoder(BaseModule):
    """Portable OPUS-V1 coarse-to-fine decoder.

    It preserves the official point-group recurrence and stop-gradient point
    proposals while using the repository's tested multi-view ``grid_sample``
    feature path instead of the legacy MMCV CUDA sampling extension.
    """

    def __init__(self, embed_dims=256, num_decoder=6, num_heads=8,
                 feedforward_channels=512, dropout=0.1,
                 num_refines=(1, 4, 16, 32, 64, 128),
                 pc_range=(-40., -40., -1., 40., 40., 5.4), init_cfg=None):
        super().__init__(init_cfg)
        if len(num_refines) != num_decoder:
            raise ValueError('num_refines must provide one point count per decoder layer')
        self.num_refines = tuple(num_refines)
        previous = (1,) + self.num_refines[:-1]
        self.layers = nn.ModuleList([
            _OfficialOPUSDecoderLayer(embed_dims, last, current, num_heads,
                                      feedforward_channels, dropout, pc_range)
            for last, current in zip(previous, self.num_refines)
        ])

    def forward(self, query_features, query_points, ms_img_feats, metas,
                query_valid_mask=None, **kwargs):
        if query_valid_mask is None:
            query_valid_mask = torch.ones(query_features.shape[:2], device=query_features.device,
                                          dtype=torch.bool)
        points = query_points.unsqueeze(2)
        representation = []
        for layer in self.layers:
            query_features, points, visible = layer(
                query_features, points, ms_img_feats, metas, query_valid_mask)
            representation.append({
                'query_features': query_features,
                'query_points': points,
                'query_valid_mask': query_valid_mask,
                'visible': visible,
            })
            # Official V1 detaches coordinate proposals between decoder stages.
            points = points.detach()
        return {'representation': representation}
