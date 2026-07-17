import torch
import torch.nn.functional as F
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from model.ops.opus_msmv_sampling import msmv_sampling


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


class _AdaptiveMixing(nn.Module):
    """Direct port of OPUS-V1 adaptive channel/point mixing."""

    def __init__(self, embed_dims, in_points, n_groups=4, out_points=32):
        super().__init__()
        self.in_points, self.n_groups, self.out_points = in_points, n_groups, out_points
        self.group_dim = embed_dims // n_groups
        self.channel_params = self.group_dim * self.group_dim
        self.point_params = in_points * out_points
        self.generator = nn.Linear(embed_dims, n_groups * (self.channel_params + self.point_params))
        self.out_proj = nn.Linear(n_groups * out_points * self.group_dim, embed_dims)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features, query):
        batch, queries, groups, points, channels = features.shape
        if (groups, points, channels) != (self.n_groups, self.in_points, self.group_dim):
            raise ValueError('OPUS sampling/mixing shape mismatch')
        params = self.generator(query).view(batch * queries, groups, -1)
        channel, point = params.split([self.channel_params, self.point_params], dim=-1)
        channel = channel.view(batch * queries, groups, self.group_dim, self.group_dim)
        point = point.view(batch * queries, groups, self.out_points, self.in_points)
        output = features.reshape(batch * queries, groups, points, channels)
        output = self.act(F.layer_norm(torch.matmul(output, channel), [points, self.group_dim]))
        output = self.act(F.layer_norm(torch.matmul(point, output), [self.out_points, self.group_dim]))
        return query + self.out_proj(output.reshape(batch, queries, -1))


class _OfficialOPUSSampling(nn.Module):
    """Official V1 fixed 4x4 sampling budget with a PyTorch projection path.

    The CUDA MSMV kernel consumes this exact tensor contract when available;
    the explicit camera loop below is its memory-bounded fallback.  Crucially,
    it samples 16 learned locations per query, never every refined child point.
    """

    def __init__(self, embed_dims, num_groups=4, num_points=4,
                 pc_range=(-40., -40., -1., 40., 40., 5.4)):
        super().__init__()
        self.num_groups, self.num_points, self.pc_range = num_groups, num_points, tuple(pc_range)
        self.offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale = nn.Linear(embed_dims, num_groups * num_points * 4)
        nn.init.zeros_(self.offset.weight)
        nn.init.uniform_(self.offset.bias.view(-1, 3), -0.5, 0.5)

    def forward(self, query_points, query_features, feature_maps, metas):
        batch, queries, _, _ = query_points.shape
        lower = query_points.new_tensor(self.pc_range[:3])
        extent = query_points.new_tensor(self.pc_range[3:]) - lower
        world = query_points * extent + lower
        center = world.mean(dim=2, keepdim=True)
        scale = world.std(dim=2, keepdim=True) if world.shape[2] > 1 else torch.zeros_like(center)
        offsets = self.offset(query_features).view(batch, queries, self.num_groups, self.num_points, 3)
        sample_points = center[:, :, None] + offsets * scale[:, :, None]
        scale_weights = self.scale(query_features).view(
            batch, queries, self.num_groups, self.num_points, 4)[..., :len(feature_maps)].softmax(dim=-1)
        sampled, visible = self._sample_single_view(sample_points.flatten(1, 3), feature_maps, metas, scale_weights)
        channels = sampled.shape[-1]
        if channels % self.num_groups:
            raise ValueError('OPUS embed_dims must be divisible by num_groups')
        sampled = sampled.view(batch, queries, self.num_groups, self.num_points,
                               self.num_groups, channels // self.num_groups)
        # The official CUDA path splits FPN channels into sampling groups
        # before sampling; select the corresponding channel group here.
        sampled = torch.stack([
            sampled[:, :, group_index, :, group_index, :]
            for group_index in range(self.num_groups)], dim=2)
        return sampled, visible.view(
            batch, queries, self.num_groups, self.num_points).any(dim=(2, 3))

    def _sample_single_view(self, points, feature_maps, metas, scale_weights):
        batch, locations, _ = points.shape
        projection = metas['projection_mat'].to(dtype=points.dtype)
        image_wh = metas['image_wh'].to(dtype=points.dtype)
        homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        camera = torch.einsum('bnij,bqj->bnqi', projection, homogeneous)
        depth = camera[..., 2]
        uv = camera[..., :2] / depth.unsqueeze(-1).clamp_min(1e-5)
        valid = (depth > 1e-5) & (uv[..., 0] > 0) & (uv[..., 1] > 0)
        valid &= (uv[..., 0] < image_wh[:, :, None, 0]) & (uv[..., 1] < image_wh[:, :, None, 1])
        selected = valid.float().argmax(dim=1)
        if projection.shape[1] == 6:
            return self._msmv_single_frame(
                uv, valid, selected, feature_maps, image_wh, scale_weights)
        output = points.new_zeros(batch, locations, feature_maps[0].shape[2])
        flattened_weights = scale_weights.flatten(1, 3)
        for level, feature in enumerate(feature_maps):
            for batch_index in range(batch):
                for camera_index in range(feature.shape[1]):
                    mask = (selected[batch_index] == camera_index) & valid[batch_index, camera_index]
                    if not mask.any():
                        continue
                    grid = uv[batch_index, camera_index, mask] / image_wh[batch_index, camera_index] * 2.0 - 1.0
                    value = F.grid_sample(feature[batch_index, camera_index][None], grid[None, :, None],
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
                    output[batch_index, mask] += value[0, :, :, 0].transpose(0, 1) * \
                        flattened_weights[batch_index, mask, level:level + 1]
        return output, valid.any(dim=1)

    def _msmv_single_frame(self, uv, valid, selected, feature_maps, image_wh, scale_weights):
        """Prepare the exact C2345 kernel contract for the 1-frame model."""
        batch, _, locations, _ = uv.shape
        queries = scale_weights.shape[1]
        normalized = uv.new_zeros(batch, locations, 3)
        for batch_index in range(batch):
            view = selected[batch_index]
            normalized[batch_index, :, :2] = (
                uv[batch_index, view, torch.arange(locations, device=uv.device)] /
                image_wh[batch_index, view])
            normalized[batch_index, :, 2] = view.to(normalized.dtype) / 5.0
        normalized = normalized.view(batch, queries, self.num_groups, self.num_points, 3)
        weight = scale_weights.permute(0, 2, 1, 3, 4).reshape(
            batch * self.num_groups, queries, self.num_points, -1).contiguous()
        location = normalized.permute(0, 2, 1, 3, 4).reshape(
            batch * self.num_groups, queries, self.num_points, 3).contiguous()
        grouped_features = []
        for feature in feature_maps:
            _, views, channels, height, width = feature.shape
            grouped = feature.view(batch, views, self.num_groups, channels // self.num_groups,
                                   height, width).permute(0, 2, 1, 4, 5, 3)
            grouped_features.append(grouped.reshape(
                batch * self.num_groups, views, height, width, channels // self.num_groups).contiguous())
        sampled = msmv_sampling(grouped_features, location, weight)
        sampled = sampled.view(batch, self.num_groups, queries, -1, self.num_points).permute(0, 2, 1, 4, 3)
        # Restore the grouped-channel layout expected by the common sampling
        # interface; each point owns exactly its corresponding channel group.
        full = sampled.new_zeros(*sampled.shape[:-1], self.num_groups, sampled.shape[-1])
        for group_index in range(self.num_groups):
            full[:, :, group_index, :, group_index, :] = sampled[:, :, group_index]
        visible = valid.any(dim=1)
        return full.reshape(batch, locations, -1), visible


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
        self.sampling = _OfficialOPUSSampling(embed_dims, pc_range=pc_range)
        self.mixing = _AdaptiveMixing(embed_dims, in_points=4, n_groups=4, out_points=32)
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

    def forward(self, query, points, feature_maps, metas, query_valid_mask):
        # The full parent point set, rather than its centroid alone, is part
        # of the next-stage query representation in OPUS-V1.
        query = query + self.position_encoder(points.flatten(2))
        sampled_feature, visible = self.sampling(points, query, feature_maps, metas)
        query = self.norm1(self.mixing(sampled_feature, query))
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


# The classes below are a line-for-line semantic port of the official V1
# transformer, adapted only at the boundary from ``img_metas`` to this
# repository's batched ``metas`` tensors.  Keep them separate from the older
# portable ``OfficialOPUSV1Encoder`` above: the latter is retained for legacy
# experiments, while this implementation is the reproduction path.
def _opus_decode_points(points, pc_range):
    lower = points.new_tensor(pc_range[:3])
    upper = points.new_tensor(pc_range[3:])
    return points * (upper - lower) + lower


def _opus_encode_points(points, pc_range):
    lower = points.new_tensor(pc_range[:3])
    upper = points.new_tensor(pc_range[3:])
    return (points - lower) / (upper - lower)


def _official_sampling_4d(sample_points, mlvl_feats, scale_weights, projection,
                          image_wh, num_views, eps=1e-5):
    """Official OPUS-V1 ``sampling_4d`` with batched image sizes.

    ``projection`` is ego-to-image here because the local Occ3D pipeline uses
    ego-frame query coordinates.  Its tensor operations otherwise match the
    official implementation, including one selected valid view per time step.
    """
    batch, queries, frames, groups, points, _ = sample_points.shape
    if projection.shape[1] != frames * num_views:
        raise ValueError('OPUS temporal sampling requires T * num_views projections')
    sample_points = sample_points.reshape(batch, queries, frames, groups * points, 3)
    projection = projection[:, :, None, None].expand(
        batch, frames * num_views, queries, groups * points, 4, 4).reshape(
        batch, frames, num_views, queries, groups * points, 4, 4)
    homogeneous = torch.cat([sample_points, torch.ones_like(sample_points[..., :1])], dim=-1)
    homogeneous = homogeneous[:, :, None, ..., None].expand(
        batch, queries, num_views, frames, groups * points, 4, 1).transpose(1, 3)
    projected = torch.matmul(projection, homogeneous).squeeze(-1)
    depth = projected[..., 2:3]
    safe_depth = torch.maximum(depth, torch.full_like(depth, eps))
    uv = projected[..., :2] / safe_depth
    image_wh = image_wh.reshape(batch, frames, num_views, 2)
    uv = uv / image_wh[:, :, :, None, None, :]
    valid = ((depth > eps) & (uv[..., 1:2] > 0.) & (uv[..., 1:2] < 1.) &
             (uv[..., 0:1] > 0.) & (uv[..., 0:1] < 1.)).squeeze(-1).float()
    valid = valid.permute(0, 1, 3, 4, 2)
    uv = uv.permute(0, 1, 3, 4, 2, 5)

    # This indexing is intentionally the same argmax-based one-view selection
    # used by the official implementation.
    selected_view = torch.argmax(valid, dim=-1, keepdim=True)
    batch_index = torch.arange(batch, device=sample_points.device).view(batch, 1, 1, 1, 1)
    time_index = torch.arange(frames, device=sample_points.device).view(1, frames, 1, 1, 1)
    query_index = torch.arange(queries, device=sample_points.device).view(1, 1, queries, 1, 1)
    point_index = torch.arange(groups * points, device=sample_points.device).view(1, 1, 1, groups * points, 1)
    selected_uv = uv[batch_index, time_index, query_index, point_index, selected_view, :]
    selected_valid = valid[batch_index, time_index, query_index, point_index, selected_view]
    selected_uv = torch.cat([
        selected_uv,
        selected_view[..., None].to(selected_uv.dtype) / (num_views - 1),
    ], dim=-1)
    locations = selected_uv.reshape(batch, frames, queries, groups, points, 1, 3)
    locations = locations.permute(0, 1, 3, 2, 4, 5, 6).reshape(
        batch * frames * groups, queries, points, 3).contiguous()
    weights = scale_weights.reshape(batch, queries, groups, frames, points, -1)
    weights = weights.permute(0, 2, 3, 1, 4, 5).reshape(
        batch * groups * frames, queries, points, -1).contiguous()
    sampled = msmv_sampling(mlvl_feats, locations, weights)
    channels = sampled.shape[2]
    sampled = sampled.reshape(batch, frames, groups, queries, channels, points)
    sampled = sampled.permute(0, 3, 2, 1, 5, 4).flatten(3, 4)
    return sampled, selected_valid.reshape(batch, frames, queries, groups, points)


class _StrictOPUSSampling(nn.Module):
    def __init__(self, embed_dims, num_frames, num_views, num_groups, num_points,
                 num_levels, pc_range):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_points = num_points
        self.num_levels = num_levels
        self.pc_range = tuple(pc_range)
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(self.sampling_offset.bias.view(num_groups * num_points, 3), -0.5, 0.5)

    def forward(self, query_points, query_features, mlvl_feats, metas):
        batch, queries = query_points.shape[:2]
        decoded = _opus_decode_points(query_points, self.pc_range)
        if decoded.shape[2] == 1:
            center, spread = decoded, torch.zeros_like(decoded)
        else:
            center = decoded.mean(dim=2, keepdim=True)
            spread = decoded.std(dim=2, keepdim=True)
        offsets = self.sampling_offset(query_features).view(batch, queries, -1, 3)
        sample_points = (center + offsets * spread).view(
            batch, queries, self.num_groups, self.num_points, 3)
        sample_points = sample_points[:, :, None].expand(
            batch, queries, self.num_frames, self.num_groups, self.num_points, 3)
        weights = self.scale_weights(query_features).view(
            batch, queries, self.num_groups, 1, self.num_points, self.num_levels)
        weights = weights.softmax(dim=-1).expand(
            batch, queries, self.num_groups, self.num_frames, self.num_points, self.num_levels)
        return _official_sampling_4d(
            sample_points, mlvl_feats, weights, metas['projection_mat'].to(query_features),
            metas['image_wh'].to(query_features), self.num_views)


class _StrictOPUSSelfAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, dropout, pc_range):
        super().__init__()
        self.pc_range = tuple(pc_range)
        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0., 2.)

    def forward(self, query_points, query_features):
        decoded = _opus_decode_points(query_points, self.pc_range).mean(dim=2)
        distance = torch.cdist(decoded, decoded)
        tau = self.gen_tau(query_features).permute(0, 2, 1)
        mask = (-distance[:, None] * tau[..., None]).flatten(0, 1)
        # Official MMCV attention consumes [B * heads, Q, Q].  PyTorch 2.0's
        # eval/no_grad native-MHA fast path forwards that 3-D mask unchanged
        # to a kernel expecting [B, heads, Q, Q], while its training fallback
        # correctly accepts the official layout.  Disable only that fast path
        # for this call to keep train/eval and PyTorch versions consistent.
        mha_backend = getattr(torch.backends, 'mha', None)
        disable_fastpath = (
            not self.training and not torch.is_grad_enabled() and
            mha_backend is not None and
            hasattr(mha_backend, 'get_fastpath_enabled') and
            hasattr(mha_backend, 'set_fastpath_enabled'))
        previous_fastpath = None
        if disable_fastpath:
            previous_fastpath = mha_backend.get_fastpath_enabled()
            mha_backend.set_fastpath_enabled(False)
        try:
            output, _ = self.attention(query_features, query_features, query_features,
                                       attn_mask=mask, need_weights=False)
        finally:
            if disable_fastpath:
                mha_backend.set_fastpath_enabled(previous_fastpath)
        # MMCV's MultiheadAttention adds the identity internally.
        return query_features + output


class _StrictAdaptiveMixing(_AdaptiveMixing):
    def __init__(self, embed_dims, in_points, n_groups=4, out_points=32):
        super().__init__(embed_dims, in_points, n_groups, out_points)
        nn.init.zeros_(self.generator.weight)


class _StrictOPUSDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_levels,
                 num_groups, num_classes, last_refine, num_refine, num_heads,
                 feedforward_channels, dropout, scale, pc_range):
        super().__init__()
        self.num_refine = num_refine
        self.scale = scale
        self.pc_range = tuple(pc_range)
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * last_refine, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.sampling = _StrictOPUSSampling(embed_dims, num_frames, num_views, num_groups,
                                            num_points, num_levels, pc_range)
        self.mixing = _StrictAdaptiveMixing(embed_dims, num_frames * num_points, num_groups, 32)
        self.self_attn = _StrictOPUSSelfAttention(embed_dims, num_heads, dropout, pc_range)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims),
                                 nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        cls_branch = []
        for _ in range(2):
            cls_branch.extend([nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True)])
        cls_branch.append(nn.Linear(embed_dims, num_classes * num_refine))
        self.cls_branch = nn.Sequential(*cls_branch)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3 * num_refine))
        nn.init.constant_(self.cls_branch[-1].bias, -4.59511985013459)

    def forward(self, query_points, query_features, mlvl_feats, metas):
        query_features = query_features + self.position_encoder(query_points.flatten(2))
        sampled, visible = self.sampling(query_points, query_features, mlvl_feats, metas)
        query_features = self.norm1(self.mixing(sampled, query_features))
        query_features = self.norm2(self.self_attn(query_points, query_features))
        query_features = self.norm3(query_features + self.ffn(query_features))
        batch, queries = query_points.shape[:2]
        logits = self.cls_branch(query_features).view(batch, queries, self.num_refine, -1)
        offsets = self.scale * self.reg_branch(query_features).view(batch, queries, self.num_refine, 3)
        proposal = _opus_decode_points(query_points, self.pc_range).mean(dim=2, keepdim=True)
        refined = _opus_encode_points(proposal + offsets, self.pc_range)
        return query_features, logits, refined, visible


@MODELS.register_module()
class StrictOPUSV1Encoder(BaseModule):
    """Official OPUS-V1 decoder semantics on the local segmentor interface."""
    def __init__(self, embed_dims=256, num_decoder=6, num_frames=8, num_views=6,
                 num_points=4, num_levels=4, num_groups=4, num_heads=8,
                 feedforward_channels=512, dropout=0.1, num_classes=17,
                 num_refines=(1, 4, 16, 32, 64, 128), scales=(0.5,),
                 pc_range=(-40., -40., -1., 40., 40., 5.4), init_cfg=None):
        super().__init__(init_cfg)
        if len(num_refines) != num_decoder:
            raise ValueError('num_refines must provide one point count per decoder layer')
        if len(scales) == 1:
            scales = tuple(scales) * num_decoder
        if len(scales) != num_decoder:
            raise ValueError('scales must have one value per decoder layer')
        previous = (1,) + tuple(num_refines[:-1])
        self.num_frames, self.num_views, self.num_groups = num_frames, num_views, num_groups
        self.layers = nn.ModuleList([
            _StrictOPUSDecoderLayer(embed_dims, num_frames, num_views, num_points, num_levels,
                                    num_groups, num_classes, last, current, num_heads,
                                    feedforward_channels, dropout, scale, pc_range)
            for last, current, scale in zip(previous, num_refines, scales)
        ])

    def forward(self, query_features, query_points, ms_img_feats, metas,
                query_valid_mask=None, **kwargs):
        if ms_img_feats[0].shape[1] != self.num_frames * self.num_views:
            raise ValueError('OPUS feature camera dimension must equal num_frames * num_views')
        if query_valid_mask is None:
            query_valid_mask = torch.ones(query_features.shape[:2], device=query_features.device,
                                          dtype=torch.bool)
        grouped_features = []
        batch = query_features.shape[0]
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
        for layer in self.layers:
            query_features, logits, points, visible = layer(
                points, query_features, grouped_features, metas)
            representation.append({
                'query_features': query_features,
                'query_points': points,
                'opus_logits': logits,
                'query_valid_mask': query_valid_mask,
                'visible': visible,
            })
            points = points.detach()
        return {'representation': representation}
