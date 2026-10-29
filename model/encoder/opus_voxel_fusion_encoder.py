"""OPUS-V1 per-decoder image/sparse-3D-voxel adaptive feature fusion."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .opus_encoder import (
    _StrictAdaptiveMixing, _StrictOPUSDecoderLayer, _StrictOPUSSampling,
    _official_sampling_4d, _opus_decode_points, _opus_encode_points,
)
from ..ops.sparse_voxel_sampling import sample_voxel_memory


class _VoxelFusionSampling(_StrictOPUSSampling):
    def __init__(self, *args, min_sampling_scale=0.4, voxel_neighborhood=1,
                 voxel_kernel_sigma=1.0, voxel_sample_chunk_size=1024, **kwargs):
        super().__init__(*args, **kwargs)
        if min_sampling_scale <= 0:
            raise ValueError('min_sampling_scale must be positive (in metres)')
        self.min_sampling_scale = float(min_sampling_scale)
        self.voxel_neighborhood = int(voxel_neighborhood)
        self.voxel_kernel_sigma = float(voxel_kernel_sigma)
        self.voxel_sample_chunk_size = int(voxel_sample_chunk_size)

    def forward(self, query_points, query_features, mlvl_feats, metas, voxel_memory):
        batch, queries = query_points.shape[:2]
        decoded = _opus_decode_points(query_points, self.pc_range)
        center = decoded.mean(dim=2, keepdim=True)
        if decoded.shape[2] == 1:
            spread = torch.full_like(center, self.min_sampling_scale)
        else:
            spread = decoded.std(dim=2, keepdim=True).clamp_min(self.min_sampling_scale)
        offsets = self.sampling_offset(query_features).view(batch, queries, -1, 3)
        locations = (center + offsets * spread).view(
            batch, queries, self.num_groups, self.num_points, 3)
        image_locations = locations[:, :, None].expand(
            batch, queries, self.num_frames, self.num_groups, self.num_points, 3)
        weights = self.scale_weights(query_features).view(
            batch, queries, self.num_groups, 1, self.num_points, self.num_levels)
        weights = weights.softmax(-1).expand(
            batch, queries, self.num_groups, self.num_frames, self.num_points, self.num_levels)
        image, visible = _official_sampling_4d(
            image_locations, mlvl_feats, weights,
            metas['projection_mat'].to(query_features),
            metas['image_wh'].to(query_features), self.num_views,
            align_scale_weights=True)
        # Explicitly zero missing image samples before point/channel mixing.
        image_valid = visible.permute(0, 2, 3, 1, 4).flatten(3, 4).bool()
        image = image * image_valid[..., None].to(image.dtype)
        lidar, lidar_valid = sample_voxel_memory(
            locations, voxel_memory, self.voxel_neighborhood,
            self.voxel_kernel_sigma, self.voxel_sample_chunk_size)
        sampled = torch.cat([image, lidar], dim=-2)
        return sampled, visible, lidar_valid


class _VoxelFusionDecoderLayer(_StrictOPUSDecoderLayer):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_levels,
                 num_groups, num_classes, last_refine, num_refine, num_heads,
                 feedforward_channels, dropout, scale, pc_range, sampling_cfg):
        super().__init__(embed_dims, num_frames, num_views, num_points, num_levels,
                         num_groups, num_classes, last_refine, num_refine, num_heads,
                         feedforward_channels, dropout, scale, pc_range)
        self.sampling = _VoxelFusionSampling(
            embed_dims, num_frames, num_views, num_groups, num_points,
            num_levels, pc_range, **sampling_cfg)
        # Official Fusion defaults out_points to the combined sample count.
        combined_points = (num_frames + 1) * num_points
        self.mixing = _StrictAdaptiveMixing(
            embed_dims, combined_points, num_groups, combined_points)

    def forward(self, query_points, query_features, mlvl_feats, metas, voxel_memory):
        query_features = query_features + self.position_encoder(query_points.flatten(2))
        sampled, visible, lidar_valid = self.sampling(
            query_points, query_features, mlvl_feats, metas, voxel_memory)
        query_features = self.norm1(self.mixing(sampled, query_features))
        query_features = self.norm2(self.self_attn(query_points, query_features))
        query_features = self.norm3(query_features + self.ffn(query_features))
        batch, queries = query_points.shape[:2]
        logits = self.cls_branch(query_features).view(batch, queries, self.num_refine, -1)
        offsets = self.scale * self.reg_branch(query_features).view(
            batch, queries, self.num_refine, 3)
        proposal = _opus_decode_points(query_points, self.pc_range).mean(dim=2, keepdim=True)
        refined = _opus_encode_points(proposal + offsets, self.pc_range)
        return query_features, logits, refined, visible, lidar_valid


@MODELS.register_module()
class OPUSVoxelFusionEncoder(BaseModule):
    """Fixed-count query refinement over images and a shared 3-D voxel memory.

Each layer uses the same query-dependent physical sampling positions for both
modalities. Gaussian pooling of a local 3x3x3 voxel stencil replaces official
Fusion's bilinear BEV sampling; concatenation and adaptive mixing follow its
point-axis fusion. Refinement and set-loss/head interfaces stay OPUS-V1.
"""

    def __init__(self, embed_dims=256, num_decoder=6, num_frames=8, num_views=6,
                 num_points=4, num_levels=4, num_groups=4, num_heads=8,
                 feedforward_channels=512, dropout=0.1, num_classes=17,
                 num_refines=(1, 4, 16, 32, 64, 128), scales=(0.5,),
                 pc_range=(-40., -40., -1., 40., 40., 5.4),
                 min_sampling_scale=0.4, voxel_neighborhood=1,
                 voxel_kernel_sigma=1.0, voxel_sample_chunk_size=1024,
                 decoder_checkpoint=True, init_cfg=None):
        super().__init__(init_cfg)
        if num_decoder < 1 or len(num_refines) != num_decoder:
            raise ValueError('num_refines must provide one count per decoder layer')
        if min(num_frames, num_points, num_levels, num_groups, num_heads) < 1 or num_views < 2:
            raise ValueError('sampling dimensions must be positive and num_views >= 2')
        if embed_dims % num_groups or embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by groups and attention heads')
        if voxel_neighborhood < 0 or voxel_kernel_sigma <= 0 or voxel_sample_chunk_size < 1:
            raise ValueError('invalid voxel sampling options')
        if len(scales) == 1:
            scales = tuple(scales) * num_decoder
        if len(scales) != num_decoder or min(num_refines) < 1:
            raise ValueError('invalid layer scales or refinement point counts')
        self.embed_dims = embed_dims
        self.num_frames, self.num_views, self.num_groups = num_frames, num_views, num_groups
        self.num_levels = num_levels
        self.decoder_checkpoint = bool(decoder_checkpoint)
        sampling_cfg = dict(
            min_sampling_scale=min_sampling_scale, voxel_neighborhood=voxel_neighborhood,
            voxel_kernel_sigma=voxel_kernel_sigma,
            voxel_sample_chunk_size=voxel_sample_chunk_size)
        previous = (1,) + tuple(num_refines[:-1])
        self.layers = nn.ModuleList([
            _VoxelFusionDecoderLayer(
                embed_dims, num_frames, num_views, num_points, num_levels, num_groups,
                num_classes, last, current, num_heads, feedforward_channels, dropout,
                scale, pc_range, sampling_cfg)
            for last, current, scale in zip(previous, num_refines, scales)
        ])

    def forward(self, query_features, query_points, ms_img_feats, metas, voxel_memory,
                query_valid_mask=None, **kwargs):
        batch, queries, channels = query_features.shape
        if query_points.shape != (batch, queries, 3) or channels != self.embed_dims:
            raise ValueError('invalid fixed query coordinates or feature dimensions')
        if len(ms_img_feats) != self.num_levels:
            raise ValueError('image pyramid does not match num_levels')
        if query_valid_mask is None:
            query_valid_mask = torch.ones((batch, queries), device=query_features.device,
                                          dtype=torch.bool)
        if not bool(query_valid_mask.all()):
            raise ValueError('fixed-count fusion expects valid queries, including fallback slots')
        grouped = []
        for feature in ms_img_feats:
            b, cameras, c, height, width = feature.shape
            if (b, cameras, c) != (batch, self.num_frames * self.num_views, channels):
                raise ValueError('image features must have shape [B,T*V,D,H,W]')
            grouped.append(feature.reshape(
                batch, self.num_frames, self.num_views, self.num_groups,
                channels // self.num_groups, height, width
            ).permute(0, 1, 3, 2, 5, 6, 4).reshape(
                batch * self.num_frames * self.num_groups, self.num_views,
                height, width, channels // self.num_groups).contiguous())
        representation, points = [], query_points.unsqueeze(2)
        for layer in self.layers:
            if self.training and torch.is_grad_enabled() and self.decoder_checkpoint:
                output = checkpoint(layer, points, query_features, grouped, metas,
                                    voxel_memory, use_reentrant=False, preserve_rng_state=True)
            else:
                output = layer(points, query_features, grouped, metas, voxel_memory)
            query_features, logits, points, visible, lidar_valid = output
            representation.append(dict(
                query_features=query_features, query_points=points, opus_logits=logits,
                query_valid_mask=query_valid_mask, visible=visible,
                lidar_sample_valid=lidar_valid))
            points = points.detach()
        return dict(representation=representation)
