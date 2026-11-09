"""Temporal camera aggregation without materializing query-wise mixing MLPs."""

import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS

from .deformable_module import DeformableFeatureAggregation


@MODELS.register_module()
class TemporalDeformableFeatureAggregation(DeformableFeatureAggregation):
    def __init__(self, num_image_frames=4, num_cameras=6,
                 initial_history_logit_bias=-2.0, history_enabled=True,
                 fuse_temporal_sampling=False, **kwargs):
        expected = num_image_frames * num_cameras
        cameras = kwargs.pop('num_cams', expected)
        if cameras != expected or num_image_frames < 1 or num_cameras < 1:
            raise ValueError('Temporal DAF num_cams must equal num_image_frames * num_cameras')
        super().__init__(num_cams=cameras, **kwargs)
        if self.camera_encoder is None:
            raise ValueError('Temporal DAF requires shared camera embedding weights')
        self.num_image_frames = int(num_image_frames)
        self.num_cameras = int(num_cameras)
        self.history_enabled = bool(history_enabled)
        self.fuse_temporal_sampling = bool(fuse_temporal_sampling)
        if self.fuse_temporal_sampling and not (self.fuse_normalized_sampling and self.fuse_point_sum):
            raise ValueError('Fused temporal sampling requires normalized sampling and point sum')
        # Additive biases live inside the joint visibility-masked softmax.
        # Current-frame bias is fixed at zero; history starts conservative
        # when warm-starting a single-frame checkpoint and remains learnable.
        self.history_logit_bias = nn.Parameter(
            torch.full((num_image_frames - 1,), float(initial_history_logit_bias)))

    def forward(self, instance_feature, anchor, anchor_embed, feature_maps, metas, **kwargs):
        if metas['projection_mat'].shape[1] != self.num_cams:
            raise ValueError('Temporal DAF needs a projection for every camera/time')
        batch = instance_feature.shape[0]
        valid = metas['image_frame_valid'].to(device=instance_feature.device, dtype=torch.bool)
        if tuple(valid.shape) != (batch, self.num_image_frames):
            raise ValueError('image_frame_valid must have shape [batch, num_image_frames]')
        if not self.history_enabled:
            valid = valid.clone()
            valid[:, 1:] = False
        bias = torch.cat([self.history_logit_bias.new_zeros(1), self.history_logit_bias])
        temporal_metas = dict(metas)
        temporal_metas['camera_valid_mask'] = valid.repeat_interleave(self.num_cameras, dim=1)
        temporal_metas['camera_logit_bias'] = bias.repeat_interleave(self.num_cameras)[None].expand(batch, -1)
        if self.fuse_temporal_sampling:
            from .temporal_ops.sampling import temporal_sampling
            points = self.kps_generator(anchor, instance_feature)
            camera_embed = self.camera_encoder(
                metas['projection_mat'][:, :, :3].reshape(batch, self.num_cams, -1))
            query = self.weights_fc(instance_feature + anchor_embed).reshape(
                batch, instance_feature.shape[1], self.num_levels, self.num_pts, self.num_groups)
            camera = F.linear(camera_embed, self.weights_fc.weight).reshape(
                batch, self.num_cams, self.num_levels, self.num_pts, self.num_groups)
            camera = camera + temporal_metas['camera_logit_bias'][:, :, None, None, None]
            features = temporal_sampling(
                points, metas['projection_mat'], metas['image_wh'],
                temporal_metas['camera_valid_mask'], kwargs['temporal_spatial_shapes'],
                kwargs['temporal_level_starts'], query, camera, kwargs['temporal_img_feats'],
                self.attn_drop if self.training else 0.)
            return self._finish(features, instance_feature)
        return super().forward(instance_feature, anchor, anchor_embed,
                               feature_maps, temporal_metas, **kwargs)
