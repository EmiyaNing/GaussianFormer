"""SparseWorldStrict: native current-runtime port of the released detector."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import deque
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS, builder

from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from model.encoder.sparseworld_strict_transformer import SparseWorldCrossAttention


NUSC_CLASS_FREQUENCIES = np.asarray([
    1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808,
    2457947, 497017, 2731022, 7224789, 214411435, 5565043, 63191967,
    76098082, 128860031, 141625221, 2307405309,
])


def _rgb_to_hsv(image, eps=1e-8):
    image = image / 255.
    maximum, argmax = image.max(-3)
    minimum = image.min(-3).values
    delta = maximum - minimum
    value = maximum
    saturation = delta / (maximum + eps)
    safe_delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    rc, gc, bc = torch.unbind(maximum.unsqueeze(-3) - image, dim=-3)
    hue = torch.stack((bc - gc, rc - bc + 2. * safe_delta,
                       gc - rc + 4. * safe_delta), dim=-3)
    hue = hue / safe_delta.unsqueeze(-3)
    hue = torch.gather(hue, -3, argmax.unsqueeze(-3)).squeeze(-3)
    hue = (hue / 6.) % 1.
    return torch.stack((hue * 360., saturation, value * 255.), dim=-3)


def _hsv_to_rgb(image):
    hue, saturation, value = image[:, 0] / 360., image[:, 1], image[:, 2] / 255.
    hi = torch.floor(hue * 6.) % 6
    fraction = (hue * 6.) % 6 - hi
    one = image.new_tensor(1.)
    p = value * (one - saturation)
    q = value * (one - fraction * saturation)
    t = value * (one - (one - fraction) * saturation)
    hi = hi.long()
    indices = torch.stack((hi, hi + 6, hi + 12), dim=-3)
    output = torch.stack((value, q, p, p, t, value,
                          t, value, value, q, p, p,
                          p, p, t, value, value, q), dim=-3)
    return torch.gather(output, -3, indices) * 255.


@SEGMENTORS.register_module()
class SparseWorldStrict(BaseModule):
    """SparseWorld trajectory model with official top-level state names."""

    def __init__(self, img_backbone, img_neck, pts_bbox_head, out_dim=256,
                 finetune_epoch=5, is_pretrain=False, drop_out=.1,
                 data_aug=None, feature_mode='official_auto', init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.img_backbone = builder.build_backbone(img_backbone)
        try:
            self.img_neck = builder.build_neck(img_neck)
        except Exception:
            self.img_neck = MODELS.build(img_neck)
        self.pts_bbox_head = MODELS.build(pts_bbox_head)
        self.num_query = self.pts_bbox_head.num_query
        self.num_fu_query = self.pts_bbox_head.num_fu_query
        self.num_fu_frames = self.pts_bbox_head.num_fu_frames
        self.num_refines = self.pts_bbox_head.num_refines[-1]
        self.pc_range = self.pts_bbox_head.pc_range
        self.out_dim = int(out_dim)
        self.finetune_epoch = int(finetune_epoch)
        self.curr_epoch = 0
        self.pretrain = bool(is_pretrain)
        self.data_aug = data_aug or {}
        self.feature_mode = feature_mode
        self.memory = {}
        self.memory_queue = deque()

        class_weights = torch.from_numpy(
            1. / np.log(NUSC_CLASS_FREQUENCIES[:17] + .001)).float()
        self.class_weights = class_weights
        self.semantic_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

        self.velocity_dim, self.past_frame = 3, 5
        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 256),
            nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim))
        self.ego_cross_attn = SparseWorldCrossAttention(
            self.out_dim, 8, drop_out, self.pts_bbox_head.pc_range.tolist())
        self.position_encoder = nn.Sequential(
            nn.Linear(4 * self.num_refines, self.out_dim),
            nn.LayerNorm(self.out_dim), nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim), nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True))
        self.reg_branch = self._branch(self.num_refines * 3)
        self.vel_branch = self._branch(self.num_refines * 2)
        self.cls_branch = self._branch(self.num_refines * 17)
        self.points_scale_branch = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3))
        self.traj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2), nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2))

    def _branch(self, output_dims):
        return nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim), nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, output_dims))

    def init_weights(self):
        self.pts_bbox_head.init_weights()
        nn.init.constant_(self.cls_branch[-1].bias, -4.59511985013459)

    @torch.no_grad()
    def set_epoch(self, epoch):
        self.curr_epoch = int(epoch)
        self.pretrain = self.curr_epoch < self.finetune_epoch
        self.pts_bbox_head.pretrain = self.pretrain
        self.memory.clear()
        self.memory_queue.clear()
        if self.pretrain:
            self.pts_bbox_head.num_stamps_all.fill_(1)
            self.pts_bbox_head.update_stamp_assignment(force=True)
        else:
            self.pts_bbox_head.update_stamp_assignment(force=True)
            self.pts_bbox_head.reset_mask()

    def _preprocess_images(self, images):
        batch, cameras, channels, height, width = images.shape
        images = images.reshape(batch * cameras, channels, height, width).float()
        # The released implementation applies GPU photo distortion only in
        # training. The data transform supplies raw BGR uint8 tensors.
        if self.training and self.data_aug.get('img_color_aug', False):
            images = self._photo_distortion(images)
        norm_cfg = self.data_aug.get('img_norm_cfg')
        if norm_cfg:
            if norm_cfg.get('to_rgb', True):
                images = images[:, [2, 1, 0]]
            mean = images.new_tensor(norm_cfg['mean']).view(1, 3, 1, 1)
            std = images.new_tensor(norm_cfg['std']).view(1, 3, 1, 1)
            images = (images - mean) / std
        divisor = self.data_aug.get('img_pad_cfg', {}).get('size_divisor')
        if divisor:
            pad_h = (-images.shape[-2]) % divisor
            pad_w = (-images.shape[-1]) % divisor
            images = F.pad(images, (0, pad_w, 0, pad_h))
        return images, batch, cameras

    @staticmethod
    def _photo_distortion(images):
        """Tensor implementation of SparseWorld's per-image GPU distortion."""
        # Keep this augmentation outside the parameter graph. It matches the
        # released operation ordering; RNG is NumPy and therefore covered by
        # the runner's seed 17.
        images = images[:, [2, 1, 0]].clone()  # BGR -> RGB
        modes = [int(np.random.randint(2)) for _ in range(images.shape[0])]
        for index in range(images.shape[0]):
            if np.random.randint(2):
                images[index].add_(float(np.random.uniform(-32., 32.)))
            if modes[index] == 0 and np.random.randint(2):
                images[index].mul_(float(np.random.uniform(.5, 1.5)))
        images = _rgb_to_hsv(images)
        for index in range(images.shape[0]):
            if np.random.randint(2):
                images[index, 1].mul_(float(np.random.uniform(.5, 1.5)))
            if np.random.randint(2):
                images[index, 0].add_(float(np.random.uniform(-18., 18.)))
        images[:, 0][images[:, 0] > 360.] -= 360.
        images[:, 0][images[:, 0] < 0.] += 360.
        images = _hsv_to_rgb(images)
        for index in range(images.shape[0]):
            if modes[index] == 1 and np.random.randint(2):
                images[index].mul_(float(np.random.uniform(.5, 1.5)))
            if np.random.randint(2):
                permutation = torch.as_tensor(np.random.permutation(3), device=images.device)
                images[index] = images[index, permutation]
        return images[:, [2, 1, 0]]  # RGB -> BGR

    def _extract_feat_batch(self, images):
        images, batch, cameras = self._preprocess_images(images)
        features = self.img_backbone(images)
        if isinstance(features, dict):
            features = list(features.values())
        features = self.img_neck(features)
        if isinstance(features, dict):
            features = features['fpn_out']
        return [feature.view(batch, cameras, *feature.shape[1:]) for feature in features]

    def extract_feat(self, images, metas=None):
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        online = (not self.training and images.shape[0] == 1 and
                  (self.feature_mode == 'official_online' or
                   (self.feature_mode == 'official_auto' and world_size == 1)))
        if not online:
            return self._extract_feat_batch(images)
        if images.shape[1] % 6:
            raise ValueError('online SparseWorld features require six cameras per frame')
        filenames = metas.get('img_filename') if metas is not None else None
        if filenames is not None:
            filenames = filenames[0]
        by_frame = []
        for frame in range(images.shape[1] // 6):
            key = (filenames[frame * 6] if filenames is not None else
                   f'uncached-{frame}')
            if key in self.memory:
                features = self.memory[key]
            else:
                features = self._extract_feat_batch(
                    images[:, frame * 6:(frame + 1) * 6])
                if filenames is not None:
                    self.memory[key] = features
                    self.memory_queue.append(key)
                    while len(self.memory_queue) >= 16:
                        self.memory.pop(self.memory_queue.popleft(), None)
            by_frame.append(features)
        return [torch.cat([frame[level] for frame in by_frame], dim=1)
                for level in range(len(by_frame[0]))]

    def refine_points(self, proposal, delta):
        batch, queries = delta.shape[:2]
        delta = delta.reshape(batch, queries, self.num_refines, 3)
        proposal = _opus_decode_points(proposal, self.pc_range).mean(2, keepdim=True)
        return _opus_encode_points(proposal + delta, self.pc_range)

    def _forecast(self, outputs, ego_feature, metas):
        batch = ego_feature.shape[0]
        stamps = self.pts_bbox_head.ind_stamps_all
        all_features = outputs['query_feat']
        all_points = outputs['all_refine_pts'][-1]
        all_classes = outputs['all_cls_scores'][-1]
        features = all_features[:, stamps == 0]
        points = all_points[:, stamps == 0].detach()
        timestamp = points.new_zeros(batch, self.num_query, self.num_refines, 1)
        horizon = (self.num_fu_frames if not self.training else
                   max(1, min(self.curr_epoch - self.finetune_epoch + 1,
                              self.num_fu_frames)))
        forecast_points, forecast_classes = [], []
        forecast_features, trajectories, range_masks = [], [], []
        for interval in range(horizon):
            fused, _ = self.ego_cross_attn(
                ego_feature.new_ones(batch, 1, 3) * .5, ego_feature,
                points.detach(), features.detach())
            trajectories.append(self.traj_head(fused))
            future = stamps == interval + 1
            features = torch.cat((features, all_features[:, future]), dim=1)
            points = torch.cat((points, all_points[:, future]), dim=1).detach()
            timestamp = torch.cat((timestamp, points.new_full(
                (batch, int(future.sum()), self.num_refines, 1), .5)), dim=1)
            position = self.position_encoder(
                torch.cat((points, timestamp), dim=-1).flatten(2, 3))
            features = features + fused + position
            offset = self.reg_branch(features).unflatten(-1, (-1, 3)) * .5
            classes = self.cls_branch(features).unflatten(-1, (-1, 17))
            velocity = self.vel_branch(features).unflatten(-1, (-1, 2))
            labels = classes.argmax(-1)
            moving = ((labels >= 2) & (labels <= 10)).unsqueeze(-1)
            offset = torch.cat((offset[..., :2] + velocity * moving,
                                offset[..., 2:]), dim=-1)
            points = self.refine_points(points, offset.flatten(2, 3))
            forecast_classes.append(classes)
            forecast_points.append(points)
            forecast_features.append(features)
            if self.training:
                ego2lidar = metas['ego2lidar'].to(points)
                world = _opus_decode_points(points, self.pc_range)
                lidar = world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
                lidar = lidar + ego2lidar[:, None, None, :3, 3]
                trajectory = metas['temporal_trajs'][:, interval].to(points)
                lidar = lidar + torch.cat((-trajectory, trajectory.new_zeros(
                    trajectory.shape[0], 1)), dim=-1)[:, None, None]
                lidar2ego = torch.linalg.inv(ego2lidar)
                shifted = lidar @ lidar2ego[:, None, :3, :3].transpose(-1, -2)
                shifted = shifted + lidar2ego[:, None, None, :3, 3]
                shifted = _opus_encode_points(shifted, self.pc_range)
                range_masks.append(shifted[..., 0] >= 0)
        if not self.pretrain and len(trajectories) < self.num_fu_frames:
            fused, _ = self.ego_cross_attn(
                ego_feature.new_zeros(batch, 1, 3), ego_feature, points, features)
            trajectories.append(self.traj_head(fused))
        return (forecast_points, forecast_classes, forecast_features,
                trajectories, range_masks)

    def forward(self, imgs=None, metas=None, **kwargs):
        if imgs is None or metas is None:
            raise ValueError('SparseWorldStrict requires imgs and metas')
        state = metas['temporal_ego_states'].to(imgs).reshape(imgs.shape[0], 1, -1)
        if state.shape[-1] != 21:
            raise ValueError(f'temporal_ego_states must have 21 values, got {state.shape[-1]}')
        ego_feature = self.plan_head(state)
        scale = torch.tanh(self.points_scale_branch(ego_feature))
        scale = (scale + 1.) * .35 + .8
        features = self.extract_feat(imgs, metas)
        outputs = self.pts_bbox_head(features, metas, points_scale=scale)
        (forecast_points, forecast_classes, forecast_features,
         trajectories, range_masks) = self._forecast(outputs, ego_feature, metas)
        stamps = self.pts_bbox_head.ind_stamps_all
        current_dict = dict(
            cls_scores=outputs['all_cls_scores'][-1][:, stamps == 0],
            refine_pts=outputs['all_refine_pts'][-1][:, stamps == 0])
        current_occ = self.pts_bbox_head.get_occ(current_dict)
        future_occ = [self.pts_bbox_head.get_occ(dict(
            cls_scores=classes, refine_pts=points))
            for points, classes in zip(forecast_points, forecast_classes)]

        result = dict(outputs)
        current_stage_points = [
            _opus_decode_points(value[:, stamps == 0], self.pc_range).flatten(1, 2)
            for value in outputs['all_refine_pts']]
        current_stage_logits = [
            value[:, stamps == 0].flatten(1, 2)
            for value in outputs['all_cls_scores']]
        result.update(
            cls_score=current_dict['cls_scores'], refine_pts=current_dict['refine_pts'],
            forecast_points_list=forecast_points,
            forecast_semantics_list=forecast_classes,
            forecast_features_list=forecast_features,
            forecast_points_mask_list=range_masks,
            pred_trajs_list=trajectories,
            pred_traj=torch.cat(trajectories, dim=1),
            sparseworld_pretrain=self.pretrain,
            query_stamps=stamps,
            stamp_statistics=self.pts_bbox_head.num_stamps_all,
            update_stamp_statistics=self.training,
            initial_pred_points=_opus_decode_points(outputs['init_points'], self.pc_range).flatten(1, 2),
            current_pred_points=current_stage_points,
            current_pred_logits=current_stage_logits,
            temporal_pred_points=[_opus_decode_points(x, self.pc_range).flatten(1, 2)
                                  for x in outputs['all_refine_pts']],
            temporal_pred_logits=[x.flatten(1, 2) for x in outputs['all_cls_scores']],
            future_pred_points=[_opus_decode_points(x, self.pc_range).flatten(1, 2)
                                for x in forecast_points],
            future_pred_logits=[x.flatten(1, 2) for x in forecast_classes],
            future_pred_masks=[x.flatten(1) for x in range_masks],
            final_occ=current_occ.flatten(1), final_occ_grid=current_occ,
            future_occ_grids=future_occ,
            sampled_label=metas.get('occ_label'),
            occ_cam_mask=metas.get('occ_cam_mask'),
            occ_lidar_mask=metas.get('occ_lidar_mask'))
        for horizon, occupancy in enumerate([current_occ] + future_occ):
            result[f'semantic_occ_{horizon}s'] = occupancy
            geometry = torch.full_like(occupancy, 17)
            geometry[occupancy != 17] = 0
            result[f'geo_occ_{horizon}s'] = geometry
        return result
