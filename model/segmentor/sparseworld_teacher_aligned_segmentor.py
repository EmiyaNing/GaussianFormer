"""Teacher-aligned SparseWorld with an isolated future-feature MLP.

This module intentionally leaves ``SparseWorldStrictSegmentor`` unchanged.
It starts from the released SparseWorld topology, freezes perception, and
exposes all six native-future query states for teacher supervision.
"""

import copy

import torch
from torch import nn
from mmseg.models import SEGMENTORS

from .sparseworld_strict_segmentor import SparseWorldStrictSegmentor


@SEGMENTORS.register_module()
class SparseWorldTeacherAlignedSegmentor(SparseWorldStrictSegmentor):
    """Fine-tune only SparseWorld future prediction against frozen teachers."""

    def __init__(self, feature_mlp_dims=512, feature_mlp_dropout=0.1,
                 alignment_dims=256, freeze_perception=True, **kwargs):
        super().__init__(**kwargs)
        self.freeze_perception = bool(freeze_perception)
        dims = int(kwargs.get('embed_dims', 256))
        # SparseWorld's original plan_head also scales perception anchors.
        # Keep that copy frozen and use this independently trainable clone for
        # future conditioning, so perception cannot drift during fine-tuning.
        self.future_plan_head = copy.deepcopy(self.plan_head)
        self.future_feature_mlp = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, int(feature_mlp_dims)),
            nn.GELU(),
            nn.Dropout(float(feature_mlp_dropout)),
            nn.Linear(int(feature_mlp_dims), dims))
        self.student_projection = nn.Linear(dims, int(alignment_dims),
                                            bias=False)
        self._reset_alignment_layers()
        self._apply_perception_freeze()

    def _reset_alignment_layers(self):
        # Zero output preserves exact SparseWorld rollout behavior at init.
        nn.init.zeros_(self.future_feature_mlp[-1].weight)
        nn.init.zeros_(self.future_feature_mlp[-1].bias)
        if self.student_projection.weight.shape[0] == \
                self.student_projection.weight.shape[1]:
            nn.init.eye_(self.student_projection.weight)
        else:
            nn.init.orthogonal_(self.student_projection.weight)

    def init_weights(self):
        super().init_weights()
        self._reset_alignment_layers()

    @torch.no_grad()
    def initialize_future_from_pretrained(self):
        """Copy the checkpoint-loaded plan encoder into the future branch."""
        self.future_plan_head.load_state_dict(self.plan_head.state_dict(),
                                              strict=True)

    def _perception_modules(self):
        # All modules that create the current sparse representation are fixed.
        # The original plan_head affects perception anchors and stays fixed;
        # future_plan_head is its independently trainable initialized clone.
        return (self.img_backbone, self.img_neck, self.lifter, self.encoder,
                self.head, self.plan_head, self.points_scale)

    def _apply_perception_freeze(self):
        if not self.freeze_perception:
            return
        for module in self._perception_modules():
            module.eval()
            module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._apply_perception_freeze()
        return self

    @torch.no_grad()
    def set_epoch(self, epoch):
        # The model is initialized from a completed SparseWorld checkpoint;
        # only the future prediction path is fine-tuned.
        self.current_epoch = int(epoch)
        self.pretrain = False
        self.lifter.set_pretrain(False)

    def _frozen_perception(self, imgs, metas):
        with torch.no_grad():
            result = dict(imgs=imgs, metas=metas, points=None)
            result.update(self.extract_img_feat(**result))
            result.update(self.lifter(**result))
            ego_state = metas['temporal_ego_states'].to(
                result['query_points']).reshape(imgs.shape[0], -1)
            frozen_ego = self.plan_head(ego_state).unsqueeze(1)
            scale = (torch.tanh(self.points_scale(frozen_ego)) + 1.) * .35 + .8
            result['query_points'] = result['query_points'] * scale
            result.update(self.encoder(**result))
            # The legacy rasterizer mixes geometry-derived FP32 thresholds
            # with decoder logits.  Give the frozen head an FP32 view so this
            # isolated AMP trainer does not mutate the shared head code.
            head_representation = []
            for stage in result['representation']:
                head_stage = dict(stage)
                head_stage['query_points'] = stage['query_points'].float()
                head_stage['opus_logits'] = stage['opus_logits'].float()
                head_representation.append(head_stage)
            result.update(self.head(
                representation=head_representation,
                query_stamps=result['query_stamps'], metas=metas))
        # Compute future conditioning with the decoupled trainable clone.
        ego_state = metas['temporal_ego_states'].to(
            result['query_points']).reshape(imgs.shape[0], -1)
        result['future_ego_feature'] = self.future_plan_head(
            ego_state).unsqueeze(1)
        return result

    def _teacher_aligned_rollout(self, result):
        encoded = result['representation'][-1]
        all_features = encoded['query_features'].detach()
        all_points = encoded['query_points'].detach()
        stamps = result['query_stamps']
        persistent = stamps == 0
        features = all_features[:, persistent]
        positions = all_points[:, persistent]
        timestamp = positions.new_zeros(*positions.shape[:-1], 1)
        ego = result['future_ego_feature']

        pred_points, pred_logits, trajectories = [], [], []
        query_states = []
        active_stamps = stamps[persistent]
        active_slot_ids = torch.nonzero(
            persistent, as_tuple=False).flatten()
        for step in range(self.future_steps):
            fused = self.ego_attention(
                ego, features.detach(), positions.detach())
            trajectories.append(self.traj(fused))

            born = stamps == step + 1
            features = torch.cat(
                [features, all_features[:, born]], dim=1)
            positions = torch.cat(
                [positions, all_points[:, born]], dim=1).detach()
            timestamp = torch.cat([
                timestamp,
                timestamp.new_full(
                    (timestamp.shape[0], int(born.sum()), self.num_refines, 1),
                    .5)
            ], dim=1)
            active_stamps = torch.cat([active_stamps, stamps[born]])
            active_slot_ids = torch.cat([
                active_slot_ids,
                torch.nonzero(born, as_tuple=False).flatten()])

            position_embedding = self.position(torch.cat(
                [positions, timestamp], dim=-1).flatten(2))
            mixed_features = (
                features + fused.expand(-1, features.shape[1], -1) +
                position_embedding)
            feature_delta = self.future_feature_mlp(mixed_features)
            features = mixed_features + feature_delta

            semantic = self.semantic(features).view(
                features.shape[0], features.shape[1], self.num_refines, 17)
            offset = self.reg(features).view(
                features.shape[0], features.shape[1], self.num_refines, 3) * .5
            velocity = self.velocity(features).view(
                features.shape[0], features.shape[1], self.num_refines, 2)
            moving = ((semantic.argmax(-1) >= 2) &
                      (semantic.argmax(-1) <= 10)).unsqueeze(-1)
            offset = torch.cat([
                offset[..., :2] + velocity * moving,
                offset[..., 2:]
            ], dim=-1)

            # Keep this step differentiable, but truncate geometry recurrence
            # before passing the state into the next future step.
            predicted_positions = self._refine(positions, offset)
            metric_points = self._world(predicted_positions)
            pred_points.append(metric_points.flatten(1, 2))
            pred_logits.append(semantic.flatten(1, 2))
            query_states.append(dict(
                query_features=features,
                aligned_features=self.student_projection(features),
                query_points_metric=metric_points,
                semantic_logits=semantic.mean(2),
                point_logits=semantic,
                feature_delta=feature_delta,
                query_stamps=active_stamps,
                slot_ids=active_slot_ids,
                persistent_mask=active_stamps == 0,
                future_born_mask=active_stamps > 0,
                coordinate_system='native_future_ego',
                horizon_index=step + 1))
            positions = predicted_positions.detach()

        return dict(
            points=pred_points, logits=pred_logits,
            trajectories=trajectories, query_states=query_states)

    def forward(self, imgs=None, metas=None, points=None, **kwargs):
        if imgs is None or metas is None:
            raise ValueError('imgs and metas are required')
        # The data loader may carry extra teacher-only history/future images.
        # Student perception is strictly limited to current + four histories.
        student_cameras = self.encoder.num_frames * self.encoder.num_views
        if imgs.shape[1] < student_cameras:
            raise ValueError('insufficient images for SparseWorld student')
        student_imgs = imgs[:, :student_cameras]
        student_metas = dict(metas)
        student_metas['projection_mat'] = metas[
            'projection_mat'][:, :student_cameras]
        student_metas['image_wh'] = metas['image_wh'][:, :student_cameras]

        result = self._frozen_perception(student_imgs, student_metas)
        result.update(
            sampled_label=metas['occ_label'].flatten(1),
            sampled_xyz=(None if metas.get('occ_xyz') is None else
                         metas['occ_xyz'].flatten(1, -2)),
            occ_cam_mask=metas.get('occ_cam_mask'),
            occ_lidar_mask=metas.get('occ_lidar_mask'),
            occ_loss_mask=metas.get('occ_loss_mask'))

        # Preserve current/all-query outputs for the established OPUS losses.
        result['temporal_pred_points'] = [
            self._world(stage['query_points'].flatten(1, 2))
            for stage in result['representation']]
        result['temporal_pred_logits'] = [
            stage['opus_logits'].flatten(1, 2)
            for stage in result['representation']]

        rollout = self._teacher_aligned_rollout(result)
        result.update(
            future_pred_points=rollout['points'],
            future_pred_logits=rollout['logits'],
            future_predictions=rollout['points'],
            future_logits=rollout['logits'],
            future_query_states=rollout['query_states'],
            pred_traj=torch.cat(rollout['trajectories'], dim=1),
            future_prediction_coordinate='native_future',
            sparseworld_pretrain=False)
        return result
