"""SparseWorld baseline with residual future feature alignment."""

from contextlib import nullcontext

import torch
from mmseg.models import SEGMENTORS

from model.future_feature_alignment import FutureFeatureAlignment
from .sparseworld_strict_segmentor import SparseWorldStrictSegmentor


@SEGMENTORS.register_module()
class SparseWorldFFASegmentor(SparseWorldStrictSegmentor):
    """Predict the sixth future sparse state from 720 persistent queries."""

    def __init__(self, target_horizon_index=6, num_student_frames=5,
                 freeze_student_perception=True, ffa=None, **kwargs):
        super().__init__(**kwargs)
        self.target_horizon_index = int(target_horizon_index)
        self.num_student_frames = int(num_student_frames)
        self.freeze_student_perception = bool(freeze_student_perception)
        ffa = dict(ffa or {})
        ffa.setdefault('embed_dims', kwargs.get('embed_dims', 256))
        ffa.setdefault('num_points', kwargs.get('num_refines', 48))
        ffa.setdefault('target_horizon_index', self.target_horizon_index)
        self.ffa = FutureFeatureAlignment(**ffa)
        self._apply_ffa_freeze()

    def _perception_modules(self):
        return (self.img_backbone, self.img_neck, self.lifter, self.encoder,
                self.plan_head, self.points_scale, self.ego_attention,
                self.position, self.reg, self.velocity, self.traj)

    def _apply_ffa_freeze(self):
        if not self.freeze_student_perception:
            return
        for module in self._perception_modules():
            module.eval()
            module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._apply_ffa_freeze()
        return self

    @torch.no_grad()
    def set_epoch(self, epoch):
        # SparseWorld's original pretrain/finetune curriculum is intentionally
        # bypassed: FFA is initialized from a finished SparseWorld checkpoint.
        self.current_epoch = int(epoch)
        self.pretrain = False
        self.ffa.set_epoch(epoch)

    @staticmethod
    def _slice_metas(metas, cameras):
        output = dict(metas)
        output['projection_mat'] = metas['projection_mat'][:, :cameras]
        output['image_wh'] = metas['image_wh'][:, :cameras]
        return output

    def _student_perception(self, imgs, metas):
        result = dict(imgs=imgs, metas=metas, points=None)
        result.update(self.extract_img_feat(**result))
        result.update(self.lifter(**result))
        ego_state = metas['temporal_ego_states'].to(
            result['query_points']).reshape(imgs.shape[0], -1)
        ego = self.plan_head(ego_state).unsqueeze(1)
        scale = (torch.tanh(self.points_scale(ego)) + 1.) * .35 + .8
        result['query_points'] = result['query_points'] * scale
        result.update(self.encoder(**result))
        result['ffa_ego_feature'] = ego
        return result

    def _baseline_rollout(self, result, encoded, persistent, horizon):
        """Run the pretrained SparseWorld transition, including born queries."""
        all_features = encoded['query_features']
        all_points = encoded['query_points']
        features = all_features[:, persistent]
        positions = all_points[:, persistent].detach()
        timestamp = positions.new_zeros(*positions.shape[:-1], 1)
        return self._forecast_rollout(dict(
            features=features,
            positions=positions,
            timestamp=timestamp,
            ego=result['ffa_ego_feature'],
            all_features=all_features,
            all_points=all_points,
            stamps=result['query_stamps'],
            slot_ids=torch.nonzero(persistent, as_tuple=False).flatten(),
            completed_steps=0), horizon)

    def _future_born_at_target(self, rollout, persistent_queries):
        target = self.target_horizon_index - 1
        target_points = rollout['points'][target]
        target_logits = rollout['logits'][target]
        first_born_point = int(persistent_queries) * self.num_refines
        return (target_points[:, first_born_point:],
                target_logits[:, first_born_point:])

    def _evaluation_predictions(self, result, encoded, persistent,
                                ffa_points, ffa_logits, rollout,
                                future_born_points, future_born_logits):
        """Expose current and intermediate SparseWorld predictions for eval.

        SparseWorld-FFA is trained to correct only the configured target
        horizon. Earlier horizons remain the pretrained SparseWorld rollout;
        returning their provenance prevents them from being mistaken for
        independently FFA-supervised predictions.
        """
        future_points = list(rollout['points'])
        future_logits = list(rollout['logits'])
        auxiliary_points = [None] * len(future_points)
        auxiliary_logits = [None] * len(future_points)
        sources = ['sparseworld_baseline'] * len(future_points)
        coordinates = ['native_future'] * len(future_points)
        target = self.target_horizon_index - 1
        if target >= len(future_points):
            raise ValueError(
                'FFA target horizon exceeds SparseWorld rollout length')
        future_points[target] = ffa_points
        future_logits[target] = ffa_logits
        auxiliary_points[target] = future_born_points
        auxiliary_logits[target] = future_born_logits
        sources[target] = 'sparseworld_ffa'
        coordinates[target] = 'current_ego'
        return dict(
            current_pred_points=self._world(
                encoded['query_points'][:, persistent]).flatten(1, 2),
            current_pred_logits=encoded['opus_logits'][
                :, persistent].flatten(1, 2),
            eval_future_pred_points=future_points,
            eval_future_pred_logits=future_logits,
            eval_future_aux_points=auxiliary_points,
            eval_future_aux_logits=auxiliary_logits,
            eval_future_sources=sources,
            eval_future_coordinates=coordinates)

    def forward(self, imgs, metas, teacher_state=None, **kwargs):
        return_all_horizons = bool(kwargs.pop('return_all_horizons', False))
        cameras = self.num_student_frames * 6
        if imgs.shape[1] < cameras:
            raise ValueError(
                f'SparseWorld-FFA requires at least {cameras} student images')
        student_imgs = imgs[:, :cameras]
        student_metas = self._slice_metas(metas, cameras)
        context = torch.no_grad() if self.freeze_student_perception else nullcontext()
        with context:
            result = self._student_perception(student_imgs, student_metas)

        encoded = result['representation'][-1]
        persistent = result['query_stamps'] == 0
        query_features = encoded['query_features'][:, persistent]
        query_points = encoded['query_points'][:, persistent]
        query_points_metric = self._world(query_points)
        # No ground-truth future pose enters the student.  FFA predicts in the
        # current ego frame; loss/evaluation adapters transform coordinates to
        # the native future frame using target metadata.
        current_to_future = torch.eye(
            4, device=query_points_metric.device, dtype=torch.float32
        ).unsqueeze(0).expand(query_points_metric.shape[0], -1, -1)
        prediction = self.ffa(
            query_features, query_points_metric, current_to_future)
        future_features = prediction['query_features']
        future_logits = self.semantic(future_features).view(
            future_features.shape[0], future_features.shape[1],
            self.num_refines, 17)
        future_points = prediction['points_metric']
        flat_points = future_points.flatten(1, 2)
        flat_logits = future_logits.flatten(1, 2)

        # Preserve SparseWorld's future-born slots. The persistent part is
        # replaced by FFA, while newly activated slots keep the baseline
        # rollout state and participate in the target occupancy objective.
        rollout_horizon = (self.future_steps if return_all_horizons else
                           self.target_horizon_index)
        rollout = self._baseline_rollout(
            result, encoded, persistent, rollout_horizon)
        born_points, born_logits = self._future_born_at_target(
            rollout, query_features.shape[1])
        if teacher_state is not None:
            expected = (query_features.shape[0], query_features.shape[1])
            if teacher_state['query_features'].shape[:2] != expected:
                raise ValueError('teacher/student query cardinality mismatch')
            teacher_state = dict(teacher_state)
            teacher_state['projected_features'] = self.ffa.project_teacher(
                teacher_state['query_features'])

        prediction.update(
            semantic_logits=future_logits.mean(2),
            point_logits=future_logits,
            teacher_state=teacher_state,
            target_horizon_index=self.target_horizon_index)
        ffa_output = dict(
            prediction=prediction,
            future_pred_points=[flat_points],
            future_pred_logits=[flat_logits],
            future_born_points_native=born_points,
            future_born_logits=born_logits)
        result.update(
            ffa_output=ffa_output,
            ffa_future_points=flat_points,
            ffa_future_logits=flat_logits,
            future_prediction_coordinate='current_ego')
        if return_all_horizons:
            if self.training:
                raise RuntimeError(
                    'return_all_horizons is an evaluation-only output')
            result.update(self._evaluation_predictions(
                result, encoded, persistent, flat_points, flat_logits,
                rollout, born_points, born_logits))
        return result
