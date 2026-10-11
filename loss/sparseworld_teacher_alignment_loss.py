"""Six-horizon occupancy and frozen-teacher alignment objectives."""

import torch
import torch.nn.functional as F
from torch import nn

from model.sparse_latent_dynamics import HungarianSparseMatcher
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class SparseWorldTeacherAlignmentLoss(nn.Module):
    """Retain SparseWorld task loss and add native-frame query supervision."""

    def __init__(self, base_loss, horizon_weights=(.5, .5, .75, .75, 1., 1.),
                 geometry_weight=.25, feature_weight=.1,
                 semantic_weight=.1, residual_weight=1e-3,
                 alignment_warmup_iters=2000, future_only=True, matcher=None):
        super().__init__()
        self.base_loss = OPENOCC_LOSS.build(base_loss)
        self.horizon_weights = tuple(float(value) for value in horizon_weights)
        self.geometry_weight = float(geometry_weight)
        self.feature_weight = float(feature_weight)
        self.semantic_weight = float(semantic_weight)
        self.residual_weight = float(residual_weight)
        self.alignment_warmup_iters = max(0, int(alignment_warmup_iters))
        self.future_only = bool(future_only)
        matcher = dict(matcher or {})
        matcher.setdefault('feature_weight', 0.)
        self.matcher = HungarianSparseMatcher(**matcher)

    @staticmethod
    def _zero(state):
        return state['query_features'].sum() * 0.

    def _matched_losses(self, student, teacher):
        matches = self.matcher(dict(
            points_metric=student['query_points_metric'],
            query_feats=student['query_features'],
            semantic_logits=student['semantic_logits']), dict(
                points_metric=teacher['query_points_metric'],
                query_feats=teacher['query_features'],
                semantic_logits=teacher['semantic_logits']))
        geometry = feature = semantic = self._zero(student)
        match_count = 0
        center_errors = []
        for batch, (student_index, teacher_index) in enumerate(zip(
                matches.prediction, matches.target)):
            if not student_index.numel():
                continue
            student_points = student['query_points_metric'][
                batch, student_index].float()
            teacher_points = teacher['query_points_metric'][
                batch, teacher_index].float()
            # OPUS queries are small unordered point sets.  Use bidirectional
            # nearest-neighbour geometry after rectangular query matching.
            distance = torch.cdist(student_points, teacher_points)
            geometry = geometry + .5 * (
                distance.min(-1).values.mean() +
                distance.min(-2).values.mean())

            student_feature = student['aligned_features'][
                batch, student_index].float()
            teacher_feature = teacher['query_features'][
                batch, teacher_index].float()
            dims = min(student_feature.shape[-1], teacher_feature.shape[-1])
            student_feature = student_feature[..., :dims]
            teacher_feature = teacher_feature[..., :dims]
            feature = feature + F.smooth_l1_loss(
                student_feature, teacher_feature, beta=1.)
            feature = feature + (1. - F.cosine_similarity(
                student_feature, teacher_feature, dim=-1)).mean()

            student_semantic = student['semantic_logits'][
                batch, student_index].float()
            teacher_semantic = teacher['semantic_logits'][
                batch, teacher_index].float().softmax(-1)
            semantic = semantic + F.kl_div(
                student_semantic.log_softmax(-1), teacher_semantic,
                reduction='batchmean')
            center_errors.append((student_points.mean(1) -
                                  teacher_points.mean(1)).norm(dim=-1).mean())
            match_count += int(student_index.numel())
        batch_size = max(student['query_features'].shape[0], 1)
        query_count = student['query_features'].shape[1]
        nan = student['query_features'].new_tensor(float('nan'))
        metrics = dict(
            matched_queries=student['query_features'].new_tensor(
                match_count / batch_size),
            matched_ratio=student['query_features'].new_tensor(
                match_count / (batch_size * query_count)),
            center_l2_m=(torch.stack(center_errors).mean()
                         if center_errors else nan))
        return (geometry / batch_size, feature / batch_size,
                semantic / batch_size, metrics)

    def _task_loss(self, inputs):
        """Skip constant current/encoder losses when perception is frozen."""
        if not self.future_only:
            return self.base_loss(inputs)
        required = ('_set_loss', 'lambda_future_cls', 'lambda_future_pts',
                    'lambda_traj')
        if not all(hasattr(self.base_loss, name) for name in required):
            raise TypeError('future_only requires SparseWorldStrictLoss')
        reference = inputs['future_pred_points'][0]
        total = reference.sum() * 0.
        details = {}
        metas = inputs['metas']
        for step, (points, logits) in enumerate(zip(
                inputs['future_pred_points'], inputs['future_pred_logits'])):
            classification = point = total * 0.
            for batch in range(points.shape[0]):
                one_cls, one_point = self.base_loss._set_loss(
                    points[batch], logits[batch],
                    metas['future_occ_labels'][batch, step].to(points.device))
                classification = classification + one_cls
                point = point + one_point
            classification = classification / points.shape[0]
            point = point / points.shape[0]
            total = total + (
                self.base_loss.lambda_future_cls * classification +
                self.base_loss.lambda_future_pts * point)
            details[f'future_{step + 1}/cls'] = classification.detach()
            details[f'future_{step + 1}/pts'] = point.detach()
        trajectory = F.mse_loss(
            inputs['pred_traj'],
            metas['temporal_trajs'].to(inputs['pred_traj'])[
                :, :inputs['pred_traj'].shape[1]])
        total = total + self.base_loss.lambda_traj * trajectory
        details.update(future=total.detach(), trajectory=trajectory.detach())
        return total, details

    def forward(self, inputs):
        teacher_states = inputs.get('teacher_states')
        student_states = inputs.get('future_query_states')
        if not teacher_states or not student_states:
            raise ValueError('six teacher and student query states are required')
        if len(teacher_states) != len(student_states):
            raise ValueError('teacher/student horizon counts do not match')
        if len(student_states) != len(self.horizon_weights):
            raise ValueError('horizon weights do not match query states')

        task_loss, details = self._task_loss(inputs)
        geometry_total = task_loss.new_zeros(())
        feature_total = task_loss.new_zeros(())
        semantic_total = task_loss.new_zeros(())
        residual_total = task_loss.new_zeros(())
        weight_sum = sum(self.horizon_weights)
        for horizon, (student, teacher, weight) in enumerate(zip(
                student_states, teacher_states, self.horizon_weights), 1):
            if student.get('coordinate_system') != 'native_future_ego':
                raise ValueError('student query state is not in native future ego')
            geometry, feature, semantic, metrics = self._matched_losses(
                student, teacher)
            residual = student['feature_delta'].float().square().mean()
            geometry_total = geometry_total + weight * geometry
            feature_total = feature_total + weight * feature
            semantic_total = semantic_total + weight * semantic
            residual_total = residual_total + weight * residual
            details.update({
                f'align_h{horizon}/geometry': geometry.detach(),
                f'align_h{horizon}/feature': feature.detach(),
                f'align_h{horizon}/semantic': semantic.detach(),
                f'align_h{horizon}/matched_ratio':
                    metrics['matched_ratio'].detach(),
                f'align_h{horizon}/center_l2_m':
                    metrics['center_l2_m'].detach()})
        geometry_total = geometry_total / weight_sum
        feature_total = feature_total / weight_sum
        semantic_total = semantic_total / weight_sum
        residual_total = residual_total / weight_sum
        warmup = 1.
        if self.alignment_warmup_iters:
            warmup = min(1., (int(inputs.get('global_iter', 0)) + 1) /
                         self.alignment_warmup_iters)
        alignment = warmup * (
            self.geometry_weight * geometry_total +
            self.feature_weight * feature_total +
            self.semantic_weight * semantic_total)
        total = task_loss + alignment + self.residual_weight * residual_total
        details.update(
            teacher_alignment=alignment.detach(),
            alignment_warmup=task_loss.new_tensor(warmup),
            alignment_geometry=geometry_total.detach(),
            alignment_feature=feature_total.detach(),
            alignment_semantic=semantic_total.detach(),
            feature_residual=residual_total.detach())
        return total, details
