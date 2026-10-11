"""Task and frozen-teacher objectives for SparseWorld-FFA."""

import torch
import torch.nn.functional as F
from torch import nn

from model.sparse_latent_dynamics import HungarianSparseMatcher
from model.sparse_latent_dynamics import transform_points
from . import OPENOCC_LOSS
from .opus_set_loss import OPUSSetLoss


@OPENOCC_LOSS.register_module()
class SparseWorldFFALoss(nn.Module):
    def __init__(self, future_occupancy_loss, target_horizon_index=6,
                 occupancy_weight=1.0, geometry_weight=2.0,
                 feature_weight=1.0, semantic_weight=0.5,
                 residual_weight=1e-3, geometry_warmup_iters=0,
                 matcher=None):
        super().__init__()
        self.future_occupancy = OPUSSetLoss(**future_occupancy_loss)
        self.target_horizon_index = int(target_horizon_index)
        self.geometry_warmup_iters = max(0, int(geometry_warmup_iters))
        self.weights = dict(
            occupancy=float(occupancy_weight),
            geometry=float(geometry_weight),
            feature=float(feature_weight),
            semantic=float(semantic_weight),
            residual=float(residual_weight))
        matcher = dict(matcher or {})
        matcher.setdefault('feature_weight', 0.0)
        self.matcher = HungarianSparseMatcher(**matcher)

    def _future_metas(self, metas):
        index = self.target_horizon_index - 1
        if metas['future_occ_labels'].shape[1] <= index:
            raise ValueError('future labels do not contain the target horizon')
        result = dict(metas)
        result['occ_label'] = metas['future_occ_labels'][:, index]
        # The voxel centers have the same native-ego grid at every horizon.
        result['occ_xyz'] = metas['occ_xyz']
        result['occ_cam_mask'] = metas['future_occ_cam_masks'][:, index]
        result['occ_lidar_mask'] = metas['future_occ_lidar_masks'][:, index]
        result['occ_loss_mask'] = torch.ones_like(
            result['occ_cam_mask'], dtype=torch.bool)
        return result

    @staticmethod
    def _zero(prediction):
        return prediction['query_features'].sum() * 0.0

    def _alignment(self, prediction, teacher):
        predicted_match = dict(
            points_metric=prediction['points_metric'],
            query_feats=prediction['query_features'],
            semantic_logits=prediction['semantic_logits'])
        teacher_match = dict(
            points_metric=teacher['points_metric'],
            query_feats=teacher['query_features'],
            semantic_logits=teacher['semantic_logits'])
        matches = self.matcher(predicted_match, teacher_match)
        geometry = self._zero(prediction)
        feature = self._zero(prediction)
        semantic = self._zero(prediction)
        centers, agreements, costs = [], [], []
        matched = 0
        for batch_index, (pred_index, teacher_index, match_cost) in enumerate(
                zip(matches.prediction, matches.target, matches.costs)):
            if pred_index.numel() == 0:
                continue
            predicted_points = prediction['points_metric'][
                batch_index, pred_index].float()
            teacher_points = teacher['points_metric'][
                batch_index, teacher_index].float()
            predicted_center = predicted_points.mean(1)
            teacher_center = teacher_points.mean(1)
            center_loss = F.smooth_l1_loss(
                predicted_center, teacher_center, beta=0.5)
            distances = torch.cdist(predicted_points, teacher_points)
            chamfer = 0.5 * (distances.min(-1).values.mean() +
                             distances.min(-2).values.mean())
            geometry = geometry + center_loss + chamfer

            predicted_feature = prediction['projected_features'][
                batch_index, pred_index].float()
            teacher_feature = teacher['projected_features'][
                batch_index, teacher_index].float()
            feature = feature + F.smooth_l1_loss(
                predicted_feature, teacher_feature, beta=1.0)
            feature = feature + (1.0 - F.cosine_similarity(
                predicted_feature, teacher_feature, dim=-1)).mean()

            predicted_semantic = prediction['semantic_logits'][
                batch_index, pred_index].float()
            teacher_semantic = teacher['semantic_logits'][
                batch_index, teacher_index].float().softmax(-1)
            semantic = semantic + F.kl_div(
                predicted_semantic.log_softmax(-1), teacher_semantic,
                reduction='batchmean')
            centers.append((predicted_center - teacher_center).norm(
                dim=-1).mean())
            agreements.append((predicted_semantic.argmax(-1) ==
                               teacher_semantic.argmax(-1)).float().mean())
            costs.append(match_cost.float().mean())
            matched += int(pred_index.numel())

        batch = max(prediction['query_features'].shape[0], 1)
        nan = prediction['query_features'].new_tensor(float('nan'))
        metrics = dict(
            matched_queries=prediction['query_features'].new_tensor(
                matched / batch),
            matched_ratio=prediction['query_features'].new_tensor(
                matched / (batch * prediction['query_features'].shape[1])),
            center_l2_m=(torch.stack(centers).mean() if centers else nan),
            semantic_agreement=(torch.stack(agreements).mean()
                                if agreements else nan),
            matching_cost=(torch.stack(costs).mean() if costs else nan))
        return geometry / batch, feature / batch, semantic / batch, metrics

    def forward(self, inputs):
        output = inputs['ffa_output']
        prediction = output['prediction']
        teacher = prediction.get('teacher_state')
        if teacher is None:
            raise ValueError('SparseWorldFFALoss requires a frozen teacher state')

        index = self.target_horizon_index - 1
        future_to_current = inputs['metas']['future_ego_to_current'][:, index]
        current_to_future = torch.linalg.inv(future_to_current.float())
        native_points = transform_points(
            prediction['points_metric'].float(), current_to_future)
        occupancy_points = native_points.flatten(1, 2)
        occupancy_logits = output['future_pred_logits'][0]
        born_points = output.get('future_born_points_native')
        born_logits = output.get('future_born_logits')
        if born_points is not None:
            if born_logits is None or born_points.shape[:2] != born_logits.shape[:2]:
                raise ValueError('future-born point/logit shapes do not match')
            occupancy_points = torch.cat(
                [occupancy_points, born_points.float()], dim=1)
            occupancy_logits = torch.cat(
                [occupancy_logits, born_logits], dim=1)
        aligned_teacher = dict(teacher)
        aligned_teacher['points_metric'] = transform_points(
            teacher['points_metric'].float(), future_to_current.float())
        occupancy, occupancy_metrics = self.future_occupancy(dict(
            opus_pred_points=[occupancy_points],
            opus_pred_logits=[occupancy_logits],
            metas=self._future_metas(inputs['metas'])))
        geometry, feature, semantic, alignment_metrics = self._alignment(
            prediction, aligned_teacher)
        residual = (prediction['feature_delta'].float().square().mean() +
                    prediction['center_delta'].float().square().mean() +
                    prediction['shape_delta'].float().square().mean())
        global_iter = int(inputs.get('global_iter', 0))
        geometry_scale = 1.0
        if self.geometry_warmup_iters:
            geometry_scale = min(
                1.0, (global_iter + 1) / self.geometry_warmup_iters)
        effective_geometry_weight = (
            self.weights['geometry'] * geometry_scale)
        total = (self.weights['occupancy'] * occupancy +
                 effective_geometry_weight * geometry +
                 self.weights['feature'] * feature +
                 self.weights['semantic'] * semantic +
                 self.weights['residual'] * residual)
        metrics = {
            'future_occupancy': occupancy.detach(),
            'geometry_alignment': geometry.detach(),
            'geometry_weight': prediction['query_features'].new_tensor(
                effective_geometry_weight),
            'feature_alignment': feature.detach(),
            'semantic_distillation': semantic.detach(),
            'residual_regularization': residual.detach(),
            'shape_scale': prediction['shape_scale'].detach(),
            'future_born_points': prediction['query_features'].new_tensor(
                0 if born_points is None else born_points.shape[1]),
        }
        metrics.update({'future/' + key: value
                        for key, value in occupancy_metrics.items()})
        metrics.update({'alignment/' + key: value.detach()
                        for key, value in alignment_metrics.items()})
        return total, metrics
