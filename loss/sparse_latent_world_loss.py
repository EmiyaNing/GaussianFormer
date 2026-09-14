"""Stage-one task-consistent sparse latent world objective."""
import torch
from torch import nn

from . import OPENOCC_LOSS
from .opus_v2_loss import OPUSV2Loss


@OPENOCC_LOSS.register_module()
class SparseLatentWorldLoss(nn.Module):
    def __init__(self, occupancy_loss, occupancy_weight=1.,
                 position_weight=2., feature_weight=1., semantic_weight=1.,
                 existence_weight=.25):
        super().__init__()
        self.occupancy = OPUSV2Loss(**occupancy_loss)
        self.weights = dict(
            position=float(position_weight), feature=float(feature_weight),
            semantic=float(semantic_weight), existence=float(existence_weight))
        self.occupancy_weight = float(occupancy_weight)

    @staticmethod
    def _future_metas(metas, horizon):
        result = dict(metas)
        target_index = horizon
        if 'future_horizon' in metas:
            selected = torch.as_tensor(
                metas['future_horizon'],
                device=metas['future_occ_labels'].device).reshape(-1)
            if not torch.all(selected == horizon):
                raise ValueError('prediction and loaded future horizon differ')
            target_index = 0
        result['occ_label'] = metas['future_occ_labels'][:, target_index]
        result['occ_cam_mask'] = metas['future_occ_cam_masks'][:, target_index]
        result['occ_lidar_mask'] = metas['future_occ_lidar_masks'][:, target_index]
        result['occ_loss_mask'] = torch.ones_like(
            result['occ_cam_mask'], dtype=torch.bool)
        return result

    def forward(self, inputs):
        output, metas = inputs['stage1_output'], inputs['metas']
        predictions = output['future_predictions']
        if not predictions:
            raise ValueError('SparseLatentWorldLoss received no future predictions')
        total = None
        metrics = {}
        component_sums = {key: None for key in self.weights}
        occupancy_sum = None
        for prediction in predictions:
            horizon = int(prediction['horizon'])
            task, task_metrics = self.occupancy(dict(
                prediction['predicted_opus'],
                metas=self._future_metas(metas, horizon)))
            occupancy_sum = task if occupancy_sum is None else occupancy_sum + task
            alignment = prediction['alignment']
            current = self.occupancy_weight * task
            for name, weight in self.weights.items():
                value = alignment[name]
                current = current + weight * value
                component_sums[name] = (value if component_sums[name] is None
                                        else component_sums[name] + value)
            total = current if total is None else total + current
            prefix = f'future{horizon + 1}/'
            metrics.update({prefix + key: value
                            for key, value in task_metrics.items()})
            metrics.update({prefix + key: value.detach()
                            for key, value in alignment['metrics'].items()})
        count = len(predictions)
        total = total / count
        metrics['future_occupancy'] = (occupancy_sum / count).detach()
        for name, value in component_sums.items():
            metrics[name + '_alignment'] = (value / count).detach()
        return total, metrics
