"""Joint OPUSv2 occupancy and future image-feature alignment loss."""
import torch
from torch import nn

from . import OPENOCC_LOSS
from .opus_v2_loss import OPUSV2Loss


@OPENOCC_LOSS.register_module()
class ForecastingSparseWorldLoss(nn.Module):
    def __init__(self, occupancy_loss, current_weight=1., future_weight=1.,
                 latent_alignment_weight=30., pyramid_alignment_weight=10.):
        super().__init__()
        self.occupancy = OPUSV2Loss(**occupancy_loss)
        self.current_weight = float(current_weight)
        self.future_weight = float(future_weight)
        self.latent_alignment_weight = float(latent_alignment_weight)
        self.pyramid_alignment_weight = float(pyramid_alignment_weight)

    @staticmethod
    def _future_metas(metas, horizon):
        future = dict(metas)
        target_index = horizon
        if 'future_horizon' in metas:
            selected = torch.as_tensor(
                metas['future_horizon'],
                device=metas['future_occ_labels'].device).reshape(-1)
            if not torch.all(selected == horizon):
                raise ValueError(
                    'prediction horizon does not match loaded future target')
            target_index = 0
        future['occ_label'] = metas['future_occ_labels'][:, target_index]
        future['occ_cam_mask'] = metas['future_occ_cam_masks'][:, target_index]
        future['occ_lidar_mask'] = metas['future_occ_lidar_masks'][:, target_index]
        future['occ_loss_mask'] = torch.ones_like(
            future['occ_cam_mask'], dtype=torch.bool)
        return future

    def forward(self, inputs):
        output = inputs['forecasting_output']
        metas = inputs['metas']
        current_inputs = dict(output['current_opus'], metas=metas)
        current_loss, current_metrics = self.occupancy(current_inputs)
        total = self.current_weight * current_loss
        metrics = {'current_total': current_loss.detach()}
        metrics.update({f'current/{key}': value
                        for key, value in current_metrics.items()})
        future_total = current_loss * 0.
        latent_total = current_loss * 0.
        pyramid_total = current_loss * 0.
        for prediction in output['future_predictions']:
            horizon = int(prediction['horizon'])
            future_inputs = dict(
                prediction['opus'],
                metas=self._future_metas(metas, horizon))
            occupancy_loss, occupancy_metrics = self.occupancy(future_inputs)
            future_total = future_total + occupancy_loss
            latent_total = latent_total + prediction['latent_loss']
            pyramid_total = pyramid_total + prediction['pyramid_loss']
            metrics.update({f'future{horizon + 1}/{key}': value
                            for key, value in occupancy_metrics.items()})
        count = max(len(output['future_predictions']), 1)
        future_total = future_total / count
        latent_total = latent_total / count
        pyramid_total = pyramid_total / count
        total = (total + self.future_weight * future_total +
                 self.latent_alignment_weight * latent_total +
                 self.pyramid_alignment_weight * pyramid_total)
        metrics.update(future_total=future_total.detach(),
                       latent_alignment=latent_total.detach(),
                       pyramid_alignment=pyramid_total.detach())
        return total, metrics
