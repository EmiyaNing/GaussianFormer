"""OPUS-style bidirectional centre supervision for decoder Gaussians."""
import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class GaussianCenterChamferLoss(nn.Module):
    """Symmetric KNN centre loss for every Gaussian-OPUS decoder stage.

    Target assignment follows OPUS-V1: exact nearest-neighbour indices are
    computed without gradients, then gradients flow through bidirectional
    Smooth-L1 point residuals.  No semantic classifier is used here; semantic
    occupancy supervision is delegated to the unmodified OccupancyLoss.
    """

    def __init__(self, stage_weights=(1.,), lambda_center=0.5,
                 smooth_l1_beta=0.2, empty_label=17, chunk_size=1024,
                 use_occ_loss_mask=False):
        super().__init__()
        self.stage_weights = tuple(stage_weights)
        self.lambda_center = lambda_center
        self.smooth_l1_beta = smooth_l1_beta
        self.empty_label = empty_label
        self.chunk_size = chunk_size
        self.use_occ_loss_mask = use_occ_loss_mask

    def _single(self, predicted_centers, gt_centers):
        pred_to_gt = _nearest_indices(predicted_centers, gt_centers, self.chunk_size)
        gt_to_pred = _nearest_indices(gt_centers, predicted_centers, self.chunk_size)
        pred_to_gt_loss = F.smooth_l1_loss(
            predicted_centers, gt_centers[pred_to_gt], beta=self.smooth_l1_beta,
            reduction='none').sum(dim=-1).mean()
        gt_to_pred_loss = F.smooth_l1_loss(
            gt_centers, predicted_centers[gt_to_pred], beta=self.smooth_l1_beta,
            reduction='none').sum(dim=-1).mean()
        with torch.no_grad():
            pred_distance = torch.linalg.vector_norm(
                predicted_centers.detach() - gt_centers[pred_to_gt], dim=-1)
            gt_distance = torch.linalg.vector_norm(
                gt_centers - predicted_centers.detach()[gt_to_pred], dim=-1)
        return pred_to_gt_loss, gt_to_pred_loss, pred_distance, gt_distance

    def forward(self, inputs):
        gaussians = inputs['gaussians']
        if len(gaussians) != len(self.stage_weights):
            raise ValueError('stage_weights must contain one entry per decoder Gaussian stage')
        metas = inputs['metas']
        device = gaussians[0].means.device
        gt_xyz = metas['occ_xyz'].to(device).flatten(1, -2)
        gt_label = metas['occ_label'].to(device).long().flatten(1)
        gt_valid = gt_label != self.empty_label
        if self.use_occ_loss_mask and 'occ_loss_mask' in metas:
            gt_valid &= metas['occ_loss_mask'].to(device).bool().flatten(1)

        total = gaussians[0].means.sum() * 0.
        stage_values, all_pred_distance, all_gt_distance = [], [], []
        for weight, gaussian in zip(self.stage_weights, gaussians):
            centers = gaussian.means
            stage_sum = centers.sum() * 0.
            valid_batch = 0
            for batch_index in range(centers.shape[0]):
                if not gt_valid[batch_index].any():
                    continue
                pred_loss, gt_loss, pred_distance, gt_distance = self._single(
                    centers[batch_index], gt_xyz[batch_index, gt_valid[batch_index]])
                stage_sum = stage_sum + pred_loss + gt_loss
                valid_batch += 1
                all_pred_distance.append(pred_distance.detach())
                all_gt_distance.append(gt_distance.detach())
            stage_value = stage_sum / max(valid_batch, 1)
            total = total + weight * stage_value
            stage_values.append(stage_value.detach())
        total = total * self.lambda_center
        zero = total.detach() * 0.
        return total, {
            'loss_center_chamfer': torch.stack(stage_values).mean() if stage_values else zero,
            'center_pred_to_gt_m': torch.cat(all_pred_distance).mean() if all_pred_distance else zero,
            'center_gt_to_pred_m': torch.cat(all_gt_distance).mean() if all_gt_distance else zero,
        }
