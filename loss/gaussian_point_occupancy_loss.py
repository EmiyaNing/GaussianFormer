"""Point-sampled occupancy classification loss for Gaussian semantic residuals."""
import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class GaussianPointOccupancyLoss(nn.Module):
    """Supervise Gaussian-refined logits at final OPUS point locations.

    Labels are assigned from the nearest non-empty GT occupancy voxel, exactly
    like OPUS point classification. This is intentionally not dense Lovasz: the
    Gaussian aggregator is queried at sparse final OPUS points, not occ_xyz.
    """
    def __init__(self, weight=1., empty_label=17, focal_gamma=2., focal_alpha=.25,
                 class_weights=None, chunk_size=1024, warmup_iters=0):
        super().__init__()
        self.weight = weight
        self.empty_label = empty_label
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.chunk_size = chunk_size
        self.warmup_iters = warmup_iters
        self.register_buffer('class_weights', torch.as_tensor(
            class_weights if class_weights is not None else [], dtype=torch.float32), persistent=False)

    def forward(self, inputs):
        logits = inputs['gaussian_point_logits']
        points = inputs['gaussian_point_points']
        valid_masks = inputs['gaussian_point_valid_mask']
        metas = inputs['metas']
        labels = metas['occ_label'].to(points.device).long().flatten(1)
        gt_points = metas['occ_xyz'].to(points.device).flatten(1, -2)
        total = logits.sum() * 0.
        valid_batches = 0
        for batch_index in range(points.shape[0]):
            prediction_mask = valid_masks[batch_index].bool()
            gt_mask = labels[batch_index] != self.empty_label
            if not prediction_mask.any() or not gt_mask.any():
                continue
            predicted_points = points[batch_index, prediction_mask]
            assignment = _nearest_indices(predicted_points, gt_points[batch_index, gt_mask], self.chunk_size)
            target = labels[batch_index, gt_mask][assignment]
            prediction = logits[batch_index, prediction_mask]
            target_one_hot = F.one_hot(target, num_classes=prediction.shape[-1]).to(prediction.dtype)
            probability = prediction.sigmoid()
            ce = F.binary_cross_entropy_with_logits(prediction, target_one_hot, reduction='none')
            pt = probability * target_one_hot + (1. - probability) * (1. - target_one_hot)
            alpha = self.focal_alpha * target_one_hot + (1. - self.focal_alpha) * (1. - target_one_hot)
            focal = alpha * (1. - pt).pow(self.focal_gamma) * ce
            if self.class_weights.numel():
                focal = focal * self.class_weights.to(prediction)[target].unsqueeze(-1)
            total = total + focal.sum(dim=-1).mean()
            valid_batches += 1
        ramp = 1. if self.warmup_iters <= 0 else min(1., float(inputs.get('global_iter', 0)) / self.warmup_iters)
        loss = self.weight * ramp * total / max(valid_batches, 1)
        return loss, {'gaussian_loss_ramp': logits.new_tensor(ramp),
                      'gaussian_valid_batches': logits.new_tensor(valid_batches)}
