"""Occupancy supervision and light Gaussian regularisation for Gaussian-OPUS."""
import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class GaussianOccupancyLoss(nn.Module):
    def __init__(self, stage_weights=(.25, .35, .5, .7, .85, 1.), empty_label=17,
                 class_weights=None, lambda_dice=0., lambda_scale=1e-3,
                 lambda_alpha=1e-4):
        super().__init__()
        self.stage_weights = tuple(stage_weights)
        self.empty_label = empty_label
        self.lambda_dice = lambda_dice
        self.lambda_scale = lambda_scale
        self.lambda_alpha = lambda_alpha
        self.register_buffer('class_weights', torch.as_tensor(
            [] if class_weights is None else class_weights, dtype=torch.float32), persistent=False)

    def forward(self, inputs):
        predictions, labels = inputs['pred_occ'], inputs['sampled_label'].long()
        if len(predictions) > len(self.stage_weights):
            raise ValueError('stage_weights must cover every supervised Gaussian-OPUS stage')
        stage_indices = inputs.get('supervised_stage_indices')
        if stage_indices is None:
            stage_indices = list(range(len(predictions)))
        if len(stage_indices) != len(predictions):
            raise ValueError('supervised_stage_indices must align with pred_occ')
        mask = inputs.get('occ_loss_mask', inputs.get('occ_mask'))
        if mask is not None:
            mask = mask.to(labels.device).bool().flatten(1)
        total = predictions[0].sum() * 0.
        ce_values = []
        for stage, logits in zip(stage_indices, predictions):
            if stage >= len(self.stage_weights):
                raise ValueError('stage weight is missing for a supervised Gaussian-OPUS stage')
            # ``local_aggregate`` follows the existing GaussianHead contract:
            # [B, C, V].  Convert only this private loss view to [B, V, C]
            # before applying the [B, V] occupancy mask.
            valid_logits, valid_labels = logits.transpose(1, 2), labels
            if mask is not None:
                valid_logits = valid_logits[mask][None]
                valid_labels = labels[mask][None]
            if valid_labels.numel() == 0:
                continue
            weight = self.class_weights.to(logits) if self.class_weights.numel() else None
            ce = F.cross_entropy(valid_logits.transpose(1, 2), valid_labels, weight=weight)
            if self.lambda_dice:
                prob = valid_logits.softmax(dim=-1)
                target = F.one_hot(valid_labels, num_classes=prob.shape[-1]).to(prob)
                dice = 1 - (2 * (prob * target).sum((0, 1)) + 1e-5) / (
                    prob.sum((0, 1)) + target.sum((0, 1)) + 1e-5)
                ce = ce + self.lambda_dice * dice.mean()
            total = total + self.stage_weights[stage] * ce
            ce_values.append(ce.detach())
        gaussian = inputs['gaussian']
        scale_penalty = gaussian.scales.square().mean()
        alpha_penalty = gaussian.opacities.mean()
        total = total + self.lambda_scale * scale_penalty + self.lambda_alpha * alpha_penalty
        return total, {'loss_semantic': torch.stack(ce_values).mean() if ce_values else total.detach() * 0.,
                       'loss_scale': scale_penalty.detach(), 'loss_alpha': alpha_penalty.detach()}
