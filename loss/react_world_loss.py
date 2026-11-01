"""Current/future occupancy, ego waypoints and small reg corrections."""
import torch
from torch import nn

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class ReactWorldLoss(nn.Module):
    def __init__(self, occupancy_loss, occupancy_weight=1., plan_weight=1.,
                 reg_weight=.01, horizon_weights=(1., 1., 1., 1., 1., 1.),
                 current_occupancy_weight=0.):
        super().__init__()
        self.occupancy = OPENOCC_LOSS.build(occupancy_loss)
        self.occupancy_weight = occupancy_weight
        self.plan_weight = plan_weight
        self.reg_weight = reg_weight
        if current_occupancy_weight < 0:
            raise ValueError('current_occupancy_weight must be nonnegative')
        self.current_occupancy_weight = current_occupancy_weight
        if not horizon_weights or any(weight < 0 for weight in horizon_weights) or sum(horizon_weights) <= 0:
            raise ValueError('horizon_weights must be nonnegative with a positive sum')
        self.horizon_weights = tuple(horizon_weights)

    def _occupancy_loss(self, predictions, label, mask=None):
        """Reuse the perception CE/Lovasz recipe and safely skip void targets."""
        label = label.flatten(1)
        valid = (label >= 0) & (label < predictions[0].shape[1])
        if mask is not None:
            valid = valid & mask.flatten(1).bool()
        if not valid.any():
            return sum(prediction.sum() * 0. for prediction in predictions)
        return self.occupancy(dict(
            pred_occ=predictions, sampled_xyz=None,
            sampled_label=label.masked_fill(~valid, 255), occ_mask=valid))

    def forward(self, inputs):
        metas = inputs['metas']
        logits = inputs['future_occ_logits']
        targets = metas['future_occ_labels']
        steps = len(self.horizon_weights)
        if len(logits) != steps or targets.shape[1] != steps:
            raise ValueError('Prediction, supervision and horizon weights must agree')
        if inputs['future_occ_frame'] != 'future_ego':
            raise ValueError('Native future labels require future-frame readout transforms')
        plan = inputs['pred_traj']
        plan_target = metas['temporal_trajs'].to(plan)
        if plan.shape != plan_target.shape or plan.shape[1:] != (steps, 2):
            raise ValueError('Planning predictions and targets must have shape [B,T,2]')

        total = plan.sum() * 0.
        details = {}
        if self.current_occupancy_weight:
            current_loss = self._occupancy_loss(
                inputs['pred_occ'], inputs['sampled_label'], inputs.get('occ_loss_mask'))
            total = total + self.current_occupancy_weight * current_loss
            details['occ_current'] = current_loss.detach()
        for step, (prediction, weight) in enumerate(zip(logits, self.horizon_weights)):
            # Preserve CE/Lovasz and the class balancing from the React recipe.
            # Skip void-only targets instead of taking an undefined CE mean.
            loss = self._occupancy_loss([prediction], targets[:, step])
            total = total + self.occupancy_weight * weight * loss / sum(self.horizon_weights)
            details[f'occ_step_{step + 1}'] = loss.detach()
        plan_loss = (plan - plan_target).square().sum(-1).mean()
        regularization = torch.stack([
            offset.square().mean() for offset in inputs['forecast_reg_offsets']]).mean()
        total = total + self.plan_weight * plan_loss + self.reg_weight * regularization
        details.update(plan_l2=plan_loss.detach(), reg_penalty=regularization.detach())
        return total, details
