"""Class-aware occupied-versus-empty supervision for rendered occupancy."""
import torch
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class OccupancyClassAwareMarginLoss(torch.nn.Module):
    """Make the true occupied class beat empty, and empty beat all foreground."""

    def __init__(self, empty_label=17, weight=1.0):
        super().__init__()
        self.empty_label = int(empty_label)
        self.weight = float(weight)

    def forward(self, inputs):
        logits = inputs['pred_occ'][-1]
        labels = inputs['sampled_label'].reshape(logits.shape[0], -1).long()
        if logits.ndim != 3 or logits.shape[1] != self.empty_label + 1:
            raise ValueError('Expected [B, C, V] logits with the empty class last')
        if labels.shape[1] != logits.shape[2]:
            raise ValueError('Label and logit voxel counts differ')
        mask = inputs.get('occ_mask')
        valid = (torch.ones_like(labels, dtype=torch.bool) if mask is None
                 else mask.reshape_as(labels).bool())
        occupied = valid & (labels != self.empty_label)
        empty = valid & (labels == self.empty_label)
        empty_score = logits[:, self.empty_label]
        true_score = logits.gather(1, labels.clamp_max(self.empty_label - 1).unsqueeze(1))[:, 0]
        top_foreground = logits[:, :self.empty_label].amax(dim=1)
        positive_loss = (F.softplus(empty_score[occupied] - true_score[occupied]).mean()
                         if occupied.any() else logits.sum() * 0.0)
        negative_loss = (F.softplus(top_foreground[empty] - empty_score[empty]).mean()
                         if empty.any() else logits.sum() * 0.0)
        loss = self.weight * (positive_loss + negative_loss)
        valid_count = valid.sum().clamp_min(1)
        return loss, {
            'positive': positive_loss.detach(),
            'negative': negative_loss.detach(),
            'predicted_occupied_fraction': ((top_foreground > empty_score) & valid).sum().float() / valid_count,
            'target_occupied_fraction': occupied.sum().float() / valid_count,
        }
