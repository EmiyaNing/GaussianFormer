"""Softmax focal with exactly the weighted-CE normalization convention."""
import torch
import torch.nn.functional as F

from loss import OPENOCC_LOSS
from loss.occupancy_loss import OccupancyLoss
from loss.utils.lovasz_softmax import lovasz_softmax


def classification_terms(logits, labels, class_weights, gamma):
    """Return unweighted -log(p_y), modulation, weights and validity per voxel."""
    if logits.ndim != 3 or tuple(labels.shape) != (logits.shape[0], logits.shape[2]):
        raise ValueError('expected logits [B,C,V] and labels [B,V]')
    if gamma < 0:
        raise ValueError('gamma must be nonnegative')
    labels = labels.long()
    valid = labels != 255
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[1])).any():
        raise ValueError('target label is outside class range')
    safe = labels.masked_fill(~valid, 0)
    log_prob = F.log_softmax(logits.float(), dim=1)
    log_pt = log_prob.gather(1, safe.unsqueeze(1)).squeeze(1)
    modulation = torch.ones_like(log_pt) if gamma == 0 else (1 - log_pt.exp()).pow(gamma)
    weights = class_weights.to(log_pt)[safe] * valid.to(log_pt)
    return -log_pt, modulation, weights, valid


def weighted_softmax_focal(logits, labels, class_weights, gamma=2.):
    ce, modulation, weights, _ = classification_terms(logits, labels, class_weights, gamma)
    # Fixed label-weight denominator: gamma=0 reproduces weighted CE.
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return (ce * modulation * weights).sum() / denominator


@OPENOCC_LOSS.register_module()
class FullspaceSoftmaxFocalOccupancyLoss(OccupancyLoss):
    """Replace voxel CE only; preserve the original occupied-only Lovasz term."""
    def __init__(self, focal_gamma=2., **kwargs):
        if kwargs.get('use_focal_loss', False):
            raise ValueError('the vendor sigmoid-focal switch must remain disabled')
        if (kwargs.get('use_sem_geo_scal_loss', True) or kwargs.get('use_dice_loss', False) or
                not kwargs.get('lovasz_use_softmax', True) or kwargs.get('ignore_empty', False)):
            raise ValueError('A changes only CE to softmax focal and must supervise free voxels')
        super().__init__(**kwargs)
        self.focal_gamma = float(focal_gamma)
        if self.focal_gamma < 0:
            raise ValueError('focal gamma must be nonnegative')

    def loss_voxel(self, pred_occ, sampled_xyz, sampled_label, occ_mask=None,
                   gaussian_sem_preds=None, gaussian_sem_label=None):
        if gaussian_sem_preds is not None or gaussian_sem_label is not None:
            raise ValueError('semantic KL is outside this single-factor ablation')
        labels = sampled_label
        mask = None if occ_mask is None else occ_mask.flatten(1).bool()
        if mask is not None:
            labels = labels[mask][None]
        total = pred_occ[0].sum() * 0.
        for logits in pred_occ:
            if mask is not None:
                logits = logits.transpose(1, 2)[mask][None].transpose(1, 2)
            total = total + self.loss_voxel_ce_weight * weighted_softmax_focal(
                logits, labels, self.class_weights, self.focal_gamma)
            if self.use_lovasz_loss:
                probabilities = logits.softmax(dim=1)
                total = total + self.loss_voxel_lovasz_weight * lovasz_softmax(
                    probabilities.transpose(1, 2).flatten(0, 1), labels.flatten(),
                    ignore=self.lovasz_ignore)
        return total / len(pred_occ)
