"""Density-aware Chamfer geometry with the original OPUS classification."""
import torch
from . import OPENOCC_LOSS
from .opus_set_loss import OPUSSetLoss, _nearest_indices


def density_aware_terms(points, target, pred_to_gt, gt_to_pred, alpha=1.,
                        distance_unit=.4, density_power=1.):
    """Official cardinality-corrected DCD directional terms before reduction.

    Counts and nearest-neighbor indices are discrete, detached assignments.
    With unequal point counts, individual terms can be negative, as in the
    authors' calc_dcd(non_reg=False). No clipping changes those gradients.
    """
    n,m=len(points),len(target)
    pred_d2=(points-target[pred_to_gt]).square().sum(-1)
    gt_d2=(target-points[gt_to_pred]).square().sum(-1)
    with torch.no_grad():
        pred_counts=torch.bincount(pred_to_gt,minlength=m)[pred_to_gt].to(points)
        gt_counts=torch.bincount(gt_to_pred,minlength=n)[gt_to_pred].to(points)
        pred_weights=(n/m)/(pred_counts.pow(density_power)+1e-6)
        gt_weights=(m/n)/(gt_counts.pow(density_power)+1e-6)
    factor=alpha/distance_unit**2
    pred_term=1-pred_weights*torch.exp(-factor*pred_d2)
    gt_term=1-gt_weights*torch.exp(-factor*gt_d2)
    return pred_term,gt_term,pred_d2,gt_d2


@OPENOCC_LOSS.register_module()
class OPUSSetDCDLoss(OPUSSetLoss):
    def __init__(self,dcd_alpha=1.,dcd_distance_unit=.4,dcd_density_power=1.,
                 dcd_retain_gt_weights=True,**kwargs):
        super().__init__(**kwargs)
        if dcd_alpha<=0 or dcd_distance_unit<=0 or dcd_density_power<0:
            raise ValueError('Invalid DCD parameters')
        self.dcd_alpha=dcd_alpha
        self.dcd_distance_unit=dcd_distance_unit
        self.dcd_density_power=dcd_density_power
        self.dcd_retain_gt_weights=dcd_retain_gt_weights
        self._density_rows=[]

    def _single(self,points,logits,gt_points,gt_labels,gt_camera_valid):
        p2g=_nearest_indices(points,gt_points,self.chunk_size)
        g2p=_nearest_indices(gt_points,points,self.chunk_size)
        pred,gt,pred_d2,gt_d2=density_aware_terms(points,gt_points,p2g,g2p,
            self.dcd_alpha,self.dcd_distance_unit,self.dcd_density_power)
        assigned=gt_labels[p2g]
        cls_weight=(self.class_weights.to(logits)[assigned] if self.class_weights.numel()
                    else torch.ones_like(pred_d2))
        cls_weight=cls_weight*self._distance_weight(gt_points[p2g])
        cls=self._sigmoid_focal(logits,assigned,cls_weight)
        if self.dcd_retain_gt_weights:
            # Preserve the original OPUS GT weighting, including its released
            # rare-class indexing behavior. Do not fix unrelated loss logic.
            weight=torch.ones_like(gt_d2)
            weight[(gt_d2.detach().sqrt()>self.empty_dist_thr)&gt_camera_valid]=self.empty_weight
            for c in self.rare_classes:weight[gt_labels==c].clamp_(min=self.rare_weight)
            gt=gt*weight
        with torch.no_grad():
            counts_pred=torch.bincount(p2g,minlength=len(gt_points))
            counts_gt=torch.bincount(g2p,minlength=len(points))
            self._density_rows.append(torch.stack((
                (counts_pred>0).float().mean(),(counts_gt>0).float().mean(),
                torch.exp(-self.dcd_alpha*pred_d2.detach()/self.dcd_distance_unit**2).mean(),
                torch.exp(-self.dcd_alpha*gt_d2.detach()/self.dcd_distance_unit**2).mean())))
        return cls,pred.mean(),gt.mean(),pred_d2.detach(),gt_d2.detach()

    def forward(self,inputs):
        self._density_rows=[]
        value,metrics=super().forward(inputs)
        if self._density_rows:
            values=torch.stack(self._density_rows).mean(0)
            metrics.update(dict(zip(('queried_gt_fraction','queried_prediction_fraction',
                'pred_exponential_mean','gt_exponential_mean'),values.detach())))
        return value,metrics
