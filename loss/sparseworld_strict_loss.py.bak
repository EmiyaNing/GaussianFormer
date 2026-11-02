"""Independent loss stack for the strict SparseWorld reproduction."""
import torch
import torch.nn.functional as F
from torch import nn
from . import OPENOCC_LOSS
from .opus_set_loss import OPUSSetLoss, _nearest_indices


@OPENOCC_LOSS.register_module()
class SparseWorldStrictLoss(nn.Module):
    """Current OPUS loss, all-query temporal warm-up and future native-frame loss."""
    def __init__(self, current_loss, pc_range=(-40.,-40.,-1.,40.,40.,5.4), grid_size=.4, empty_label=17, lambda_future_cls=2., lambda_future_pts=.5, lambda_traj=1., chunk_size=1024):
        super().__init__(); self.current_loss=OPENOCC_LOSS.build(current_loss); self.pc_range=tuple(pc_range); self.grid_size=grid_size; self.empty_label=empty_label; self.lambda_future_cls=lambda_future_cls; self.lambda_future_pts=lambda_future_pts; self.lambda_traj=lambda_traj; self.chunk_size=chunk_size
        if not isinstance(self.current_loss, OPUSSetLoss): raise TypeError('current_loss must be OPUSSetLoss')

    def _targets(self, dense, template):
        valid=dense!=self.empty_label; index=valid.nonzero().to(template); lower=template.new_tensor(self.pc_range[:3])
        return lower+(index+.5)*self.grid_size, dense[valid].long()

    def _set_loss(self, points, logits, dense):
        target, labels=self._targets(dense,points)
        if not len(target): return points.sum()*0.,points.sum()*0.
        p2g=_nearest_indices(points,target,self.chunk_size); g2p=_nearest_indices(target,points,self.chunk_size)
        onehot=F.one_hot(labels[p2g],logits.shape[-1]).to(logits); bce=F.binary_cross_entropy_with_logits(logits,onehot,reduction='none'); prob=logits.sigmoid(); pt=prob*onehot+(1-prob)*(1-onehot)
        cls=(((.25*onehot+.75*(1-onehot))*(1-pt).pow(2)*bce).sum(-1)).mean()
        pts=F.smooth_l1_loss(points,target[p2g],beta=.2,reduction='none').sum(-1).mean()+F.smooth_l1_loss(target,points[g2p],beta=.2,reduction='none').sum(-1).mean()
        return cls,pts

    def _temporal_targets(self, metas, batch, template):
        """Stack current/future occupied voxels in current ego coordinates."""
        points, labels = self._targets(metas['occ_label'][batch].to(template.device), template)
        all_points, all_labels = [points], [labels]
        for step in range(metas['future_occ_labels'].shape[1]):
            future_points, future_labels = self._targets(
                metas['future_occ_labels'][batch, step].to(template.device), template)
            transform = metas['future_ego_to_current'][batch, step].to(template)
            future_points = future_points @ transform[:3, :3].T + transform[:3, 3]
            all_points.append(future_points); all_labels.append(future_labels)
        return torch.cat(all_points), torch.cat(all_labels)

    def _temporal_loss(self, inputs, all_stages):
        total = all_stages[0][0].sum() * 0.; details = {}
        # Source pretraining supervises every decoder stage, while finetuning
        # retains this term only on the final full-query stage.
        stages = all_stages if inputs['sparseworld_pretrain'] else all_stages[-1:]
        for stage, (points, logits) in enumerate(stages):
            cls = pts = total * 0.
            for batch in range(points.shape[0]):
                target, labels = self._temporal_targets(inputs['metas'], batch, points)
                p2g = _nearest_indices(points[batch], target, self.chunk_size)
                g2p = _nearest_indices(target, points[batch], self.chunk_size)
                onehot = F.one_hot(labels[p2g], logits.shape[-1]).to(logits)
                cls = cls + F.binary_cross_entropy_with_logits(logits[batch], onehot)
                pts = pts + F.smooth_l1_loss(points[batch], target[p2g], beta=.2) + F.smooth_l1_loss(target, points[batch][g2p], beta=.2)
            cls, pts = cls / points.shape[0], pts / points.shape[0]
            total = total + self.lambda_future_cls * cls + self.lambda_future_pts * pts
            details[f'temporal/{stage}/cls'], details[f'temporal/{stage}/pts'] = cls.detach(), pts.detach()
        return total, details

    def forward(self, inputs):
        all_stages = list(zip(inputs['temporal_pred_points'], inputs['temporal_pred_logits']))
        temporal, temporal_details = self._temporal_loss(inputs, all_stages)
        if inputs['sparseworld_pretrain']:
            # This is exactly the source curriculum: no current-only or
            # future-forecast occupancy term before the finetune boundary.
            return temporal, temporal_details
        current, details=self.current_loss(inputs); details.update(temporal_details); total=current + temporal; metas=inputs['metas']; future_points=inputs.get('future_pred_points',()); future_logits=inputs.get('future_pred_logits',())
        future=current.new_zeros(())
        for step,(points,logits) in enumerate(zip(future_points,future_logits)):
            cls=pts=current.new_zeros(())
            for b in range(points.shape[0]):
                c,p=self._set_loss(points[b],logits[b],metas['future_occ_labels'][b,step].to(points.device)); cls,pts=cls+c,p+pts
            cls,pts=cls/points.shape[0],pts/points.shape[0]; future=future+self.lambda_future_cls*cls+self.lambda_future_pts*pts; details[f'future_{step+1}/cls']=cls.detach(); details[f'future_{step+1}/pts']=pts.detach()
        trajectory=F.mse_loss(inputs['pred_traj'],metas['temporal_trajs'].to(inputs['pred_traj'])[:,:inputs['pred_traj'].shape[1]])
        total=total+future+self.lambda_traj*trajectory; details.update(future=future.detach(),trajectory=trajectory.detach()); return total,details
