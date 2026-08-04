"""Losses for the local SparseWorld trajectory migration.

Current-frame supervision remains the repository's strict ``OPUSSetLoss``.
This module only adds the future occupancy and ego-displacement terms.
"""
import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import OPUSSetLoss, _nearest_indices


@OPENOCC_LOSS.register_module()
class SparseWorldTrajectoryLoss(nn.Module):
    def __init__(self, current_loss, lambda_traj=1., lambda_future_cls=2.,
                 lambda_future_pts=.5, smooth_l1_beta=.2, empty_label=17,
                 pc_range=(-40., -40., -1., 40., 40., 5.4), grid_size=.4,
                 chunk_size=1024):
        super().__init__()
        self.current_loss = OPENOCC_LOSS.build(current_loss)
        if not isinstance(self.current_loss, OPUSSetLoss):
            raise TypeError('current_loss must be the local OPUSSetLoss')
        self.lambda_traj, self.lambda_future_cls, self.lambda_future_pts = lambda_traj, lambda_future_cls, lambda_future_pts
        self.beta, self.empty_label, self.pc_range, self.grid_size, self.chunk_size = smooth_l1_beta, empty_label, tuple(pc_range), grid_size, chunk_size

    def _targets(self, labels, mask, template):
        # Dense voxel labels are converted to world-coordinate sparse targets
        # in current ego coordinates.  Camera validity is retained for the
        # source-style unmatched-target weighting.
        valid = labels != self.empty_label
        xyz = valid.nonzero(as_tuple=False).to(template.dtype)
        lower = template.new_tensor(self.pc_range[:3])
        points = lower + (xyz + .5) * self.grid_size
        return points, labels[valid].long(), valid

    def _future_single(self, points, logits, labels, camera_mask, transform):
        # ``Tensor.to(points)`` also adopts the floating prediction dtype.  A
        # camera validity tensor is a logical mask, so normalize it at the
        # loss boundary before combining it with the boolean unmatched mask.
        # This keeps both train (warm-up/future loss) and eval on the same
        # dtype-safe path even if a caller has already cast the input.
        camera_mask = camera_mask.to(device=points.device, dtype=torch.bool)
        target, target_labels, valid = self._targets(labels, camera_mask, points)
        rotation, translation = transform[:3, :3].to(points), transform[:3, 3].to(points)
        target = target @ rotation.T + translation
        pred_to_gt = _nearest_indices(points, target, self.chunk_size)
        gt_to_pred = _nearest_indices(target, points, self.chunk_size)
        assigned = target_labels[pred_to_gt]
        focal_target = F.one_hot(assigned, num_classes=logits.shape[-1]).to(logits.dtype)
        cls = F.binary_cross_entropy_with_logits(logits, focal_target, reduction='none')
        prob = logits.sigmoid(); p_t = prob * focal_target + (1 - prob) * (1 - focal_target)
        cls = (.25 * focal_target + .75 * (1 - focal_target)) * (1 - p_t).pow(2.) * cls
        cls = cls.sum(-1).mean()
        forward = F.smooth_l1_loss(points, target[pred_to_gt], beta=self.beta, reduction='none').sum(-1).mean()
        backward = F.smooth_l1_loss(target, points[gt_to_pred], beta=self.beta, reduction='none').sum(-1)
        # Preserve SparseWorld's stronger unmatched-camera target penalty.
        unmatched = (target - points.detach()[gt_to_pred]).norm(dim=-1) > .2
        camera_valid = camera_mask[valid]
        weight = torch.where(
            unmatched & camera_valid,
            backward.new_tensor(5.), backward.new_tensor(1.))
        backward = (backward * weight).mean()
        return cls, forward + backward

    def forward(self, inputs):
        current, metrics = self.current_loss(inputs)
        pred_points, pred_logits = inputs['future_pred_points'], inputs['future_pred_logits']
        metas = inputs['metas']
        horizon = min(len(pred_points), metas['future_occ_labels'].shape[1])
        future_total = current.new_zeros(())
        detail = dict(metrics)
        for step in range(horizon):
            cls_step, pts_step = current.new_zeros(()), current.new_zeros(())
            for batch in range(pred_points[step].shape[0]):
                cls, pts = self._future_single(pred_points[step][batch], pred_logits[step][batch],
                                               metas['future_occ_labels'][batch, step].to(
                                                   device=pred_points[step].device),
                                               metas['future_occ_cam_masks'][batch, step].to(
                                                   device=pred_points[step].device),
                                               metas['future_ego_to_current'][batch, step].to(pred_points[step]))
                cls_step, pts_step = cls_step + cls, pts_step + pts
            cls_step, pts_step = cls_step / pred_points[step].shape[0], pts_step / pred_points[step].shape[0]
            future_total = future_total + self.lambda_future_cls * cls_step + self.lambda_future_pts * pts_step
            detail[f'future_{step + 1}/cls'] = cls_step.detach()
            detail[f'future_{step + 1}/pts'] = pts_step.detach()
        trajectory_horizon = min(inputs['pred_traj'].shape[1], metas['temporal_trajs'].shape[1])
        trajectory = F.mse_loss(inputs['pred_traj'][:, :trajectory_horizon],
                                metas['temporal_trajs'].to(inputs['pred_traj'])[:, :trajectory_horizon])
        total = current + future_total + self.lambda_traj * trajectory
        detail.update(future= future_total.detach(), trajectory=trajectory.detach())
        return total, detail
