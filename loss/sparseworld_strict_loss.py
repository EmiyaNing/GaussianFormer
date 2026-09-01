"""Training objective and curriculum used by the released SparseWorld model."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class SparseWorldStrictLoss(nn.Module):
    """OPUS bidirectional set loss, temporal warm-up, future and trajectory.

    This follows the source control flow: epochs 0--4 supervise the expanded
    seven-timestamp stack on all decoder stages plus initial anchors; from
    epoch 5 the temporal term keeps stage 0 plus initial anchors, all six
    current-only stages become active, and future horizons grow to six.
    """

    def __init__(self, pc_range=(-40., -40., -1., 40., 40., 5.4),
                 voxel_size=.4, empty_label=17, num_query=720,
                 future_queries=(60, 60, 60, 60, 40, 40),
                 class_weights=(3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1),
                 focal_alpha=.25, focal_gamma=2., focal_weight=2.,
                 smooth_l1_beta=.2, point_weight=.5, empty_dist_thr=.2,
                 empty_weight=3., chunk_size=1024):
        super().__init__()
        self.pc_range = tuple(pc_range)
        self.voxel_size = float(voxel_size)
        self.empty_label = int(empty_label)
        self.num_query = int(num_query)
        self.future_queries = tuple(future_queries)
        self.class_weights = tuple(float(x) for x in class_weights)
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.focal_weight, self.point_weight = focal_weight, point_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.empty_dist_thr, self.empty_weight = empty_dist_thr, empty_weight
        self.chunk_size = int(chunk_size)
        self.expanded_range = (-80., -60., -3., 80., 60., 6.6)
        self.expanded_shape = (400, 300, 24)

    def _dense_points(self, labels, template):
        occupied = labels != self.empty_label
        indices = occupied.nonzero().to(template)
        lower = template.new_tensor(self.pc_range[:3])
        points = lower + (indices + .5) * self.voxel_size
        return points, labels[occupied].long()

    def _focal(self, logits, labels, weights):
        target = F.one_hot(labels, logits.shape[-1]).to(logits)
        probability = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = probability * target + (1. - probability) * (1. - target)
        alpha = self.focal_alpha * target + (1. - self.focal_alpha) * (1. - target)
        return (alpha * (1. - pt).pow(self.focal_gamma) * ce * weights).sum() \
            / max(logits.shape[0], 1) * self.focal_weight

    def _set_loss(self, points, logits, gt_points, gt_labels, gt_stamps=None,
                  stamp_statistics=None, query_count=None, temporal_reweight=False,
                  prediction_mask=None):
        if gt_points.numel() == 0 or points.numel() == 0:
            zero = points.sum() * 0.
            return zero, zero
        if temporal_reweight:
            keep = gt_stamps[:, 0].bool()
            keep |= gt_points[:, 0] > 40
            keep |= (gt_points[:, 0] > 35) & gt_stamps[:, 2].bool()
            keep |= (gt_points[:, 0] > 30) & gt_stamps[:, 1].bool()
            gt_points, gt_labels, gt_stamps = (
                gt_points[keep], gt_labels[keep], gt_stamps[keep])
        gt_to_pred = _nearest_indices(gt_points, points, self.chunk_size)
        pred_to_gt = _nearest_indices(points, gt_points, self.chunk_size)
        paired_gt = gt_points[pred_to_gt]
        paired_pred = points[gt_to_pred]
        assigned = gt_labels[pred_to_gt]

        scene = points.new_tensor(self.pc_range[3:]) - points.new_tensor(self.pc_range[:3])
        center = (points.new_tensor(self.pc_range[3:]) +
                  points.new_tensor(self.pc_range[:3])) / 2
        distance_weight = torch.norm(paired_gt[:, :2] - center[:2], dim=-1)
        distance_weight = distance_weight / torch.norm(scene[:2]) + 1.
        class_weight = logits.new_tensor(self.class_weights)[None].expand_as(logits)
        class_weight = class_weight * distance_weight[:, None]
        nearest_distance = torch.norm(paired_gt - points, dim=-1, keepdim=True)
        class_weight = class_weight * torch.clamp(
            nearest_distance.clamp_min(1e-12).reciprocal(), min=.4, max=1.)
        if prediction_mask is not None:
            class_weight = class_weight * prediction_mask.reshape(-1, 1).to(class_weight)
        loss_cls = self._focal(logits, assigned, class_weight)

        gt_weight = points.new_ones(len(gt_points))
        gt_distance = torch.norm(gt_points - paired_pred, dim=-1)
        gt_weight[gt_distance > self.empty_dist_thr] = self.empty_weight
        gt_weight[(gt_labels >= 15) | (gt_labels == 11)] *= .5
        gt_term = F.smooth_l1_loss(
            gt_points, paired_pred, beta=self.smooth_l1_beta, reduction='none')
        gt_term = (gt_term * gt_weight[:, None]).sum() / max(len(gt_points), 1)
        pred_term = F.smooth_l1_loss(
            points, paired_gt, beta=self.smooth_l1_beta, reduction='none')
        if prediction_mask is not None:
            pred_term = pred_term * prediction_mask.reshape(-1, 1).to(pred_term)
        pred_term = pred_term.sum() / max(len(points), 1)
        loss_points = (gt_term + pred_term) * self.point_weight

        if stamp_statistics is not None and gt_stamps is not None:
            children = len(points) // query_count
            if children * query_count != len(points):
                raise ValueError('prediction count is not Q * refinements')
            assigned_stamps = gt_stamps[pred_to_gt].reshape(
                query_count, children, -1)
            foreground = ((assigned.reshape(query_count, children, 1) < 2) |
                          (assigned.reshape(query_count, children, 1) > 10))
            with torch.no_grad():
                stamp_statistics.add_((assigned_stamps * foreground).sum(1).long())
        return loss_cls, loss_points

    def _expanded_targets(self, metas, batch_index, template):
        shape = self.expanded_shape
        count = shape[0] * shape[1] * shape[2]
        semantics = torch.full((count,), self.empty_label, dtype=torch.long,
                               device=template.device)
        stamps = torch.zeros((count, 7), dtype=torch.bool, device=template.device)
        labels_by_time = [metas['occ_label'][batch_index]] + [
            metas['future_occ_labels'][batch_index, step] for step in range(6)]
        transforms = [None] + [metas['future_ego_to_current'][batch_index, step]
                               for step in range(6)]
        lower = template.new_tensor(self.expanded_range[:3])
        # Source overwrite order: current, t6, t5, ..., t1.
        for stamp in (0, 6, 5, 4, 3, 2, 1):
            points, labels = self._dense_points(labels_by_time[stamp].to(template.device), template)
            if stamp:
                transform = transforms[stamp].to(template)
                points = points @ transform[:3, :3].T + transform[:3, 3]
            indices = torch.floor((points - lower) / self.voxel_size).long()
            valid = ((indices >= 0) & (indices < indices.new_tensor(shape))).all(-1)
            indices, labels = indices[valid], labels[valid]
            flat = indices[:, 0] * shape[1] * shape[2] + indices[:, 1] * shape[2] + indices[:, 2]
            stamps[flat, stamp] = True
            semantics[flat] = labels
        occupied = semantics != self.empty_label
        flat = occupied.nonzero(as_tuple=True)[0]
        x = flat // (shape[1] * shape[2])
        remainder = flat % (shape[1] * shape[2])
        y, z = remainder // shape[2], remainder % shape[2]
        points = torch.stack((x, y, z), dim=-1).to(template)
        points = lower + (points + .5) * self.voxel_size
        return points, semantics[occupied], stamps[occupied]

    def _batch_set_loss(self, points, logits, target_provider, **kwargs):
        loss_cls = points.sum() * 0.
        loss_pts = points.sum() * 0.
        for batch in range(points.shape[0]):
            targets = target_provider(batch, points)
            cls, pts = self._set_loss(points[batch], logits[batch], *targets, **kwargs)
            loss_cls, loss_pts = loss_cls + cls, loss_pts + pts
        return loss_cls / points.shape[0], loss_pts / points.shape[0]

    def forward(self, inputs):
        metas = inputs['metas']
        details = {}
        total = inputs['initial_pred_points'].sum() * 0.
        statistics = (inputs['stamp_statistics']
                      if inputs.get('update_stamp_statistics', True) else None)
        total_queries = self.num_query + sum(self.future_queries)

        def expanded(batch, template):
            return self._expanded_targets(metas, batch, template)

        # Initial anchors are always part of temporal stack supervision.
        initial = inputs['initial_pred_points']
        pseudo = initial.new_zeros(*initial.shape[:-1], 17)
        init_cls, init_pts = self._batch_set_loss(
            initial, pseudo, expanded, stamp_statistics=statistics,
            query_count=total_queries, temporal_reweight=True)
        total = total + init_pts
        details['init_loss_pts'] = init_pts.detach()

        temporal_stages = (list(zip(inputs['temporal_pred_points'],
                                    inputs['temporal_pred_logits']))
                           if inputs['sparseworld_pretrain'] else
                           [(inputs['temporal_pred_points'][0],
                             inputs['temporal_pred_logits'][0])])
        for stage, (points, logits) in enumerate(temporal_stages):
            cls, pts = self._batch_set_loss(
                points, logits, expanded, stamp_statistics=statistics,
                query_count=total_queries, temporal_reweight=True)
            total = total + cls + pts
            details[f'd{stage}.loss_cls'] = cls.detach()
            details[f'd{stage}.loss_pts'] = pts.detach()

        if not inputs['sparseworld_pretrain']:
            def current(batch, template):
                return self._dense_points(metas['occ_label'][batch].to(template.device), template)
            for stage, (points, logits) in enumerate(zip(
                    inputs['current_pred_points'], inputs['current_pred_logits'])):
                cls, pts = self._batch_set_loss(points, logits, current)
                total = total + cls + pts
                details[f'current.d{stage}.loss_cls'] = cls.detach()
                details[f'current.d{stage}.loss_pts'] = pts.detach()
            for horizon, (points, logits) in enumerate(zip(
                    inputs['future_pred_points'], inputs['future_pred_logits'])):
                def future(batch, template, step=horizon):
                    return self._dense_points(
                        metas['future_occ_labels'][batch, step].to(template.device), template)
                cls = points.sum() * 0.
                pts = points.sum() * 0.
                masks = inputs.get('future_pred_masks', ())
                for batch in range(points.shape[0]):
                    targets = future(batch, points)
                    mask = masks[horizon][batch] if len(masks) > horizon else None
                    current_cls, current_pts = self._set_loss(
                        points[batch], logits[batch], *targets,
                        prediction_mask=mask)
                    cls, pts = cls + current_cls, pts + current_pts
                cls, pts = cls / points.shape[0], pts / points.shape[0]
                total = total + cls + pts
                details[f'fu{horizon + 1}.loss_cls'] = cls.detach()
                details[f'fu{horizon + 1}.loss_pts'] = pts.detach()

        predictions = inputs['pred_traj']
        targets = metas['temporal_trajs'].to(predictions)
        for horizon in range(predictions.shape[1]):
            # Source l2_loss: sum(mean(abs(error)^2, dim=batch)).
            trajectory = (predictions[:, horizon] - targets[:, horizon]).abs().square()
            trajectory = trajectory.mean(dim=0).sum()
            total = total + trajectory
            details[f'loss_traj_{horizon + 1}s'] = trajectory.detach()
        return total, details
