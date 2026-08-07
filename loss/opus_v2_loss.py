"""Official OPUSv2 point/densified-voxel training objective."""
import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class OPUSV2Loss(nn.Module):
    """Reproduce ``OPUSV2Head.loss`` on the local runner interface.

    Every decoder stage receives the bidirectional Smooth-L1 point loss and
    sparse densifier focal loss.  The learned initial points receive the same
    GT-to-pred/pred-to-GT point term without classification.
    """

    def __init__(self, num_classes=17, empty_label=17, pc_range=(),
                 class_weights=(10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1),
                 focal_gamma=2., focal_alpha=.25, cls_loss_weight=2.,
                 pts_loss_weight=.5, smooth_l1_beta=.2,
                 empty_dist_thr=.2, empty_weight=5.,
                 rare_classes=(0, 2, 5, 8), rare_weight=10., chunk_size=1024):
        super().__init__()
        self.num_classes, self.empty_label = num_classes, empty_label
        self.pc_range = tuple(pc_range)
        self.focal_gamma, self.focal_alpha = focal_gamma, focal_alpha
        self.cls_loss_weight, self.pts_loss_weight = cls_loss_weight, pts_loss_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.empty_dist_thr, self.empty_weight = empty_dist_thr, empty_weight
        self.rare_classes, self.rare_weight = tuple(rare_classes), rare_weight
        self.chunk_size = chunk_size
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32),
                             persistent=False)

    def _decode(self, points):
        lower = points.new_tensor(self.pc_range[:3])
        upper = points.new_tensor(self.pc_range[3:])
        return points * (upper - lower) + lower

    def _point_loss_single(self, predicted, gt_points, gt_camera_valid, gt_labels):
        # Official names: gt_paired_idx maps every GT point to a prediction;
        # pred_paired_idx maps every prediction to a GT point.
        gt_to_pred = _nearest_indices(gt_points, predicted, self.chunk_size)
        pred_to_gt = _nearest_indices(predicted, gt_points, self.chunk_size)
        paired_prediction = predicted[gt_to_pred]
        with torch.no_grad():
            distance = torch.linalg.vector_norm(gt_points - paired_prediction.detach(), dim=-1)
            gt_weights = torch.ones_like(distance)
            gt_weights[(distance > self.empty_dist_thr) & gt_camera_valid] = self.empty_weight
            for class_index in self.rare_classes:
                mask = (gt_labels == class_index) & gt_camera_valid
                gt_weights[mask] = gt_weights[mask].clamp_min(self.rare_weight)
        gt_term = F.smooth_l1_loss(
            gt_points, paired_prediction, beta=self.smooth_l1_beta, reduction='none')
        gt_term = (gt_term * gt_weights[:, None]).sum()
        pred_term = F.smooth_l1_loss(
            predicted, gt_points[pred_to_gt], beta=self.smooth_l1_beta, reduction='sum')
        return gt_term, pred_term, gt_points.shape[0], predicted.shape[0]

    def _point_loss(self, normalized_points, gt_points, gt_valid,
                    gt_camera_valid, gt_labels):
        predicted = self._decode(normalized_points.float()).flatten(1, 2)
        gt_sum = predicted.sum() * 0.
        pred_sum = predicted.sum() * 0.
        gt_count = predicted_count = 0
        for batch_index in range(predicted.shape[0]):
            mask = gt_valid[batch_index]
            if not mask.any():
                continue
            sample_gt, sample_pred, current_gt, current_pred = self._point_loss_single(
                predicted[batch_index], gt_points[batch_index, mask].float(),
                gt_camera_valid[batch_index, mask], gt_labels[batch_index, mask])
            gt_sum = gt_sum + sample_gt
            pred_sum = pred_sum + sample_pred
            gt_count += int(current_gt)
            predicted_count += int(current_pred)
        return self.pts_loss_weight * (
            gt_sum / max(gt_count, 1) + pred_sum / max(predicted_count, 1))

    def _focal_loss(self, logits, labels):
        if logits is None or labels is None or logits.numel() == 0:
            reference = logits if logits is not None else self.class_weights
            return reference.sum() * 0.
        logits = logits.float()
        valid_class = labels < self.num_classes
        targets = logits.new_zeros(logits.shape)
        if valid_class.any():
            targets[valid_class] = F.one_hot(
                labels[valid_class], self.num_classes).to(logits.dtype)
        probability = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = probability * targets + (1. - probability) * (1. - targets)
        alpha = self.focal_alpha * targets + (1. - self.focal_alpha) * (1. - targets)
        loss = alpha * (1. - pt).pow(self.focal_gamma) * ce
        loss = loss * self.class_weights.to(loss)[None]
        average = (labels != self.empty_label).sum().clamp_min(1)
        return self.cls_loss_weight * loss.sum() / average

    def forward(self, inputs):
        initial = inputs['init_points']
        refined = inputs['all_refine_pts']
        sparse_logits = inputs['all_cls_scores']
        sparse_coords = inputs['all_voxel_coors']
        metas = inputs['metas']
        dense_labels = metas['occ_label'].long()
        camera_mask = metas.get('occ_cam_mask', torch.ones_like(dense_labels)).bool()
        gt_valid = dense_labels != self.empty_label
        if 'occ_xyz' in metas:
            gt_points = metas['occ_xyz'].float().flatten(1, -2)
        else:
            raise KeyError('OPUSV2Loss requires metas["occ_xyz"]')
        flat_labels = dense_labels.flatten(1)
        flat_valid = gt_valid.flatten(1)
        flat_camera = camera_mask.flatten(1)

        init_loss = self._point_loss(
            initial, gt_points, flat_valid, flat_camera, flat_labels)
        point_losses, cls_losses = [], []
        total = init_loss
        for points, logits, coords in zip(refined, sparse_logits, sparse_coords):
            point_loss = self._point_loss(
                points, gt_points, flat_valid, flat_camera, flat_labels)
            if logits is None or coords is None:
                cls_loss = point_loss * 0.
            else:
                batch, x, y, z = coords.long().unbind(-1)
                target = dense_labels[batch, x, y, z]
                cls_loss = self._focal_loss(logits, target)
            point_losses.append(point_loss)
            cls_losses.append(cls_loss)
            total = total + point_loss + cls_loss

        metrics = {'init_loss_pts': init_loss.detach()}
        for index, (cls_loss, point_loss) in enumerate(zip(cls_losses, point_losses)):
            prefix = '' if index == len(point_losses) - 1 else f'd{index}.'
            metrics[prefix + 'loss_cls'] = cls_loss.detach()
            metrics[prefix + 'loss_pts'] = point_loss.detach()
        return total, metrics
