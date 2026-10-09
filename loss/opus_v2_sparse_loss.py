"""OPUS-V2 point Chamfer and sparse discretized voxel classification loss."""
import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class OPUSV2SparseLoss(nn.Module):
    def __init__(self, num_classes=17, empty_label=17,
                 lambda_cls=2.0, lambda_pts=0.5, beta=0.2,
                 focal_alpha=0.25, focal_gamma=2.0,
                 empty_dist_thr=0.2, empty_weight=5.0,
                 rare_classes=(0, 2, 5, 8), rare_weight=10.0,
                 class_weights=(10, 5, 10, 5, 5, 10, 10, 5, 10, 5,
                                5, 1, 5, 1, 1, 2, 1),
                 chunk_size=1024):
        super().__init__()
        self.num_classes = num_classes
        self.empty_label = empty_label
        self.lambda_cls = lambda_cls
        self.lambda_pts = lambda_pts
        self.beta = beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.empty_dist_thr = empty_dist_thr
        self.empty_weight = empty_weight
        self.rare_classes = tuple(rare_classes)
        self.rare_weight = rare_weight
        self.chunk_size = chunk_size
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def _points_loss(self, pred, gt, labels, camera_valid):
        pred_to_gt = _nearest_indices(pred, gt, self.chunk_size)
        gt_to_pred = _nearest_indices(gt, pred, self.chunk_size)
        nearest_pred = pred[gt_to_pred]
        nearest_gt = gt[pred_to_gt]
        pred_loss = F.smooth_l1_loss(pred, nearest_gt, beta=self.beta,
                                     reduction='none').sum() / max(len(pred), 1)
        gt_error = F.smooth_l1_loss(gt, nearest_pred, beta=self.beta,
                                    reduction='none').sum(dim=-1)
        weights = torch.ones_like(gt_error)
        gap = torch.linalg.vector_norm(gt - nearest_pred.detach(), dim=-1)
        weights[(gap > self.empty_dist_thr) & camera_valid] = self.empty_weight
        for cls in self.rare_classes:
            rare = (labels == cls) & camera_valid
            weights[rare] = weights[rare].clamp_min(self.rare_weight)
        gt_loss = (gt_error * weights).sum() / max(len(gt), 1)
        return self.lambda_pts * (pred_loss + gt_loss)

    def _classification_loss(self, logits, coords, dense_labels):
        if logits is None or coords is None or logits.numel() == 0:
            return dense_labels.sum() * 0.0
        batch, x, y, z = coords.unbind(dim=-1)
        labels = dense_labels[batch, x, y, z].long()
        occupied = labels != self.empty_label
        target = F.one_hot(labels.clamp(0, self.num_classes - 1),
                           num_classes=self.num_classes).to(logits.dtype)
        target = target * occupied.unsqueeze(-1)
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        weight = self.class_weights.to(logits)[None]
        value = ce * (1 - p_t).pow(self.focal_gamma) * alpha_t * weight
        return self.lambda_cls * value.sum() / occupied.sum().clamp_min(1)

    def forward(self, inputs):
        pred_points = inputs['v2_pred_points']
        sparse_logits = inputs['v2_sparse_logits']
        sparse_coords = inputs['v2_sparse_coords']
        metas = inputs['metas']
        dense = metas['occ_label'].to(pred_points[0].device).long()
        xyz = metas['occ_xyz'].to(pred_points[0].device).reshape(dense.shape[0], -1, 3)
        flat_labels = dense.flatten(1)
        camera = metas['occ_cam_mask'].to(dense.device).bool().flatten(1)
        total = pred_points[0].sum() * 0.0
        point_values, class_values = [], []
        for stage, predicted in enumerate(pred_points):
            stage_points = predicted.sum() * 0.0
            valid_samples = 0
            for index in range(predicted.shape[0]):
                mask = flat_labels[index] != self.empty_label
                if not mask.any():
                    continue
                stage_points = stage_points + self._points_loss(
                    predicted[index], xyz[index, mask], flat_labels[index, mask],
                    camera[index, mask])
                valid_samples += 1
            stage_points = stage_points / max(valid_samples, 1)
            stage_cls = self._classification_loss(
                sparse_logits[stage], sparse_coords[stage], dense)
            total = total + stage_points + stage_cls
            point_values.append(stage_points.detach())
            class_values.append(stage_cls.detach())
        initial = inputs.get('v2_init_points')
        if initial is not None:
            initial_loss = initial.sum() * 0.0
            valid_samples = 0
            for index in range(initial.shape[0]):
                mask = flat_labels[index] != self.empty_label
                if not mask.any():
                    continue
                initial_loss = initial_loss + self._points_loss(
                    initial[index], xyz[index, mask], flat_labels[index, mask],
                    camera[index, mask])
                valid_samples += 1
            total = total + initial_loss / max(valid_samples, 1)
        return total, {'loss_pts': torch.stack(point_values).mean(),
                       'loss_cls': torch.stack(class_values).mean()}
