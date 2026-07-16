import torch
import torch.nn.functional as F
from torch import nn

from . import OPENOCC_LOSS


def _nearest_distances(source, target, chunk_size):
    """Return squared nearest-neighbour distances and target indices.

    The operation is exact: chunking limits only the temporary distance
    matrix, never the point set.  This is important for OPUS-V1, whose
    point-wise heads must all receive a matching loss.
    """
    values, indices = [], []
    with torch.autocast(device_type=source.device.type, enabled=False):
        source, target = source.float(), target.float()
        target_norm = target.square().sum(dim=1).unsqueeze(0)
        target_t = target.transpose(0, 1).contiguous()
        for start in range(0, source.shape[0], chunk_size):
            source_chunk = source[start:start + chunk_size]
            distance = (source_chunk.square().sum(dim=1, keepdim=True) +
                        target_norm - 2.0 * source_chunk.matmul(target_t))
            value, index = distance.clamp_min_(0.0).min(dim=1)
            values.append(value)
            indices.append(index)
    return torch.cat(values), torch.cat(indices)


@OPENOCC_LOSS.register_module()
class OPUSSetLoss(nn.Module):
    """The OPUS-V1 full-set, bidirectional nearest-neighbour objective.

    In contrast to the former integration, no deterministic point truncation
    is performed.  The official V1 implementation uses an exact GPU KNN;
    this portable implementation obtains the same assignments in chunks.
    """

    def __init__(self, stage_weights=(1.,), lambda_cls=2.0,
                 focal_gamma=2.0, focal_alpha=0.25, lambda_pts=0.5,
                 smooth_l1_beta=0.2, empty_dist_thr=0.2,
                 empty_weight=5.0, rare_classes=(0, 2, 5, 8),
                 rare_weight=10.0, class_weights=None,
                 pc_range=(-40., -40., -1., 40., 40., 5.4),
                 chunk_size=1024, loss_mode='official_v1',
                 # Kept only so older configs construct successfully.  It is
                 # intentionally ignored in official_v1 mode.
                 lambda_cd=None, max_match_points=None):
        super().__init__()
        if loss_mode != 'official_v1':
            raise ValueError('Only the full-set official_v1 OPUS loss is supported')
        self.stage_weights = tuple(stage_weights)
        self.lambda_cls = lambda_cls
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.lambda_pts = lambda_pts
        self.smooth_l1_beta = smooth_l1_beta
        self.empty_dist_thr = empty_dist_thr
        self.empty_weight = empty_weight
        self.rare_classes = tuple(rare_classes)
        self.rare_weight = rare_weight
        self.pc_range = tuple(pc_range)
        self.chunk_size = chunk_size
        self.register_buffer(
            'class_weights',
            torch.as_tensor(class_weights if class_weights is not None else [],
                            dtype=torch.float32), persistent=False)

    def _distance_weight(self, points):
        lower = points.new_tensor(self.pc_range[:3])
        upper = points.new_tensor(self.pc_range[3:])
        center = (lower + upper) / 2.0
        max_distance = torch.linalg.vector_norm((upper - lower)[:2])
        return torch.linalg.vector_norm(points[:, :2] - center[:2], dim=-1) / max_distance + 1.0

    def _sigmoid_focal(self, logits, labels, sample_weights):
        targets = F.one_hot(labels, num_classes=logits.shape[-1]).to(logits.dtype)
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce
        return loss.sum(dim=-1).mul(sample_weights).sum() / max(logits.shape[0], 1)

    def _single(self, points, logits, gt_points, gt_labels, gt_camera_valid):
        # pred_to_gt: semantic assignment and pred-side geometric loss.
        pred_distance2, pred_to_gt = _nearest_distances(points, gt_points, self.chunk_size)
        # gt_to_pred: coverage loss.  Camera validity only reweights this
        # direction, matching OPUS-V1's sparse-voxel target construction.
        gt_distance2, gt_to_pred = _nearest_distances(gt_points, points, self.chunk_size)
        assigned_labels = gt_labels[pred_to_gt]

        if self.class_weights.numel():
            if self.class_weights.numel() != logits.shape[-1]:
                raise ValueError('class_weights must match OPUS logits classes')
            class_weight = self.class_weights.to(logits)[assigned_labels]
        else:
            class_weight = torch.ones_like(pred_distance2)
        cls_weight = class_weight * self._distance_weight(gt_points[pred_to_gt])

        gt_weight = torch.ones_like(gt_distance2)
        gt_weight[(gt_distance2.sqrt() > self.empty_dist_thr) & gt_camera_valid] = self.empty_weight
        for class_index in self.rare_classes:
            gt_weight[gt_labels == class_index].clamp_(min=self.rare_weight)

        pred_target = gt_points[pred_to_gt]
        gt_prediction = points[gt_to_pred]
        pred_loss = F.smooth_l1_loss(
            points, pred_target, beta=self.smooth_l1_beta, reduction='none').sum(dim=-1).mean()
        gt_loss = F.smooth_l1_loss(
            gt_points, gt_prediction, beta=self.smooth_l1_beta, reduction='none').sum(dim=-1)
        gt_loss = (gt_loss * gt_weight).mean()
        cls_loss = self._sigmoid_focal(logits, assigned_labels, cls_weight)
        return cls_loss, self.lambda_pts * (pred_loss + gt_loss), pred_distance2, gt_distance2

    def forward(self, inputs):
        pred_points = inputs['opus_pred_points']
        pred_logits = inputs['opus_pred_logits']
        pred_valid_masks = inputs.get('opus_point_valid_masks')
        metas = inputs['metas']
        gt_points = metas['opus_gt_points'].to(pred_points[0].device)
        gt_labels = metas['opus_gt_labels'].to(pred_points[0].device).long()
        gt_valid = metas['opus_gt_valid'].to(pred_points[0].device).bool()
        gt_camera_valid = metas.get('opus_gt_camera_valid', gt_valid)
        gt_camera_valid = gt_camera_valid.to(pred_points[0].device).bool()
        if len(pred_points) != len(self.stage_weights):
            raise ValueError('stage_weights must match OPUS prediction stages')

        total = pred_points[0].sum() * 0.0
        cls_values, pts_values, pred_distances, gt_distances = [], [], [], []
        for stage_index, (weight, stage_points, stage_logits) in enumerate(
                zip(self.stage_weights, pred_points, pred_logits)):
            stage_cls = stage_logits.sum() * 0.0
            stage_pts = stage_points.sum() * 0.0
            valid_batches = 0
            for batch_index in range(stage_points.shape[0]):
                target_mask = gt_valid[batch_index]
                if not target_mask.any():
                    continue
                point_mask = (torch.ones(stage_points.shape[1], device=stage_points.device,
                                         dtype=torch.bool) if pred_valid_masks is None else
                              pred_valid_masks[stage_index][batch_index].bool())
                if not point_mask.any():
                    continue
                cls, pts, p2g, g2p = self._single(
                    stage_points[batch_index, point_mask], stage_logits[batch_index, point_mask],
                    gt_points[batch_index, target_mask], gt_labels[batch_index, target_mask],
                    gt_camera_valid[batch_index, target_mask])
                stage_cls = stage_cls + cls
                stage_pts = stage_pts + pts
                pred_distances.append(p2g.detach())
                gt_distances.append(g2p.detach())
                valid_batches += 1
            if valid_batches:
                stage_cls = stage_cls / valid_batches
                stage_pts = stage_pts / valid_batches
            total = total + weight * (self.lambda_cls * stage_cls + stage_pts)
            cls_values.append(stage_cls.detach())
            pts_values.append(stage_pts.detach())

        metrics = {
            'loss_cls': torch.stack(cls_values).mean(),
            'loss_pts': torch.stack(pts_values).mean(),
            'pred_to_gt_distance_m': torch.cat(pred_distances).sqrt().mean(),
            'gt_to_pred_distance_m': torch.cat(gt_distances).sqrt().mean(),
            'valid_gt_points': gt_valid.sum().detach(),
            'valid_pred_points': (pred_valid_masks[-1].sum().detach()
                                  if pred_valid_masks is not None else
                                  pred_points[-1].new_tensor(pred_points[-1].shape[1])),
        }
        return total, metrics
