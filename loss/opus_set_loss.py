import torch
import torch.nn.functional as F
from torch import nn
from . import OPENOCC_LOSS


def _nearest_distances(source, target, chunk_size):
    """Stable min squared distances without a P x M allocation.

    ``torch.cdist(...).square()`` differentiates through a square root. Near
    coincident points this may yield non-finite gradients under AMP. Computing
    squared distances directly has the same nearest neighbours and a defined
    zero-distance derivative.
    """
    mins, indices = [], []
    with torch.autocast(device_type=source.device.type, enabled=False):
        source, target = source.float(), target.float()
        target_norm = target.square().sum(dim=1).unsqueeze(0)
        target_t = target.transpose(0, 1).contiguous()
        for start in range(0, source.shape[0], chunk_size):
            source_chunk = source[start:start + chunk_size]
            distance = (source_chunk.square().sum(dim=1, keepdim=True) +
                        target_norm - 2.0 * source_chunk.matmul(target_t))
            value, index = distance.clamp_min(0.0).min(dim=1)
            mins.append(value)
            indices.append(index)
    return torch.cat(mins), torch.cat(indices)


def _even_sample(values, limit):
    """Deterministic bounded fallback when no dedicated GPU KNN is installed."""
    if values.shape[0] <= limit:
        return values
    indices = torch.linspace(0, values.shape[0] - 1, limit,
                             device=values.device).long()
    return values[indices]


@OPENOCC_LOSS.register_module()
class OPUSSetLoss(nn.Module):
    """Chamfer geometry plus nearest-neighbour semantic focal supervision."""

    def __init__(self, stage_weights=(0.25, 0.35, 0.5, 0.7, 0.85, 1.0),
                 lambda_cd=1.0, lambda_cls=1.0, focal_gamma=2.0,
                 pc_range=(-40., -40., -1., 40., 40., 5.4),
                 chunk_size=1024, max_match_points=8192):
        super().__init__()
        self.stage_weights = tuple(stage_weights)
        self.lambda_cd = lambda_cd
        self.lambda_cls = lambda_cls
        self.focal_gamma = focal_gamma
        self.pc_range = tuple(pc_range)
        self.chunk_size = chunk_size
        self.max_match_points = max_match_points

    def _normalize_points(self, points):
        lower = points.new_tensor(self.pc_range[:3])
        extent = points.new_tensor(self.pc_range[3:]) - lower
        return (points - lower) / extent.clamp_min(1e-6)

    def forward(self, inputs):
        pred_points, pred_logits = inputs['opus_pred_points'], inputs['opus_pred_logits']
        metas = inputs['metas']
        gt_points = metas['opus_gt_points'].to(pred_points[0].device)
        gt_labels = metas['opus_gt_labels'].to(pred_points[0].device).long()
        gt_valid = metas['opus_gt_valid'].to(pred_points[0].device).bool()
        if len(pred_points) != len(self.stage_weights):
            raise ValueError('stage_weights must match OPUS prediction stages')

        total = pred_points[0].sum() * 0.0
        cd_values, cls_values = [], []
        for weight, stage_points, stage_logits in zip(self.stage_weights, pred_points, pred_logits):
            stage_cd, stage_cls, valid_batches = stage_points.sum() * 0.0, stage_logits.sum() * 0.0, 0
            for batch_idx in range(stage_points.shape[0]):
                valid = gt_valid[batch_idx]
                if not valid.any():
                    continue
                target_points, target_labels = gt_points[batch_idx, valid], gt_labels[batch_idx, valid]
                if target_points.shape[0] > self.max_match_points:
                    select = torch.linspace(0, target_points.shape[0] - 1,
                                            self.max_match_points,
                                            device=target_points.device).long()
                    target_points, target_labels = target_points[select], target_labels[select]
                point_indices = None
                match_points = stage_points[batch_idx]
                if match_points.shape[0] > self.max_match_points:
                    point_indices = torch.linspace(0, match_points.shape[0] - 1,
                                                   self.max_match_points,
                                                   device=match_points.device).long()
                    match_points = match_points[point_indices]
                normalized_predictions = self._normalize_points(match_points)
                normalized_targets = self._normalize_points(target_points)
                p2g, nearest = _nearest_distances(
                    normalized_predictions, normalized_targets, self.chunk_size)
                g2p, _ = _nearest_distances(
                    normalized_targets, normalized_predictions, self.chunk_size)
                stage_cd = stage_cd + p2g.mean() + g2p.mean()
                assigned = target_labels[nearest]
                match_logits = stage_logits[batch_idx] if point_indices is None else stage_logits[batch_idx, point_indices]
                ce = F.cross_entropy(match_logits, assigned, reduction='none')
                pt = torch.exp(-ce.detach())
                stage_cls = stage_cls + ((1.0 - pt).pow(self.focal_gamma) * ce).mean()
                valid_batches += 1
            if valid_batches:
                stage_cd, stage_cls = stage_cd / valid_batches, stage_cls / valid_batches
            total = total + weight * (self.lambda_cd * stage_cd + self.lambda_cls * stage_cls)
            cd_values.append(stage_cd.detach())
            cls_values.append(stage_cls.detach())
        metrics = {
            'loss_cd': torch.stack(cd_values).mean(),
            'loss_cls': torch.stack(cls_values).mean(),
            'mean_nn_distance': torch.sqrt(cd_values[-1].clamp_min(0.0)),
            'valid_gt_points': gt_valid.sum().detach(),
            'valid_pred_points': pred_points[-1].new_tensor(pred_points[-1].shape[1]),
        }
        return total, metrics
