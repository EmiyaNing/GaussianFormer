"""Same predicted-class matching/reduction for both CD directions and focal."""
import torch

from . import OPENOCC_LOSS
from .classwise_chamfer_set_loss import ClasswiseChamferSetLoss
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class StrictClasswiseSetLoss(ClasswiseChamferSetLoss):
    """Frozen previous geometry, class-mean focal with a single outer weight.

    An active class has both predictions and camera-filtered GT. Focal targets
    are the labels of nearest same-class GT, hence equal detached argmax labels.
    The original spatial GT weight is retained, but neither an inner class
    weight nor the geometry-only distance-threshold multiplier is applied.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._classification_rows = []

    def classification(self, points, logits, gt_points, gt_labels, global_labels):
        classes = logits.detach().argmax(-1)
        value = logits.sum() * 0.
        denominator = logits.new_zeros(())
        rows = logits.new_zeros((17, 4))
        # raw class focal, outer-weighted focal, active predictions, agreement
        for c in range(17):
            pm, gm = classes == c, gt_labels == c
            if not pm.any() or not gm.any():
                continue
            p, g, z = points[pm], gt_points[gm], logits[pm]
            match = _nearest_indices(p, g, self.chunk_size)
            targets = gt_labels[gm][match]
            term = self._sigmoid_focal(z, targets, self._distance_weight(g[match]))
            weight = self.geometry_class_weights[c].to(logits)
            value = value + weight * term
            denominator = denominator + weight
            rows[c] = torch.stack((term.detach(), (weight * term).detach(),
                                  pm.sum().to(logits),
                                  (global_labels[pm] == c).sum().to(logits)))
        return value / denominator.clamp_min(torch.finfo(logits.dtype).tiny), rows

    def _single(self, points, logits, gt_points, gt_labels, gt_camera_valid):
        # Geometry calls the unmodified, previously audited implementation.
        pred, gt, geometry_rows = self.geometry(
            points, logits, gt_points, gt_labels, gt_camera_valid)
        self._class_rows.append(geometry_rows)
        p2g = _nearest_indices(points, gt_points, self.chunk_size)
        g2p = _nearest_indices(gt_points, points, self.chunk_size)
        with torch.no_grad():
            pd = (points.detach() - gt_points[p2g]).square().sum(-1)
            gd = (gt_points - points.detach()[g2p]).square().sum(-1)
        cls, cls_rows = self.classification(points, logits, gt_points, gt_labels,
                                            gt_labels[p2g])
        self._classification_rows.append(cls_rows.detach())
        return cls, pred, gt, pd, gd

    def components(self, inputs):
        self._classification_rows = []
        parts, metrics = super().components(inputs)
        batch = inputs['opus_pred_points'][0].shape[0]
        assert len(self._classification_rows) == len(self.stage_weights) * batch
        for s in range(len(self.stage_weights)):
            rows = torch.stack(self._classification_rows[s*batch:(s+1)*batch])
            prefix = f'stage_{s}/strict_classification'
            metrics[prefix + '/agreement_with_nearest_GT'] = (
                rows[..., 3].sum() / rows[..., 2].sum().clamp_min(1))
            for c in range(17):
                metrics[f'{prefix}/class_{c}/raw_term'] = rows[:, c, 0].mean()
                metrics[f'{prefix}/class_{c}/weighted_term'] = rows[:, c, 1].mean()
        return parts, metrics
