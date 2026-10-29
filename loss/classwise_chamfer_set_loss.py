"""Class-conditional bidirectional OPUS geometry; original focal is unchanged."""
import torch
import torch.nn.functional as F
from . import OPENOCC_LOSS
from .center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss
from .opus_set_loss import _nearest_indices


@OPENOCC_LOSS.register_module()
class ClasswiseChamferSetLoss(CenterSupportDiagnosticSetLoss):
    """Weighted average of 17 per-class, per-direction Smooth-L1 CD terms.

    Partition predictions using detached stage logits argmax. A class is active
    only when both sets are nonempty; unsupported counts are always logged.
    The original all-class focal target assignment and sample weights remain.
    """
    def __init__(self, geometry_class_weights=None, **kwargs):
        super().__init__(**kwargs)
        weights = geometry_class_weights
        if weights is None:
            weights = self.class_weights.tolist() if self.class_weights.numel() else [1.] * 17
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.shape != (17,) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError('geometry_class_weights must contain 17 finite positive weights')
        self.register_buffer('geometry_class_weights', weights, persistent=False)
        self._class_rows = []

    def geometry(self, points, logits, gt_points, gt_labels, gt_camera_valid):
        if logits.shape[-1] != 17:
            raise ValueError('Classwise geometry requires 17 foreground logits')
        classes = logits.detach().argmax(-1)
        zero = points.sum() * 0.
        pred_sum, gt_sum, denominator = zero, zero, points.new_zeros(())
        stats = points.new_zeros((17, 8))
        # Columns: Npred, Ngt, active, p_loss, g_loss, sum_p_distance,
        # sum_g_distance, geometry_weight. Empty classes have no geometry term.
        for c in range(17):
            pm, gm = classes == c, gt_labels == c
            n, m = int(pm.sum()), int(gm.sum())
            stats[c, 0], stats[c, 1] = n, m
            if n == 0 or m == 0:
                continue
            p, g = points[pm], gt_points[gm]
            p2g = _nearest_indices(p, g, self.chunk_size)
            g2p = _nearest_indices(g, p, self.chunk_size)
            pd2 = (p.detach() - g[p2g]).square().sum(-1)
            gd2 = (g - p.detach()[g2p]).square().sum(-1)
            weights = torch.ones_like(gd2)
            weights[(gd2.sqrt() > self.empty_dist_thr) & gt_camera_valid[gm]] = self.empty_weight
            # Same existing rare-indexing behavior as OPUSSetLoss. No unrelated fix.
            for rare in self.rare_classes:
                weights[gt_labels[gm] == rare].clamp_(min=self.rare_weight)
            lp = F.smooth_l1_loss(p, g[p2g], beta=self.smooth_l1_beta, reduction='none').sum(-1).mean()
            lg = (F.smooth_l1_loss(g, p[g2p], beta=self.smooth_l1_beta,
                                   reduction='none').sum(-1) * weights).mean()
            wc = self.geometry_class_weights[c].to(points)
            pred_sum, gt_sum = pred_sum + wc * lp, gt_sum + wc * lg
            denominator = denominator + wc
            stats[c, 2:] = torch.stack((points.new_tensor(1.), lp.detach(), lg.detach(),
                                       pd2.sqrt().sum(), gd2.sqrt().sum(), wc))
        return pred_sum / denominator.clamp_min(torch.finfo(points.dtype).tiny), gt_sum / denominator.clamp_min(torch.finfo(points.dtype).tiny), stats.detach()

    def _single(self, points, logits, gt_points, gt_labels, gt_camera_valid):
        # Reusing the original function guarantees its focal, global assignments,
        # distance weighting and global-distance diagnostics stay exactly intact.
        cls, _, _, global_pd, global_gd = super()._single(
            points, logits, gt_points, gt_labels, gt_camera_valid)
        pred, gt, stats = self.geometry(points, logits, gt_points, gt_labels, gt_camera_valid)
        self._class_rows.append(stats)
        return cls, pred, gt, global_pd, global_gd

    def components(self, inputs):
        self._class_rows = []
        components, metrics = super().components(inputs)
        batch = inputs['opus_pred_points'][0].shape[0]
        # Experiment uses batch=1 with nonempty camera foreground every frame.
        # Keep explicit stage tags and never silently mislabel empty-sample rows.
        if len(self._class_rows) != len(self.stage_weights) * batch:
            raise ValueError('Classwise diagnostics require nonempty GT and predictions in every sample')
        for stage in range(len(self.stage_weights)):
            rows = torch.stack(self._class_rows[stage*batch:(stage+1)*batch])
            counts = rows[..., :2].sum(0)
            active = rows[..., 2].bool()
            tag = f'stage_{stage}/classwise'
            metrics[tag + '/active_classes_mean'] = active.sum(-1).float().mean()
            metrics[tag + '/weight_sum_mean'] = rows[..., 7].sum(-1).mean()
            metrics[tag + '/unmatched_pred_fraction'] = (rows[..., 0] * ~active).sum() / rows[..., 0].sum().clamp_min(1)
            metrics[tag + '/unmatched_GT_fraction'] = (rows[..., 1] * ~active).sum() / rows[..., 1].sum().clamp_min(1)
            metrics[tag + '/p2g_same_m'] = rows[..., 5].sum() / (rows[..., 0] * active).sum().clamp_min(1)
            metrics[tag + '/g2p_same_m'] = rows[..., 6].sum() / (rows[..., 1] * active).sum().clamp_min(1)
            for c in range(17):
                ct = f'{tag}/class_{c}'
                metrics[ct + '/pred_count'], metrics[ct + '/GT_count'] = counts[c]
                metrics[ct + '/pred_term'] = rows[:, c, 3].mean()
                metrics[ct + '/GT_term'] = rows[:, c, 4].mean()
        self._class_rows = []
        return components, metrics
