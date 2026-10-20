"""Controlled set-loss ablations; the historical OPUSSetLoss is unchanged."""
import torch

from . import OPENOCC_LOSS
from .opus_set_loss import OPUSSetLoss


@OPENOCC_LOSS.register_module()
class CenterSupportDiagnosticSetLoss(OPUSSetLoss):
    def __init__(self, gt_domain='all', final_gt_domain=None,
                 final_geometry_weight=1.0, final_classification_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.gt_domain = gt_domain
        self.final_gt_domain = final_gt_domain or gt_domain
        if {self.gt_domain, self.final_gt_domain} - {'all', 'camera'}:
            raise ValueError('GT domain must be all or camera')
        self.final_geometry_weight = float(final_geometry_weight)
        self.final_classification_weight = float(final_classification_weight)
        if min(self.final_geometry_weight, self.final_classification_weight) < 0:
            raise ValueError('Loss multipliers must be nonnegative')

    def components(self, inputs):
        """Differentiable components, including actual stage/lambda weights.

        _single is inherited so assignment, focal, distance/rare weighting and
        Smooth-L1 exactly match the implementation being diagnosed. Reduction
        uses prediction counts for p2g/cls and GT counts for g2p, as in OPUS.
        """
        points, logits = inputs['opus_pred_points'], inputs['opus_pred_logits']
        masks = inputs.get('opus_point_valid_masks')
        metas = inputs['metas']
        if len(points) != len(self.stage_weights):
            raise ValueError('stage_weights must match prediction stages')
        device = points[0].device
        gt_xyz = metas['occ_xyz'].to(device).flatten(1, -2)
        gt_y = metas['occ_label'].to(device).long().flatten(1)
        # Invalid/unknown labels must never become occupied set targets.
        valid = (gt_y >= 0) & (gt_y < self.empty_label)
        camera = metas.get('occ_cam_mask')
        if camera is None:
            if 'camera' in (self.gt_domain, self.final_gt_domain):
                raise ValueError('camera-domain set loss requires occ_cam_mask')
            camera = valid
        else:
            camera = camera.to(device).bool().flatten(1)
        result, metrics = {}, {}
        last = len(points) - 1
        for s, (weight, xyz, z) in enumerate(zip(self.stage_weights, points, logits)):
            domain = self.final_gt_domain if s == last else self.gt_domain
            target_valid = valid & camera if domain == 'camera' else valid
            cls_sum, p_sum, g_sum = z.sum() * 0, xyz.sum() * 0, xyz.sum() * 0
            npred = ngt = 0
            distances_p, distances_g = [], []
            for b in range(xyz.shape[0]):
                pm = (torch.ones(xyz.shape[1], device=device, dtype=torch.bool)
                      if masks is None else masks[s][b].bool())
                tm = target_valid[b]
                if not pm.any() or not tm.any():
                    continue
                cls, p, g, pd, gd = self._single(
                    xyz[b, pm], z[b, pm], gt_xyz[b, tm], gt_y[b, tm], camera[b, tm])
                pn, gn = int(pm.sum()), int(tm.sum())
                cls_sum, p_sum, g_sum = cls_sum + cls * pn, p_sum + p * pn, g_sum + g * gn
                npred, ngt = npred + pn, ngt + gn
                distances_p.append(pd)
                distances_g.append(gd)
            gm = self.final_geometry_weight if s == last else 1.0
            cm = self.final_classification_weight if s == last else 1.0
            prefix = f'stage_{s}'
            result[prefix + '/classification'] = weight * self.lambda_cls * cm * cls_sum / max(npred, 1)
            result[prefix + '/pred_to_gt'] = weight * self.lambda_pts * gm * p_sum / max(npred, 1)
            result[prefix + '/gt_to_pred'] = weight * self.lambda_pts * gm * g_sum / max(ngt, 1)
            metrics[prefix + '/gt_count'] = ngt
            metrics[prefix + '/pred_count'] = npred
            if distances_p:
                metrics[prefix + '/p2g_m'] = torch.cat(distances_p).sqrt().mean().detach()
                metrics[prefix + '/g2p_m'] = torch.cat(distances_g).sqrt().mean().detach()
        return result, metrics

    def forward(self, inputs):
        components, metrics = self.components(inputs)
        metrics.update({name: value.detach() for name, value in components.items()})
        return sum(components.values()), metrics
