"""GT-SDF center and locally fixed-normal scale losses; no renderer changes."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from . import OPENOCC_LOSS
from .opus_set_loss import OPUSSetLoss, _nearest_indices


@OPENOCC_LOSS.register_module()
class OPUSSetClassificationLoss(OPUSSetLoss):
    """Identical full-set focal classification, omitting both geometric terms."""
    def __init__(self, **kwargs):
        kwargs['lambda_pts'] = 0.
        super().__init__(**kwargs)

    def forward(self, inputs):
        points = inputs['opus_pred_points']; logits = inputs['opus_pred_logits']
        valid = inputs.get('opus_point_valid_masks'); metas = inputs['metas']
        gt_labels = metas['occ_label'].to(points[0].device).long().flatten(1)
        gt_points = metas['occ_xyz'].to(points[0].device).flatten(1, -2)
        if len(points) != len(self.stage_weights): raise ValueError('Wrong stage count')
        total = logits[0].sum()*0; values = []
        for stage, (weight, p, z) in enumerate(zip(self.stage_weights, points, logits)):
            numerator = z.sum()*0; denominator = 0
            for batch in range(p.shape[0]):
                target_mask = gt_labels[batch] != self.empty_label
                mask = torch.ones(p.shape[1], dtype=torch.bool, device=p.device) if valid is None else valid[stage][batch].bool()
                if not target_mask.any() or not mask.any(): continue
                target = gt_points[batch, target_mask]; labels = gt_labels[batch, target_mask]
                chosen = _nearest_indices(p[batch, mask], target, self.chunk_size)
                assigned = labels[chosen]
                class_weight = self.class_weights.to(z)[assigned] if self.class_weights.numel() else torch.ones_like(assigned, dtype=z.dtype)
                sample_weight = class_weight*self._distance_weight(target[chosen])
                count = int(mask.sum().item())
                numerator = numerator + self._sigmoid_focal(z[batch, mask], assigned, sample_weight)*count
                denominator += count
            value = numerator/max(denominator, 1)
            total = total + weight*self.lambda_cls*value
            values.append(value.detach())
        return total, dict(loss_cls=torch.stack(values).mean(), loss_pts=total.detach()*0,
                          valid_gt_points=(gt_labels != self.empty_label).sum().detach())


def sample_geometry(gaussian, fields, pc_range=(-40., -40., -1., 40., 40., 5.4)):
    means = gaussian.means
    lower = means.new_tensor(pc_range[:3])
    extent = means.new_tensor(pc_range[3:]) - lower
    grid = (2*(means-lower)/extent-1).reshape(means.shape[0], 1, 1, -1, 3)
    sampled = F.grid_sample(fields, grid, mode='bilinear', padding_mode='border',
                            align_corners=False).squeeze(2).squeeze(2).transpose(1, 2)
    inside = ((means > lower) & (means < lower+extent)).all(-1)
    center_valid = inside & (sampled[..., 4].detach() > 1-1e-5)
    n = sampled[..., 1:4].detach()
    norm = n.norm(dim=-1, keepdim=True)
    shape_valid = center_valid & (sampled[..., 5].detach() > 1-1e-5) & (norm[..., 0] > .9)
    n = n / norm.clamp_min(1e-6)
    q = F.normalize(gaussian.rotations, dim=-1)
    # Local renderer uses Sigma=R.T @ S^2 @ R; apply R to the normal.
    uv = 2*torch.cross(q[..., 1:], n, dim=-1)
    rotated = n + q[..., :1]*uv + torch.cross(q[..., 1:], uv, dim=-1)
    sigma = (gaussian.scales.square()*rotated.square()).sum(-1).clamp_min(1e-12).sqrt()
    return sampled[..., 0], sigma, center_valid, shape_valid


@OPENOCC_LOSS.register_module()
class GaussianSDFGeometryLoss(nn.Module):
    def __init__(self, weight_center=.5, weight_shape=.5, delta=0., rho=.1,
                 pc_range=(-40., -40., -1., 40., 40., 5.4)):
        super().__init__()
        if not 0 < rho < .5:
            raise ValueError('rho must lie in (0,.5)')
        self.weight_center = weight_center
        self.weight_shape = weight_shape
        self.delta = delta
        self.rho = rho
        self.quantile = math.sqrt(2)*torch.erfinv(torch.tensor(1-2*rho, dtype=torch.float64)).item()
        self.pc_range = tuple(pc_range)

    def forward(self, inputs):
        gaussian = inputs['gaussian']
        value, sigma, cv, sv = sample_geometry(gaussian, inputs['metas']['sdf_fields'], self.pc_range)
        zero = gaussian.means.sum()*0 + gaussian.scales.sum()*0 + gaussian.rotations.sum()*0
        center = F.relu(value[cv]-self.delta).square().mean() if cv.any() else zero
        shape = F.relu(value[sv]+self.quantile*sigma[sv]-self.delta).square().mean() if sv.any() else zero
        total = self.weight_center*center + self.weight_shape*shape
        return total, dict(center=center.detach(), shape=shape.detach(),
            valid_center=cv.sum().detach(), valid_shape=sv.sum().detach(),
            center_outside_fraction=((value[cv] > 0).float().mean() if cv.any() else zero).detach(),
            normal_scale=(sigma[sv].mean() if sv.any() else zero).detach())
