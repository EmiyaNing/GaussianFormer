"""Final-stage supervision experiments; the native renderer is unchanged."""
from itertools import product

import torch
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss
from .opus_set_loss import _nearest_indices
from model.utils.utils import get_rotation_matrix


@torch.no_grad()
def sparse_support(gaussian, gt_xyz, gt_labels, *, pc_min=(-40., -40., -1.),
                   shape=(200, 200, 16), grid_size=.4, scale_multiplier=3.,
                   chunk_size=4096):
    """Exact renderer boxes and Gaussian density, only for foreground GT.

    Returns COO edges (Gaussian, foreground voxel, density). No V x G tensor
    is constructed. There is no opacity or spherical cutoff beyond the
    native integer-box support; underflowed zero weights are discarded.
    """
    means, scales = gaussian.means.detach(), gaussian.scales.detach()
    if means.ndim != 2:
        raise ValueError('sparse_support expects an unbatched Gaussian')
    rotation = get_rotation_matrix(gaussian.rotations.detach())
    transform = torch.diag_embed(scales) @ rotation
    inverse = torch.linalg.inv(transform.transpose(-1, -2) @ transform)
    opacity = gaussian.opacities.detach().reshape(-1)
    if not opacity.numel():
        opacity = means.new_ones(len(means))
    coefficient = torch.linalg.det(inverse).clamp_min(0).sqrt() * opacity / (2 * torch.pi) ** 1.5
    lower = means.new_tensor(pc_min)
    dims = torch.tensor(shape, device=means.device)
    cells = ((gt_xyz - lower) / grid_size).long()
    if not ((cells >= 0) & (cells < dims)).all():
        raise ValueError('GT outside renderer grid')
    flat = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
    lookup = torch.full((int(dims.prod()),), -1, dtype=torch.long, device=means.device)
    lookup[flat] = torch.arange(len(gt_xyz), device=means.device)
    centers = ((means - lower) / grid_size).long()
    radii = torch.ceil(scales.amax(-1) * scale_multiplier / grid_size).long()
    radius = int(radii.max())
    if radius > 3:
        raise ValueError('This mini protocol registers scale <= .4 and radius <= 3')
    offsets = torch.tensor(list(product(range(-radius, radius + 1), repeat=3)), device=means.device)
    rows, columns, weights = [], [], []
    for start in range(0, len(means), chunk_size):
        stop = min(start + chunk_size, len(means))
        bins = centers[start:stop, None] + offsets[None]
        keep = ((bins >= 0) & (bins < dims)).all(-1)
        keep &= (offsets.abs()[None] <= radii[start:stop, None, None]).all(-1)
        row, offset = keep.nonzero(as_tuple=True)
        selected = bins[row, offset]
        voxel = lookup[(selected[:, 0] * shape[1] + selected[:, 1]) * shape[2] + selected[:, 2]]
        present = voxel >= 0
        row, voxel = row[present] + start, voxel[present]
        delta = means[row] - gt_xyz[voxel]
        power = -.5 * torch.einsum('ei,eij,ej->e', delta, inverse[row], delta)
        weight = coefficient[row] * power.exp()
        positive = weight > 0
        rows.append(row[positive]); columns.append(voxel[positive]); weights.append(weight[positive])
    return torch.cat(rows), torch.cat(columns), torch.cat(weights)


def scatter_sum(values, indices, size):
    return values.new_zeros(size).scatter_add_(0, indices, values)


@torch.no_grad()
def allocation_targets(rows, columns, weights, labels, nearest_labels, gt_nearest,
                       *, transport=False, iterations=24, tau_row=1., tau_voxel=10.):
    """Renderer responsibility, or KL-unbalanced transport on the same graph.

    Both treatments give unsupported GT a nearest-Gaussian guard edge and
    unsupported Gaussians their original nearest label. These guard edges
    are reported separately from actual rendering support.
    """
    ng, nv = len(nearest_labels), len(labels)
    density = scatter_sum(weights, columns, nv)
    rendered = density > 1e-9
    active = rendered[columns]
    row, col = rows[active], columns[active]
    kernel = weights[active] / density[col]
    missing = (~rendered).nonzero().flatten()
    row = torch.cat([row, gt_nearest[missing]])
    col = torch.cat([col, missing])
    kernel = torch.cat([kernel, weights.new_ones(len(missing))])
    base_mass = scatter_sum(kernel, row, ng)
    plan = kernel
    residual = weights.new_tensor(0.)
    if transport:
        # Soft capacity: flatten the original responsibility mass using sqrt.
        # Rows without candidate support keep zero capacity, not forced mass.
        capacity = base_mass.sqrt()
        capacity *= nv / capacity.sum().clamp_min(1e-30)
        u, v = torch.ones_like(capacity), weights.new_ones(nv)
        rho_row, rho_voxel = tau_row / (tau_row + 1), tau_voxel / (tau_voxel + 1)
        for _ in range(iterations):
            u = (capacity / scatter_sum(kernel * v[col], row, ng).clamp_min(1e-30)).pow(rho_row)
            v = scatter_sum(kernel * u[row], col, nv).clamp_min(1e-30).pow(-rho_voxel)
        plan = kernel * u[row] * v[col]
        proposed = (capacity / scatter_sum(kernel * v[col], row, ng).clamp_min(1e-30)).pow(rho_row)
        good = capacity > 0
        residual = (proposed[good].clamp_min(1e-30).log() - u[good].clamp_min(1e-30).log()).abs().max()
    mass = scatter_sum(plan, row, ng)
    hist = scatter_sum(plan, row * 17 + labels[col], ng * 17).reshape(ng, 17)
    targets = hist / mass[:, None].clamp_min(1e-30)
    supported = mass > 1e-12
    targets[~supported] = F.one_hot(nearest_labels[~supported], 17).to(targets)
    nearest_share = targets.gather(1, nearest_labels[:, None]).squeeze(-1)
    mismatch = (targets.argmax(-1) != nearest_labels) & supported
    voxel_mass = scatter_sum(plan, col, nv)
    diagnostics = dict(edges=len(row), renderer_edges=len(rows), gt_voxels=nv,
        guard_voxels=len(missing), gaussians=ng, supported_gaussians=int(supported.sum()),
        mismatched_gaussians=int(mismatch.sum()),
        supported_nearest_share_sum=float(nearest_share[supported].sum()),
        supported_entropy_sum=float((-(targets * targets.clamp_min(1e-30).log()).sum(-1))[supported].sum()),
        column_mass_mean=float(voxel_mass.mean()),
        column_mass_abs_error=float((voxel_mass - 1).abs().mean()),
        transport_fixed_point_log_residual=float(residual),
        max_mass_fraction=float(mass.max() / mass.sum().clamp_min(1e-30)))
    return targets, diagnostics, dict(rows=row, columns=col, plan=plan, mass=mass,
        base_mass=base_mass, rendered=rendered, targets=targets, mismatch=mismatch,
        nearest_share=nearest_share)


def match_gradient_norm(reference, proposed, logits):
    """Equal final-logit L2 gradient budget, with a detached scale factor."""
    if torch.is_grad_enabled() and logits.requires_grad:
        old = torch.autograd.grad(reference, logits, retain_graph=True)[0].detach().norm()
        new = torch.autograd.grad(proposed, logits, retain_graph=True)[0].detach().norm()
        scale = old / new.clamp_min(1e-20)
        return proposed * scale, scale.detach(), old, new * scale
    return proposed, proposed.new_tensor(1.), proposed.new_tensor(0.), proposed.new_tensor(0.)


@OPENOCC_LOSS.register_module()
class GaussianResponsibilitySetLoss(CenterSupportDiagnosticSetLoss):
    def __init__(self, assignment='nearest', **kwargs):
        super().__init__(**kwargs)
        if assignment not in ('nearest', 'render', 'transport'):
            raise ValueError(assignment)
        if self.final_gt_domain != 'all':
            raise ValueError('Keep the original full-foreground Set target domain')
        self.assignment = assignment
        self.last_diagnostics = {}

    def forward(self, inputs):
        parts, details = self.components(inputs)
        z = inputs['opus_pred_logits'][-1]
        if len(z) != 1:
            raise ValueError('Registered batch size is 1')
        valid = inputs['opus_point_valid_masks'][-1][0].bool()
        xyz = inputs['opus_pred_points'][-1][0, valid]
        gt_y = inputs['metas']['occ_label'][0].flatten().long()
        selected = (gt_y >= 0) & (gt_y < 17)
        gt_xyz = inputs['metas']['occ_xyz'][0].reshape(-1, 3)[selected]
        gt_y = gt_y[selected]
        if not len(gt_y):
            return sum(parts.values()), details
        nearest = _nearest_indices(xyz, gt_xyz, self.chunk_size)
        labels = gt_y[nearest]
        if self.assignment == 'nearest':
            targets = F.one_hot(labels, 17).to(z)
            diagnostic = {}
        else:
            gaussian = inputs['gaussian']
            gaussian = gaussian._replace(**{key: getattr(gaussian, key)[0, valid]
                for key in gaussian._fields if getattr(gaussian, key) is not None})
            edges = sparse_support(gaussian, gt_xyz, gt_y)
            gt_nearest = _nearest_indices(gt_xyz, xyz, self.chunk_size)
            targets, diagnostic, _ = allocation_targets(*edges, gt_y, labels, gt_nearest,
                transport=self.assignment == 'transport')
        class_weight = self.class_weights.to(z)[labels] if self.class_weights.numel() else torch.ones_like(labels).to(z)
        sample_weight = class_weight * self._distance_weight(gt_xyz[nearest])
        loss = (-(targets * z[0, valid].log_softmax(-1)).sum(-1) * sample_weight).mean()
        index = len(inputs['opus_pred_logits']) - 1
        key = f'stage_{index}/classification'
        loss *= self.stage_weights[index] * self.lambda_cls * self.final_classification_weight
        loss, scale, old, new = match_gradient_norm(parts[key], loss, z)
        parts[key] = loss
        details.update({k: v.detach() for k, v in parts.items()})
        details['final_cls_scale'] = scale
        details['final_cls_reference_grad_norm'] = old
        details['final_cls_matched_grad_norm'] = new
        self.last_diagnostics = dict(diagnostic, cls_scale=float(scale),
            reference_grad_norm=float(old), matched_grad_norm=float(new))
        return sum(parts.values()), details
