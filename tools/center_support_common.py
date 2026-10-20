"""Small, bounded references for center/support diagnostics (no CUDA imports)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    # Scene checkpoints and queue status must survive interruption mid-write.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as target:
        target.write(encoded)
        temporary = target.name
    os.replace(temporary, path)


def kernel_weights(xyz, means, opacity, scales, inverse_covariance,
                   pc_min, grid_size=0.4, scale_multiplier=3.0):
    """Exact localagg candidate-box convention, then alpha exp(-d'Qd/2).

    Inputs are unbatched. Caller bounds the voxel and Gaussian chunk sizes.
    The box/radius is detached, as in local_aggregate/__init__.py. There is
    no extra spherical or ellipsoid cutoff in the production forward kernel.
    """
    lower = xyz.new_tensor(pc_min)
    vi = ((xyz - lower) / grid_size).to(torch.int64)
    mi = ((means.detach() - lower) / grid_size).to(torch.int64)
    radius = torch.ceil(scales.detach().amax(-1) * scale_multiplier / grid_size).long()
    candidate = ((vi[:, None] - mi[None]).abs() <= radius[None, :, None]).all(-1)
    delta = xyz[:, None] - means[None]
    power = torch.einsum('vgi,gij,vgj->vg', delta, inverse_covariance, delta)
    return opacity.reshape(1, -1) * torch.exp(-0.5 * power) * candidate


def gradient_summary(a, b, eps=1e-12):
    """Missing and zero gradients are reported explicitly, never cosine=0."""
    na = float(a.double().norm()) if a is not None else 0.0
    nb = float(b.double().norm()) if b is not None else 0.0
    cosine = None
    if na > eps and nb > eps:
        cosine = float((a.double().flatten() @ b.double().flatten()) / (na * nb))
    return dict(norm_a=na, norm_b=nb, cosine=cosine,
                missing_a=a is None, missing_b=b is None,
                near_zero_a=na <= eps, near_zero_b=nb <= eps)


def field_metrics(logits, labels, empty_label):
    pred = logits.argmax(-1)
    occupied, predicted = labels != empty_label, pred != empty_label
    tp = (occupied & predicted).sum().item()
    union = (occupied | predicted).sum().item()
    return dict(ce=float(F.cross_entropy(logits, labels)),
                error=float((pred != labels).float().mean()),
                binary_iou=tp / union if union else None,
                occupied_precision=tp / int(predicted.sum()) if predicted.any() else None,
                occupied_recall=tp / int(occupied.sum()) if occupied.any() else None)


@torch.no_grad()
def contribution_diagnostic(xyz, labels, means, opacity, scales, inv_cov,
                            semantics, nearest_labels, foreground_classes,
                            pc_min, grid_size=0.4, scale_multiplier=3.,
                            voxel_chunk=32, gaussian_chunk=2048,
                            minimum_mass=1e-3, minimum_effective_voxels=1.0):
    """Two-pass responsibility accumulation; never materialize V x G.

    All sampled occupied voxels must have equal inclusion probability. Target
    reliability here concerns this sampled set, not full-scene support.
    """
    g, c = len(means), foreground_classes
    mass = means.new_zeros(g)
    square_mass = means.new_zeros(g)
    histogram = means.new_zeros(g, c)
    correct_fraction, kernel_mass, label_out = [], [], []
    for v in range(0, len(xyz), voxel_chunk):
        x, y = xyz[v:v + voxel_chunk], labels[v:v + voxel_chunk]
        total = means.new_zeros(len(x))
        fg = means.new_zeros(len(x), c)
        for k in range(0, g, gaussian_chunk):
            w = kernel_weights(x, means[k:k + gaussian_chunk], opacity[k:k + gaussian_chunk],
                               scales[k:k + gaussian_chunk], inv_cov[k:k + gaussian_chunk],
                               pc_min, grid_size, scale_multiplier)
            total += w.sum(-1)
            fg += w @ semantics[k:k + gaussian_chunk, :c]
        occ = (y >= 0) & (y < c)
        onehot = F.one_hot(y.clamp(0, c - 1), c).to(means) * occ[:, None]
        for k in range(0, g, gaussian_chunk):
            w = kernel_weights(x, means[k:k + gaussian_chunk], opacity[k:k + gaussian_chunk],
                               scales[k:k + gaussian_chunk], inv_cov[k:k + gaussian_chunk],
                               pc_min, grid_size, scale_multiplier)
            r = w / total[:, None].clamp_min(1e-12)
            histogram[k:k + gaussian_chunk] += r.T @ onehot
            mass[k:k + gaussian_chunk] += (r * occ[:, None]).sum(0)
            square_mass[k:k + gaussian_chunk] += (r.square() * occ[:, None]).sum(0)
        correct = fg.gather(1, y.clamp(0, c - 1)[:, None]).squeeze(-1)
        contributed = occ & (fg.sum(-1) > 1e-12)
        correct_fraction.append(correct[contributed] / fg[contributed].sum(-1))
        kernel_mass.append(total)
        label_out.append(y)
    effective_voxels = mass.square() / square_mass.clamp_min(1e-12)
    supported = (mass >= minimum_mass) & (effective_voxels >= minimum_effective_voxels)
    target = histogram / mass[:, None].clamp_min(1e-12)
    nearest_fraction = target.gather(1, nearest_labels[:, None]).squeeze(-1)
    majority = target.argmax(-1)
    entropy = -(target * target.clamp_min(1e-12).log()).sum(-1)
    correct = torch.cat(correct_fraction)
    totals, ys = torch.cat(kernel_mass), torch.cat(label_out)
    occ = (ys >= 0) & (ys < c)
    mismatch = supported & (majority != nearest_labels)
    def mean_or_none(values):
        return float(values.mean()) if values.numel() else None
    report = dict(
        sampled_voxels=len(xyz), gaussians=g,
        occupied_voxels=int(occ.sum()), empty_voxels=int((ys == c).sum()),
        minimum_sampled_responsibility_mass=minimum_mass,
        minimum_effective_sampled_voxels=minimum_effective_voxels,
        supported_gaussians=int(supported.sum()),
        weak_or_no_support_gaussians=int((~supported).sum()),
        majority_mismatch_fraction=mean_or_none(mismatch[supported].float()),
        nearest_label_support_fraction=mean_or_none(nearest_fraction[supported]),
        target_entropy_nats=mean_or_none(entropy[supported]),
        occupied_voxels_with_semantic_contribution=len(correct),
        occupied_correct_semantic_contribution_fraction=mean_or_none(correct),
        occupied_wrong_semantic_contribution_fraction=mean_or_none(1 - correct),
        occupied_no_kernel_fraction=mean_or_none((totals[occ] <= 1e-12).float()),
        occupied_kernel_mass_mean=mean_or_none(totals[occ]),
        empty_kernel_mass_mean=mean_or_none(totals[ys == c]),
    )
    return report, dict(target=target, mass=mass, supported=supported,
                        mismatch=mismatch, nearest_fraction=nearest_fraction,
                        effective_voxels=effective_voxels)
