"""Pure geometry/semantic interventions; GT is used only in diagnostic outputs."""

import numpy as np
from scipy.spatial import cKDTree
import torch


VARIANTS = ("D00", "D10_r08", "D10_r20", "D01", "D11_r08", "D11_r20",
            "amplitude_half", "amplitude_double", "position_random_direction", "semantic_wrong_control")


def bounded_xy_displacement(means, targets, radius):
    displacement = np.asarray(targets)[..., :2] - np.asarray(means)[..., :2]
    lengths = np.linalg.norm(displacement, axis=-1, keepdims=True)
    return displacement * np.minimum(1., radius / np.maximum(lengths, 1e-12))


def associate(state, xyz, target, fast, current_queries, radius=2.):
    """Fixed nearest-fast-voxel association, without assuming physical identity."""
    means = state.means[0, :current_queries].reshape(-1, 3).detach().cpu().numpy()
    opacity = state.opacities[0, :current_queries].reshape(-1).detach().cpu().numpy()
    fast_ids = np.flatnonzero(fast)
    valid = np.isfinite(means).all(-1) & (opacity > 0)
    eligible = np.flatnonzero(valid)
    if not len(fast_ids) or not len(eligible):
        return dict(indices=np.empty(0, np.int64), targets=np.empty((0, 3), np.float32),
                    labels=np.empty(0, np.int64), gt_indices=np.empty(0, np.int64),
                    distances=np.empty(0, np.float64), fast_voxels=len(fast_ids), active_gaussians=int(valid.sum()))
    distance, nearest = cKDTree(xyz[fast_ids]).query(means[eligible], workers=1)
    keep = distance <= radius
    gaussian_ids, gt_ids = eligible[keep], fast_ids[nearest[keep]]
    return dict(indices=gaussian_ids, targets=xyz[gt_ids], labels=target[gt_ids],
                gt_indices=gt_ids, distances=distance[keep],
                fast_voxels=len(fast_ids), active_gaussians=int(valid.sum()))


def replace_semantic_direction(semantics, indices, labels):
    flat = semantics.reshape(-1, 17).clone()
    ids = torch.as_tensor(indices, device=flat.device, dtype=torch.long)
    labels = torch.as_tensor(labels, device=flat.device, dtype=torch.long)
    mass = flat[ids].sum(-1)
    flat[ids] = 0
    flat[ids, labels] = mass
    return flat.reshape_as(semantics)


def make_variants(state, mapping, current_queries, seed):
    """Every state derives directly from the same native state and fixed mapping."""
    ids = mapping["indices"]
    output = {"D00": state}
    if not len(ids):
        return {name: state for name in VARIANTS}, dict(mass_max_absolute_error=0., max_displacement_r08=0., max_displacement_r20=0.)
    original = state.means[0, :current_queries].reshape(-1, 3).detach().cpu().numpy()[ids]
    labels = mapping["labels"]
    semantics = replace_semantic_direction(state.semantics, ids, labels)
    wrong = replace_semantic_direction(state.semantics, ids, (labels + 1) % 17)
    output["D01"] = state._replace(semantics=semantics)
    output["semantic_wrong_control"] = state._replace(semantics=wrong)
    displacement = {}
    for radius, suffix in ((.8, "r08"), (2., "r20")):
        delta = bounded_xy_displacement(original, mapping["targets"], radius)
        displacement[suffix] = delta
        means = state.means.clone()
        flat = means.reshape(-1, 3)
        flat[torch.as_tensor(ids, device=means.device), :2] += torch.as_tensor(delta, device=means.device, dtype=means.dtype)
        output[f"D10_{suffix}"] = state._replace(means=means)
        output[f"D11_{suffix}"] = state._replace(means=means, semantics=semantics)
    angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(ids))
    lengths = np.linalg.norm(displacement["r20"], axis=-1)
    delta = np.stack((np.cos(angles), np.sin(angles)), -1) * lengths[:, None]
    means = state.means.clone()
    flat = means.reshape(-1, 3)
    tensor_ids = torch.as_tensor(ids, device=means.device)
    flat[tensor_ids, :2] += torch.as_tensor(delta, device=means.device, dtype=means.dtype)
    # Match the native renderer's ROI validity rule for the random-direction control.
    lower = means.new_tensor([-40., -40., -1.])
    upper = means.new_tensor([40., 40., 5.4])
    valid = ((means >= lower) & (means < upper)).all(-1)
    random_opacity = state.opacities * valid[..., None]
    means[..., :2] = means[..., :2].clamp(min=lower[:2] + 1e-4, max=upper[:2] - 1e-4)
    output["position_random_direction"] = state._replace(means=means, opacities=random_opacity)
    for factor, name in ((.5, "amplitude_half"), (2., "amplitude_double")):
        changed = state.semantics.clone()
        changed.reshape(-1, 17)[tensor_ids] *= factor
        output[name] = state._replace(semantics=changed)
    mass_error = float((semantics.sum(-1) - state.semantics.sum(-1)).abs().max())
    assert torch.allclose(semantics.sum(-1), state.semantics.sum(-1), rtol=1e-6, atol=1e-5)
    for name, variant in output.items():
        assert torch.equal(variant.scales, state.scales)
        assert torch.equal(variant.rotations, state.rotations)
        assert torch.equal(variant.means[..., 2], state.means[..., 2])
        assert torch.equal(variant.means[:, current_queries:], state.means[:, current_queries:])
        assert torch.equal(variant.semantics[:, current_queries:], state.semantics[:, current_queries:])
        if name != "position_random_direction":
            assert torch.equal(variant.opacities, state.opacities)
    checks = dict(mass_max_absolute_error=mass_error,
                  max_displacement_r08=float(np.linalg.norm(displacement["r08"], axis=-1).max()),
                  max_displacement_r20=float(np.linalg.norm(displacement["r20"], axis=-1).max()),
                  clipped_random_gaussians=int((~valid).sum()))
    assert checks["max_displacement_r08"] <= .80001
    assert checks["max_displacement_r20"] <= 2.00001
    return output, checks
