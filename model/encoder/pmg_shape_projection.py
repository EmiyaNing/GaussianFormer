"""Parameter-free motion/shape operators for the PMG architecture experiment.

Coordinates are metric xyz with shape [batch, query, child, 3]. The canonical
template stays fixed during a rollout. A radius bounds the TOTAL deformation
relative to that template, rather than each recurrent increment.
"""

import torch


def centered(points):
    return points - points.mean(-2, keepdim=True)


def yaw_matrix(angle):
    cosine, sine = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((cosine, -sine, zero, sine, cosine, zero,
                        zero, zero, one), -1).reshape(*angle.shape, 3, 3)


def apply_rotation(points, rotation):
    return points @ rotation.transpose(-1, -2)


def fit_yaw(template, candidate, epsilon=1e-8):
    """Best planar rigid alignment; identity for an unidentifiable rotation.

    The covariance is computed from centered, unordered-with-fixed-slot inputs.
    Unlike the offline diagnosis, child identities here are explicit persistent
    model slots. No ground truth, instance identity or future observation enters.
    """
    dot = (template[..., :2] * candidate[..., :2]).sum((-1, -2))
    cross = (template[..., 0] * candidate[..., 1]
             - template[..., 1] * candidate[..., 0]).sum(-1)
    valid = dot.square() + cross.square() > epsilon ** 2
    # Avoid atan2(0, 0), including its undefined backward on a masked branch.
    return torch.atan2(torch.where(valid, cross, torch.zeros_like(cross)),
                       torch.where(valid, dot, torch.ones_like(dot)))


def rigid_tangent(template):
    x, y, z = template.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*template.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=template.dtype, device=template.device)
    return torch.cat((identity.expand_as(skew), -skew), -1).flatten(-3, -2)


def shape_prox(proposal, template, penalty=2.0, radius_rms=0.10):
    """Exact subspace/ball proximal map, with a rank-aware pseudoinverse.

    radius_rms is converted into a group L2 budget sqrt(children)*radius_rms.
    The template is a detached reference; gradients flow through the candidate.
    Non-expansiveness applies with the reference and hyperparameters fixed.
    """
    if penalty < 0 or radius_rms < 0:
        raise ValueError("shape penalty and radius must be nonnegative")
    basis = rigid_tangent(template.detach())
    gram_inverse = torch.linalg.pinv(basis.transpose(-1, -2) @ basis,
                                     hermitian=True, rtol=1e-6)
    vector = proposal.flatten(-2)
    coefficients = gram_inverse @ (basis.transpose(-1, -2) @ vector[..., None])
    perpendicular = vector - (basis @ coefficients).squeeze(-1)
    update = perpendicular / (1.0 + penalty)
    radius = radius_rms * template.shape[-2] ** 0.5
    norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
    update = update * (radius / norm.clamp_min(1e-12)).clamp(max=1.0)
    return update.reshape_as(proposal)


def structured_update(previous, template, offsets, mode, penalty=2.0,
                      radius_rms=0.10):
    """Project a native point-offset proposal onto the selected state model.

    Translation uses the mean of the unchanged native motion proposal in all
    structured arms. A1 keeps canonical shape; A3 permits unconstrained shape;
    A4 permits only the bounded non-rigid part. No additional trainable weights.
    """
    location = previous.mean(-2, keepdim=True) + offsets.mean(-2, keepdim=True)
    candidate = centered(previous) + centered(offsets)
    if mode == "translation":
        angle = torch.zeros_like(previous[..., 0, 0])
        local = template
    else:
        angle = fit_yaw(template, candidate)
        rotation = yaw_matrix(angle)
        proposed_residual = apply_rotation(candidate, rotation.transpose(-1, -2)) - template
        if mode == "rigid":
            residual = torch.zeros_like(proposed_residual)
        elif mode == "free":
            residual = proposed_residual
        elif mode == "prox":
            residual = shape_prox(proposed_residual, template, penalty, radius_rms)
        else:
            raise ValueError(f"unknown geometry mode: {mode}")
        local = apply_rotation(template + residual, rotation)
    return location + local, angle
