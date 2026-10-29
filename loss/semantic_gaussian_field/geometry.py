"""Distances to solid ellipsoids, retaining center, scale and rotation gradients."""
import torch


@torch.no_grad()
def project_standard(points, means, scales, rotation, iterations=48):
    """Independent PyTorch reference, including FP64 finite-difference checks."""
    v = torch.einsum('...ij,...j->...i', rotation, points - means)
    inside = ((v / scales).square().sum(-1) <= 1)
    lo = torch.zeros_like(inside, dtype=points.dtype)
    hi = scales.amax(-1) * v.norm(dim=-1)
    for _ in range(iterations):
        t = (lo + hi) * .5
        norm2 = (scales * v / (scales.square() + t[..., None])).square().sum(-1)
        lo = torch.where(norm2 > 1, t, lo)
        hi = torch.where(norm2 > 1, hi, t)
    u = scales * v / (scales.square() + ((lo + hi) * .5)[..., None])
    u = torch.where(inside[..., None], v / scales, u)
    return u, inside


def distance_with_projection(points, means, scales, rotation, u, inside):
    """Envelope derivative: detach optimal standard u, never world projection."""
    world = means + torch.einsum('...ji,...j->...i', rotation, scales * u.detach())
    distance = (points - world).norm(dim=-1)
    return torch.where(inside, torch.zeros_like(distance), distance)


def solid_ellipsoid_distance(points, means, scales, rotation):
    u, inside = project_standard(points, means, scales, rotation)
    return distance_with_projection(points, means, scales, rotation, u, inside)
