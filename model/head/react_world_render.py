"""Render predicted Gaussians with the unchanged React aggregation operator."""
from __future__ import annotations


def render_world_gaussians(head, gaussians, points, context, current_to_output=None):
    """Read out one state in the current frame or a benchmark's future frame.

    Frame transforms are readout metadata: they never enter the world decoder.
    The inverse covariance rotates with the centres. Support boxes use a
    conservative rotated AABB, so anisotropic supports are not cut off.
    """
    means, opacity, semantics, scales, inverse = head.prepare_gaussian_args(
        gaussians, append_empty=False)
    if current_to_output is not None:
        rotation = current_to_output[:, :3, :3]
        translation = current_to_output[:, :3, 3]
        means = means @ rotation.transpose(-1, -2) + translation[:, None]
        inverse = (rotation[:, None] @ inverse @
                   rotation[:, None].transpose(-1, -2))
        scales = scales @ rotation.abs().transpose(-1, -2)
    batch, count = means.shape[:2]
    logits = head.aggregator(
        points.float(), means, opacity.reshape(batch, count), semantics,
        scales, inverse, points_int=context['points_int'])
    if head.separate_empty:
        logits = logits + context['empty_weights'][:, None] * context['background_sem'][None]
    return logits[None].transpose(1, 2)
