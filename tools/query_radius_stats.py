"""Read-only radius measurements around the actual final decoder sampler center."""
from __future__ import annotations
import torch
from model.encoder.opus_encoder import _opus_decode_points
from model.utils.utils import get_rotation_matrix

FIELDS = (
    'center_radius_mean', 'center_radius_rms', 'center_radius_p90', 'center_radius_max',
    'internal_radius_mean', 'internal_radius_rms', 'internal_radius_max', 'centroid_shift',
    'gs_major_sigma_mean', 'gs_major_sigma_max',
    'core_outer_radius_lower', 'core_outer_radius_upper',
    'three_sigma_outer_radius_lower', 'three_sigma_outer_radius_upper',
    'feature_sampling_radius_mean', 'feature_sampling_radius_max', 'input_std_norm',
    'core_bbox_span_x', 'core_bbox_span_y', 'core_bbox_span_z',
    'three_sigma_bbox_span_x', 'three_sigma_bbox_span_y', 'three_sigma_bbox_span_z',
    'native_candidate_radius_max', 'native_bbox_span_x', 'native_bbox_span_y', 'native_bbox_span_z',
    'reference_to_fg_center', 'reference_to_fg_volume',
)


def capture_final_sampler(network):
    """Observe actual sampler inputs after positional encoding; never mutate."""
    module = network.encoder.layers[-1].sampling
    assert type(module).__name__ == '_StrictOPUSSampling'

    def hook(sampler, inputs):
        points, features = inputs[:2]
        decoded = _opus_decode_points(points, sampler.pc_range)
        center = decoded.mean(2)
        spread = decoded.std(2) if decoded.shape[2] > 1 else torch.zeros_like(center)
        offsets = sampler.sampling_offset(features).view(*features.shape[:2], -1, 3)
        samples = center[:, :, None] + offsets * spread[:, :, None]
        network._query_radius_capture = dict(center=center, spread=spread, samples=samples)

    return module.register_forward_pre_hook(hook)


def ellipsoid_radius_bounds(delta, scales, rotation, k):
    """Bounds on max distance from query center to each untruncated ellipsoid.

    Lower: two points that actually lie on its surface (radial support point,
    and the most distant endpoint of a major axis). Upper: triangle inequality.
    """
    norm = delta.norm(dim=-1)
    largest = scales.amax(-1)
    transform = torch.diag_embed(scales) @ rotation
    covariance = transform.transpose(-1, -2) @ transform
    direction = delta / norm.clamp_min(1e-20)[..., None]
    ce = (covariance @ direction[..., None]).squeeze(-1)
    support = (direction * ce).sum(-1).clamp_min(0).sqrt()
    radial = (delta + k * ce / support.clamp_min(1e-20)[..., None]).norm(dim=-1)
    axis_id = scales.argmax(-1)
    axis = rotation.gather(-2, axis_id[..., None, None].expand(*axis_id.shape, 1, 3)).squeeze(-2)
    endpoint = (norm.square() + (k * largest).square() +
                2 * k * largest * (delta * axis).sum(-1).abs()).clamp_min(0).sqrt()
    lower = torch.maximum(radial, endpoint)
    upper = norm + k * largest
    assert bool((lower <= upper + 1e-4).all())
    return torch.minimum(lower, upper), upper, covariance


@torch.inference_mode()
def measure(network, out, batch, q, children, expected_core):
    import numpy as np
    from tools.probe_gaussian_scale_supervision import layout
    from tools.query_final_support import extension as support_extension
    from tools.query_foreground_distance import extension as distance_extension

    capture = network._query_radius_capture
    center = capture['center'][0]
    spread, samples = capture['spread'][0], capture['samples'][0]
    assert center.shape == (q, 3)
    raw_previous = _opus_decode_points(out['representation'][-2]['query_points'],
                                     network.encoder.layers[-1].sampling.pc_range)[0]
    torch.testing.assert_close(center, raw_previous.mean(1), rtol=0, atol=0)
    assert bool(out['representation'][-1]['query_valid_mask'].all())
    _, values = layout(network.head, out['gaussian'])
    means, scales, inverse, alpha, semantic, cells, radii = values
    assert len(means) == q * children
    means = means.reshape(q, children, 3)
    scale = scales.reshape(q, children, 3)
    h = float(network.head.aggregator.grid_size)
    assert abs(h - .4) < 1e-7
    delta = means - center[:, None]
    center_radius = delta.norm(dim=-1) / h
    centroid = means.mean(1)
    internal = (means - centroid[:, None]).norm(dim=-1) / h
    rotations = get_rotation_matrix(out['gaussian'].rotations)[0].reshape(q, children, 3, 3)
    lo1, hi1, covariance = ellipsoid_radius_bounds(delta, scale, rotations, 1.)
    lo3, hi3, _ = ellipsoid_radius_bounds(delta, scale, rotations, 3.)
    axis_sigma = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    bbox1 = ((means + axis_sigma).amax(1) - (means - axis_sigma).amin(1)) / h
    bbox3 = ((means + 3 * axis_sigma).amax(1) - (means - 3 * axis_sigma).amin(1)) / h
    agg = network.head.aggregator
    shape = torch.tensor((agg.H, agg.W, agg.D), device=center.device)
    low = (cells.long() - radii[:, None]).clamp_min(0)
    high = torch.minimum(cells.long() + radii[:, None], shape - 1)
    low_world = agg.pc_min + (low.float() + .5) * h
    high_world = agg.pc_min + (high.float() + .5) * h
    ref = center[:, None].expand(-1, children, -1).reshape(-1, 3)
    candidate_far = torch.maximum((low_world - ref).abs(), (high_world - ref).abs()).norm(dim=-1)
    native_span = (high.reshape(q, children, 3).amax(1) - low.reshape(q, children, 3).amin(1) + 1).float()
    sampling_radius = (samples - center[:, None]).norm(dim=-1) / h
    xyz = out['sampled_xyz'][0].contiguous()
    y = batch['occ_label'].flatten().long()
    camera = batch['occ_cam_mask'].flatten().bool()
    targets = torch.where(camera & (y >= 0) & (y < 17))[0]
    gt = xyz[targets].contiguous()
    distance = distance_extension().distances(center.contiguous(), gt, h)
    lookup = torch.full((len(y),), -1, dtype=torch.int32, device=y.device)
    lookup[targets] = torch.arange(len(targets), device=y.device, dtype=torch.int32)
    bits, coverage, distinct = support_extension().support(means.reshape(-1, 3).contiguous(),
        inverse, alpha, cells, radii, xyz, lookup, len(gt), q, children, agg.H, agg.W, agg.D)
    np.testing.assert_array_equal(coverage[0].cpu().numpy(), expected_core)
    ref_cells = torch.floor((center - agg.pc_min) / h).long()
    inside = ((ref_cells >= 0) & (ref_cells < shape)).all(1)
    safe = torch.minimum(torch.maximum(ref_cells, torch.zeros_like(ref_cells)), shape - 1)
    flat = (safe[:, 0] * agg.W + safe[:, 1]) * agg.D + safe[:, 2]
    visible = inside & camera[flat] & (y[flat] >= 0) & (y[flat] < 18)
    ref_kind = torch.full((q,), 3, device=y.device, dtype=torch.int8)  # outside valid camera / unknown
    ref_kind[~inside] = 4  # outside pc box
    ref_kind[visible & (y[flat] == 17)] = 2
    ref_kind[visible & (y[flat] < 17)] = 1
    data = {
        'center_radius_mean': center_radius.mean(1),
        'center_radius_rms': center_radius.square().mean(1).sqrt(),
        'center_radius_p90': torch.quantile(center_radius, .9, dim=1),
        'center_radius_max': center_radius.amax(1),
        'internal_radius_mean': internal.mean(1),
        'internal_radius_rms': internal.square().mean(1).sqrt(),
        'internal_radius_max': internal.amax(1),
        'centroid_shift': (centroid - center).norm(dim=-1) / h,
        'gs_major_sigma_mean': scale.amax(-1).mean(1) / h,
        'gs_major_sigma_max': scale.amax(-1).amax(1) / h,
        'core_outer_radius_lower': lo1.amax(1) / h,
        'core_outer_radius_upper': hi1.amax(1) / h,
        'three_sigma_outer_radius_lower': lo3.amax(1) / h,
        'three_sigma_outer_radius_upper': hi3.amax(1) / h,
        'feature_sampling_radius_mean': sampling_radius.mean(1),
        'feature_sampling_radius_max': sampling_radius.amax(1),
        'input_std_norm': spread.norm(dim=-1) / h,
        'native_candidate_radius_max': candidate_far.reshape(q, children).amax(1) / h,
        'reference_to_fg_center': distance[:, 0], 'reference_to_fg_volume': distance[:, 1],
    }
    for i, axis in enumerate('xyz'):
        data['core_bbox_span_' + axis] = bbox1[:, i]
        data['three_sigma_bbox_span_' + axis] = bbox3[:, i]
        data['native_bbox_span_' + axis] = native_span[:, i]
    metrics = torch.stack([data[f] for f in FIELDS], -1)
    assert bool(torch.isfinite(metrics).all()) and bool((metrics >= 0).all())
    return dict(metrics=metrics.cpu().numpy(), reference_center=center.cpu().numpy(),
        gs_centroid=centroid.cpu().numpy(), reference_kind=ref_kind.cpu().numpy()), dict(
        foreground=len(gt), covered=int((distinct[0] > 0).sum()),
        zero_opacity_gs=int((alpha == 0).sum()),
        core_radius_bound_max_width=float(((hi1 - lo1).amax(1) / h).max()),
        three_sigma_radius_bound_max_width=float(((hi3 - lo3).amax(1) / h).max()))
