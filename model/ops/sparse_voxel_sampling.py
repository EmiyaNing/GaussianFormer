"""Fixed query selection and local sampling of sparse, regular 3-D grids.

Features stay sparse. Only the int32 coordinate-to-feature-row map is dense;
the same map is reused by every decoder layer in a forward pass.
"""

import itertools

import torch


@torch.no_grad()
def voxel_fps_indices(coordinates_zyx, spatial_shape, voxel_size, count):
    """Select voxel rows by FPS after canonical integer-coordinate sorting.

The starting point is the smallest flattened grid key. CUDA uses MMCV's
fixed-start FPS; CPU has a small reference implementation. Selection is
repeatable and invariant to input row permutations on a fixed backend. The
CPU/CUDA tie reductions need not choose the same set on a regular lattice.
"""
    if count < 1:
        raise ValueError('FPS count must be positive')
    if coordinates_zyx.ndim != 2 or coordinates_zyx.shape[1] != 3:
        raise ValueError('voxel coordinates must have shape [N, 3] in ZYX order')
    _, height, width = spatial_shape
    coords = coordinates_zyx.long()
    keys = (coords[:, 0] * height + coords[:, 1]) * width + coords[:, 2]
    order = keys.argsort()
    take = min(count, order.numel())
    if take == order.numel():
        return order
    # Distances are in metres, not scene-normalized coordinates. Translation
    # is irrelevant to FPS, so integer indices times the pitch suffice.
    xyz = coords[order].flip(-1).float()
    xyz = xyz * xyz.new_tensor(voxel_size)
    if xyz.is_cuda:
        from mmcv.ops import furthest_point_sample
        selected = furthest_point_sample(xyz[None].contiguous(), take)[0].long()
    else:
        selected = torch.empty(take, dtype=torch.long, device=xyz.device)
        distance = xyz.new_full((xyz.shape[0],), float('inf'))
        current = torch.zeros((), dtype=torch.long, device=xyz.device)
        for step in range(take):
            selected[step] = current
            distance = torch.minimum(distance, (xyz - xyz[current]).square().sum(-1))
            current = distance.argmax()
    return order[selected]


def build_voxel_memory(features, indices, spatial_shape, batch_size,
                       voxel_size, origin):
    """Map [batch,z,y,x] coordinates to rows of a differentiable [N,C] array."""
    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('spatial_shape must contain three positive ZYX dimensions')
    if features.ndim != 2 or indices.shape != (features.shape[0], 4):
        raise ValueError('features/indices must have shapes [N,C] and [N,4]')
    if features.device != indices.device:
        raise ValueError('voxel features and indices must be on the same device')
    index_map = torch.full((batch_size, *shape), -1, dtype=torch.int32,
                           device=features.device)
    coords = indices.long()
    index_map[tuple(coords.unbind(-1))] = torch.arange(
        features.shape[0], dtype=torch.int32, device=features.device)
    return dict(features=features, indices=indices, index_map=index_map,
                spatial_shape=shape, voxel_size=tuple(voxel_size), origin=tuple(origin))


def sample_voxel_memory(sample_points, memory, neighborhood=1, sigma=1.0,
                        chunk_size=1024):
    """Gaussian-weighted local voxel sampling, without global neighbor search.

Args:
    sample_points: Physical XYZ positions, [B,Q,G,K,3]. Channels of memory
        features are split into G groups, just as in OPUS image sampling.
    neighborhood: Grid radius; one reads a 3x3x3 stencil per sampling point.
    sigma: Distance-kernel width in units of the feature voxel pitch.

Returns [B,Q,G,K,C/G] features and [B,Q,G,K] valid-neighborhood flags. Bounds
and missing cells are masked independently. Distances provide piecewise
coordinate gradients; floor/index lookup itself is intentionally discrete.
"""
    if sample_points.ndim != 5 or sample_points.shape[-1] != 3:
        raise ValueError('sample_points must have shape [B,Q,G,K,3]')
    if neighborhood < 0 or sigma <= 0 or chunk_size < 1:
        raise ValueError('invalid voxel sampling neighborhood, sigma or chunk size')
    batch, queries, groups, points, _ = sample_points.shape
    features = memory['features']
    if features.shape[-1] % groups:
        raise ValueError('voxel feature channels must be divisible by the sampling groups')
    if memory['index_map'].shape[0] != batch:
        raise ValueError('voxel memory batch size does not match query batch size')
    channels = features.shape[-1] // groups
    # Row zero is a constant empty-cell feature, including completely empty
    # LiDAR samples. Never interpret -1 as the final valid feature row.
    padded = torch.cat([features.new_zeros((1, groups, channels)),
                        features.reshape(-1, groups, channels)], dim=0)
    positions = sample_points.reshape(-1, 3).float()
    batch_ids = torch.arange(batch, device=positions.device).view(batch, 1, 1, 1)
    batch_ids = batch_ids.expand(batch, queries, groups, points).reshape(-1)
    group_ids = torch.arange(groups, device=positions.device).view(1, 1, groups, 1)
    group_ids = group_ids.expand(batch, queries, groups, points).reshape(-1)
    offsets = list(itertools.product(range(-neighborhood, neighborhood + 1), repeat=3))
    offsets = torch.tensor(offsets, dtype=torch.long, device=positions.device)
    shape_xyz = positions.new_tensor(memory['spatial_shape'][::-1], dtype=torch.long)
    pitch = positions.new_tensor(memory['voxel_size'])
    origin = positions.new_tensor(memory['origin'])
    sampled, validity = [], []
    for start in range(0, positions.shape[0], chunk_size):
        end = start + chunk_size
        position = positions[start:end]
        cell = torch.floor((position - origin) / pitch).long()
        neighbors = cell[:, None, :] + offsets[None]
        inside = ((neighbors >= 0) & (neighbors < shape_xyz)).all(-1)
        safe = torch.minimum(neighbors.clamp_min(0), shape_xyz - 1)
        row = memory['index_map'][batch_ids[start:end, None],
                                   safe[..., 2], safe[..., 1], safe[..., 0]].long()
        valid = inside & (row >= 0)
        values = padded[(row + 1).clamp_min(0), group_ids[start:end, None]].float()
        delta = neighbors.to(position.dtype) + 0.5 - (position - origin)[:, None] / pitch
        weights = torch.exp(-0.5 * delta.square().sum(-1) / (sigma * sigma))
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        sampled.append((values * weights[..., None]).sum(-2))
        validity.append(valid.any(-1))
    output = torch.cat(sampled).reshape(batch, queries, groups, points, channels)
    valid = torch.cat(validity).reshape(batch, queries, groups, points)
    return output, valid
