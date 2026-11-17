"""Grid-free exact reference for the Phase-B semantic Gaussian local aggregator.

This mirrors the *semantics* of ``ForwardKernel`` in
``model/ops/semantic_gaussian_localagg/semantic_gaussian_localagg.cu`` while
enumerating every Gaussian pair directly instead of using the world-space cell
acceleration structure.

Two consequences make it the right audit target:

1. It is a plain differentiable PyTorch expression, so ``torch.autograd``
   produces the exact gradient of the same mathematical function the CUDA
   ``BackwardKernel`` is supposed to implement.  Agreement therefore validates
   the hand-written backward kernel, not just the forward one.
2. It never bins.  The CUDA op bins into ``cell_size`` cells and scans
   ``neighbour_cells = ceil(support_sigma * max_scale / cell_size)`` cells in
   each direction.  Agreement therefore also validates the claim that binning
   is an *exact* acceleration structure rather than an approximation -- but
   only when the caller supplies a ``max_scale`` that really bounds the scales
   (see ``test_binning_completeness_requires_adequate_max_scale``).

Semantics copied from the CUDA kernel, in the order they appear there:

* neighbour scales are clamped with ``max(scale, scale_eps)`` per axis;
* the support cutoff is a **per-axis box** -- ``|r_d| <= support_sigma * s_d``
  for every axis ``d`` -- *not* an ellipsoid ``||r / s|| <= support_sigma``.
  A neighbour at offset ``(3s, 3s, 0)`` is therefore *included* even though
  ``||r/s||^2 = 18 > 9``;
* aggregation is normalized: ``sum_h w_h seed_h / (sum_h w_h + denom_eps)``;
* an invalid query yields an all-zero residual (the CUDA path returns early);
* ``include_self`` controls whether ``h == g`` participates;
* the operator is **not** differentiable w.r.t. point coordinates -- the CUDA
  backward writes only ``grad_scales``, ``grad_opacities`` and ``grad_seeds``.
"""
from __future__ import annotations

import torch


def g2q_bruteforce(points, scales, opacities, semantic_seed, valid_mask, *,
                   support_sigma=3.0, scale_eps=1e-4, denom_eps=1e-6,
                   include_self=True, query_chunk=256, return_denominator=False):
    """Exhaustive-enumeration reference for ``semantic_gaussian_localagg``.

    Args:
        points: ``[B, G, 3]`` Gaussian centres in metres.
        scales: ``[B, G, 3]`` positive per-axis scales.
        opacities: ``[B, G]`` opacity in ``[0, 1]``.
        semantic_seed: ``[B, G, C]`` per-Gaussian semantic seed logits.
        valid_mask: ``[B, G]`` bool validity.
        support_sigma: per-axis support cutoff in scale units.
        scale_eps: lower bound applied to every scale axis.
        denom_eps: additive stabilizer on the aggregation denominator.
        include_self: whether a Gaussian aggregates itself.
        query_chunk: number of queries per pairwise block (memory control).
        return_denominator: also return ``sum_h w_h + denom_eps`` per query.

    Returns:
        ``[B, G, C]`` normalized semantic residual, or a tuple with the
        ``[B, G]`` denominator when ``return_denominator`` is set.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('points must be [B,G,3]')
    if scales.shape != points.shape:
        raise ValueError('scales must match points')
    if semantic_seed.shape[:2] != points.shape[:2]:
        raise ValueError('semantic_seed must be [B,G,C]')
    if valid_mask.shape != points.shape[:2]:
        raise ValueError('valid_mask must be [B,G]')

    batch, count, _ = points.shape
    classes = semantic_seed.shape[-1]
    dense = points.dtype
    result = points.new_zeros(batch, count, classes)
    denominators = points.new_zeros(batch, count)

    for scene in range(batch):
        scene_points = points[scene]
        # Clamp only for the kernel geometry; the gradient path below has to
        # see the clamp so that d/dscale is zero at or below scale_eps, exactly
        # like the `if (scale[d] > scale_eps)` guards in BackwardKernel.
        scene_scales = scales[scene].clamp_min(scale_eps)
        scene_opacity = opacities[scene]
        scene_seed = semantic_seed[scene]
        scene_valid = valid_mask[scene].bool()

        for start in range(0, count, query_chunk):
            stop = min(start + query_chunk, count)
            queries = slice(start, stop)
            # [q, 1, 3] - [1, G, 3] -> query minus neighbour.
            delta = scene_points[queries, None, :] - scene_points[None, :, :]
            # Divide by the *neighbour* scale, matching the kernel.
            relative = delta / scene_scales[None, :, :]

            inside = (relative.abs() <= support_sigma).all(dim=-1)
            kernel = torch.exp(-0.5 * relative.square().sum(dim=-1))
            weight = scene_opacity[None, :] * kernel

            mask = inside & scene_valid[None, :] & scene_valid[queries, None]
            if not include_self:
                rows = torch.arange(stop - start, device=points.device)
                columns = torch.arange(start, stop, device=points.device)
                mask = mask.clone()
                mask[rows, columns] = False

            weight = torch.where(mask, weight, torch.zeros_like(weight))
            numerator = weight @ scene_seed
            denominator = weight.sum(dim=-1) + denom_eps
            residual = numerator / denominator.unsqueeze(-1)
            residual = torch.where(
                scene_valid[queries, None], residual, torch.zeros_like(residual))

            result[scene, queries] = residual
            denominators[scene, queries] = torch.where(
                scene_valid[queries], denominator, torch.ones_like(denominator))

    if return_denominator:
        return result, denominators
    return result


def ellipsoid_reference(points, scales, opacities, semantic_seed, valid_mask, *,
                        support_sigma=3.0, scale_eps=1e-4, denom_eps=1e-6,
                        include_self=True):
    """Same as :func:`g2q_bruteforce` but with an *ellipsoid* support check.

    Only used to demonstrate that the CUDA operator uses a per-axis box: the
    two references disagree exactly on neighbours that are inside the box but
    outside the ellipsoid, e.g. offsets like ``(3s, 3s, 0)``.
    """
    batch, count, _ = points.shape
    classes = semantic_seed.shape[-1]
    result = points.new_zeros(batch, count, classes)
    for scene in range(batch):
        scene_points = points[scene]
        scene_scales = scales[scene].clamp_min(scale_eps)
        scene_valid = valid_mask[scene].bool()
        delta = scene_points[:, None, :] - scene_points[None, :, :]
        relative = delta / scene_scales[None, :, :]
        squared = relative.square().sum(dim=-1)
        inside = squared <= support_sigma ** 2
        kernel = torch.exp(-0.5 * squared)
        weight = opacities[scene][None, :] * kernel
        mask = inside & scene_valid[None, :] & scene_valid[:, None]
        if not include_self:
            mask = mask & ~torch.eye(count, dtype=torch.bool, device=points.device)
        weight = torch.where(mask, weight, torch.zeros_like(weight))
        residual = (weight @ semantic_seed[scene]) / (
            weight.sum(dim=-1, keepdim=True) + denom_eps)
        result[scene] = torch.where(
            scene_valid[:, None], residual, torch.zeros_like(residual))
    return result


def g2q_cuda(points, scales, opacities, semantic_seed, valid_mask, pc_range, *,
             cell_size=1.8, support_sigma=3.0, scale_eps=1e-4, denom_eps=1e-6,
             include_self=True, max_scale=None):
    """Thin wrapper around the production CUDA op with an explicit ``max_scale``.

    ``max_scale`` is the static bound the production head passes to size the
    cell scan.  When omitted it is derived from the data, which forces a
    device synchronization and is fine for tests but not for a hot path.
    """
    from model.ops.semantic_gaussian_localagg import semantic_gaussian_localagg

    if max_scale is None:
        max_scale = float(scales.detach().max())
    if semantic_seed.shape[1] != points.shape[1]:
        raise ValueError('semantic_seed must be [B,G,C]')
    opacity = opacities
    if opacity.ndim == 3 and opacity.shape[-1] == 1:
        opacity = opacity.squeeze(-1)
    return semantic_gaussian_localagg(
        points.contiguous(), scales.contiguous(), opacity.contiguous(),
        semantic_seed.contiguous(), valid_mask.contiguous(), pc_range,
        cell_size, support_sigma, scale_eps, denom_eps, include_self, max_scale)


def neighbour_cells(support_sigma, max_scale, cell_size):
    """Replicate the wrapper's cell-scan radius so tests can probe it."""
    import math
    return int(math.ceil(support_sigma * max_scale / cell_size))


def cell_coordinates(points, pc_range, cell_size):
    """Integer world-space cell index per point, clamped to the cell grid."""
    low = points.new_tensor(pc_range[:3])
    high = points.new_tensor(pc_range[3:])
    shape = ((high - low) / cell_size).ceil().long()
    coord = ((points - low) / cell_size).floor().long()
    return torch.maximum(torch.minimum(coord, shape - 1), torch.zeros_like(coord)), shape


def shell_cap_selection(points, scales, valid, pc_range, cell_size, support_sigma,
                        scale_eps, max_scale, max_neighbours, include_self=True):
    """Replicate the CUDA density-cap neighbour selection for one scene.

    Mirrors the capped branch of ``ForwardKernel`` exactly:

    * cell shells are visited in increasing Chebyshev radius from the query's
      own cell, and within a shell the offsets are visited in lexicographic
      ``(dx, dy, dz)`` order;
    * out-of-grid cells are skipped;
    * within a cell, members are visited in ascending original index order,
      which is what ``cub::DeviceRadixSort`` produces (it is stable, and the
      keys were built from an ascending index);
    * the cap counts neighbours that pass validity **and** the per-axis box
      test, matching ``AccumulateForwardNeighbour``'s return value.

    Returns:
        ``[G, G]`` bool selection matrix, plus the number of candidates scanned.
    """
    count = points.shape[0]
    device = points.device
    coord, shape = cell_coordinates(points, pc_range, cell_size)
    gx, gy, gz = (int(v) for v in shape)
    flat_valid = valid.bool()
    clamped = scales.clamp_min(scale_eps)

    # Cell members in ascending index order.
    members = {}
    for index in range(count):
        if not flat_valid[index]:
            continue
        key = (int(coord[index, 0]), int(coord[index, 1]), int(coord[index, 2]))
        members.setdefault(key, []).append(index)

    radius = neighbour_cells(support_sigma, max_scale, cell_size)
    selection = torch.zeros(count, count, dtype=torch.bool, device=device)
    scans = 0
    for query in range(count):
        if not flat_valid[query]:
            continue
        base = (int(coord[query, 0]), int(coord[query, 1]), int(coord[query, 2]))
        accepted = 0
        for shell in range(radius + 1):
            if max_neighbours > 0 and accepted >= max_neighbours:
                break
            offsets = [(dx, dy, dz)
                       for dx in range(-shell, shell + 1)
                       for dy in range(-shell, shell + 1)
                       for dz in range(-shell, shell + 1)
                       if max(abs(dx), abs(dy), abs(dz)) == shell]
            for dx, dy, dz in offsets:
                if max_neighbours > 0 and accepted >= max_neighbours:
                    break
                x, y, z = base[0] + dx, base[1] + dy, base[2] + dz
                if not (0 <= x < gx and 0 <= y < gy and 0 <= z < gz):
                    continue
                for candidate in members.get((x, y, z), ()):
                    if max_neighbours > 0 and accepted >= max_neighbours:
                        break
                    scans += 1
                    if not include_self and candidate == query:
                        continue
                    relative = (points[query] - points[candidate]).abs() / clamped[candidate]
                    if bool((relative <= support_sigma).all()):
                        selection[query, candidate] = True
                        accepted += 1
    return selection, scans


def g2q_capped_reference(points, scales, opacities, semantic_seed, valid_mask,
                         pc_range, cell_size, support_sigma, max_scale,
                         max_neighbours, scale_eps=1e-4, denom_eps=1e-6,
                         include_self=True):
    """Capped-mode reference: shell selection followed by the same algebra.

    The selection mask is non-differentiable (it depends on points, scales,
    validity and the cap), so autograd through this function yields the exact
    gradient of the capped operator, including the truncation.
    """
    clamped = scales.clamp_min(scale_eps)
    delta = points[:, None, :] - points[None, :, :]
    relative = delta / clamped[None, :, :]
    inside = (relative.abs() <= support_sigma).all(dim=-1)
    kernel = torch.exp(-0.5 * relative.square().sum(dim=-1))
    weight = opacities[None, :] * kernel
    selection, _ = shell_cap_selection(
        points, scales, valid_mask, pc_range, cell_size, support_sigma,
        scale_eps, max_scale, max_neighbours, include_self=include_self)
    mask = inside & valid_mask[None, :] & valid_mask[:, None] & selection
    weight = torch.where(mask, weight, torch.zeros_like(weight))
    residual = (weight @ semantic_seed) / (
        weight.sum(dim=-1, keepdim=True) + denom_eps)
    return torch.where(valid_mask[:, None], residual,
                       torch.zeros_like(residual))
