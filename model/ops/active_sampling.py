"""P2 T2.3 prerequisite: construct the query-conditioned active set.

Plan reference: ``plans/sparse_render_plans/P2_sparse_renderer_completion.md``
task T2.3.  The plan defines the active set as

    A = predicted points
      u point-neighbour boundary samples
      u uncertainty-selected voxels
      u uniformly sampled empty voxels

and requires the ablation ladder to add these sources one at a time.  This
module builds that set and, crucially, records **which source each query came
from**, so the ladder can be run as a single training job per rung rather than
re-deriving the union five times.

This module is deliberately independent of the CUDA operator: the operator
still evaluates the Gaussian field only at Gaussian centres (its ``points``
tensor doubles as both the query set and the Gaussian set).  Decoupling those
two is the separate P2 task T2.1; until that lands, this module produces the
*locations* the extended operator will need.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Provenance codes, ordered as the ablation ladder in the plan.
SOURCE_PREDICTED = 0
SOURCE_BOUNDARY = 1
SOURCE_UNCERTAIN = 2
SOURCE_EMPTY = 3
SOURCE_NAMES = {
    SOURCE_PREDICTED: 'predicted_points',
    SOURCE_BOUNDARY: 'boundary_samples',
    SOURCE_UNCERTAIN: 'uncertainty_voxels',
    SOURCE_EMPTY: 'empty_samples',
}


def voxel_centres_from_indices(indices, pc_range, voxel_size):
    """Convert integer voxel indices ``[N, 3]`` to world-space centres."""
    lower = indices.new_tensor(pc_range[:3], dtype=torch.float32)
    return lower + (indices.float() + 0.5) * voxel_size


def occupied_boundary_voxels(occupancy, empty_label=17):
    """Indices of empty voxels that touch at least one occupied voxel.

    ``occupancy`` is ``[D, H, W]`` (or a batch of them).  The returned indices
    are in the same ``(d, h, w)`` order as the input.
    """
    if occupancy.dim() == 3:
        occupancy = occupancy.unsqueeze(0)
    occupied = (occupancy != empty_label) & (occupancy >= 0)
    occupied = occupied.unsqueeze(1).float()
    dilated = F.max_pool3d(occupied, 3, stride=1, padding=1) > 0
    boundary = dilated & ~occupied.bool()
    # Drop both the batch and channel axes: nonzero() on a [1, D, H, W]
    # slice would return 4 columns, not the (d, h, w) triples we want.
    indices = boundary[0, 0].nonzero(as_tuple=False)
    return indices


def uniform_empty_voxels(occupancy, count, generator=None, empty_label=17):
    """Sample ``count`` empty voxel indices uniformly from one scene."""
    if occupancy.dim() == 3:
        occupancy = occupancy.unsqueeze(0)
    empty = ((occupancy == empty_label) | (occupancy < 0))[0]
    candidates = empty.nonzero(as_tuple=False)
    if candidates.shape[0] == 0 or count <= 0:
        return candidates[:0]
    take = min(count, candidates.shape[0])
    order = torch.randperm(candidates.shape[0], generator=generator,
                           device=candidates.device)[:take]
    return candidates[order]


def uncertain_voxels(probability, low, high):
    """Indices whose occupancy probability falls inside ``[low, high]``."""
    if probability.dim() == 3:
        probability = probability.unsqueeze(0)
    band = (probability[0] >= low) & (probability[0] <= high)
    return band.nonzero(as_tuple=False)


def build_active_set(predicted_points, predicted_valid, occupancy,
                     pc_range, voxel_size, empty_label=17,
                     include_boundary=True, boundary_stride=1,
                     include_uncertain=False, occupancy_probability=None,
                     uncertain_band=(0.4, 0.6),
                     empty_samples=0, uniform_empty=True,
                     seed=0, max_empty_stride=1):
    """Assemble the active set for one scene and tag each query's source.

    Args:
        predicted_points: ``[N, 3]`` world-space points from the backbone.
        predicted_valid: ``[N]`` bool validity for those points.
        occupancy: ``[D, H, W]`` integer GT labels, used for boundary and empty
            sampling.  Training-time supervision may use GT for *sampling*
            locations; this never leaks labels into the prediction.
        pc_range: six world-coordinate floats.
        voxel_size: metric edge length of a voxel.
        boundary_stride: keep every N-th boundary voxel (cost control).
        include_uncertain: add voxels whose predicted occupancy is near the
            decision boundary.
        occupancy_probability: ``[D, H, W]`` float probability, required when
            ``include_uncertain`` is set.
        uncertain_band: inclusive probability band treated as uncertain.
        empty_samples: number of uniform empty voxels to add (0 disables).
        uniform_empty: when False, empty voxels come from the boundary shell
            only, which is the "hard negatives" rung of the ladder.
        seed: sampling seed.

    Returns:
        ``(points, valid, source)`` where ``points`` is ``[Q, 3]``,
        ``valid`` is ``[Q]`` bool and ``source`` is ``[Q]`` int64 provenance.
    """
    device = predicted_points.device
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks_points, chunks_valid, chunks_source = [], [], []

    valid_points = predicted_points[predicted_valid.bool()]
    chunks_points.append(valid_points)
    chunks_valid.append(torch.ones(valid_points.shape[0], dtype=torch.bool,
                                   device=device))
    chunks_source.append(torch.full((valid_points.shape[0],), SOURCE_PREDICTED,
                                    dtype=torch.long, device=device))

    if include_boundary:
        indices = occupied_boundary_voxels(occupancy, empty_label)
        if boundary_stride > 1 and indices.shape[0] > 0:
            indices = indices[::boundary_stride]
        boundary = voxel_centres_from_indices(indices, pc_range, voxel_size)
        chunks_points.append(boundary)
        chunks_valid.append(torch.ones(boundary.shape[0], dtype=torch.bool,
                                       device=device))
        chunks_source.append(torch.full((boundary.shape[0],), SOURCE_BOUNDARY,
                                        dtype=torch.long, device=device))

    if include_uncertain and occupancy_probability is not None:
        indices = uncertain_voxels(occupancy_probability, *uncertain_band)
        uncertain = voxel_centres_from_indices(indices, pc_range, voxel_size)
        chunks_points.append(uncertain)
        chunks_valid.append(torch.ones(uncertain.shape[0], dtype=torch.bool,
                                       device=device))
        chunks_source.append(torch.full((uncertain.shape[0],), SOURCE_UNCERTAIN,
                                        dtype=torch.long, device=device))

    if empty_samples > 0 and uniform_empty:
        indices = uniform_empty_voxels(occupancy, empty_samples,
                                       generator=generator,
                                       empty_label=empty_label)
        empty = voxel_centres_from_indices(indices, pc_range, voxel_size)
        chunks_points.append(empty)
        chunks_valid.append(torch.ones(empty.shape[0], dtype=torch.bool,
                                       device=device))
        chunks_source.append(torch.full((empty.shape[0],), SOURCE_EMPTY,
                                        dtype=torch.long, device=device))

    points = torch.cat(chunks_points, dim=0)
    valid = torch.cat(chunks_valid, dim=0)
    source = torch.cat(chunks_source, dim=0)
    return points, valid, source


def source_histogram(source):
    """Human-readable counts per provenance, for reports."""
    return {SOURCE_NAMES[int(code)]: int((source == code).sum().item())
            for code in SOURCE_NAMES if bool((source == code).any())}
