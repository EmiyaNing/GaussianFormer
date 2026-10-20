"""Unit tests for the P2 active-set construction.

Every case checks counts and provenance against a hand-built occupancy grid, so
a silently wrong sampler cannot pass.
"""
from __future__ import annotations

import pytest
import torch

from model.ops.active_sampling import (
    SOURCE_BOUNDARY, SOURCE_EMPTY, SOURCE_PREDICTED, SOURCE_UNCERTAIN,
    build_active_set, occupied_boundary_voxels, source_histogram,
    uniform_empty_voxels, voxel_centres_from_indices)

PC_RANGE = (-4.0, -4.0, -4.0, 4.0, 4.0, 4.0)
VOXEL = 1.0


def block_grid(shape=(8, 8, 8)):
    """All-empty grid with one 2x2x2 occupied block starting at (3, 3, 3)."""
    grid = torch.full(shape, 17, dtype=torch.long)
    grid[3:5, 3:5, 3:5] = 1
    return grid


def test_voxel_centre_conversion():
    indices = torch.tensor([[0, 0, 0], [1, 2, 3]])
    centres = voxel_centres_from_indices(indices, PC_RANGE, VOXEL)
    # lower corner is -4, so voxel 0 centre is -3.5 and voxel (1,2,3) is
    # (-2.5, -1.5, -0.5).
    assert torch.allclose(centres[0], torch.tensor([-3.5, -3.5, -3.5]))
    assert torch.allclose(centres[1], torch.tensor([-2.5, -1.5, -0.5]))


def test_boundary_of_a_solid_block_is_the_surrounding_shell():
    grid = block_grid()
    indices = occupied_boundary_voxels(grid, empty_label=17)
    # The 4x4x4 dilation of a 2x2x2 block minus the block itself is 64 - 8 = 56.
    assert indices.shape[0] == 56
    # No returned index may itself be occupied.
    assert bool((grid[indices[:, 0], indices[:, 1], indices[:, 2]] == 17).all())


def test_boundary_of_an_empty_grid_is_empty():
    grid = torch.full((6, 6, 6), 17, dtype=torch.long)
    indices = occupied_boundary_voxels(grid, empty_label=17)
    assert indices.shape[0] == 0


def test_uniform_empty_sampling_respects_the_count_and_avoids_occupied():
    grid = block_grid()
    indices = uniform_empty_voxels(grid, 20, empty_label=17)
    assert indices.shape[0] == 20
    assert bool((grid[indices[:, 0], indices[:, 1], indices[:, 2]] == 17).all())
    # Distinct samples.
    unique = {tuple(row.tolist()) for row in indices}
    assert len(unique) == 20


def test_uniform_empty_sampling_clamps_to_available_voxels():
    grid = torch.full((2, 2, 2), 17, dtype=torch.long)
    grid[0, 0, 0] = 1
    indices = uniform_empty_voxels(grid, 999, empty_label=17)
    assert indices.shape[0] == 7


def test_build_active_set_tags_each_source_and_covers_all_four():
    grid = block_grid()
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]])
    valid = torch.tensor([True, False, True])          # middle point dropped
    probability = torch.full((8, 8, 8), 0.1)
    probability[0, 0, 0] = 0.5                          # exactly one uncertain

    active, active_valid, source = build_active_set(
        points, valid, grid, PC_RANGE, VOXEL, occupancy_probability=probability,
        include_boundary=True, include_uncertain=True, empty_samples=5, seed=0)

    histogram = source_histogram(source)
    assert histogram[ 'predicted_points'] == 2          # invalid point dropped
    assert histogram['boundary_samples'] == 56
    assert histogram['uncertainty_voxels'] == 1
    assert histogram['empty_samples'] == 5
    assert active.shape == (2 + 56 + 1 + 5, 3)
    assert bool(active_valid.all())
    # Provenance blocks appear in the plan's ladder order:
    # 2 predicted | 56 boundary | 1 uncertain | 5 empty
    assert int(source[0]) == SOURCE_PREDICTED
    assert int(source[1]) == SOURCE_PREDICTED
    assert int(source[2]) == SOURCE_BOUNDARY
    assert int(source[57]) == SOURCE_BOUNDARY
    assert int(source[58]) == SOURCE_UNCERTAIN
    assert int(source[59]) == SOURCE_EMPTY
    assert int(source[63]) == SOURCE_EMPTY


def test_build_active_set_is_deterministic_for_a_fixed_seed():
    grid = block_grid()
    points = torch.randn(10, 3)
    valid = torch.ones(10, dtype=torch.bool)
    first, _, _ = build_active_set(points, valid, grid, PC_RANGE, VOXEL,
                                   empty_samples=7, seed=3)
    second, _, _ = build_active_set(points, valid, grid, PC_RANGE, VOXEL,
                                    empty_samples=7, seed=3)
    assert torch.equal(first, second)


def test_build_active_set_boundary_stride_reduces_samples():
    grid = block_grid()
    points = torch.zeros(0, 3)
    valid = torch.zeros(0, dtype=torch.bool)
    full, _, source_full = build_active_set(points, valid, grid, PC_RANGE,
                                            VOXEL, boundary_stride=1)
    every_fourth, _, source_fourth = build_active_set(points, valid, grid,
                                                      PC_RANGE, VOXEL,
                                                      boundary_stride=4)
    assert source_histogram(source_full)['boundary_samples'] == 56
    assert source_histogram(source_fourth)['boundary_samples'] == 14
    assert every_fourth.shape[0] == 14


def test_disabling_everything_yields_only_predicted_points():
    grid = block_grid()
    points = torch.randn(6, 3)
    valid = torch.ones(6, dtype=torch.bool)
    active, _, source = build_active_set(
        points, valid, grid, PC_RANGE, VOXEL, include_boundary=False,
        include_uncertain=False, empty_samples=0)
    assert active.shape[0] == 6
    assert set(source.tolist()) == {SOURCE_PREDICTED}
