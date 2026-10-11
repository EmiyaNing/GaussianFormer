import numpy as np
import torch

from misc.error_propagation import (
    evaluation_to_injection_transform, paired_damage, region_masks,
    sum_confusions, transform_points, validate_partition, voxel_centers)
from misc.sparseworld_diagnostics import confusion_matrix


def poses():
    result = torch.eye(4).repeat(3, 1, 1)
    result[1, 0, 3] = 2.
    result[2, 1, 3] = -3.
    return result


def test_future_frame_transform_roundtrip():
    transforms = poses()
    point_eval = torch.tensor([[1., 2., 0.]])
    evaluation_to_injection = evaluation_to_injection_transform(transforms, 1, 3)
    point_injection = transform_points(point_eval, evaluation_to_injection)
    expected = transform_points(point_eval, torch.linalg.inv(transforms[0]) @ transforms[2])
    assert torch.allclose(point_injection, expected)


def test_region_masks_are_exclusive_and_complete():
    grid = voxel_centers((20, 20, 2), (-10., -10., -1., 10., 10., 1.), 1.)
    masks, distance, _ = region_masks(grid, (-2., 2., -2., 2.), poses(), 1, 2)
    assert validate_partition(masks)
    assert sum(int(mask.sum()) for mask in masks.values()) == grid.shape[0] * grid.shape[1] * grid.shape[2]
    assert (distance >= 0).all()


def test_regional_confusion_sums_to_global_confusion():
    grid_points = voxel_centers((8, 8, 1), (-4., -4., 0., 4., 4., 1.), 1.)
    transforms = torch.eye(4).repeat(2, 1, 1)
    masks, _, _ = region_masks(grid_points, (-1., 1., -1., 1.), transforms, 1, 2)
    label = torch.arange(64).reshape(8, 8, 1) % 3
    clean = label.clone()
    branch = label.roll(1, 0)
    regional = paired_damage(clean, branch, label, masks, num_classes=3)
    global_hist = confusion_matrix(branch.numpy(), label.numpy(), num_classes=3)
    assert np.array_equal(sum_confusions(regional, 'branch_confusion'), global_hist)


def test_zero_branch_has_no_damage():
    grid_points = voxel_centers((8, 8, 1), (-4., -4., 0., 4., 4., 1.), 1.)
    transforms = torch.eye(4).repeat(2, 1, 1)
    masks, _, _ = region_masks(grid_points, (-1., 1., -1., 1.), transforms, 1, 2)
    label = torch.zeros(8, 8, 1, dtype=torch.long)
    clean = torch.ones_like(label)
    regional = paired_damage(clean, clean.clone(), label, masks, num_classes=3)
    assert all(value['new_errors'] == 0 and value['recovered_errors'] == 0 and
               value['changed_voxels'] == 0 for value in regional.values())
