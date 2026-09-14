"""Local-region geometry and paired damage statistics for E01 onward."""
import numpy as np
import torch

from .sparseworld_diagnostics import confusion_matrix


def voxel_centers(grid_shape, pc_range, grid_size, device='cpu'):
    """Return native ego-frame voxel centers as [X,Y,Z,3]."""
    axes = [torch.arange(size, device=device, dtype=torch.float32) * grid_size +
            float(pc_range[axis]) + grid_size / 2.
            for axis, size in enumerate(grid_shape)]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)


def transform_points(points, transform):
    """Apply a column-vector homogeneous transform to [...,3] points."""
    transform = torch.as_tensor(transform, device=points.device, dtype=points.dtype)
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.matmul(homogeneous, transform.transpose(-1, -2))[..., :3]


def evaluation_to_injection_transform(future_ego_to_current,
                                      injection_step, evaluation_step):
    """Map an evaluation future-ego point into the injection future-ego frame.

    Steps are one-based, matching the six SparseWorld forecast steps.
    """
    transforms = torch.as_tensor(future_ego_to_current)
    injection = transforms[injection_step - 1]
    evaluation = transforms[evaluation_step - 1]
    return torch.linalg.inv(injection) @ evaluation


def rectangle_distance_xy(points, bounds):
    """Horizontal distance to an axis-aligned rectangle; zero inside."""
    xmin, xmax, ymin, ymax = map(float, bounds)
    dx = torch.maximum(torch.maximum(points[..., 0].new_tensor(xmin) - points[..., 0],
                                     points[..., 0] - xmax),
                       torch.zeros_like(points[..., 0]))
    dy = torch.maximum(torch.maximum(points[..., 1].new_tensor(ymin) - points[..., 1],
                                     points[..., 1] - ymax),
                       torch.zeros_like(points[..., 1]))
    return torch.sqrt(dx.square() + dy.square())


def region_masks(grid_points_evaluation, bounds_injection,
                 future_ego_to_current, injection_step, evaluation_step):
    """Build mutually exclusive R0--R4 masks for one physical source block."""
    transform = evaluation_to_injection_transform(
        future_ego_to_current, injection_step, evaluation_step)
    injection_points = transform_points(grid_points_evaluation, transform)
    distance = rectangle_distance_xy(injection_points, bounds_injection)
    xmin, xmax, ymin, ymax = map(float, bounds_injection)
    inside = ((injection_points[..., 0] >= xmin) &
              (injection_points[..., 0] < xmax) &
              (injection_points[..., 1] >= ymin) &
              (injection_points[..., 1] < ymax))
    masks = dict(
        R0=inside,
        R1=(~inside) & (distance < 2.),
        R2=(distance >= 2.) & (distance < 4.),
        R3=(distance >= 4.) & (distance < 8.),
        R4=distance >= 8.)
    return masks, distance, injection_points


def validate_partition(masks):
    stacked = torch.stack(tuple(masks.values()), dim=0).to(torch.int8)
    return bool((stacked.sum(0) == 1).all().item())


def eligible_block(query_centers_m, label, bounds, minimum_queries=5,
                   minimum_occupied_voxels=50, free_label=17):
    xmin, xmax, ymin, ymax = map(float, bounds)
    query_mask = ((query_centers_m[..., 0] >= xmin) &
                  (query_centers_m[..., 0] < xmax) &
                  (query_centers_m[..., 1] >= ymin) &
                  (query_centers_m[..., 1] < ymax))
    # Labels use [X,Y,Z], and pc_range/grid size are fixed by the caller's
    # block mask. This function receives the already selected label block.
    occupied = label != free_label
    return (int(query_mask.sum().item()) >= int(minimum_queries) and
            int(occupied.sum().item()) >= int(minimum_occupied_voxels)), query_mask


def paired_damage(clean_grid, branch_grid, label, masks, valid_mask=None,
                  num_classes=18):
    """Return regional confusion and new/recovered error counts."""
    clean = torch.as_tensor(clean_grid)
    branch = torch.as_tensor(branch_grid)
    label = torch.as_tensor(label)
    valid = torch.ones_like(label, dtype=torch.bool)
    if valid_mask is not None:
        valid &= torch.as_tensor(valid_mask).bool()
    valid &= (label >= 0) & (label < num_classes)
    clean_wrong = clean != label
    branch_wrong = branch != label
    output = {}
    for name, region in masks.items():
        selected = valid & region.to(valid.device)
        new_error = selected & (~clean_wrong) & branch_wrong
        recovered = selected & clean_wrong & (~branch_wrong)
        changed = selected & (clean != branch)
        output[name] = dict(
            valid_voxels=int(selected.sum().item()),
            occupied_voxels=int((selected & (label != 17)).sum().item()),
            new_errors=int(new_error.sum().item()),
            recovered_errors=int(recovered.sum().item()),
            net_extra_errors=int(new_error.sum().item() - recovered.sum().item()),
            changed_voxels=int(changed.sum().item()),
            clean_confusion=confusion_matrix(
                clean.cpu().numpy(), label.cpu().numpy(), num_classes,
                selected.cpu().numpy()).tolist(),
            branch_confusion=confusion_matrix(
                branch.cpu().numpy(), label.cpu().numpy(), num_classes,
                selected.cpu().numpy()).tolist())
    return output


def sum_confusions(regional, key):
    return np.sum([np.asarray(value[key], dtype=np.int64)
                   for value in regional.values()], axis=0)
