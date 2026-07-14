import os

import numpy as np
import torch

__all__ = ['load_augmented_point_cloud', 'reduce_LiDAR_beams']


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    """Load nuScenes points together with optional virtual-point annotations."""
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    tokens = path.split('/')
    vp_dir = '_VIRTUAL' if reduce_beams == 32 else f'_VIRTUAL_{reduce_beams}BEAMS'
    seg_path = os.path.join(
        *tokens[:-3], 'virtual_points', tokens[-3], tokens[-2] + vp_dir,
        tokens[-1] + '.pkl.npy')
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f'Virtual point annotation not found: {seg_path}')
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict['real_points']
    virtual_points2 = np.concatenate([
        data_dict['virtual_points'][:, :3],
        np.zeros([data_dict['virtual_points'].shape[0], 1]),
        data_dict['virtual_points'][:, 3:],
    ], axis=-1)
    points = np.concatenate([
        points,
        np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
    ], axis=1)
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1)
    if len(data_dict['real_points_indice']) > 0:
        points[data_dict['real_points_indice']] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -np.ones([virtual_points2.shape[0], 1])], axis=1)
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    """Keep the standard nuScenes laser rings used by the legacy data loader."""
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    theta = torch.asin(pts[:, 2] / radius.clamp_min(1e-6))
    top_ang, down_ang = 0.1862, -0.5353
    beam_range = torch.zeros(32, device=pts.device)
    beam_range[0], beam_range[31] = top_ang, down_ang
    for index in range(1, 31):
        beam_range[index] = beam_range[index - 1] - 0.023275

    if reduce_beams_to == 16:
        beam_ids = range(1, 32, 2)
    elif reduce_beams_to == 4:
        beam_ids = [7, 9, 11, 13]
    elif reduce_beams_to == 1:
        beam_ids = [9]
    else:
        raise NotImplementedError(f'Unsupported beam count: {reduce_beams_to}')

    mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
    for beam_id in beam_ids:
        mask |= ((theta < beam_range[beam_id - 1] - 0.012) &
                 (theta > beam_range[beam_id] - 0.012))
    return pts[mask].cpu().numpy()
