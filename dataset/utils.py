import numpy as np
from pyquaternion import Quaternion
import torch
from collections import OrderedDict


_OCC_GRID_CACHE = OrderedDict()


def static_occ_grid_to_device(specs, device):
    """Construct the exact CPU float32 reference grid once per spec/device.

    The explicit pipeline marker is emitted only for unperturbed ego grids;
    arbitrary/augmented occ_xyz tensors continue through the normal transfer.
    """
    if not specs or any(spec != specs[0] for spec in specs):
        raise ValueError('A static Occ3D batch must have one grid specification')
    spec = specs[0]
    device = torch.device(device) if not isinstance(device, int) else torch.device('cuda', device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    key = (tuple(spec['origin']), tuple(spec['shape']), spec['resolution'], device)
    grid = _OCC_GRID_CACHE.get(key)
    if grid is None:
        # Keep the original operation order and rounding instead of fusing
        # arange/scale/offset on GPU. Construction is outside the hot path.
        with torch.inference_mode(False), torch.no_grad():
            axes = [torch.arange(size, dtype=torch.float32) * spec['resolution'] +
                    .5 * spec['resolution'] + origin
                    for size, origin in zip(spec['shape'], spec['origin'])]
            grid = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).to(device)
        _OCC_GRID_CACHE[key] = grid
        if len(_OCC_GRID_CACHE) > 8:
            _OCC_GRID_CACHE.popitem(last=False)
    _OCC_GRID_CACHE.move_to_end(key)
    return grid.unsqueeze(0).expand(len(specs), *grid.shape)


def get_rm(angle, axis, deg=False):
    if deg:
        angle = np.deg2rad(angle)
    rm = np.eye(3)
    if axis == 'x':
        rm[1, 1] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[1, 2] = - np.sin(angle)
        rm[2, 1] = np.sin(angle)
    elif axis == 'y':
        rm[0, 0] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[0, 2] = np.sin(angle)
        rm[2, 0] = - np.sin(angle)
    elif axis == 'z':
        rm[0, 0] = np.cos(angle)
        rm[1, 1] = np.cos(angle)
        rm[0, 1] = - np.sin(angle)
        rm[1, 0] = np.sin(angle)
    return rm


def get_xyz(pose_dict):
    return np.array(pose_dict['translation'])

def get_img2global(calib_dict, pose_dict):
    
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    img2cam = np.linalg.inv(cam2img)

    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    img2global = ego2global @ cam2ego @ img2cam
    return img2global

def get_lidar2global(calib_dict, pose_dict):

    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T

    lidar2global = ego2global @ lidar2ego
    return lidar2global


def custom_collate_fn_temporal(instances):
    return_dict = {}
    for k, v in instances[0].items():
        if k == 'react_voxels':
            prepared = [instance[k] for instance in instances]
            spec = prepared[0]['spec']
            if any(item['spec'] != spec for item in prepared):
                raise ValueError('A React batch must use one voxelization specification')
            coordinates = []
            for batch_index, item in enumerate(prepared):
                coords = torch.as_tensor(item['voxel_coords'], dtype=torch.int32)
                prefix = coords.new_full((len(coords), 1), batch_index)
                coordinates.append(torch.cat([prefix, coords], dim=1))
            return_dict[k] = dict(
                voxels=torch.cat([torch.as_tensor(item['voxels']) for item in prepared]),
                voxel_num_points=torch.cat([torch.as_tensor(item['voxel_num_points'])
                                            for item in prepared]),
                voxel_coords=torch.cat(coordinates), spec=spec)
        elif k == 'lidar_points':
            # Sweeps contain a different number of points per sample. Keep the
            # batch ragged; sparse voxelization is performed sample by sample.
            return_dict[k] = [instance[k] for instance in instances]
        elif isinstance(v, np.ndarray):
            return_dict[k] = torch.stack([
                torch.from_numpy(instance[k]) for instance in instances])
        elif isinstance(v, torch.Tensor):
            return_dict[k] = torch.stack([instance[k] for instance in instances])
        elif isinstance(v, (dict, str)):
            return_dict[k] = [instance[k] for instance in instances]
        elif isinstance(v, (list, tuple)):
            return_dict[k] = [instance[k] for instance in instances]
        elif isinstance(v, (int, float)):
            return_dict[k] = [instance[k] for instance in instances]
        elif v is None:
            return_dict[k] = [None] * len(instances)
        else:
            raise NotImplementedError
    return return_dict


def move_batch_to_device(batch, device, non_blocking=False):
    """Transfer tensors and prepared voxels; retain raw LiDAR on the CPU.

    Other metadata containers keep the existing training-loop behavior. Raw
    ragged points must not be uploaded only to be read by the CPU voxelizer.
    """
    result = {}
    for key, value in batch.items():
        if key == 'static_occ_grid':
            if 'occ_xyz' in batch:
                raise ValueError('Specify either occ_xyz or static_occ_grid')
            result['occ_xyz'] = static_occ_grid_to_device(value, device)
            continue
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=non_blocking)
        elif key == 'react_voxels':
            value = {name: tensor.to(device, non_blocking=non_blocking)
                     if isinstance(tensor, torch.Tensor) else tensor
                     for name, tensor in value.items()}
        result[key] = value
    return result
