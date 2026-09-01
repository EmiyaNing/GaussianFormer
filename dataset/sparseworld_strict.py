"""SparseWorld five-frame/six-future adapter for local Occ3D annotations."""
from __future__ import annotations

from copy import deepcopy
import os
import pickle

import numpy as np
import torch
from pyquaternion import Quaternion

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .dataset import NuScenesDataset


def _ego_pose(info):
    sensor = info.get('data', {}).get('LIDAR_TOP')
    if sensor is None:
        sensor = next((value for value in info.get('data', {}).values()
                       if isinstance(value, dict) and value.get('pose')), None)
    if sensor is None:
        raise KeyError('frame has no sensor ego pose')
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Quaternion(sensor['pose']['rotation']).rotation_matrix
    pose[:3, 3] = np.asarray(sensor['pose']['translation'], dtype=np.float32)
    return pose


@OPENOCC_DATASET.register_module()
class NuScenesSparseWorldStrictDataset(NuScenesDataset):
    """Token-aligned SparseWorld data flow without importing its old runtime."""

    def __init__(self, admlp_state_path, trajectory_path, future_steps=6,
                 history_frames=4, **kwargs):
        super().__init__(**kwargs)
        self.future_steps = int(future_steps)
        self.history_frames = int(history_frames)
        all_keyframes = list(self.keyframes)
        self.keyframe_indices_by_scene = {}
        for scene, index in all_keyframes:
            self.keyframe_indices_by_scene.setdefault(scene, []).append(index)
        with open(admlp_state_path, 'rb') as handle:
            self.admlp_state = pickle.load(handle)
        with open(trajectory_path, 'rb') as handle:
            trajectory = pickle.load(handle)
        self.trajectory_info = trajectory.get('infos', trajectory)

        self.trajectory_by_token = {}
        for scene_name, records in self.trajectory_info.items():
            for index, record in enumerate(records):
                token = record.get('token')
                if token in self.trajectory_by_token:
                    raise ValueError(f'duplicate trajectory token: {token}')
                self.trajectory_by_token[token] = (scene_name, index, record)
        self.annotation_by_token = {}
        for records in self.scene_infos.values():
            for record in records:
                token = record.get('token')
                if token:
                    self.annotation_by_token[token] = record

        valid = []
        for scene, index in self.keyframes:
            records = self.scene_infos[scene]
            if index < self.history_frames:
                continue
            if not all(all(camera in records[history].get('data', {})
                           for camera in self.sensor_types)
                       for history in range(index - self.history_frames, index + 1)):
                continue
            token = records[index].get('token')
            if token not in self.admlp_state or token not in self.trajectory_by_token:
                continue
            trajectory_scene, trajectory_index, _ = self.trajectory_by_token[token]
            trajectory_records = self.trajectory_info[trajectory_scene]
            # The released dataset builds temp2nusc_map from base indices
            # satisfying base+12 < scene_len and evaluates base+5. Hence the
            # actual current keyframes are [5, scene_len-8].
            if trajectory_index < 5 or trajectory_index + 7 >= len(trajectory_records):
                continue
            future = [self.annotation_by_token.get(
                trajectory_records[trajectory_index + step]['token'])
                for step in range(1, self.future_steps + 1)]
            if any(record is None or not record.get('occ_path') for record in future):
                continue
            valid.append((scene, index))
        self.keyframes = valid

    @staticmethod
    def _state_vector(record):
        # Official code sorts keys and excludes ``gt``: a, cmd, v, x = 21.
        values = []
        for key in sorted(record):
            if key == 'gt':
                continue
            values.extend(np.asarray(record[key], dtype=np.float32).reshape(-1).tolist())
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (21,):
            raise ValueError(f'AD-MLP state must flatten to 21 values, got {vector.shape}')
        return vector

    def _future_records(self, token):
        trajectory_scene, trajectory_index, current = self.trajectory_by_token[token]
        records = self.trajectory_info[trajectory_scene]
        future = []
        for step in range(1, self.future_steps + 1):
            future_token = records[trajectory_index + step]['token']
            future.append(self.annotation_by_token[future_token])
        return current, future

    def __getitem__(self, item):
        scene, index = self.keyframes[item]
        info = deepcopy(self.scene_infos[scene][index])
        result = self.get_data_info(info, scene_token=scene, frame_index=index)
        scene_keyframes = self.keyframe_indices_by_scene[scene]
        keyframe_position = scene_keyframes.index(index)
        result['history_context']['history_indices'] = [
            scene_keyframes[keyframe_position - step]
            for step in range(1, self.history_frames + 1)]
        token = info['token']
        trajectory, future = self._future_records(token)
        result['temporal_trajs'] = torch.from_numpy(np.asarray(
            trajectory['gt_ego_fut_trajs'][:self.future_steps, :2],
            dtype=np.float32))
        result['temporal_ego_states'] = torch.from_numpy(
            self._state_vector(self.admlp_state[token]))
        current_pose = _ego_pose(info)
        result['future_occ_paths'] = [record['occ_path'] for record in future]
        result['future_ego_to_current'] = np.stack([
            np.linalg.inv(current_pose) @ _ego_pose(record) for record in future
        ]).astype(np.float32)
        if self.data_aug_conf is not None:
            result['aug_configs'] = self._sample_augmentation()
        for transform in self.pipeline:
            result = transform(result)
        return {key: result[key] for key in self.return_keys}


@OPENOCC_TRANSFORMS.register_module()
class LoadSparseWorldFutureOccupancy:
    def __init__(self, occ3d_path, empty_label=17):
        self.occ3d_path = occ3d_path
        self.empty_label = int(empty_label)

    def _resolve(self, path):
        candidates = (path, os.path.join(self.occ3d_path, path),
                      os.path.join(os.path.dirname(self.occ3d_path), path))
        for candidate in candidates:
            if os.path.isdir(candidate):
                candidate = os.path.join(candidate, 'labels.npz')
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f'future occupancy not found: {path}; checked {candidates}')

    def __call__(self, results):
        semantics, camera, lidar = [], [], []
        for path in results['future_occ_paths']:
            data = np.load(self._resolve(path))
            semantics.append(data['semantics'].astype(np.int64))
            camera.append(data['mask_camera'].astype(np.bool_))
            lidar.append(data['mask_lidar'].astype(np.bool_))
        results['future_occ_labels'] = np.stack(semantics)
        results['future_occ_cam_masks'] = np.stack(camera)
        results['future_occ_lidar_masks'] = np.stack(lidar)
        return results
