"""SparseWorld trajectory data adapter.

This module deliberately consumes AD-MLP/OccWorld *pickle artifacts* only.  It
never imports AD-MLP, STP3, Paddle, or changes the local OpenMMLab packages.
"""
from copy import deepcopy
import multiprocessing as mp
import os
import pickle

import numpy as np
import torch

from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .dataset import NuScenesDataset
from .utils import get_img2global


def _pose(info):
    """Return ego-to-global pose from any synchronized sensor record.

    The Occ3D sweep annotation contains camera-only intermediate frames, so
    ``LIDAR_TOP`` is not guaranteed to exist even when the frame is otherwise
    valid.  Sensor poses are ego poses, independent of the chosen sensor;
    prefer lidar for compatibility with the base dataset and fall back to a
    camera/sensor record that carries a pose.
    """
    sensors = info.get('data', {})
    sensor = sensors.get('LIDAR_TOP')
    if sensor is None:
        sensor = next(
            (item for item in sensors.values()
             if isinstance(item, dict) and item.get('pose') is not None),
            None)
    if sensor is None:
        raise KeyError(
            'frame has no sensor record with an ego pose (expected LIDAR_TOP '
            'or a camera pose)')
    from pyquaternion import Quaternion
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = Quaternion(sensor['pose']['rotation']).rotation_matrix
    matrix[:3, 3] = np.asarray(sensor['pose']['translation'], dtype=np.float32)
    return matrix


@OPENOCC_DATASET.register_module()
class NuScenesSparseWorldTrajectoryDataset(NuScenesDataset):
    """Scene-safe five-history/six-future SparseWorld dataset.

    ``admlp_state_path`` and ``trajectory_path`` are data files generated
    offline.  Keeping their parsing here makes the training process independent
    of the AD-MLP implementation and its optional dependencies.
    """
    def __init__(self, admlp_state_path, trajectory_path, future_steps=6,
                 history_frames=4, official_boundary_filter=False,
                 history_keyframes_only=False, max_samples=None,
                 single_future_training=False,
                 forecast_curriculum_start_epoch=0,
                 forecast_curriculum_epochs_per_horizon=1, **kwargs):
        super().__init__(**kwargs)
        self.future_steps = future_steps
        self.history_frames = history_frames
        self.official_boundary_filter = bool(official_boundary_filter)
        self.history_keyframes_only = bool(history_keyframes_only)
        self.single_future_training = bool(single_future_training)
        self.forecast_curriculum_start_epoch = int(
            forecast_curriculum_start_epoch)
        self.forecast_curriculum_epochs_per_horizon = max(
            1, int(forecast_curriculum_epochs_per_horizon))
        # Shared storage lets persistent DataLoader workers observe epoch
        # changes made by the main training process.
        self._shared_epoch = mp.Value('i', 0, lock=False)
        with open(admlp_state_path, 'rb') as handle:
            self.admlp_state = pickle.load(handle)
        with open(trajectory_path, 'rb') as handle:
            trajectory = pickle.load(handle)
        self.trajectory_info = trajectory.get('infos', trajectory)
        # The local Occ3D annotation is indexed by NuScenes ``scene_token`` and
        # by an index in its dense sweep sequence.  SparseWorld's trajectory
        # pickle is instead indexed by ``scene-XXXX`` and by its own sparse
        # keyframe index.  A sample token is the shared, stable identifier.
        # Build that correspondence once here; using ``scene``/``index`` below
        # would either fail at the scene lookup or silently select a wrong
        # trajectory for almost every keyframe.
        self.trajectory_by_token = {}
        for trajectory_scene, records in self.trajectory_info.items():
            if not isinstance(records, (list, tuple)):
                continue
            for trajectory_index, record in enumerate(records):
                token = record.get('token')
                if not token:
                    raise KeyError(
                        'trajectory record is missing its NuScenes sample token')
                if token in self.trajectory_by_token:
                    raise ValueError(
                        'trajectory pickle contains duplicate sample token: '
                        f'{token}')
                self.trajectory_by_token[token] = (
                    trajectory_scene, trajectory_index, record)
        self.annotation_by_token = {}
        for records in self.scene_infos.values():
            for frame in records:
                token = frame.get('token')
                if not token:
                    continue
                if token in self.annotation_by_token:
                    raise ValueError(
                        'annotation pickle contains duplicate sample token: '
                        f'{token}')
                self.annotation_by_token[token] = frame
        candidate_keyframes = self.keyframes
        if self.official_boundary_filter:
            candidate_keyframes = self._official_protocol_keyframes(
                candidate_keyframes)
        valid = []
        for scene, index in candidate_keyframes:
            frames = self.scene_infos[scene]
            history_indices = self._history_indices(
                frames, index, history_frames, self.history_keyframes_only)
            if len(history_indices) != history_frames:
                continue
            if not all(all(name in frames[frame].get('data', {})
                           for name in self.sensor_types)
                       for frame in history_indices + [index]):
                continue
            token = frames[index].get('token')
            if token not in self.trajectory_by_token or token not in self.admlp_state:
                continue
            trajectory_scene, trajectory_index, _ = self.trajectory_by_token[token]
            trajectory_records = self.trajectory_info[trajectory_scene]
            if trajectory_index + future_steps >= len(trajectory_records):
                continue
            future_tokens = [
                trajectory_records[trajectory_index + step]['token']
                for step in range(1, future_steps + 1)]
            future_infos = [self.annotation_by_token.get(future_token)
                            for future_token in future_tokens]
            if any(future is None or not future.get('occ_path')
                   for future in future_infos):
                continue
            if any(not any(isinstance(sensor, dict) and sensor.get('pose') is not None
                           for sensor in future.get('data', {}).values())
                   for future in future_infos):
                continue
            valid.append((scene, index))
        self.keyframes = valid
        if max_samples is not None:
            self.keyframes = self.keyframes[:int(max_samples)]

    @staticmethod
    def _official_protocol_keyframes(keyframes):
        """Match upstream's five-past/seven-future scene boundary rule."""
        by_scene = {}
        for keyframe in keyframes:
            by_scene.setdefault(keyframe[0], []).append(keyframe)
        selected = []
        for records in by_scene.values():
            selected.extend(records[5:-7])
        return selected

    @staticmethod
    def _history_indices(frames, index, count, keyframes_only):
        indices = []
        for previous in range(index - 1, -1, -1):
            if keyframes_only and not frames[previous].get('token'):
                continue
            indices.append(previous)
            if len(indices) == count:
                break
        return indices

    @staticmethod
    def _state_vector(record):
        values = []
        for key in sorted(record):
            if key == 'gt':
                continue
            values.extend(np.asarray(record[key], dtype=np.float32).reshape(-1).tolist())
        return np.asarray(values, dtype=np.float32)

    def _trajectory_record(self, token):
        try:
            _, _, record = self.trajectory_by_token[token]
        except KeyError as exc:
            raise KeyError(
                f'trajectory pickle has no entry for sample token {token}') from exc
        try:
            state = self.admlp_state[token]
        except KeyError as exc:
            raise KeyError(
                f'AD-MLP state pickle has no entry for sample token {token}') from exc
        return np.asarray(record['gt_ego_fut_trajs'], dtype=np.float32), self._state_vector(state)

    def _future_infos(self, token):
        """Return the next SparseWorld keyframes, not dense camera sweeps."""
        try:
            trajectory_scene, trajectory_index, _ = self.trajectory_by_token[token]
        except KeyError as exc:
            raise KeyError(
                f'trajectory pickle has no entry for sample token {token}') from exc
        records = self.trajectory_info[trajectory_scene]
        if trajectory_index + self.future_steps >= len(records):
            raise IndexError(
                f'sample token {token} has fewer than {self.future_steps} '
                'future SparseWorld keyframes')
        future_infos = []
        for step in range(1, self.future_steps + 1):
            future_token = records[trajectory_index + step]['token']
            try:
                future_infos.append(self.annotation_by_token[future_token])
            except KeyError as exc:
                raise KeyError(
                    f'annotation pickle has no entry for future sample token '
                    f'{future_token}') from exc
        return future_infos

    def set_epoch(self, epoch):
        self._shared_epoch.value = int(epoch)

    def _training_horizon(self, item):
        elapsed = (int(self._shared_epoch.value) -
                   self.forecast_curriculum_start_epoch)
        active = max(1, min(
            self.future_steps,
            elapsed // self.forecast_curriculum_epochs_per_horizon + 1))
        return (int(item) + int(self._shared_epoch.value)) % active

    def __getitem__(self, item):
        scene, index = self.keyframes[item]
        frames = self.scene_infos[scene]
        info = deepcopy(frames[index])
        result = self.get_data_info(info, scene_token=scene, frame_index=index)
        if result.get('history_context') is not None:
            result['history_context']['keyframes_only'] = \
                self.history_keyframes_only
        traj, state = self._trajectory_record(info.get('token', ''))
        if traj.shape[0] < self.future_steps:
            raise ValueError('trajectory annotation has fewer than future_steps entries')
        result['temporal_trajs'] = torch.from_numpy(traj[:self.future_steps, :2])
        result['temporal_ego_states'] = torch.from_numpy(state)
        future_paths, transforms = [], []
        current_pose = _pose(info)
        future_infos = self._future_infos(info.get('token', ''))
        selected_horizon = None
        if self.single_future_training and not self.test_mode:
            selected_horizon = self._training_horizon(item)
            future_infos = [future_infos[selected_horizon]]
        for future in future_infos:
            future_paths.append(future['occ_path'])
            transforms.append(np.linalg.inv(current_pose) @ _pose(future))
        result['future_occ_paths'] = future_paths
        result['future_img_filenames'] = []
        result['future_native_ego2img'] = []
        for future in future_infos:
            result['future_img_filenames'].append([
                os.path.join(self.data_path, future['data'][camera]['filename'])
                for camera in self.sensor_types])
            future_pose = _pose(future)
            result['future_native_ego2img'].append([
                np.linalg.inv(get_img2global(
                    future['data'][camera]['calib'],
                    future['data'][camera]['pose'])) @ future_pose
                for camera in self.sensor_types])
        result['future_ego_to_current'] = np.stack(transforms).astype(np.float32)
        if selected_horizon is not None:
            result['future_horizon'] = torch.tensor(
                selected_horizon, dtype=torch.long)
        if self.data_aug_conf is not None:
            result['aug_configs'] = self._sample_augmentation()
        for transform in self.pipeline:
            result = transform(result)
        output = {key: result[key] for key in self.return_keys}
        if selected_horizon is not None:
            output['future_horizon'] = result['future_horizon']
        return output


@OPENOCC_TRANSFORMS.register_module()
class LoadSparseWorldFutureOccupancy:
    """Load six future Occ3D labels without importing upstream data pipelines."""
    def __init__(self, occ3d_path, empty_label=17):
        self.occ3d_path = occ3d_path
        self.empty_label = empty_label

    def _resolve_label_file(self, path):
        candidates = [
            path,
            os.path.join(self.occ3d_path, path),
            os.path.join(os.path.dirname(self.occ3d_path), path),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                candidate = os.path.join(candidate, 'labels.npz')
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            'future occupancy label was not found for path '
            f'{path!r}; checked: {candidates}')

    def __call__(self, results):
        labels, camera_masks, lidar_masks = [], [], []
        for path in results['future_occ_paths']:
            filename = self._resolve_label_file(path)
            data = np.load(filename)
            labels.append(data['semantics'].astype(np.int64))
            camera_masks.append(data['mask_camera'].astype(np.bool_))
            lidar_masks.append(data['mask_lidar'].astype(np.bool_))
        results['future_occ_labels'] = np.stack(labels)
        results['future_occ_cam_masks'] = np.stack(camera_masks)
        results['future_occ_lidar_masks'] = np.stack(lidar_masks)
        return results
