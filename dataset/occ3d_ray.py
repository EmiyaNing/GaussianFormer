"""Occ3D RayIoU ray origins on the local sweep metadata format.

The official RayIoU casts the LiDAR ray pattern from the LiDAR positions of a
scene's keyframes, expressed in the current ego frame.  This transform mirrors
the reference ``EgoPoseDataset`` protocol: every keyframe of the scene is a
candidate origin, origins outside ``|x| < 39`` / ``|y| < 39`` are dropped, and
at most ``max_origins`` are kept with an even ``linspace`` selection.
"""

import numpy as np
from pyquaternion import Quaternion

from . import OPENOCC_TRANSFORMS
from .utils import get_lidar2global


def lidar2ego(calib):
    """LiDAR-to-ego matrix from one ``LIDAR_TOP`` calib record."""
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(calib['rotation']).rotation_matrix
    matrix[:3, 3] = np.asarray(calib['translation'], dtype=np.float64)
    return matrix


@OPENOCC_TRANSFORMS.register_module()
class LoadOcc3DRayOrigins(object):
    """Attach ``ray_origins`` ``[T, 3]`` (ego frame) to one sample."""

    def __init__(self, max_origins=8, range_limit=39.0):
        self.max_origins = int(max_origins)
        self.range_limit = float(range_limit)
        if self.max_origins < 1:
            raise ValueError('max_origins must be positive')
        self._scenes = {}

    def _scene(self, scene_infos, scene_token):
        cached = self._scenes.get(scene_token)
        if cached is not None:
            return cached
        frames = scene_infos[scene_token]
        indices, poses = [], {}
        for index, info in enumerate(frames):
            if not info.get('is_key_frame', False):
                continue
            lidar = info.get('data', {}).get('LIDAR_TOP')
            if lidar is None:
                continue
            indices.append(index)
            poses[index] = (get_lidar2global(lidar['calib'], lidar['pose']),
                            lidar2ego(lidar['calib']))
        self._scenes[scene_token] = (indices, poses)
        return indices, poses

    def __call__(self, results):
        context = results.get('history_context')
        if context is None:
            raise ValueError('Ray origins require history_context metadata')
        scene_token = context['scene_token']
        current = context['frame_index']
        indices, poses = self._scene(context['scene_infos'], scene_token)
        if current not in poses:
            # Evaluation samples are keyframes; keep a clear failure otherwise.
            raise ValueError(
                f'Ray origins need a keyframe sample, got frame {current} '
                f'of scene {scene_token}')
        reference_global, reference_ego_from_lidar = poses[current]
        inverse_reference = np.linalg.inv(reference_global)

        origins = []
        for index in indices:
            if index == current:
                translation = np.zeros(3)
            else:
                translation = (inverse_reference @ poses[index][0])[:3, 3]
            origin = (reference_ego_from_lidar[:3, :3] @ translation
                      + reference_ego_from_lidar[:3, 3])
            if abs(origin[0]) < self.range_limit and abs(origin[1]) < self.range_limit:
                origins.append(origin)
        if not origins:
            origins = [reference_ego_from_lidar[:3, 3].copy()]
        if len(origins) > self.max_origins:
            select = np.round(np.linspace(0, len(origins) - 1, self.max_origins)).astype(np.int64)
            origins = [origins[index] for index in select]
        results['ray_origins'] = np.asarray(origins, dtype=np.float32)
        return results
