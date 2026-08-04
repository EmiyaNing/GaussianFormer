"""基于组合熵增益的自适应历史帧选择引擎。"""

import os
from collections import OrderedDict

import numpy as np

from dataset.entropy_core import compute_composite_entropy
from dataset.utils import get_lidar2global, get_img2global
from model.lifter.spconv_voxelize import VoxelGeneratorWrapper


class EntropyBasedHistoryLoader:
    """按组合熵增益选择与当前帧对齐的历史 LiDAR/图像帧。

    组合熵为 ``0.4 * scene_entropy + 0.6 * topk_deci_mean``。
    当完整窗口剩余收益比例低于阈值时，保留当前候选并停止继续搜索。
    """

    SENSOR_TYPES = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    ]

    def __init__(
        self,
        max_window,
        min_window,
        entropy_gain_ratio_threshold,
        data_root,
        pc_range,
        voxel_size=(0.5, 0.5, 0.5),
        max_points_per_voxel=20,
        max_voxels=1600000,
        deci_batch_size=65536,
        deci_topk=64,
        point_cache_size=64,
        include_history_images=True,
    ):
        if not max_window >= min_window >= 0:
            raise ValueError(
                f'max_window({max_window}) >= min_window({min_window}) >= 0 '
                'is required.'
            )
        if not 0.0 <= entropy_gain_ratio_threshold <= 1.0:
            raise ValueError(
                'entropy_gain_ratio_threshold must be within [0, 1].'
            )

        self.max_window = int(max_window)
        self.min_window = int(min_window)
        self.entropy_gain_ratio_threshold = float(
            entropy_gain_ratio_threshold
        )
        self.data_root = data_root
        self.pc_range = tuple(pc_range)
        self.voxel_size = tuple(voxel_size)
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)
        self.deci_batch_size = int(deci_batch_size)
        if deci_topk < 1:
            raise ValueError('deci_topk must be at least 1.')
        self.deci_topk = int(deci_topk)
        if point_cache_size < 0:
            raise ValueError('point_cache_size must be non-negative.')
        self.point_cache_size = int(point_cache_size)
        self.include_history_images = bool(include_history_images)
        self._voxel_generator = None
        self._point_cache = OrderedDict()

    def _get_voxel_generator(self):
        # DataLoader worker 首次使用时创建，避免跨进程传递 C++ 对象。
        if self._voxel_generator is None:
            self._voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.pc_range,
                num_point_features=4,
                max_num_points_per_voxel=self.max_points_per_voxel,
                max_num_voxels=self.max_voxels,
            )
        return self._voxel_generator

    def forward(self, scene_infos, scene_token, frame_index):
        """返回选择结果以及可供 Pipeline 记录的诊断信息。"""
        current_frame = scene_infos[scene_token][frame_index]
        lidar_info = current_frame.get('data', {}).get('LIDAR_TOP')
        if lidar_info is None:
            raise KeyError(
                f'Missing LIDAR_TOP: scene={scene_token}, frame={frame_index}'
            )

        current_pose = get_lidar2global(lidar_info['calib'], lidar_info['pose'])
        ego2global = (
            self._get_ego2global(lidar_info['pose'])
            if self.include_history_images else None
        )
        current_points = self._filter_points(
            self._load_lidar(lidar_info['filename'])
        )
        history = self._load_all_history(
            scene_infos, scene_token, frame_index, current_pose, ego2global
        )

        base_entropy = self._entropy(current_points)['combined']
        entropy_values = [base_entropy]

        total_capacity = len(current_points) + sum(
            len(frame['points']) for frame in history
        )
        fused_buffer = np.empty(
            (total_capacity, current_points.shape[1]), dtype=current_points.dtype
        )
        current_size = len(current_points)
        fused_buffer[:current_size] = current_points

        # 先构造完整窗口，仅计算 C0 和 C_M 得到 p2；中间 level 按需计算。
        level_sizes = []
        for candidate in history:
            candidate_size = len(candidate['points'])
            next_size = current_size + candidate_size
            fused_buffer[current_size:next_size] = candidate['points']
            level_sizes.append(next_size)
            current_size = next_size

        full_window_entropy = (
            self._entropy(fused_buffer[:current_size])['combined']
            if history else base_entropy
        )
        total_gain = float(full_window_entropy - base_entropy)
        candidate_gains = []
        selected_count = 0
        remaining_gain_ratios = []
        stop_reason = 'no_more_history'

        if history and total_gain <= 0.0:
            stop_reason = 'nonpositive_total_gain'
        else:
            previous_entropy = base_entropy
            for level, level_size in enumerate(level_sizes, start=1):
                current_entropy = (
                    full_window_entropy
                    if level == len(level_sizes)
                    else self._entropy(fused_buffer[:level_size])['combined']
                )
                entropy_values.append(current_entropy)
                candidate_gains.append(
                    float(current_entropy - previous_entropy)
                )
                previous_entropy = current_entropy
                cumulative_gain = float(current_entropy - base_entropy)
                remaining_ratio = 1.0 - cumulative_gain / total_gain
                remaining_gain_ratios.append(float(remaining_ratio))
                selected_count = level
                if (level >= self.min_window and
                        remaining_ratio < self.entropy_gain_ratio_threshold):
                    stop_reason = 'gain_ratio_threshold'
                    break
            if (selected_count == self.max_window and
                    stop_reason == 'no_more_history'):
                stop_reason = 'max_history'

        selected = history[:selected_count]
        accepted_gains = candidate_gains[:selected_count]
        accepted_size = (
            level_sizes[selected_count - 1]
            if selected_count > 0 else len(current_points)
        )

        fused_points = fused_buffer[:accepted_size].copy()
        return dict(
            selected_frames=selected,
            current_points=current_points,
            fused_points=fused_points,
            candidate_gains=candidate_gains,
            accepted_gains=accepted_gains,
            entropy_values=entropy_values,
            full_window_entropy=full_window_entropy,
            total_gain=total_gain,
            remaining_gain_ratios=remaining_gain_ratios,
            stop_reason=stop_reason,
        )

    def _entropy(self, points):
        return compute_composite_entropy(
            points,
            self._get_voxel_generator(),
            voxel_size=self.voxel_size,
            scene_weight=0.4,
            local_weight=0.6,
            deci_batch_size=self.deci_batch_size,
            deci_topk=self.deci_topk,
        )

    def _load_lidar(self, lidar_filename):
        lidar_path = (
            lidar_filename if os.path.isabs(lidar_filename)
            else os.path.join(self.data_root, lidar_filename)
        )
        points = self._point_cache.pop(lidar_path, None)
        if points is not None:
            self._point_cache[lidar_path] = points
            return points

        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        if self.point_cache_size > 0:
            self._point_cache[lidar_path] = points
            while len(self._point_cache) > self.point_cache_size:
                self._point_cache.popitem(last=False)
        return points

    def _filter_points(self, points):
        mask = (
            (points[:, 0] > self.pc_range[0]) &
            (points[:, 0] < self.pc_range[3]) &
            (points[:, 1] > self.pc_range[1]) &
            (points[:, 1] < self.pc_range[4]) &
            (points[:, 2] > self.pc_range[2]) &
            (points[:, 2] < self.pc_range[5])
        )
        return points[mask]

    @staticmethod
    def _get_ego2global(pose_dict):
        from pyquaternion import Quaternion

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            pose_dict['rotation']
        ).rotation_matrix
        ego2global[:3, 3] = np.asarray(pose_dict['translation']).T
        return ego2global

    @staticmethod
    def _transform_points(points, source_pose, target_pose):
        if points.shape[0] == 0:
            return points
        points_homogeneous = np.concatenate(
            [points[:, :3], np.ones((len(points), 1), dtype=points.dtype)],
            axis=-1,
        )
        target_from_source = np.linalg.inv(target_pose) @ source_pose
        transformed_xyz = (
            target_from_source @ points_homogeneous.T
        ).T[:, :3]
        return np.concatenate(
            [transformed_xyz, points[:, 3:]], axis=-1
        ).astype(points.dtype, copy=False)

    def _has_all_cameras(self, frame):
        data = frame.get('data', {})
        return all(camera in data for camera in self.SENSOR_TYPES)

    def _load_all_history(
        self, scene_infos, scene_token, frame_index, target_pose, ego2global
    ):
        history = []
        for previous_index in range(frame_index - 1, -1, -1):
            previous_frame = scene_infos[scene_token][previous_index]
            lidar_info = previous_frame.get('data', {}).get('LIDAR_TOP')
            if lidar_info is None:
                continue
            if self.include_history_images and not self._has_all_cameras(previous_frame):
                continue

            source_pose = get_lidar2global(
                lidar_info['calib'], lidar_info['pose']
            )
            # 必须先变换到当前 LiDAR 坐标系，再按目标范围过滤。
            transformed = self._transform_points(
                self._load_lidar(lidar_info['filename']),
                source_pose,
                target_pose,
            )
            transformed = self._filter_points(transformed)

            candidate = dict(
                frame_index=previous_index,
                points=transformed,
                pts_filename=os.path.abspath(os.path.join(
                    self.data_root, lidar_info['filename']
                )),
                lidar_pose=source_pose,
            )
            if self.include_history_images:
                image_files = {}
                lidar2img = {}
                ego2img = {}
                for camera in self.SENSOR_TYPES:
                    camera_info = previous_frame['data'][camera]
                    image_files[camera] = os.path.join(
                        self.data_root, camera_info['filename']
                    )
                    img2global = get_img2global(
                        camera_info['calib'], camera_info['pose']
                    )
                    lidar2img[camera] = np.linalg.inv(img2global) @ target_pose
                    ego2img[camera] = np.linalg.inv(img2global) @ ego2global
                candidate.update(
                    img_files=image_files,
                    lidar2img=lidar2img,
                    ego2img=ego2img,
                )
            history.append(candidate)
            if len(history) >= self.max_window:
                break

        return history
