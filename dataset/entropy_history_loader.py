"""
基于熵增益的自适应历史帧窗口选择器。

在数据加载阶段，根据融合历史帧后点云复合场景熵的变化量（熵增益），
动态决定实际使用的历史帧数量，避免无效帧的冗余计算。

复合场景熵 = 0.4 × scene_entropy + 0.6 × max_voxel_deci
熵增益 = 复合场景熵(融合后) - 复合场景熵(融合前)
"""

import os
import math
import numpy as np
from model.lifter.spconv_voxelize import VoxelGeneratorWrapper
from dataset.utils import get_lidar2global, get_img2global


# ---------------------------------------------------------------------------
# 场景熵计算函数
# ---------------------------------------------------------------------------

def _scene_entropy(points, voxel_generator):
    """体素化点云的香农熵: H = -Σ p_i·ln(p_i), p_i = 体素内点数/总点数"""
    if points.shape[0] == 0:
        return 0.0
    _, _, num_points = voxel_generator.generate(points)
    if num_points.shape[0] == 0:
        return 0.0
    total_points = num_points.sum()
    probs = num_points.astype(np.float64) / total_points
    valid = probs > 0
    return float(-np.sum(probs[valid] * np.log(probs[valid])))


def _compute_voxel_deci(points_in_voxel, vsize=0.5):
    """单个体素的 DECI (Differential Entropy-based Compactness Index)"""
    k = points_in_voxel.shape[0]
    if k <= 1:
        return 0.0

    points_normalized = points_in_voxel / vsize
    mean = points_normalized.mean(axis=0)
    centered = points_normalized - mean
    cov = (centered.T @ centered) / (k - 1)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues, 1e-10, None)

    lambda_max = eigenvalues[0]
    r = int(np.sum(eigenvalues > 0.01 * lambda_max))
    r = min(r, min(k - 1, 3))

    eigenvalues = eigenvalues / eigenvalues.sum()
    if r == 0:
        return 0.0

    prod = np.prod(eigenvalues[:r])
    constant = (2.0 * math.pi * math.e) ** r
    h = 0.5 * math.log(constant * prod + 1.0)
    return float(1.0 / h) if h > 0 else 0.0


def _max_voxel_deci(points, voxel_generator):
    """点云中所有体素 DECI 的最大值"""
    if points.shape[0] == 0:
        return 0.0
    voxels, coords, num_pts = voxel_generator.generate(points)
    M = coords.shape[0]
    if M == 0:
        return 0.0

    max_val = 0.0
    for i in range(M):
        k = int(num_pts[i])
        pts_xyz = voxels[i, :k, :3]
        deci = _compute_voxel_deci(pts_xyz)
        if deci > max_val:
            max_val = deci
    return max_val


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class EntropyBasedHistoryLoader:
    """基于熵增益的自适应历史帧窗口选择器。

    在 get_data_info() 阶段调用 forward()，根据融合历史帧后
    复合场景熵的变化量动态选择实际使用的历史帧数量。

    Args:
        max_window:            最大历史帧融合窗口
        min_window:            最小历史帧融合窗口
        entropy_gain_threshold: 熵增益阈值，低于此值时停止融合
        data_root:             NuScenes 数据根目录
        pc_range:              点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    """

    SENSOR_TYPES = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    ]

    def __init__(self, max_window, min_window, entropy_gain_threshold,
                 data_root, pc_range):
        assert max_window >= min_window >= 1, (
            f"max_window({max_window}) >= min_window({min_window}) >= 1"
        )
        self.max_window = max_window
        self.min_window = min_window
        self.entropy_gain_threshold = entropy_gain_threshold
        self.data_root = data_root
        self.pc_range = pc_range

        self._voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=[0.5, 0.5, 0.5],
            coors_range_xyz=pc_range,
            num_point_features=4,
            max_num_points_per_voxel=20,
            max_num_voxels=1600000,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def forward(self, scene_infos, scene_token, frame_index):
        """主入口：根据熵增益选择历史帧，返回选中帧的元信息列表。

        Returns:
            list[dict]: 选中的历史帧信息（按时间从近到远排列），每个 dict 包含:
                pts_filename  -- 点云文件绝对路径
                lidar_pose    -- LiDAR→global 的 4×4 变换矩阵
                img_files     -- dict[cam_type] → 图像绝对路径
                lidar2img     -- dict[cam_type] → lidar2img 4×4 矩阵
                ego2img       -- dict[cam_type] → ego2img 4×4 矩阵
        """
        current_frame = scene_infos[scene_token][frame_index]
        lidar_info = current_frame['data']['LIDAR_TOP']
        current_pose = get_lidar2global(lidar_info['calib'], lidar_info['pose'])
        ego2global = self._get_ego2global(lidar_info['pose'])

        # 加载当前帧点云用于初始熵计算
        current_points = self._load_and_filter_lidar(lidar_info['filename'])

        # 批量加载 max_window 帧历史
        all_history = self._load_all_history(
            scene_infos, scene_token, frame_index, current_pose, ego2global
        )

        if len(all_history) == 0:
            return []

        # 逐帧融合 + 熵增益判定
        fused = current_points
        E_prev = self._composite_entropy(fused)
        selected = []

        for h in all_history:
            fused = np.concatenate([fused, h['points']], axis=0)
            E_curr = self._composite_entropy(fused)
            gain = max(0.0, E_curr - E_prev)

            # 记录选中帧（去除临时的大块点云数据）
            selected.append({
                'pts_filename': h['pts_filename'],
                'lidar_pose': h['lidar_pose'],
                'img_files': h['img_files'],
                'lidar2img': h['lidar2img'],
                'ego2img': h['ego2img'],
            })

            if self._should_stop(gain, len(selected)):
                break

            E_prev = E_curr

        return selected

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _composite_entropy(self, points):
        """复合场景熵 = 0.4 × scene_entropy + 0.6 × max_voxel_deci"""
        if points.shape[0] == 0:
            return 0.0
        se = _scene_entropy(points, self._voxel_generator)
        max_deci = _max_voxel_deci(points, self._voxel_generator)
        return 0.4 * se + 0.6 * max_deci

    def _should_stop(self, gain, fused_count):
        """判定是否停止融合"""
        if fused_count >= self.max_window:
            return True
        if fused_count >= self.min_window and gain < self.entropy_gain_threshold:
            return True
        return False

    def _load_and_filter_lidar(self, lidar_filename):
        """加载单帧 LiDAR 点云并按 pc_range 过滤"""
        lidar_path = (
            lidar_filename if os.path.isabs(lidar_filename)
            else os.path.join(self.data_root, lidar_filename)
        )
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        mask = (
            (points[:, 0] > self.pc_range[0]) & (points[:, 0] < self.pc_range[3]) &
            (points[:, 1] > self.pc_range[1]) & (points[:, 1] < self.pc_range[4]) &
            (points[:, 2] > self.pc_range[2]) & (points[:, 2] < self.pc_range[5])
        )
        return points[mask]

    @staticmethod
    def _get_ego2global(pose_dict):
        """从 LiDAR pose 提取 ego2global 矩阵"""
        from pyquaternion import Quaternion
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(pose_dict['translation']).T
        return ego2global

    @staticmethod
    def _transform_points(points, source_pose, target_pose):
        """将点云从 source_pose 变换到 target_pose 坐标系"""
        if points.shape[0] == 0:
            return points
        points_hom = np.concatenate(
            [points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)],
            axis=-1,
        )
        T = np.linalg.inv(target_pose) @ source_pose
        transformed_xyz = (T @ points_hom.T).T[:, :3]
        if points.shape[1] > 3:
            return np.concatenate(
                [transformed_xyz, points[:, 3:]], axis=-1
            ).astype(points.dtype, copy=False)
        return transformed_xyz.astype(points.dtype, copy=False)

    def _has_all_cameras(self, frame):
        """检查帧是否包含全部 6 视角相机数据"""
        data = frame.get('data', {})
        return all(cam_type in data for cam_type in self.SENSOR_TYPES)

    def _load_all_history(self, scene_infos, scene_token, frame_index,
                          target_pose, ego2global):
        """一次性加载最多 max_window 帧历史（点云 + 图像路径 + 变换矩阵）"""
        history = []
        scene_frames = scene_infos[scene_token]

        for prev_idx in range(frame_index - 1, -1, -1):
            prev_frame = scene_frames[prev_idx]

            lidar_info = prev_frame.get('data', {}).get('LIDAR_TOP')
            if lidar_info is None:
                continue
            if not self._has_all_cameras(prev_frame):
                continue

            # 加载并变换点云
            source_points = self._load_and_filter_lidar(lidar_info['filename'])
            source_pose = get_lidar2global(lidar_info['calib'], lidar_info['pose'])
            transformed = self._transform_points(source_points, source_pose, target_pose)

            # 收集图像路径与变换矩阵
            img_files = {}
            lidar2img = {}
            ego2img_dict = {}
            for cam_type in self.SENSOR_TYPES:
                cam_info = prev_frame['data'][cam_type]
                img_files[cam_type] = os.path.join(
                    self.data_root, cam_info['filename']
                )
                img2global = get_img2global(cam_info['calib'], cam_info['pose'])
                lidar2img[cam_type] = np.linalg.inv(img2global) @ target_pose
                ego2img_dict[cam_type] = np.linalg.inv(img2global) @ ego2global

            history.append({
                'points': transformed,
                'pts_filename': os.path.abspath(
                    os.path.join(self.data_root, lidar_info['filename'])
                ),
                'lidar_pose': source_pose,
                'img_files': img_files,
                'lidar2img': lidar2img,
                'ego2img': ego2img_dict,
            })

            if len(history) >= self.max_window:
                break

        return history
