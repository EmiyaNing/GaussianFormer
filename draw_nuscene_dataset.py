import pdb
import os
import torch
import pickle
import argparse
import math

import numpy as np

from tqdm import tqdm
from open3d_vis_utils import draw_scenes
from model.lifter.spconv_voxelize import VoxelGeneratorWrapper
from dataset.utils import get_lidar2global

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]


voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=[0.5, 0.5, 0.5],
            coors_range_xyz=pc_range,
            num_point_features=4,
            max_num_points_per_voxel=5,
            max_num_voxels=1600000
)

# ===========================================================================
# [保留] 原始数据加载与处理函数
# ===========================================================================

def load_pkl(pkl_path):
    """加载pkl文件，返回infos字典"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']


def collect_lidar_history_from_infos(infos, scene_token, frame_index, num_history=5):
    """
    从infos字典中收集历史帧的LiDAR信息。
    返回列表，每个元素为 dict(pts_filename=..., lidar_pose=...)
    """
    if num_history <= 0:
        return []

    history = []
    scene_frames = infos[scene_token]
    for prev_idx in range(frame_index - 1, -1, -1):
        prev_frame = scene_frames[prev_idx]
        prev_lidar_info = prev_frame.get('data', {}).get('LIDAR_TOP')
        if prev_lidar_info is None:
            continue

        history.append(
            dict(
                pts_filename=prev_lidar_info['filename'],
                lidar_pose=get_lidar2global(
                    prev_lidar_info['calib'],
                    prev_lidar_info['pose'],
                ),
            )
        )
        if len(history) >= num_history:
            break

    return history


def transform_points_to_target(points, source_pose, target_pose):
    """
    将点云从source_pose坐标系变换到target_pose坐标系。
    points: [N, 3+] (x, y, z, ...)
    """
    if points.shape[0] == 0:
        return points

    points_hom = np.concatenate(
        [points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=-1,
    )
    target_from_source = np.linalg.inv(target_pose) @ source_pose
    transformed_xyz = (target_from_source @ points_hom.T).T[:, :3]

    if points.shape[1] > 3:
        transformed_points = np.concatenate([transformed_xyz, points[:, 3:]], axis=-1)
    else:
        transformed_points = transformed_xyz
    return transformed_points.astype(points.dtype, copy=False)


def load_lidar_points(lidar_path):
    """加载单帧LiDAR点云，返回 [N, 4] (x, y, z, intensity)"""
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points = points[:, :4]  # x, y, z, intensity
    return points


def filter_points_by_range(points, pc_range):
    """根据pc_range过滤点云"""
    mask_x = (points[:, 0] > pc_range[0]) & (points[:, 0] < pc_range[3])
    mask_y = (points[:, 1] > pc_range[1]) & (points[:, 1] < pc_range[4])
    mask_z = (points[:, 2] > pc_range[2]) & (points[:, 2] < pc_range[5])
    mask = mask_x & mask_y & mask_z
    return points[mask]

# ===========================================================================
# [新增] 子任务 A: 体素化模块
# ===========================================================================

def voxelize_point_cloud(points, pc_range, vsize, voxel_generator):
    """
    子任务 A: 将原始LiDAR点云转换为3D体素占据网格。
    
    处理流程:
      A1: 调用 VoxelGeneratorWrapper.generate() 获取体素坐标
      A2: 根据 pc_range / vsize 计算稠密网格维度 (D, H, W)
      A3: 构建二值占据网格 — 有体素则标记为 1.0
      A4: 为每个点计算其所属的体素索引，建立 coord → point_indices 映射
    
    Args:
        points:      (N, 4)  numpy 数组 — (x, y, z, intensity)
        pc_range:    [xmin, ymin, zmin, xmax, ymax, zmax]
        vsize:       [vx, vy, vz] 体素尺寸
        voxel_generator: VoxelGeneratorWrapper 实例
    
    Returns:
        occupancy_grid:  (D, H, W) float32 二值占据网格
        coord_to_points: dict, key=(z,y,x) → np.array of point indices
        grid_shape:      (D, H, W) 元组
    """
    # A1: 调用体素生成器
    voxels, coordinates, num_points = voxel_generator.generate(points)
    # coordinates: (M, 3) 或 (M, 4)，spconv 格式 [z, y, x] 或 [batch, z, y, x]
    
    # 处理可能的 batch 维度
    if coordinates.shape[1] == 4:
        coordinates = coordinates[:, 1:]  # 去掉 batch 索引，保留 [z, y, x]
    # coordinates 现在是 (M, 3): [z_idx, y_idx, x_idx]
    
    # A2: 计算网格维度
    D = int(math.ceil((pc_range[5] - pc_range[2]) / vsize[2]))  # Z
    H = int(math.ceil((pc_range[4] - pc_range[1]) / vsize[1]))  # Y
    W = int(math.ceil((pc_range[3] - pc_range[0]) / vsize[0]))  # X
    grid_shape = (D, H, W)
    
    # A3: 构建二值占据网格
    occupancy_grid = np.zeros(grid_shape, dtype=np.float32)
    for i in range(coordinates.shape[0]):
        z, y, x = int(coordinates[i, 0]), int(coordinates[i, 1]), int(coordinates[i, 2])
        if 0 <= z < D and 0 <= y < H and 0 <= x < W:
            occupancy_grid[z, y, x] = 1.0
    
    # A4: 建立体素 → 点云映射
    voxel_x = np.floor((points[:, 0] - pc_range[0]) / vsize[0]).astype(np.int32)
    voxel_y = np.floor((points[:, 1] - pc_range[1]) / vsize[1]).astype(np.int32)
    voxel_z = np.floor((points[:, 2] - pc_range[2]) / vsize[2]).astype(np.int32)
    
    valid = (
        (voxel_x >= 0) & (voxel_x < W) &
        (voxel_y >= 0) & (voxel_y < H) &
        (voxel_z >= 0) & (voxel_z < D)
    )
    
    coord_to_points = {}
    for i in range(len(points)):
        if not valid[i]:
            continue
        key = (int(voxel_z[i]), int(voxel_y[i]), int(voxel_x[i]))
        if key not in coord_to_points:
            coord_to_points[key] = []
        coord_to_points[key].append(i)
    
    # 列表转 numpy 数组
    for key in coord_to_points:
        coord_to_points[key] = np.array(coord_to_points[key], dtype=np.int32)
    
    return occupancy_grid, coord_to_points, grid_shape


# ===========================================================================
# [新增] 子任务 B: 3D 离散傅里叶变换
# ===========================================================================

def apply_3d_fft(occupancy_grid):
    """
    子任务 B: 将实空间体素占据网格变换到频域。
    
    处理流程:
      B1: 实值 → 复数 (虚部 = 0)
      B2: numpy.fft.fftn() 执行 3D FFT
      B3: numpy.fft.fftshift() 将零频分量移至频谱中心
      B4: 计算频谱幅度 (调试用)
    
    Args:
        occupancy_grid: (D, H, W) float32 占据网格
    
    Returns:
        freq_shifted:   (D, H, W) complex128 中心化频域表示
        freq_magnitude: (D, H, W) float64 频谱幅度
    """
    # B1 & B2: 3D FFT
    freq_domain = np.fft.fftn(occupancy_grid.astype(np.float64))
    
    # B3: 频谱中心化
    freq_shifted = np.fft.fftshift(freq_domain)
    
    # B4: 频谱幅度
    freq_magnitude = np.abs(freq_shifted)
    
    return freq_shifted, freq_magnitude


# ===========================================================================
# [新增] 子任务 C: 频域高通滤波
# ===========================================================================

def high_pass_filter(freq_shifted, cutoff_ratio=0.15, filter_type='gaussian',
                     gaussian_sigma=None, butterworth_order=2):
    """
    子任务 C: 在频域中抑制低频成分，保留高频成分。
    
    处理流程:
      C1: 获取频谱尺寸 D, H, W
      C2: 为每个维度生成归一化频率坐标网格 (Fz, Fy, Fx)
      C3: 计算每个体素的归一化频率距离 dist ∈ [0, 1]
      C4: 根据 filter_type 构建高通掩膜
          - 'ideal':       硬截止 mask = (dist >= cutoff_ratio)
          - 'gaussian':    mask = 1 - exp(-dist² / (2·σ²))
                           σ 默认 = cutoff_ratio (使 dist=cutoff 时 mask≈0.39)
          - 'butterworth': mask = 1 / (1 + (cutoff / dist)^(2n))
      C5: 保留 DC 分量 (dist=0 处强制 mask=1，避免信号漂移)
      C6: 应用掩膜: filtered = freq_shifted * mask
    
    Args:
        freq_shifted:       (D, H, W) complex128 中心化频域
        cutoff_ratio:       截止频率比例 (0~1), 默认 0.15
                            gaussian 模式下同时作为 σ (若未指定 gaussian_sigma)
        filter_type:        滤波器类型 ('ideal' | 'gaussian' | 'butterworth')
        gaussian_sigma:     gaussian σ, None 则取 cutoff_ratio
        butterworth_order:  Butterworth 阶数, 默认 2
    
    Returns:
        filtered_freq_shifted: (D, H, W) complex128 滤波后频域
        hpf_mask:              (D, H, W) float64 高通掩膜 (调试用)
    """
    D, H, W = freq_shifted.shape
    
    # C2: 生成频率坐标网格
    fz = np.arange(D) - D // 2          # (D,)
    fy = np.arange(H) - H // 2          # (H,)
    fx = np.arange(W) - W // 2          # (W,)
    
    Fz, Fy, Fx = np.meshgrid(fz, fy, fx, indexing='ij')  # 每个 (D, H, W)
    
    # C3: 归一化频率距离 — 除以各自维度半径后再除以 √3，确保 dist ∈ [0, 1]
    Fz_norm = Fz / max(D // 2, 1)
    Fy_norm = Fy / max(H // 2, 1)
    Fx_norm = Fx / max(W // 2, 1)
    
    dist = np.sqrt(Fz_norm**2 + Fy_norm**2 + Fx_norm**2)       # [0, √3]
    dist = dist / np.sqrt(3.0)                                  # [0, 1] 真正归一化
    
    # C4: 构建高通掩膜
    if filter_type == 'ideal':
        # 理想高通: 距离 >= cutoff 的频率通过
        hpf_mask = (dist >= cutoff_ratio).astype(np.float64)
    
    elif filter_type == 'gaussian':
        # 高斯高通: mask = 1 - exp(-dist² / (2σ²))
        # σ 默认取 cutoff_ratio，使得 cutoff_ratio 真正控制截止频率
        sigma = cutoff_ratio if gaussian_sigma is None else gaussian_sigma
        sigma = max(sigma, 1e-8)
        hpf_mask = 1.0 - np.exp(- (dist ** 2) / (2.0 * sigma ** 2))
    
    elif filter_type == 'butterworth':
        # Butterworth 高通: mask = 1 / (1 + (cutoff / dist)^(2n))
        eps = 1e-8
        dist_safe = np.maximum(dist, eps)
        hpf_mask = 1.0 / (1.0 + (cutoff_ratio / dist_safe) ** (2 * butterworth_order))
    
    else:
        raise ValueError(f"Unknown filter_type: '{filter_type}'. "
                         f"Choose from: 'ideal', 'gaussian', 'butterworth'.")
    
    # C5: 保留 DC 分量 — 零频 (dist=0) 不参与高通滤波
    #     否则信号完全失去 DC 偏移，IFFT 后重建网格值域大幅漂移
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    hpf_mask[center_z, center_y, center_x] = 1.0
    
    # C6: 应用掩膜
    filtered_freq_shifted = freq_shifted * hpf_mask
    
    return filtered_freq_shifted, hpf_mask


# ===========================================================================
# [新增] 子任务 D: 逆变换与点云重建
# ===========================================================================

def reconstruct_points_from_fft(filtered_freq_shifted, coord_to_points, original_points,
                                 pc_range, vsize, threshold=0.4):
    """
    子任务 D: 从滤波后频域恢复实空间体素占据，重建滤波后点云。
    
    处理流程:
      D1: ifftshift 逆中心化
      D2: ifftn 3D 逆 FFT，取实部
      D3: 阈值二值化: filtered_grid > threshold
      D4: 对每个保留的体素重建点云坐标:
          - 若体素在 coord_to_points 中有映射 → 取原始点坐标
          - 若体素无映射 (新建高频体素) → 用体素中心坐标代替
    
    Args:
        filtered_freq_shifted: (D, H, W) complex128 滤波后频域
        coord_to_points:       dict, (z,y,x) → np.array of point indices
        original_points:       (N, 4) 原始点云 (x, y, z, intensity)
        pc_range:              [xmin, ymin, zmin, xmax, ymax, zmax]
        vsize:                 [vx, vy, vz]
        threshold:             二值化阈值, 默认 0.4
    
    Returns:
        filtered_points:       (K, 3) 滤波后保留的点云坐标
        filtered_grid_binary:  (D, H, W) bool 滤波后二值占据网格
    """
    # D1: 逆中心化
    filtered_freq = np.fft.ifftshift(filtered_freq_shifted)
    
    # D2: 3D 逆 FFT — numpy.fft 的 ifftn 自带 1/N 归一化
    #     高通滤波去除大量频域能量后，输出值域远小于 [0,1]
    #     用 min-max 归一化使 threshold 参数语义稳定
    filtered_grid_complex = np.fft.ifftn(filtered_freq)
    filtered_grid = np.real(filtered_grid_complex)  # (D, H, W) float64
    
    # D3: min-max 归一化 → 阈值二值化
    #     threshold=0.5 表示"保留重建值排前 50% 的体素"
    grid_min = filtered_grid.min()
    grid_max = filtered_grid.max()
    if grid_max - grid_min > 1e-12:
        filtered_grid_norm = (filtered_grid - grid_min) / (grid_max - grid_min)
    else:
        filtered_grid_norm = filtered_grid
    filtered_grid_binary = filtered_grid_norm > threshold
    
    # D4: 从保留体素重建点云坐标
    kept_indices = np.where(filtered_grid_binary)  # (z_idxs, y_idxs, x_idxs)
    num_kept = len(kept_indices[0])
    
    filtered_points_list = []
    
    for i in range(num_kept):
        z = kept_indices[0][i]
        y = kept_indices[1][i]
        x = kept_indices[2][i]
        key = (z, y, x)
        
        if key in coord_to_points:
            # 该体素有原始点云对应 → 取原始点坐标
            pt_indices = coord_to_points[key]
            # 取所有对应点的 xyz 坐标
            pts_xyz = original_points[pt_indices, :3].copy()
            filtered_points_list.append(pts_xyz)
        else:
            # 该体素无原始对应 (可能是滤波后"新出现"的高频体素)
            # → 使用体素中心坐标
            x_center = (x + 0.5) * vsize[0] + pc_range[0]
            y_center = (y + 0.5) * vsize[1] + pc_range[1]
            z_center = (z + 0.5) * vsize[2] + pc_range[2]
            filtered_points_list.append(
                np.array([[x_center, y_center, z_center]], dtype=np.float64)
            )
    
    if len(filtered_points_list) > 0:
        filtered_points = np.concatenate(filtered_points_list, axis=0)
    else:
        filtered_points = np.empty((0, 3), dtype=np.float64)
    
    return filtered_points, filtered_grid_binary


# ===========================================================================
# [新增] 子任务 A+B+C+D 组合管线
# ===========================================================================

def voxelize_and_filter_pipeline(points, pc_range, vsize, voxel_generator,
                                  cutoff_ratio=0.15, filter_type='gaussian',
                                  gaussian_sigma=None, butterworth_order=2,
                                  threshold=0.4):
    """
    组合子任务 A→B→C→D 的完整高通滤波管线。
    
    管线:
      A: 体素化          → occupancy_grid, coord_to_points
      B: 3D FFT         → freq_shifted
      C: 高通滤波        → filtered_freq_shifted
      D: 逆变换 + 点云重建 → filtered_points
    
    Args:
        points:            (N, 4) 原始点云
        pc_range:          点云范围
        vsize:             体素尺寸
        voxel_generator:   VoxelGeneratorWrapper 实例
        cutoff_ratio:      截止频率比例
        filter_type:       滤波器类型
        gaussian_sigma:    高斯 sigma
        butterworth_order: Butterworth 阶数
        threshold:         重建阈值
    
    Returns:
        filtered_points:       (K, 3) 滤波后点云坐标
        occupancy_grid:        (D, H, W) 原始占据网格 (调试)
        filtered_grid_binary:  (D, H, W) 滤波后占据网格 (调试)
        hpf_mask:              (D, H, W) 高通掩膜 (调试)
    """
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), None, None, None
    
    # A: 体素化
    occupancy_grid, coord_to_points, grid_shape = voxelize_point_cloud(
        points, pc_range, vsize, voxel_generator
    )
    
    # B: 3D FFT
    freq_shifted, _ = apply_3d_fft(occupancy_grid)
    
    # C: 高通滤波
    filtered_freq, hpf_mask = high_pass_filter(
        freq_shifted, cutoff_ratio, filter_type, gaussian_sigma, butterworth_order
    )
    
    # D: 逆变换 + 重建
    filtered_points, filtered_grid_binary = reconstruct_points_from_fft(
        filtered_freq, coord_to_points, points, pc_range, vsize, threshold
    )
    
    return filtered_points, occupancy_grid, filtered_grid_binary, hpf_mask


# ===========================================================================
# [新增] 子任务 E: 统计与可视化
# ===========================================================================

def compute_retention_ratio(original_count, filtered_count, label=''):
    """
    子任务 E1: 计算高通滤波点云保留比率。
    
    Args:
        original_count: 原始点云数量
        filtered_count: 滤波后点云数量
        label:          标签前缀 (如 'Current' / 'History-1')
    
    Returns:
        ratio: float, 保留比率
    """
    if original_count == 0:
        ratio = 0.0
    else:
        ratio = filtered_count / original_count
    print(f"  [{label}] 原始点数: {original_count:>8d}, "
          f"滤波后: {filtered_count:>8d}, 保留率: {ratio:.2%}")
    return ratio


def visualize_voxel_points(points, point_colors=None, window_title=None):
    """
    子任务 E2: 体素化/滤波后点云可视化独立函数。
    
    功能: 封装 draw_scenes() 调用，独立于主循环可复用。
    
    Args:
        points:       (M, 3) 点云坐标
        point_colors: (M, 3) 颜色, None 则使用白色
        window_title: 窗口标题 (当前仅终端打印)
    """
    if window_title is not None:
        print(f"\n  >>> 可视化窗口: {window_title}")
    
    if point_colors is not None:
        draw_scenes(points=points, point_colors=point_colors)
    else:
        draw_scenes(points=points)


def build_history_colors(all_points_list, current_color=None, history_cmap_start=None,
                         history_cmap_end=None):
    """
    为多帧点云构建颜色数组 — 复现原有历史帧可视化颜色方案。
    
    颜色方案 (默认):
      当前帧 (idx=0): 白色 [1, 1, 1]
      历史帧 (idx>0): 蓝(0,0,1) → 青(0,1,1) → 黄(1,1,0) → 红(1,0,0)
    
    可通过 current_color / history_cmap_start / history_cmap_end 自定义。
    
    Args:
        all_points_list:     list of (N_i, 3+) arrays, 每帧的点云
        current_color:       当前帧颜色, 默认 [1, 1, 1] (白色)
        history_cmap_start:  历史帧起点颜色, 默认 [0, 0, 1] (蓝色)
        history_cmap_end:    历史帧终点颜色, 默认 [1, 0, 0] (红色)
    
    Returns:
        point_colors:       (total_N, 3) float64
        all_frame_indices:  (total_N,) int32
    """
    if current_color is None:
        current_color = np.array([1.0, 1.0, 1.0])  # 白色
    if history_cmap_start is None:
        history_cmap_start = np.array([0.0, 0.0, 1.0])  # 蓝色
    if history_cmap_end is None:
        history_cmap_end = np.array([1.0, 0.0, 0.0])  # 红色
    
    num_frames = len(all_points_list)
    
    # 收集帧索引
    all_frame_indices_list = []
    for fi, pts in enumerate(all_points_list):
        all_frame_indices_list.append(np.full(len(pts), fi, dtype=np.int32))
    
    all_indices = np.concatenate(all_frame_indices_list, axis=0) if all_frame_indices_list \
                  else np.array([], dtype=np.int32)
    
    total_points = sum(len(pts) for pts in all_points_list)
    point_colors = np.ones((total_points, 3), dtype=np.float64)
    
    for fi in range(num_frames):
        mask_fi = (all_indices == fi)
        count = mask_fi.sum()
        if count == 0:
            continue
        
        if fi == 0:
            point_colors[mask_fi] = current_color
        else:
            # 蓝色渐变到红色: 蓝 → 青 → 黄 → 红
            t = fi / max(num_frames - 1, 1)  # 0~1
            if t < 0.33:
                r, g, b = 0.0, t / 0.33, 1.0
            elif t < 0.66:
                r, g, b = (t - 0.33) / 0.33, 1.0, 1.0 - (t - 0.33) / 0.33
            else:
                r, g, b = 1.0, 1.0 - (t - 0.66) / 0.34, 0.0
            point_colors[mask_fi] = np.array([r, g, b])
    
    return point_colors, all_indices


# ===========================================================================
# [修改] main() — 子任务 F: 主流程整合
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='nuscenes LiDAR点云可视化 — 支持 FFT 高通滤波'
    )
    # --- 原有参数 ---
    parser.add_argument('--pkl_path', default='./data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl',
                        help='相机参数pkl文件路径')
    parser.add_argument('--occ_data_root', default='./data/surroundocc/samples',
                        help='surroundocc数据目录，包含.npy文件')
    parser.add_argument('--num_history', type=int, default=5,
                        help='可视化的历史帧数量')
    parser.add_argument('--data_root', default='./data/nuscenes',
                        help='nuscenes原始数据根目录')
    
    # --- 新增: 高通滤波参数 ---
    parser.add_argument('--enable_hpf', action='store_true', default=False,
                        help='启用 FFT 高通滤波管线 (默认关闭，按需开启)')
    parser.add_argument('--hpf_cutoff', type=float, default=0.15,
                        help='高通滤波截止频率比例 (0~1), 默认 0.15')
    parser.add_argument('--hpf_type', type=str, default='gaussian',
                        choices=['ideal', 'gaussian', 'butterworth'],
                        help='滤波器类型, 默认 gaussian')
    parser.add_argument('--hpf_sigma', type=float, default=None,
                        help='高斯滤波器 sigma (None 则使用 cutoff_ratio)')
    parser.add_argument('--hpf_order', type=int, default=2,
                        help='Butterworth 滤波器阶数, 默认 2')
    parser.add_argument('--hpf_threshold', type=float, default=0.4,
                        help='重建二值化阈值, 默认 0.4')
    
    # --- 新增: 可视化控制参数 ---
    parser.add_argument('--show_original', action='store_true', default=True,
                        help='显示原始点云 (默认开启)')
    parser.add_argument('--no_show_original', action='store_false', dest='show_original',
                        help='不显示原始点云')
    parser.add_argument('--show_filtered', action='store_true', default=True,
                        help='显示滤波后点云 (默认开启)')
    parser.add_argument('--no_show_filtered', action='store_false', dest='show_filtered',
                        help='不显示滤波后点云')
    parser.add_argument('--compare_mode', action='store_true', default=False,
                        help='对比模式: 在同一窗口同时显示原始和滤波后点云')
    
    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"Camera pkl file not found: {args.pkl_path}")

    if args.enable_hpf:
        gs_display = args.hpf_sigma if args.hpf_sigma is not None else f"<cutoff_ratio={args.hpf_cutoff}>"
        print(f"\n{'='*60}")
        print(f"  FFT 高通滤波参数:")
        print(f"    cutoff_ratio  = {args.hpf_cutoff}")
        print(f"    filter_type   = {args.hpf_type}")
        print(f"    gaussian_sigma= {gs_display}")
        print(f"    hpf_order     = {args.hpf_order}")
        print(f"    threshold     = {args.hpf_threshold}")
        print(f"    compare_mode  = {args.compare_mode}")
        print(f"{'='*60}\n")

    infos = load_pkl(args.pkl_path)

    vsize = [0.5, 0.5, 0.5]  # 体素尺寸

    for scene_token, frames in tqdm(infos.items()):
        for frame_idx, frame in enumerate(frames):
            if 'LIDAR_TOP' not in frame['data'].keys():
                continue

            # ============================================================
            # [保留] 1. 加载当前帧点云
            # ============================================================
            lidar_info = frame['data']['LIDAR_TOP']
            lidar_filename = lidar_info['filename']
            lidar_path = os.path.join(args.data_root, lidar_filename)
            current_points = load_lidar_points(lidar_path)
            current_points = filter_points_by_range(current_points, pc_range)

            # ============================================================
            # [保留] 2. 获取当前帧的 lidar2global 位姿
            # ============================================================
            current_lidar_pose = get_lidar2global(
                lidar_info['calib'],
                lidar_info['pose'],
            )

            # ============================================================
            # [保留] 3. 收集历史帧信息并变换到当前帧坐标系
            # ============================================================
            history_list = collect_lidar_history_from_infos(
                infos, scene_token, frame_idx,
                num_history=args.num_history,
            )

            # --- 收集原始点云列表 (保留原逻辑) ---
            all_points_list = []
            all_points_list.append(current_points)
            for hist_i, sweep in enumerate(history_list):
                sweep_path = os.path.join(args.data_root, sweep['pts_filename'])
                sweep_points = load_lidar_points(sweep_path)
                sweep_points = transform_points_to_target(
                    sweep_points,
                    sweep['lidar_pose'],
                    current_lidar_pose,
                )
                sweep_points = filter_points_by_range(sweep_points, pc_range)
                all_points_list.append(sweep_points)

            # ============================================================
            # [新增] 4. 高通滤波管线 (如果启用)
            # ============================================================
            filtered_points_list = []   # 滤波后点云, 顺序与 all_points_list 一致
            retention_ratios = []

            if args.enable_hpf:
                print(f"\n{'─'*50}")
                print(f"  Scene: {scene_token[:16]}... | Frame: {frame_idx}")
                print(f"  Running FFT High-Pass Filter...")

                # 对每一帧 (当前帧 + 历史帧) 执行高通滤波
                for fi, pts in enumerate(all_points_list):
                    label = f"Frame-{fi}" if fi > 0 else "Current"

                    filtered_pts, occup_grid, filt_grid, hpf_mask = voxelize_and_filter_pipeline(
                        pts, pc_range, vsize, voxel_generator,
                        cutoff_ratio=args.hpf_cutoff,
                        filter_type=args.hpf_type,
                        gaussian_sigma=args.hpf_sigma,
                        butterworth_order=args.hpf_order,
                        threshold=args.hpf_threshold,
                    )

                    filtered_points_list.append(filtered_pts)

                    ratio = compute_retention_ratio(len(pts), len(filtered_pts), label=label)
                    retention_ratios.append(ratio)

                # 打印汇总
                if retention_ratios:
                    avg_ratio = sum(retention_ratios) / len(retention_ratios)
                    print(f"  [Avg] 平均保留率: {avg_ratio:.2%} "
                          f"({sum(r>0 for r in retention_ratios)}/{len(retention_ratios)} 帧有效)")
                print(f"{'─'*50}")

            # ============================================================
            # [保留 + 增强] 5. 原始点云可视化
            # ============================================================
            if args.show_original and not args.compare_mode:
                orig_colors, _ = build_history_colors(all_points_list)
                all_orig_pts = np.concatenate(
                    [p[:, :3] for p in all_points_list], axis=0
                )
                visualize_voxel_points(
                    all_orig_pts, orig_colors,
                    window_title=f"Original — Scene:{scene_token[:12]} Frame:{frame_idx}"
                )

            # ============================================================
            # [新增] 6. 滤波后点云可视化
            # ============================================================
            if args.enable_hpf and args.show_filtered and not args.compare_mode:
                if filtered_points_list:
                    # 使用品红渐变色方案
                    current_color_filt = np.array([0.0, 1.0, 0.0])     # 绿色 = 滤波后当前帧
                    history_start_filt = np.array([1.0, 0.0, 1.0])     # 品红 = 最近历史
                    history_end_filt = np.array([1.0, 0.5, 0.0])       # 橙色 = 最旧历史

                    filt_colors, _ = build_history_colors(
                        filtered_points_list,
                        current_color=current_color_filt,
                        history_cmap_start=history_start_filt,
                        history_cmap_end=history_end_filt,
                    )
                    all_filt_pts = np.concatenate(
                        [fp[:, :3] for fp in filtered_points_list if fp.shape[0] > 0], axis=0
                    )
                    visualize_voxel_points(
                        all_filt_pts, filt_colors,
                        window_title=f"HPF Filtered — cutoff={args.hpf_cutoff} "
                                     f"Scene:{scene_token[:12]} Frame:{frame_idx}"
                    )

            # ============================================================
            # [新增] 7. 对比模式: 原始 + 滤波后 在同一窗口显示
            # ============================================================
            if args.enable_hpf and args.compare_mode:
                if filtered_points_list and all_points_list:
                    N = len(all_points_list)
                    
                    # 滤掉空帧，只保留非空帧的点和颜色
                    nonempty_orig = [all_points_list[i] for i in range(N)
                                     if all_points_list[i].shape[0] > 0]
                    nonempty_filt = [filtered_points_list[i] for i in range(N)
                                     if filtered_points_list[i].shape[0] > 0]
                    
                    # 构建颜色
                    current_color_filt = np.array([0.0, 1.0, 0.0])
                    history_start_filt = np.array([1.0, 0.0, 1.0])
                    history_end_filt   = np.array([1.0, 0.5, 0.0])
                    
                    orig_colors, _ = build_history_colors(nonempty_orig)
                    filt_colors, _ = build_history_colors(
                        nonempty_filt,
                        current_color=current_color_filt,
                        history_cmap_start=history_start_filt,
                        history_cmap_end=history_end_filt,
                    )
                    
                    # 合并点云
                    orig_pts_combined = np.concatenate([p[:, :3] for p in nonempty_orig], axis=0)
                    filt_pts_combined = np.concatenate([p[:, :3] for p in nonempty_filt], axis=0)
                    combined_points = np.concatenate([orig_pts_combined, filt_pts_combined], axis=0)
                    combined_colors = np.concatenate([orig_colors, filt_colors], axis=0)
                    
                    visualize_voxel_points(
                        combined_points, combined_colors,
                        window_title=f"对比模式 — 白/蓝→红=原始 | 绿/品红→橙=滤波 | "
                                     f"cutoff={args.hpf_cutoff}"
                    )


if __name__ == '__main__':
    main()