#!/usr/bin/env python3
"""
dataset_analysis_cuda.py — NuScenes 数据集场景融合熵统计脚本 (CUDA 精确版)

从融合 0 帧逐步统计到融合 16 帧的:
  - 平均场景熵 (基于体素点分布的香农熵)
  - 平均局部 DECI 熵 (基于体素内点云几何紧凑度的微分熵)

使用 multiprocessing.Pool 实现场景级并行: 每个 worker 进程独立处理
一个或多个场景，主进程负责汇总全局统计并写入结果文件。

结果输出到 entropy_result.txt
"""

import os
import pickle
import argparse

# 避免每个多进程 worker 内部的 BLAS/OpenMP 再开启大量线程。
# setdefault 保留用户在运行脚本前显式指定的配置。
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import torch
from tqdm import tqdm

from draw_nuscene_dataset import (
    load_pkl,
    load_lidar_points,
    filter_points_by_range,
    collect_lidar_history_from_frames,
    transform_points_to_target,
)
from dataset.utils import get_lidar2global

# ===========================================================================
# 全局配置 (与 draw_nuscene_dataset.py 保持一致)
# ===========================================================================

PC_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
VOXEL_SIZE = [0.5, 0.5, 0.5]
MAX_POINTS_PER_VOXEL = 20
MAX_VOXELS = 1600000

# 组合熵 C = 场景熵权重 * H + 局部熵权重 * DECI_mean
SCENE_ENTROPY_WEIGHT = 0.4
LOCAL_ENTROPY_WEIGHT = 0.6

# 增益正负判断阈值。默认严格按照 < 0 / > 0 判断。
GAIN_EPS = 0.0

# 批量计算 DECI 时的最大体素批大小，限制临时矩阵的峰值内存。
DECI_BATCH_SIZE = 65536

# DECI 只可能取 0~3 维。常数表在模块加载时计算一次，避免每个批次
# 重复执行幂运算；这只是缓存原公式中的精确常数，不改变数学定义。
DECI_CONSTANTS = np.power(2.0 * np.pi * np.e, np.arange(4))


class CUDAVoxelGeneratorWrapper:
    """确定性的 CUDA 硬体素化器。

    稳定排序保证每个体素保留输入顺序中的前 max_points 个点，与 CPU
    Point2Voxel 的选择语义一致，避免 hash/atomic kernel 的随机写入顺序。
    """

    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features,
                 max_num_points_per_voxel, max_num_voxels, device='cuda:0'):
        self.device = torch.device(device)
        self.num_point_features = num_point_features
        self.max_num_voxels = max_num_voxels
        self.max_points = max_num_points_per_voxel
        self.vsize = torch.tensor(
            vsize_xyz, dtype=torch.float32, device=self.device
        )
        self.range_min = torch.tensor(
            coors_range_xyz[:3], dtype=torch.float32, device=self.device
        )
        self.grid_size = torch.tensor(
            [
                round((coors_range_xyz[i + 3] - coors_range_xyz[i])
                      / vsize_xyz[i])
                for i in range(3)
            ],
            dtype=torch.int64,
            device=self.device,
        )

    @torch.inference_mode()
    def generate(self, points):
        # np.ascontiguousarray 不改变点顺序。
        points = np.ascontiguousarray(points, dtype=np.float32)
        points_cuda = torch.from_numpy(points).to(self.device)
        coords_xyz = torch.floor(
            (points_cuda[:, :3] - self.range_min) / self.vsize
        ).to(torch.int64)
        in_range = torch.all(
            (coords_xyz >= 0) & (coords_xyz < self.grid_size), dim=1
        )
        points_cuda = points_cuda[in_range]
        coords_xyz = coords_xyz[in_range]
        if points_cuda.shape[0] == 0:
            return (
                torch.empty((0, self.max_points, self.num_point_features),
                            dtype=torch.float32, device=self.device),
                torch.empty((0, 3), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.int32, device=self.device),
            )

        keys = (
            coords_xyz[:, 0]
            + self.grid_size[0] * (
                coords_xyz[:, 1] + self.grid_size[1] * coords_xyz[:, 2]
            )
        )
        order = torch.argsort(keys, stable=True)
        keys = keys[order]
        sorted_points = points_cuda[order]
        sorted_coords = coords_xyz[order]

        is_first = torch.ones_like(keys, dtype=torch.bool)
        is_first[1:] = keys[1:] != keys[:-1]
        group_ids = torch.cumsum(is_first, dim=0) - 1
        positions = torch.arange(keys.numel(), device=self.device)
        group_starts = torch.cummax(
            torch.where(is_first, positions, torch.zeros_like(positions)), dim=0
        ).values
        point_slots = positions - group_starts
        keep = point_slots < self.max_points

        group_ids = group_ids[keep]
        point_slots = point_slots[keep]
        sorted_points = sorted_points[keep]
        num_voxels = int(group_ids[-1].item()) + 1
        if num_voxels > self.max_num_voxels:
            # 当前配置的空间网格最多 640000 个体素，小于 1600000 上限；
            # 保留通用保护，避免静默改变 max_voxels 语义。
            raise RuntimeError(
                '唯一体素数超过 max_num_voxels；需实现首遇体素截断语义'
            )

        voxels = torch.zeros(
            (num_voxels, self.max_points, self.num_point_features),
            dtype=torch.float32, device=self.device,
        )
        voxels[group_ids, point_slots] = sorted_points
        counts = torch.bincount(
            group_ids, minlength=num_voxels
        ).to(torch.int32)
        # spconv 坐标采用 z,y,x；熵计算不依赖坐标，仅保持接口一致。
        first_indices = torch.nonzero(is_first, as_tuple=False).flatten()
        coordinates = sorted_coords[first_indices]
        coordinates = coordinates[:, [2, 1, 0]].to(torch.int32)
        return voxels, coordinates, counts


def create_voxel_generator(device='cuda:0'):
    """创建与现有代码参数一致的体素生成器"""
    return CUDAVoxelGeneratorWrapper(
        vsize_xyz=VOXEL_SIZE,
        coors_range_xyz=PC_RANGE,
        num_point_features=4,
        max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
        max_num_voxels=MAX_VOXELS,
        device=device,
    )


def compute_deci_values_batched(voxels, num_points, batch_size=DECI_BATCH_SIZE):
    """在 CUDA 上批量计算 DECI，数学公式与 CPU 版本严格一致。

    优化仅减少无效计算和临时数组：单点体素的 DECI 按定义恒为 0，
    因此不送入协方差和特征值分解；中心化则复用点坐标缓冲区。
    """
    counts_all = num_points.to(dtype=torch.int64)
    voxel_count = counts_all.numel()
    deci_values = torch.zeros(
        voxel_count, dtype=torch.float64, device=voxels.device
    )
    if voxel_count == 0:
        return deci_values

    if not torch.any(counts_all > 1):
        return deci_values

    # 原实现 compute_voxel_deci() 使用标量 vsize=0.5。
    vsize = float(VOXEL_SIZE[0])
    max_points = voxels.shape[1]
    point_indices = torch.arange(max_points, device=voxels.device)[None, :]
    constants = torch.as_tensor(
        DECI_CONSTANTS, dtype=torch.float64, device=voxels.device
    )

    # batch_size 仍表示输入体素批大小，确保优化版的峰值内存不会因
    # “只按有效体素计数”而意外超过原版。
    for start in range(0, voxel_count, batch_size):
        end = min(start + batch_size, voxel_count)
        selected_indices = (
            start + torch.nonzero(
                counts_all[start:end] > 1, as_tuple=False
            ).flatten()
        )
        if selected_indices.numel() == 0:
            continue
        counts = counts_all[selected_indices]

        # 保持输入点云 dtype，确保与原逐体素实现的数值精度一致。
        # 高级索引已经返回独立数组，原地归一化可避免再复制一份坐标。
        points = voxels[selected_indices, :, :3] / vsize
        mask = point_indices < counts[:, None]
        points *= mask[..., None]
        means = points.sum(axis=1) / counts[:, None]

        # 只为有效体素生成中心化数组；相比原实现仍按单点体素占比同比
        # 降低峰值内存，同时保留 NumPy 原表达式的求值和舍入路径。
        centered = (points - means[:, None, :]) * mask[..., None]
        covariances = torch.bmm(
            centered.transpose(1, 2), centered
        ) / (counts - 1)[:, None, None]

        eigenvalues = torch.linalg.eigvalsh(covariances)
        eigenvalues = torch.sort(
            torch.abs(eigenvalues), dim=1, descending=True
        ).values
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)

        ranks = torch.sum(
            eigenvalues > 0.01 * eigenvalues[:, :1], dim=1
        ).to(torch.int64)
        ranks = torch.minimum(
            ranks, torch.minimum(counts - 1, torch.full_like(counts, 3))
        )
        normalized = eigenvalues / eigenvalues.sum(dim=1, keepdim=True)

        products = torch.ones(
            selected_indices.numel(), dtype=torch.float64,
            device=voxels.device,
        )
        for rank in (1, 2, 3):
            selected = ranks == rank
            if torch.any(selected):
                products[selected] = torch.prod(
                    normalized[selected, :rank], dim=1
                ).to(torch.float64)

        entropy = 0.5 * torch.log(constants[ranks] * products + 1.0)
        valid_entropy = (ranks > 0) & (entropy > 0)
        batch_deci = torch.zeros(
            selected_indices.numel(), dtype=torch.float64,
            device=voxels.device,
        )
        batch_deci[valid_entropy] = 1.0 / entropy[valid_entropy]
        deci_values[selected_indices] = batch_deci

    return deci_values


@torch.inference_mode()
def compute_both_entropies(points, vg):
    """
    对单组点云计算场景熵 + 局部 DECI 熵统计。

    Returns:
        dict: {'H': float, 'deci_max': float, 'deci_min': float,
               'deci_mean': float, 'deci_above': float,
               'combined': float}
    """
    if points.shape[0] == 0:
        H = 0.0
        deci = {
            'max': 0.0, 'min': 0.0, 'mean': 0.0,
            'above_mean_ratio': 0.0,
        }
    else:
        # 场景熵与局部熵共享一次体素化结果。
        voxels, coords, num_points = vg.generate(points)
        if num_points.numel() == 0:
            H = 0.0
            deci = {
                'max': 0.0, 'min': 0.0, 'mean': 0.0,
                'above_mean_ratio': 0.0,
            }
        else:
            counts = num_points.to(dtype=torch.int64)
            total_points = counts.sum()
            # -sum(p*log(p)) 的严格恒等形式：
            # log(N) - sum(n_i*log(n_i))/N。避免 probabilities 中间数组
            # 以及对一次除法结果再取对数，不使用任何数学近似。
            counts_float = counts.to(dtype=torch.float64)
            H = (
                torch.log(total_points.to(torch.float64))
                - torch.dot(counts_float, torch.log(counts_float))
                / total_points
            )

            deci_values = compute_deci_values_batched(voxels, counts)
            deci_mean = torch.dot(
                deci_values, counts_float
            ) / total_points
            above_ratio = (
                counts[deci_values > deci_mean].sum() / total_points
            )

            # 一次同步取回全部标量，避免多个 .item() 形成串行同步点。
            scalar_values = torch.stack((
                H,
                torch.max(deci_values),
                torch.min(deci_values),
                deci_mean,
                above_ratio,
            )).cpu().tolist()
            H, deci_max, deci_min, deci_mean_value, above_ratio_value = (
                scalar_values
            )
            deci = {
                'max': deci_max,
                'min': deci_min,
                'mean': deci_mean_value,
                'above_mean_ratio': above_ratio_value,
            }
    combined = (
        SCENE_ENTROPY_WEIGHT * H
        + LOCAL_ENTROPY_WEIGHT * deci['mean']
    )
    return {
        'H': H,
        'deci_max': deci['max'],
        'deci_min': deci['min'],
        'deci_mean': deci['mean'],
        'deci_above': deci['above_mean_ratio'],
        'combined': combined,
    }


def summarize_frame_gains(frame_H, frame_combined, N):
    """汇总单个关键帧在相邻融合级别间的增益。"""
    gain_H_sum = np.zeros(N)
    gain_combined_sum = np.zeros(N)
    gain_count = np.zeros(N, dtype=int)

    delta_H = np.diff(np.asarray(frame_H, dtype=np.float64))
    delta_combined = np.diff(np.asarray(frame_combined, dtype=np.float64))

    for level, (gain_H, gain_combined) in enumerate(
            zip(delta_H, delta_combined), start=1):
        gain_H_sum[level] = gain_H
        gain_combined_sum[level] = gain_combined
        gain_count[level] = 1

    negative_count = int(np.sum(delta_combined < -GAIN_EPS))
    recovered_negative_count = 0
    future_has_positive = False
    for gain in reversed(delta_combined):
        if gain < -GAIN_EPS and future_has_positive:
            recovered_negative_count += 1
        if gain > GAIN_EPS:
            future_has_positive = True

    return (gain_H_sum, gain_combined_sum, gain_count,
            negative_count, recovered_negative_count, len(delta_combined))


# ===========================================================================
# Worker 入口函数 (模块级, 可被 pickle 序列化)
# ===========================================================================

def process_single_scene(args):
    """
    多进程 worker 入口: 处理单个场景的全部帧和融合级别。

    单 GPU 进程创建确定性 CUDA 体素生成器，场景内部由 CUDA 并行。

    Args:
        args: tuple of (scene_token, frames, data_root, max_fusion, N,
                         pc_range, voxel_size, max_pts_per_voxel, max_voxels,
                         device)

    Returns:
        tuple: 场景标识、帧数、逐级均值及供主进程合并的累加统计。
    """
    (scene_token, frames, data_root, max_fusion, N,
     pc_range, voxel_size, max_pts_per_voxel, max_voxels, device) = args

    # 每个 worker 独立创建 VoxelGenerator
    vg = CUDAVoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=pc_range,
        num_point_features=4,
        max_num_points_per_voxel=max_pts_per_voxel,
        max_num_voxels=max_voxels,
        device=device,
    )

    # 场景级累加器
    scene_H = np.zeros(N)
    scene_deci_max = np.zeros(N)
    scene_deci_min = np.zeros(N)
    scene_deci_mean = np.zeros(N)
    scene_deci_above = np.zeros(N)
    scene_combined = np.zeros(N)
    scene_count = np.zeros(N, dtype=int)
    scene_gain_H = np.zeros(N)
    scene_gain_combined = np.zeros(N)
    scene_gain_count = np.zeros(N, dtype=int)
    scene_negative_gain_count = 0
    scene_recovered_negative_count = 0
    scene_total_gain_count = 0
    valid_keyframes = 0
    # 同一场景中的点云会被多个后续关键帧重复引用，缓存原始读取结果。
    point_cache = {}

    def load_points_cached(path):
        points = point_cache.get(path)
        if points is None:
            points = load_lidar_points(path)
            point_cache[path] = points
        return points

    for frame_idx, frame in enumerate(frames):
        if 'LIDAR_TOP' not in frame.get('data', {}):
            continue
        valid_keyframes += 1

        # ----------------------------------------------------------------
        # 1. 加载当前帧点云
        # ----------------------------------------------------------------
        lidar_info = frame['data']['LIDAR_TOP']
        lidar_path = os.path.join(data_root, lidar_info['filename'])
        current_points = load_points_cached(lidar_path)
        current_points = filter_points_by_range(current_points, pc_range)

        current_pose = get_lidar2global(
            lidar_info['calib'], lidar_info['pose']
        )

        # ----------------------------------------------------------------
        # 2. 收集历史帧 (多进程友好版本, 直接使用 frames 列表)
        # ----------------------------------------------------------------
        history_list = collect_lidar_history_from_frames(
            frames, frame_idx, num_history=max_fusion
        )

        # ----------------------------------------------------------------
        # 3. 渐进式融合: 0 帧 → min(max_fusion, len(history_list)) 帧
        # ----------------------------------------------------------------
        actual_levels = min(max_fusion, len(history_list)) + 1
        frame_H = []
        frame_combined = []

        # 历史点云先完成变换和过滤，以便一次性分配融合缓冲区。
        transformed_sweeps = []
        for sweep in history_list[:actual_levels - 1]:
            sweep_path = os.path.join(data_root, sweep['pts_filename'])
            sweep_points = load_points_cached(sweep_path)
            sweep_points = transform_points_to_target(
                sweep_points, sweep['lidar_pose'], current_pose
            )
            transformed_sweeps.append(
                filter_points_by_range(sweep_points, pc_range)
            )

        total_merged_points = len(current_points) + sum(
            len(sweep_points) for sweep_points in transformed_sweeps
        )
        merged_buffer = np.empty(
            (total_merged_points, current_points.shape[1]),
            dtype=current_points.dtype,
        )
        used_points = len(current_points)
        merged_buffer[:used_points] = current_points

        for level in range(actual_levels):
            if level == 0:
                merged = merged_buffer[:used_points]
            else:
                sweep_points = transformed_sweeps[level - 1]
                next_used_points = used_points + len(sweep_points)
                merged_buffer[used_points:next_used_points] = sweep_points
                used_points = next_used_points
                merged = merged_buffer[:used_points]

            # ------------------------------------------------------------
            # 4. 双路熵计算
            # ------------------------------------------------------------
            stats = compute_both_entropies(merged, vg)

            # 场景级累加
            scene_H[level] += stats['H']
            scene_deci_max[level] += stats['deci_max']
            scene_deci_min[level] += stats['deci_min']
            scene_deci_mean[level] += stats['deci_mean']
            scene_deci_above[level] += stats['deci_above']
            scene_combined[level] += stats['combined']
            scene_count[level] += 1

            frame_H.append(stats['H'])
            frame_combined.append(stats['combined'])

        # 同一关键帧内计算相邻融合级别的增益，避免不同样本集合相减。
        (frame_gain_H, frame_gain_combined, frame_gain_count,
         negative_count, recovered_count, total_gain_count) = (
            summarize_frame_gains(frame_H, frame_combined, N)
        )
        scene_gain_H += frame_gain_H
        scene_gain_combined += frame_gain_combined
        scene_gain_count += frame_gain_count
        scene_negative_gain_count += negative_count
        scene_recovered_negative_count += recovered_count
        scene_total_gain_count += total_gain_count

    # 计算场景均值
    scene_avgs = []
    for level in range(N):
        if scene_count[level] > 0:
            scene_avgs.append({
                'count': int(scene_count[level]),
                'H': scene_H[level] / scene_count[level],
                'deci_max': scene_deci_max[level] / scene_count[level],
                'deci_min': scene_deci_min[level] / scene_count[level],
                'deci_mean': scene_deci_mean[level] / scene_count[level],
                'deci_above': scene_deci_above[level] / scene_count[level],
                'combined': scene_combined[level] / scene_count[level],
                'gain_H': (
                    scene_gain_H[level] / scene_gain_count[level]
                    if scene_gain_count[level] > 0 else None
                ),
                'gain_combined': (
                    scene_gain_combined[level] / scene_gain_count[level]
                    if scene_gain_count[level] > 0 else None
                ),
                'gain_count': int(scene_gain_count[level]),
            })
        else:
            scene_avgs.append(None)

    return (scene_token, len(frames), valid_keyframes, scene_avgs,
            scene_H, scene_deci_max, scene_deci_min,
            scene_deci_mean, scene_deci_above, scene_combined, scene_count,
            scene_gain_H, scene_gain_combined, scene_gain_count,
            scene_negative_gain_count, scene_recovered_negative_count,
            scene_total_gain_count)


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NuScenes 场景融合熵统计 (CUDA 确定性版)'
    )
    parser.add_argument(
        '--pkl_path',
        default='./data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl',
        help='pkl 文件路径',
    )
    parser.add_argument(
        '--data_root',
        default='./data/nuscenes',
        help='NuScenes 原始数据根目录',
    )
    parser.add_argument(
        '--num_history',
        type=int,
        default=16,
        help='最大融合历史帧数 (默认 16)',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='CUDA 场景 worker 数；单 GPU 必须为 1 (默认 1)',
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='CUDA 设备 (默认 cuda:0)',
    )
    parser.add_argument(
        '--output',
        default='entropy_result_cuda.txt',
        help='输出文件路径',
    )
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = 1
    if args.num_workers != 1:
        raise ValueError(
            '单 GPU CUDA 版本要求 --num_workers=1，避免多进程争抢显存'
        )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；请检查 NVIDIA 驱动与执行环境')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('--device 必须是 CUDA 设备，例如 cuda:0')
    torch.cuda.set_device(device)

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"pkl 文件不存在: {args.pkl_path}")

    # ------------------------------------------------------------------
    # 1. 主进程加载 pkl
    # ------------------------------------------------------------------
    print(f"Loading pkl from: {args.pkl_path}")
    infos = load_pkl(args.pkl_path)

    max_fusion = args.num_history
    N = max_fusion + 1  # 融合级别数: 0 ~ max_fusion
    num_scenes = len(infos)

    print(f"Total scenes: {num_scenes}, "
          f"fusion levels: 0~{max_fusion}, "
          f"workers: {args.num_workers}")

    # ------------------------------------------------------------------
    # 2. 构建任务列表 (每场景一个 task)
    # ------------------------------------------------------------------
    tasks = [
        (scene_token, frames, args.data_root, max_fusion, N,
         PC_RANGE, VOXEL_SIZE, MAX_POINTS_PER_VOXEL, MAX_VOXELS, args.device)
        for scene_token, frames in infos.items()
    ]

    # ------------------------------------------------------------------
    # 3. 全局累加器
    # ------------------------------------------------------------------
    global_H = np.zeros(N)
    global_deci_max = np.zeros(N)
    global_deci_min = np.zeros(N)
    global_deci_mean = np.zeros(N)
    global_deci_above = np.zeros(N)
    global_combined = np.zeros(N)
    global_count = np.zeros(N, dtype=int)
    global_gain_H = np.zeros(N)
    global_gain_combined = np.zeros(N)
    global_gain_count = np.zeros(N, dtype=int)
    global_negative_gain_count = 0
    global_recovered_negative_count = 0
    global_total_gain_count = 0
    per_scene_data = []

    # ------------------------------------------------------------------
    # 4. 单 GPU 顺序调度场景；每个场景内部由 CUDA 大规模并行
    # ------------------------------------------------------------------
    print(f"\nDispatching {num_scenes} scenes to {args.device}...")
    results = map(process_single_scene, tasks)
    for result in tqdm(results, total=len(tasks), desc='Scenes (CUDA)'):
        (scene_token, total_frames, valid_keyframes, scene_avgs,
         s_H, s_dmax, s_dmin, s_dmean, s_dabove, s_combined, s_count,
         s_gain_H, s_gain_combined, s_gain_count,
         s_negative_count, s_recovered_count, s_total_gain_count) = result

        per_scene_data.append(
            (scene_token, total_frames, valid_keyframes, scene_avgs)
        )

        global_H += s_H
        global_deci_max += s_dmax
        global_deci_min += s_dmin
        global_deci_mean += s_dmean
        global_deci_above += s_dabove
        global_combined += s_combined
        global_count += s_count
        global_gain_H += s_gain_H
        global_gain_combined += s_gain_combined
        global_gain_count += s_gain_count
        global_negative_gain_count += s_negative_count
        global_recovered_negative_count += s_recovered_count
        global_total_gain_count += s_total_gain_count

    # imap_unordered 的完成顺序不固定，排序后保证输出可复现。
    per_scene_data.sort(key=lambda item: item[0])

    # ==================================================================
    # 5. 写入结果文件
    # ==================================================================
    with open(args.output, 'w', encoding='utf-8') as f:
        sep = "=" * 80
        sub = "-" * 80

        f.write(sep + "\n")
        f.write("  NuScenes 数据集场景融合熵统计结果 (CUDA 确定性版)\n")
        f.write(f"  pkl 路径: {args.pkl_path}\n")
        f.write(f"  数据根目录: {args.data_root}\n")
        f.write(f"  体素配置: vsize={VOXEL_SIZE}, max_pts={MAX_POINTS_PER_VOXEL}\n")
        f.write(f"  最大融合帧数: {max_fusion}\n")
        f.write(f"  Worker 数量: {args.num_workers}\n")
        f.write(f"  CUDA 设备: {args.device}\n")
        f.write(
            "  组合熵: "
            f"{SCENE_ENTROPY_WEIGHT:.1f} * 场景熵 + "
            f"{LOCAL_ENTROPY_WEIGHT:.1f} * DECI_mean\n"
        )
        f.write("  增益: 同一关键帧 level k 减 level k-1\n")
        f.write(f"  增益正负判断阈值: {GAIN_EPS:g}\n")
        f.write("  增益比例分母: 全部有效的相邻融合级别增益观测\n")
        f.write(sep + "\n\n")

        # ---------- 一、全局统计 ----------
        f.write(sub + "\n")
        f.write("  一、全局统计 (所有场景平均)\n")
        f.write(sub + "\n\n")

        header = (
            f"{'融合帧数':>8}  {'场景熵':>10}  {'组合熵':>10}  "
            f"{'场景熵增益':>12}  {'组合熵增益':>12}  "
            f"{'DECI_max':>10}  "
            f"{'DECI_mean':>10}  {'DECI_min':>10}  {'超均值比(%)':>12}  "
            f"{'有效帧数':>8}  {'有效增益数':>10}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for level in range(N):
            if global_count[level] > 0:
                n = global_count[level]
                if global_gain_count[level] > 0:
                    gain_H_text = (
                        f"{global_gain_H[level] / global_gain_count[level]:>12.4f}"
                    )
                    gain_C_text = (
                        f"{global_gain_combined[level] / global_gain_count[level]:>12.4f}"
                    )
                else:
                    gain_H_text = f"{'N/A':>12}"
                    gain_C_text = f"{'N/A':>12}"
                f.write(
                    f"{level:>8}  "
                    f"{global_H[level] / n:>10.4f}  "
                    f"{global_combined[level] / n:>10.4f}  "
                    f"{gain_H_text}  "
                    f"{gain_C_text}  "
                    f"{global_deci_max[level] / n:>10.4f}  "
                    f"{global_deci_mean[level] / n:>10.4f}  "
                    f"{global_deci_min[level] / n:>10.4f}  "
                    f"{global_deci_above[level] / n * 100:>12.2f}  "
                    f"{n:>8}  "
                    f"{global_gain_count[level]:>10}\n"
                )
        f.write("\n")

        f.write("  全局组合熵增益事件统计\n\n")
        f.write(f"  有效增益观测总数: {global_total_gain_count}\n")
        f.write(f"  负组合熵增益数: {global_negative_gain_count}\n")
        f.write(
            "  负组合熵增益占全部有效增益: "
            + (f"{global_negative_gain_count / global_total_gain_count * 100:.4f}%"
               if global_total_gain_count > 0 else "N/A")
            + "\n"
        )
        f.write(
            "  后续恢复为正的负增益数: "
            f"{global_recovered_negative_count}\n"
        )
        f.write(
            "  后续恢复为正的负增益占全部有效增益: "
            + (f"{global_recovered_negative_count / global_total_gain_count * 100:.4f}%"
               if global_total_gain_count > 0 else "N/A")
            + "\n"
        )
        f.write(
            "  负增益条件恢复率: "
            + (f"{global_recovered_negative_count / global_negative_gain_count * 100:.4f}%"
               if global_negative_gain_count > 0 else "N/A")
            + "\n\n"
        )

        # ---------- 二、逐场景统计 ----------
        f.write(sub + "\n")
        f.write("  二、逐场景统计\n")
        f.write(sub + "\n\n")

        for (scene_token, total_frames, valid_keyframes,
             scene_avgs) in per_scene_data:
            f.write(
                f"Scene: {scene_token} (场景共 {total_frames} 帧, "
                f"有效关键帧 {valid_keyframes} 帧)\n\n"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for level in range(N):
                sa = scene_avgs[level]
                if sa is not None:
                    gain_H_text = (
                        f"{sa['gain_H']:>12.4f}"
                        if sa['gain_H'] is not None else f"{'N/A':>12}"
                    )
                    gain_C_text = (
                        f"{sa['gain_combined']:>12.4f}"
                        if sa['gain_combined'] is not None else f"{'N/A':>12}"
                    )
                    f.write(
                        f"{level:>8}  "
                        f"{sa['H']:>10.4f}  "
                        f"{sa['combined']:>10.4f}  "
                        f"{gain_H_text}  "
                        f"{gain_C_text}  "
                        f"{sa['deci_max']:>10.4f}  "
                        f"{sa['deci_mean']:>10.4f}  "
                        f"{sa['deci_min']:>10.4f}  "
                        f"{sa['deci_above'] * 100:>12.2f}  "
                        f"{sa['count']:>8}  "
                        f"{sa['gain_count']:>10}\n"
                    )
            f.write("\n")

    print(f"\n统计完成, 结果已写入: {args.output}")


if __name__ == '__main__':
    main()
