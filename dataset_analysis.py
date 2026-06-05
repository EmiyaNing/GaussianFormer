#!/usr/bin/env python3
"""
dataset_analysis.py — NuScenes 数据集场景融合熵统计脚本 (多进程版本)

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
import numpy as np
from tqdm import tqdm

from model.lifter.spconv_voxelize import VoxelGeneratorWrapper
from draw_nuscene_dataset import (
    load_pkl,
    load_lidar_points,
    filter_points_by_range,
    collect_lidar_history_from_frames,
    transform_points_to_target,
    scene_entropy,
    local_geometry_entropy,
)
from dataset.utils import get_lidar2global

# ===========================================================================
# 全局配置 (与 draw_nuscene_dataset.py 保持一致)
# ===========================================================================

PC_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
VOXEL_SIZE = [0.5, 0.5, 0.5]
MAX_POINTS_PER_VOXEL = 20
MAX_VOXELS = 1600000


def create_voxel_generator():
    """创建与现有代码参数一致的体素生成器"""
    return VoxelGeneratorWrapper(
        vsize_xyz=VOXEL_SIZE,
        coors_range_xyz=PC_RANGE,
        num_point_features=4,
        max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
        max_num_voxels=MAX_VOXELS,
    )


def compute_both_entropies(points, vg):
    """
    对单组点云计算场景熵 + 局部 DECI 熵统计。

    Returns:
        dict: {'H': float, 'deci_max': float, 'deci_min': float,
               'deci_mean': float, 'deci_above': float}
    """
    H = scene_entropy(points, vg)
    deci = local_geometry_entropy(points, vg)
    return {
        'H': H,
        'deci_max': deci['max'],
        'deci_min': deci['min'],
        'deci_mean': deci['mean'],
        'deci_above': deci['above_mean_ratio'],
    }


# ===========================================================================
# Worker 入口函数 (模块级, 可被 pickle 序列化)
# ===========================================================================

def process_single_scene(args):
    """
    多进程 worker 入口: 处理单个场景的全部帧和融合级别。

    每个 worker 独立创建 VoxelGeneratorWrapper, 因为 spconv/cumm 内部
    的 C++ 对象不可跨进程 pickle 序列化。

    Args:
        args: tuple of (scene_token, frames, data_root, max_fusion, N,
                         pc_range, voxel_size, max_pts_per_voxel, max_voxels)

    Returns:
        tuple: (scene_token, total_frames, scene_avgs,
                scene_H, scene_deci_max, scene_deci_min,
                scene_deci_mean, scene_deci_above, scene_count)
    """
    (scene_token, frames, data_root, max_fusion, N,
     pc_range, voxel_size, max_pts_per_voxel, max_voxels) = args

    # 每个 worker 独立创建 VoxelGenerator
    vg = VoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=pc_range,
        num_point_features=4,
        max_num_points_per_voxel=max_pts_per_voxel,
        max_num_voxels=max_voxels,
    )

    # 场景级累加器
    scene_H = np.zeros(N)
    scene_deci_max = np.zeros(N)
    scene_deci_min = np.zeros(N)
    scene_deci_mean = np.zeros(N)
    scene_deci_above = np.zeros(N)
    scene_count = np.zeros(N, dtype=int)

    for frame_idx, frame in enumerate(frames):
        if 'LIDAR_TOP' not in frame.get('data', {}):
            continue

        # ----------------------------------------------------------------
        # 1. 加载当前帧点云
        # ----------------------------------------------------------------
        lidar_info = frame['data']['LIDAR_TOP']
        lidar_path = os.path.join(data_root, lidar_info['filename'])
        current_points = load_lidar_points(lidar_path)
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
        accumulated_pts = current_points.copy()
        actual_levels = min(max_fusion, len(history_list)) + 1

        for level in range(actual_levels):
            if level == 0:
                merged = current_points
            else:
                sweep = history_list[level - 1]
                sweep_path = os.path.join(
                    data_root, sweep['pts_filename']
                )
                sweep_points = load_lidar_points(sweep_path)
                sweep_points = transform_points_to_target(
                    sweep_points, sweep['lidar_pose'], current_pose
                )
                sweep_points = filter_points_by_range(
                    sweep_points, pc_range
                )
                accumulated_pts = np.concatenate(
                    [accumulated_pts, sweep_points], axis=0
                )
                merged = accumulated_pts

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
            scene_count[level] += 1

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
            })
        else:
            scene_avgs.append(None)

    return (scene_token, len(frames), scene_avgs,
            scene_H, scene_deci_max, scene_deci_min,
            scene_deci_mean, scene_deci_above, scene_count)


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='NuScenes 场景融合熵统计 (多进程)')
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
        help='并行 worker 进程数 (默认: CPU 核心数)',
    )
    parser.add_argument(
        '--output',
        default='entropy_result.txt',
        help='输出文件路径',
    )
    args = parser.parse_args()

    import multiprocessing as mp

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    elif args.num_workers < 1:
        args.num_workers = 1

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
         PC_RANGE, VOXEL_SIZE, MAX_POINTS_PER_VOXEL, MAX_VOXELS)
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
    global_count = np.zeros(N, dtype=int)
    per_scene_data = []  # list of (scene_token, total_frames, scene_avgs)

    # ------------------------------------------------------------------
    # 4. 多进程执行
    # ------------------------------------------------------------------
    print(f"\nDispatching {num_scenes} scenes to {args.num_workers} workers...")
    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_scene, tasks),
            total=len(tasks),
            desc='Scenes (parallel)',
        ))

        for result in results:
            (scene_token, total_frames, scene_avgs,
             s_H, s_dmax, s_dmin, s_dmean, s_dabove, s_count) = result

            per_scene_data.append((scene_token, total_frames, scene_avgs))

            # 合并全局累加
            global_H += s_H
            global_deci_max += s_dmax
            global_deci_min += s_dmin
            global_deci_mean += s_dmean
            global_deci_above += s_dabove
            global_count += s_count

    # ==================================================================
    # 5. 写入结果文件
    # ==================================================================
    with open(args.output, 'w', encoding='utf-8') as f:
        sep = "=" * 80
        sub = "-" * 80

        f.write(sep + "\n")
        f.write("  NuScenes 数据集场景融合熵统计结果 (多进程)\n")
        f.write(f"  pkl 路径: {args.pkl_path}\n")
        f.write(f"  数据根目录: {args.data_root}\n")
        f.write(f"  体素配置: vsize={VOXEL_SIZE}, max_pts={MAX_POINTS_PER_VOXEL}\n")
        f.write(f"  最大融合帧数: {max_fusion}\n")
        f.write(f"  Worker 数量: {args.num_workers}\n")
        f.write(sep + "\n\n")

        # ---------- 一、全局统计 ----------
        f.write(sub + "\n")
        f.write("  一、全局统计 (所有场景平均)\n")
        f.write(sub + "\n\n")

        header = (
            f"{'融合帧数':>8}  {'场景熵':>10}  {'DECI_max':>10}  "
            f"{'DECI_mean':>10}  {'DECI_min':>10}  {'超均值比(%)':>12}  "
            f"{'有效帧数':>8}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for level in range(N):
            if global_count[level] > 0:
                n = global_count[level]
                f.write(
                    f"{level:>8}  "
                    f"{global_H[level] / n:>10.4f}  "
                    f"{global_deci_max[level] / n:>10.4f}  "
                    f"{global_deci_mean[level] / n:>10.4f}  "
                    f"{global_deci_min[level] / n:>10.4f}  "
                    f"{global_deci_above[level] / n * 100:>12.2f}  "
                    f"{n:>8}\n"
                )
        f.write("\n")

        # ---------- 二、逐场景统计 ----------
        f.write(sub + "\n")
        f.write("  二、逐场景统计\n")
        f.write(sub + "\n\n")

        for scene_token, total_frames, scene_avgs in per_scene_data:
            f.write(f"Scene: {scene_token} (场景共 {total_frames} 帧)\n\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for level in range(N):
                sa = scene_avgs[level]
                if sa is not None:
                    f.write(
                        f"{level:>8}  "
                        f"{sa['H']:>10.4f}  "
                        f"{sa['deci_max']:>10.4f}  "
                        f"{sa['deci_mean']:>10.4f}  "
                        f"{sa['deci_min']:>10.4f}  "
                        f"{sa['deci_above'] * 100:>12.2f}  "
                        f"{sa['count']:>8}\n"
                    )
            f.write("\n")

    print(f"\n统计完成, 结果已写入: {args.output}")


if __name__ == '__main__':
    main()
