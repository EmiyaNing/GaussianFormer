#!/usr/bin/env python3
"""
dataset_analysis.py — NuScenes 数据集场景融合熵统计脚本

从融合 0 帧逐步统计到融合 16 帧的:
  - 平均场景熵 (基于体素点分布的香农熵)
  - 平均局部 DECI 熵 (基于体素内点云几何紧凑度的微分熵)

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
    collect_lidar_history_from_infos,
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


def main():
    parser = argparse.ArgumentParser(description='NuScenes 场景融合熵统计')
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
        '--output',
        default='entropy_result.txt',
        help='输出文件路径',
    )
    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"pkl 文件不存在: {args.pkl_path}")

    vg = create_voxel_generator()
    infos = load_pkl(args.pkl_path)

    max_fusion = args.num_history
    N = max_fusion + 1  # 融合级别数: 0 ~ max_fusion

    # ------------------------------------------------------------------
    # 全局累加器  (按融合级别索引)
    # ------------------------------------------------------------------
    global_H = np.zeros(N)
    global_deci_max = np.zeros(N)
    global_deci_min = np.zeros(N)
    global_deci_mean = np.zeros(N)
    global_deci_above = np.zeros(N)
    global_count = np.zeros(N, dtype=int)

    # ------------------------------------------------------------------
    # 逐场景累加器
    # ------------------------------------------------------------------
    per_scene_data = []  # list of (scene_token, total_frames, scene_avgs)

    for scene_token, frames in tqdm(infos.items(), desc='Scenes'):
        scene_H = np.zeros(N)
        scene_deci_max = np.zeros(N)
        scene_deci_min = np.zeros(N)
        scene_deci_mean = np.zeros(N)
        scene_deci_above = np.zeros(N)
        scene_count = np.zeros(N, dtype=int)

        for frame_idx, frame in enumerate(frames):
            if 'LIDAR_TOP' not in frame.get('data', {}):
                continue

            # ------------------------------------------------------------
            # 1. 加载当前帧点云
            # ------------------------------------------------------------
            lidar_info = frame['data']['LIDAR_TOP']
            lidar_path = os.path.join(args.data_root, lidar_info['filename'])
            current_points = load_lidar_points(lidar_path)
            current_points = filter_points_by_range(current_points, PC_RANGE)

            current_pose = get_lidar2global(
                lidar_info['calib'], lidar_info['pose']
            )

            # ------------------------------------------------------------
            # 2. 收集历史帧
            # ------------------------------------------------------------
            history_list = collect_lidar_history_from_infos(
                infos, scene_token, frame_idx, num_history=max_fusion
            )

            # ------------------------------------------------------------
            # 3. 渐进式融合: 0 帧 → min(max_fusion, len(history_list)) 帧
            # ------------------------------------------------------------
            accumulated_pts = current_points.copy()
            actual_levels = min(max_fusion, len(history_list)) + 1

            for level in range(actual_levels):
                if level == 0:
                    merged = current_points
                else:
                    sweep = history_list[level - 1]
                    sweep_path = os.path.join(
                        args.data_root, sweep['pts_filename']
                    )
                    sweep_points = load_lidar_points(sweep_path)
                    sweep_points = transform_points_to_target(
                        sweep_points, sweep['lidar_pose'], current_pose
                    )
                    sweep_points = filter_points_by_range(
                        sweep_points, PC_RANGE
                    )
                    accumulated_pts = np.concatenate(
                        [accumulated_pts, sweep_points], axis=0
                    )
                    merged = accumulated_pts

                # --------------------------------------------------------
                # 4. 双路熵计算
                # --------------------------------------------------------
                stats = compute_both_entropies(merged, vg)

                # 场景级累加
                scene_H[level] += stats['H']
                scene_deci_max[level] += stats['deci_max']
                scene_deci_min[level] += stats['deci_min']
                scene_deci_mean[level] += stats['deci_mean']
                scene_deci_above[level] += stats['deci_above']
                scene_count[level] += 1

                # 全局累加
                global_H[level] += stats['H']
                global_deci_max[level] += stats['deci_max']
                global_deci_min[level] += stats['deci_min']
                global_deci_mean[level] += stats['deci_mean']
                global_deci_above[level] += stats['deci_above']
                global_count[level] += 1

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

        per_scene_data.append((scene_token, len(frames), scene_avgs))

    # ==================================================================
    # 写入结果文件
    # ==================================================================
    with open(args.output, 'w', encoding='utf-8') as f:
        sep = "=" * 80
        sub = "-" * 80

        f.write(sep + "\n")
        f.write("  NuScenes 数据集场景融合熵统计结果\n")
        f.write(f"  pkl 路径: {args.pkl_path}\n")
        f.write(f"  数据根目录: {args.data_root}\n")
        f.write(f"  体素配置: vsize={VOXEL_SIZE}, max_pts={MAX_POINTS_PER_VOXEL}\n")
        f.write(f"  最大融合帧数: {max_fusion}\n")
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
