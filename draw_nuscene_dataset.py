import os
#import torch
import pickle
import argparse
import math

import cv2
import numpy as np

from tqdm import tqdm
from open3d_vis_utils import draw_scenes
from model.lifter.spconv_voxelize import VoxelGeneratorWrapper
from dataset.utils import get_img2global, get_lidar2global

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]


voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=[0.5, 0.5, 0.5],
            coors_range_xyz=pc_range,
            num_point_features=4,
            max_num_points_per_voxel=20,
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


def collect_lidar_history_from_frames(frames, frame_index, num_history=5):
    """
    从帧列表中收集历史帧 LiDAR 信息（多进程友好版本）。

    与 collect_lidar_history_from_infos() 功能相同，但直接接收该场景的
    frames 列表而非整个 infos 字典，避免跨进程访问全局 infos。

    Args:
        frames:       该场景的完整帧列表 list[dict]
        frame_index:  当前帧在列表中的索引
        num_history:  最大历史帧数

    Returns:
        list[dict]: 每个元素为 dict(pts_filename=..., lidar_pose=...)
    """
    if num_history <= 0:
        return []

    history = []
    for prev_idx in range(frame_index - 1, -1, -1):
        prev_frame = frames[prev_idx]
        prev_lidar_info = prev_frame.get('data', {}).get('LIDAR_TOP')
        if prev_lidar_info is None:
            continue

        history.append(dict(
            pts_filename=prev_lidar_info['filename'],
            lidar_pose=get_lidar2global(
                prev_lidar_info['calib'],
                prev_lidar_info['pose'],
            ),
        ))
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
# [新增] 场景熵统计
# ===========================================================================

def scene_entropy(points, voxel_generator):
    """
    计算单帧点云的场景熵。
    
    H = -Σ p_i · ln(p_i), p_i = num_points_in_voxel / max_num_points_per_voxel
    规定 0·ln(0)=0，底数 e。
    """
    if points.shape[0] == 0:
        return 0.0

    _, _, num_points = voxel_generator.generate(points)

    if num_points.shape[0] == 0:
        return 0.0

    total_points = num_points.sum()
    probs = num_points.astype(np.float64) / total_points

    valid = probs > 0
    entropy = -np.sum(probs[valid] * np.log(probs[valid]))

    return float(entropy)


# ===========================================================================
# [新增] 局部几何熵 (Local Geometry Entropy) 统计
# ===========================================================================

def compute_voxel_deci(points_in_voxel, vsize=0.5):
    """
    子任务 B: 计算单个体素的 DECI (Differential Entropy-based Compactness Index)。

    基于 deci_solver.md 中的三维 DECI 公式:
      - 将体素内点坐标除以 vsize 归一化到体素网格单位
      - 计算点集的协方差矩阵 Σ (3×3)
      - 特征值分解 → 按秩 r 分类计算微分熵 h
      - DECI = 1/h

    坐标归一化原因:
      LiDAR 点在世界坐标系中 (单位: 米), 0.5m 体素内点的特征值量级
      仅 ~0.01 m², 导致 (2πe)^r · ∏λ ≪ 1, 公式中的 "+1" 反客为主,
      压缩 h 到 ~0.004, DECI 爆炸到 ~200+。
      除以 vsize 后将坐标缩放到体素网格单位, 特征值量级提升,
      使 h 回归合理范围 (0.1~2.0), DECI 回归 0.5~10。

    秩检测策略:
      使用相对阈值 (最大特征值的 1%) 替代绝对阈值 (1e-8), 以正确识别
      planar (r=2) / linear (r=1) 降秩分布; 同时以 min(k-1, 3) 作为
      理论上限, 与 deci_solver.md 中 k=1→r=0, k=2→r=1, k=3→r=2, k≥4→r=3 一致。

    Args:
        points_in_voxel: (k, 3) array — 体素内的点坐标 (世界坐标系, 单位: 米)
        vsize:           体素尺寸 (默认 0.5m)

    Returns:
        deci: float — DECI 值 (h=0 时返回 0.0)
    """
    k = points_in_voxel.shape[0]

    if k <= 1:
        return 0.0

    # 坐标归一化: 从世界坐标(米) → 体素网格单位, 解决 h 过小的问题
    points_normalized = points_in_voxel / vsize

    # 协方差矩阵 (无偏估计)
    mean = points_normalized.mean(axis=0)
    centered = points_normalized - mean

    cov = (centered.T @ centered) / (k - 1)   # (3, 3)

    # 特征值分解 (使用 eigvalsh 因为协方差矩阵是对称的)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]   # 降序
    eigenvalues = np.clip(eigenvalues, 1e-10, None)

    # 有效秩: 相对阈值检测 + 理论上限约束
    # 问题: 绝对阈值 1e-8 对归一化后仍过小, planar/linear 分布的微小 λ 仍被判为满秩
    # 修复: 相对最大特征值的 1% 为阈值, 并结合 k 的理论上限 min(k-1, 3)
    lambda_max = eigenvalues[0]
    r = int(np.sum(eigenvalues > 0.01 * lambda_max))
    r = min(r, min(k - 1, 3))  # 协方差矩阵秩 ≤ min(点数-1, 维度)

    # normalize the eigenvalues
    eigenvalues = eigenvalues / eigenvalues.sum()
    if r == 0:
        return 0.0

    # 微分熵 h = ½ · ln( (2πe)^r · ∏λ_j + 1 )
    prod = np.prod(eigenvalues[:r])
    constant = (2.0 * math.pi * math.e) ** r
    h = 0.5 * math.log(constant * prod + 1.0)

    # DECI, 加上限防止 h 过小时 DECI 爆炸
    if h > 0:
        return float(1.0 / h)
    else:
        return 0.0


def local_geometry_entropy(points, voxel_generator):
    """
    子任务 A+B+C 组合: 计算单帧点云中每个点的局部几何熵 (DECI) 并返回统计信息。

    管线:
      A: voxel_generator.generate() → 获取每体素内的点坐标
      B: 逐体素 compute_voxel_deci()
      C: 体素 DECI 赋给体内所有点 → 统计 max / min / mean

    Args:
        points:         (N, 4) 原始点云 (x, y, z, intensity)
        voxel_generator: VoxelGeneratorWrapper 实例

    Returns:
        stats: dict with keys 'max', 'min', 'mean', 'total_points', 'nonzero_points'
    """
    if points.shape[0] == 0:
        return {'max': 0.0, 'min': 0.0, 'mean': 0.0,
                'total_points': 0, 'nonzero_points': 0,
                'above_mean_ratio': 0.0}

    voxels, coords, num_pts = voxel_generator.generate(points)
    M = coords.shape[0]  # 非空体素数量

    if M == 0:
        return {'max': 0.0, 'min': 0.0, 'mean': 0.0,
                'total_points': 0, 'nonzero_points': 0,
                'above_mean_ratio': 0.0}

    # 处理可能的 batch 维度
    if coords.shape[1] == 4:
        coords = coords[:, 1:]

    all_deci = []

    for i in range(M):
        k = int(num_pts[i])
        pts_xyz = voxels[i, :k, :3]   # 仅取 xyz，丢弃 intensity

        deci = compute_voxel_deci(pts_xyz)
        # 该体素内每个点获得相同的 DECI 值
        all_deci.extend([deci] * k)

    all_deci = np.array(all_deci, dtype=np.float64)

    if len(all_deci) == 0:
        return {'max': 0.0, 'min': 0.0, 'mean': 0.0,
                'total_points': 0, 'nonzero_points': 0,
                'above_mean_ratio': 0.0}

    stats = {
        'max': float(np.max(all_deci)),
        'min': float(np.min(all_deci)),
        'mean': float(np.mean(all_deci)),
        'total_points': len(all_deci),
        'nonzero_points': int(np.sum(all_deci > 0)),
        'above_mean_ratio': float(np.sum(all_deci > np.mean(all_deci)) / len(all_deci)) if len(all_deci) > 0 else 0.0,
    }
    return stats


# ===========================================================================
# [新增] 图像层面场景熵统计
# ===========================================================================

sensor_types = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


def load_frame_images(frame, data_root):
    """
    子任务 A: 读取当前帧 6 视角图像并转为灰度图。

    Args:
        frame:     pkl 中单帧数据字典 frame['data'][cam_type]
        data_root: NuScenes 原始数据根目录

    Returns:
        images: dict[cam_type] → 灰度图 ndarray (H, W) uint8
    """
    images = {}
    for cam_type in sensor_types:
        cam_info = frame['data'].get(cam_type)
        if cam_info is None:
            continue

        img_path = os.path.join(data_root, cam_info['filename'])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [WARNING] 图像读取失败: {img_path}")
            continue

        images[cam_type] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return images


def compute_image_gradient_entropy(img_gray):
    """
    子任务 C: 计算单张灰度图像的梯度熵。

    流程:
      Sobel 梯度 → 256-bin 直方图 → 香农熵 H = -Σ p_i · ln(p_i)

    Args:
        img_gray: (H, W) uint8 灰度图像

    Returns:
        entropy: float
    """
    if img_gray is None or img_gray.size == 0:
        return 0.0

    # Sobel 梯度 (dx + dy)
    gx = cv2.Sobel(img_gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    gmag = np.sqrt(gx ** 2 + gy ** 2)

    # 256-bin 直方图
    hist, _ = np.histogram(gmag, bins=256, range=(0, 255))
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    probs = hist.astype(np.float64) / hist.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_scene_image_entropy(images):
    """
    子任务 C 汇总: 对 6 视角图像计算图像梯度熵并返回统计信息。

    Args:
        images: dict[cam_type] → 灰度图 ndarray

    Returns:
        stats: dict with keys 'per_view', 'max', 'min', 'mean', 'sum', 'valid_views'
    """
    per_view = {}
    for cam_type, img in images.items():
        per_view[cam_type] = compute_image_gradient_entropy(img)

    if len(per_view) == 0:
        return {'per_view': {}, 'max': 0.0, 'min': 0.0,
                'mean': 0.0, 'sum': 0.0, 'valid_views': 0}

    values = list(per_view.values())
    return {
        'per_view': per_view,
        'max': max(values),
        'min': min(values),
        'mean': sum(values) / len(values),
        'sum': sum(values),
        'valid_views': len(values),
    }


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
    parser.add_argument('--scene_token', type=str, default=None,
                        help='指定要可视化的单个场景token。若不指定则遍历所有场景。')

    
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

    infos = load_pkl(args.pkl_path)

    # --- 按 scene_token 过滤 ---
    if args.scene_token is not None:
        if args.scene_token not in infos:
            available = '\n'.join(f'  {t}' for t in list(infos.keys())[:20])
            raise ValueError(
                f"指定的 scene_token 不存在: {args.scene_token}\n"
                f"pkl 中可用的 scene_token (前20个):\n{available}"
            )
        scenes_to_process = {args.scene_token: infos[args.scene_token]}
        print(f"仅可视化指定场景: {args.scene_token}")
    else:
        scenes_to_process = infos

    for scene_token, frames in tqdm(scenes_to_process.items()):
        # [新增] 场景级熵统计累加器
        scene_entropy_sum = 0.0
        scene_entropy_with_history_sum = 0.0
        scene_frame_count = 0

        # [新增] 场景级局部几何熵统计累加器
        scene_deci_max_sum = 0.0
        scene_deci_min_sum = 0.0
        scene_deci_mean_sum = 0.0
        scene_deci_above_ratio_sum = 0.0

        merged_scene_deci_max_sum = 0.0
        merged_scene_deci_min_sum = 0.0
        merged_scene_deci_mean_sum = 0.0
        merged_scene_deci_above_ratio_sum = 0.0

        # [新增] 场景级图像梯度熵统计累加器
        scene_img_entropy_sum = 0.0
        scene_img_entropy_valid_frames = 0

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

            # [新增] 计算并输出当前帧的场景熵
            cur_entropy = scene_entropy(current_points, voxel_generator)
            scene_entropy_sum += cur_entropy
            scene_frame_count += 1
            print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] Entropy = {cur_entropy:.4f}")

            # [新增] 计算并输出当前帧的局部几何熵 (DECI)
            deci_stats = local_geometry_entropy(current_points, voxel_generator)
            scene_deci_max_sum += deci_stats['max']
            scene_deci_min_sum += deci_stats['min']
            scene_deci_mean_sum += deci_stats['mean']
            scene_deci_above_ratio_sum += deci_stats['above_mean_ratio']
            print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] "
                  f"Local Geometry Entropy — max={deci_stats['max']:.4f}  "
                  f"min={deci_stats['min']:.4f}  mean={deci_stats['mean']:.4f}  "
                  f"above_ratio={deci_stats['above_mean_ratio']:.4%}  "
                  f"(points={deci_stats['total_points']}, nonzero={deci_stats['nonzero_points']})")

            # ============================================================
            # [保留] 2. 获取当前帧的 lidar2global 位姿
            # ============================================================
            current_lidar_pose = get_lidar2global(
                lidar_info['calib'],
                lidar_info['pose'],
            )

            # ============================================================
            # [新增] 子任务 A+B+C: 图像读取 + 变换矩阵 + 图像梯度熵
            # ============================================================
            images = load_frame_images(frame, args.data_root)
            if len(images) > 0:
                print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] "
                      f"已加载 {len(images)} 张图像")

                # 计算 lidar2img 变换矩阵 (可选打印验证)
                lidar2img_dict = {}
                for cam_type in sensor_types:
                    cam_info = frame['data'].get(cam_type)
                    if cam_info is None or cam_type not in images:
                        continue
                    img2global = get_img2global(cam_info['calib'], cam_info['pose'])
                    lidar2img_dict[cam_type] = np.linalg.inv(img2global) @ current_lidar_pose

                # 计算图像梯度熵
                img_entropy_stats = compute_scene_image_entropy(images)
                scene_img_entropy_sum += img_entropy_stats['mean']
                scene_img_entropy_valid_frames += 1

                print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] "
                      f"Image Gradient Entropy (6 views):")
                for cam_type, h in img_entropy_stats['per_view'].items():
                    print(f"    {cam_type:16s} = {h:.4f}")
                print(f"    {'─' * 30}")
                print(f"    max = {img_entropy_stats['max']:.4f}  "
                      f"min = {img_entropy_stats['min']:.4f}  "
                      f"mean = {img_entropy_stats['mean']:.4f}  "
                      f"sum = {img_entropy_stats['sum']:.4f}")
            else:
                print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] "
                      f"[WARNING] 未加载到任何图像，跳过图像梯度熵计算")

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

            # [新增] 计算叠加历史帧后的场景熵
            merged_points = np.concatenate(all_points_list, axis=0)
            merged_entropy = scene_entropy(merged_points, voxel_generator)
            scene_entropy_with_history_sum += merged_entropy
            print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] "
                  f"Entropy (with history) = {merged_entropy:.4f}  "
                  f"(history_frames={len(history_list)})")
            
            merged_deci_stats = local_geometry_entropy(merged_points, voxel_generator)
            merged_scene_deci_max_sum += merged_deci_stats['max']
            merged_scene_deci_min_sum += merged_deci_stats['min']
            merged_scene_deci_mean_sum += merged_deci_stats['mean']
            merged_scene_deci_above_ratio_sum += merged_deci_stats['above_mean_ratio']
            print(f"[Scene: {scene_token[:12]}... | Frame: {frame_idx}] "
                  f"Local Geometry Entropy-with-history — max={merged_deci_stats['max']:.4f}  "
                  f"min={merged_deci_stats['min']:.4f}  mean={merged_deci_stats['mean']:.4f}  "
                  f"above_ratio={merged_deci_stats['above_mean_ratio']:.4%}  "
                  f"(points={merged_deci_stats['total_points']}, nonzero={merged_deci_stats['nonzero_points']})")



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



        # [新增] 场景切换时输出平均熵
        if scene_frame_count > 0:
            avg_entropy = scene_entropy_sum / scene_frame_count
            avg_merged_entropy = scene_entropy_with_history_sum / scene_frame_count
            avg_deci_max = scene_deci_max_sum / scene_frame_count
            avg_deci_min = scene_deci_min_sum / scene_frame_count
            avg_deci_mean = scene_deci_mean_sum / scene_frame_count
            avg_deci_above_ratio = scene_deci_above_ratio_sum / scene_frame_count
            avg_merged_deci_max = merged_scene_deci_max_sum / scene_frame_count
            avg_merged_deci_min = merged_scene_deci_min_sum / scene_frame_count
            avg_merged_deci_mean = merged_scene_deci_mean_sum / scene_frame_count
            avg_merged_deci_above_ratio = merged_scene_deci_above_ratio_sum / scene_frame_count
            print(f"{'='*60}")
            print(f"[Scene: {scene_token[:12]}...] 平均场景熵 = {avg_entropy:.4f} "
                  f"(共 {scene_frame_count} 帧)")
            print(f"[Scene: {scene_token[:12]}...] 叠加历史帧后平均场景熵 = {avg_merged_entropy:.4f}")
            print(f"[Scene: {scene_token[:12]}...] 平均局部几何熵 (DECI): "
                  f"max={avg_deci_max:.4f}  min={avg_deci_min:.4f}  mean={avg_deci_mean:.4f}  "
                  f"above_ratio={avg_deci_above_ratio:.4%}")
            print(f"[Scene: {scene_token[:12]}...] 叠加历史帧后平均局部几何熵 (DECI): "
                  f"max={avg_merged_deci_max:.4f}  min={avg_merged_deci_min:.4f}  "
                  f"mean={avg_merged_deci_mean:.4f}  above_ratio={avg_merged_deci_above_ratio:.4%}")
            # [新增] 场景级图像梯度熵均值输出
            if scene_img_entropy_valid_frames > 0:
                avg_img_entropy = scene_img_entropy_sum / scene_img_entropy_valid_frames
                print(f"[Scene: {scene_token[:12]}...] 平均图像梯度熵 = {avg_img_entropy:.4f} "
                      f"(共 {scene_img_entropy_valid_frames} 帧)")
            print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
