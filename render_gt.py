#!/usr/bin/env python3
"""
根据surroundocc数据集的occ标注生成固定尺寸的3D高斯球，并使用gsplat渲染6个视角的surround-view语义图像。
此脚本读取./data/nuscenes_cam中的pkl文件，根据每个相机矩阵对应的场景名称加载./data/surroundocc中的occ gt标签。
颜色映射参考vis_surround_data.py中的colors_map。
"""

import os
import numpy as np
import torch
import gsplat
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion

# 颜色映射（从vis_surround_data.py复制）
colors_map = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            [139, 137, 137, 255],       # other_flat
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
        ]
    ).astype(np.float32) / 255.

# 只取RGB（去掉Alpha）
colors_map_rgb = colors_map[:, :3]

def load_surroundocc_data(file_path):
    """加载surroundocc的.npy文件，返回世界坐标和语义标签"""
    data = np.load(file_path)  # (N, 4)
    voxel_coords = data[:, :3]  # 体素索引
    sem_labels = data[:, 3].astype(int)  # 语义标签 (0-16)
    # 转换为世界坐标 (与vis_surround_data.py相同)
    world_coords = voxel_coords * 0.5 - np.array([50.0, 50.0, 5.0])
    return world_coords, sem_labels

lidar_top = None

def get_camera_params_from_frame(frame):
    """
    从单个帧信息中提取六个相机的视图矩阵、内参矩阵、图像尺寸。
    使用 dataset/dataset.py 中的坐标变换逻辑，计算 lidar2cam 矩阵（LiDAR坐标系到相机坐标系）。
    返回 viewmats (lidar2cam), Ks, W, H, cam_names, cam_filenames
    """
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    viewmats = []
    Ks = []
    out_cam_names = []
    out_cam_filenames = []
    W = None
    H = None
    
    # 计算 LiDAR 到全局的变换矩阵（所有相机共享）
    lidar_calib = frame['data']['LIDAR_TOP']['calib']
    lidar_pose = frame['data']['LIDAR_TOP']['pose']
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(lidar_calib['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.array(lidar_calib['translation'], dtype=np.float32).flatten()
    ego2global_lidar = np.eye(4)
    ego2global_lidar[:3, :3] = Quaternion(lidar_pose['rotation']).rotation_matrix
    ego2global_lidar[:3, 3] = np.array(lidar_pose['translation'], dtype=np.float32).flatten()
    lidar2global = ego2global_lidar @ lidar2ego
    
    for cam_name in cam_names:
        cam_info = frame['data'][cam_name]
        calib = cam_info['calib']
        pose = cam_info['pose']
        
        # 内参矩阵
        intrinsic = np.array(calib['camera_intrinsic'], dtype=np.float32)
        # 图像尺寸
        if W is None:
            W = cam_info['width']
            H = cam_info['height']
        
        # 计算相机到 ego 的变换矩阵 cam2ego
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = Quaternion(calib['rotation']).rotation_matrix
        cam2ego[:3, 3] = np.array(calib['translation'], dtype=np.float32).flatten()
        
        # 计算 ego 到全局的变换矩阵 ego2global
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(pose['rotation']).rotation_matrix
        ego2global[:3, 3] = np.array(pose['translation'], dtype=np.float32).flatten()
        
        # 相机到全局的变换矩阵 cam2global = ego2global @ cam2ego
        cam2global = ego2global @ cam2ego
        # LiDAR 到相机坐标系的变换矩阵 lidar2cam = inv(cam2global) @ lidar2global
        lidar2cam = np.linalg.inv(cam2global) @ lidar2global
        viewmats.append(lidar2cam)
        
        # 内参矩阵转为 3x3
        K = np.eye(3)
        K[:2, :3] = intrinsic[:2, :3]  # 可能 intrinsic 已经是 3x3
        Ks.append(K)
        
        # 保存相机名称和文件名
        out_cam_names.append(cam_name)
        out_cam_filenames.append(cam_info['filename'])
    
    # 转换为 torch 张量
    viewmats = torch.from_numpy(np.stack(viewmats, axis=0)).float()
    Ks = torch.from_numpy(np.stack(Ks, axis=0)).float()
    return viewmats, Ks, W, H, out_cam_names, out_cam_filenames

def create_gaussians(world_coords, sem_labels, fixed_scale=0.1):
    """
    根据坐标和语义标签创建高斯球参数。
    返回torch张量：means, quats, scales, opacities, colors
    """
    n = len(world_coords)
    # 位置 (N, 3)
    means = torch.from_numpy(world_coords).float()
    # 旋转：单位四元数 (w, x, y, z)
    quats = torch.tensor([1., 0., 0., 0.], dtype=torch.float).repeat(n, 1)
    # 尺度：固定尺寸，各向同性 (N, 3)
    scales = torch.full((n, 3), fixed_scale, dtype=torch.float)
    # 不透明度：固定为1.0 (N,)
    opacities = torch.ones(n, dtype=torch.float)
    # 颜色：根据语义标签选择RGB (N, 3)
    colors = torch.from_numpy(colors_map_rgb[sem_labels]).float()
    return means, quats, scales, opacities, colors

def render_one_frame(frame, occ_data_root, output_dir, fixed_scale=0.1, rgb_root='./data/nuscenes'):
    """渲染一个帧的6个视角并保存图像，按相机文件夹组织，并复制对应的RGB图像"""
    # 检查是否存在LIDAR_TOP数据
    if 'LIDAR_TOP' not in frame['data']:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    # 提取LiDAR文件名
    lidar_filename = frame['data']['LIDAR_TOP']['filename']  # 如 'samples/LIDAR_TOP/n015-...pcd.bin'
    # 获取基本名称（去掉目录）
    lidar_basename = os.path.basename(lidar_filename)  # 'n015-...pcd.bin'
    # 对应的occ标签文件路径
    occ_file = os.path.join(occ_data_root, lidar_basename + '.npy')
    if not os.path.exists(occ_file):
        print(f"Warning: occ file {occ_file} does not exist, skipping.")
        return
    
    # 加载occ数据
    world_coords, sem_labels = load_surroundocc_data(occ_file)
    
    # 创建高斯参数
    means, quats, scales, opacities, colors = create_gaussians(world_coords, sem_labels, fixed_scale)
    
    # 从帧中获取相机参数（包括相机名称和文件名）
    viewmats, Ks, W, H, cam_names, cam_filenames = get_camera_params_from_frame(frame)
    
    # 将数据移到GPU（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    means = means.to(device)
    quats = quats.to(device)
    scales = scales.to(device)
    opacities = opacities.to(device)
    colors = colors.to(device)
    viewmats = viewmats.to(device)
    Ks = Ks.to(device)
    bg_color = torch.zeros(3, device=device)
    
    # 使用传入的RGB根目录
    
    # 逐个相机渲染
    for cam_idx in range(viewmats.shape[0]):
        cam_name = cam_names[cam_idx]
        cam_filename = cam_filenames[cam_idx]  # 如 'samples/CAM_FRONT/...jpg'
        # 提取基本文件名（包含扩展名）
        basename = os.path.basename(cam_filename)  # 'n015-...__CAM_FRONT__...jpg'
        # 将扩展名改为 .png 用于渲染输出
        name_without_ext = os.path.splitext(basename)[0]
        output_filename = name_without_ext + '.png'
        # 创建相机文件夹
        cam_dir = os.path.join(output_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        output_path = os.path.join(cam_dir, output_filename)
        
        # 选取当前相机的视图矩阵和内参
        viewmat = viewmats[cam_idx].unsqueeze(0)  # (1,4,4)
        K = Ks[cam_idx].unsqueeze(0)  # (1,3,3)
        render, _, _ = gsplat.rasterization(
            means, quats, scales, opacities, colors, viewmat, K, W, H, backgrounds=bg_color
        )
        render = render[0]
        render_np = render.detach().cpu().numpy()  # (H, W, 3)
        # 保存渲染图像为PNG
        plt.imsave(output_path, np.clip(render_np, 0, 1))
        #print(f"Saved camera {cam_name} image to {output_path}")
        
        # 复制对应的原始RGB图像（如果存在）
        rgb_source = os.path.join(rgb_root, cam_filename)
        if os.path.exists(rgb_source):
            rgb_dest = os.path.join(cam_dir, basename)  # 保持原始文件名（.jpg）
            import shutil
            shutil.copy2(rgb_source, rgb_dest)
            #print(f"Copied RGB image to {rgb_dest}")
        #else:
            #print(f"Warning: RGB source {rgb_source} does not exist, skipping copy.")
    
    #print(f"All images saved to {output_dir}")

def load_pkl(pkl_path):
    """加载pkl文件，返回infos字典"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', default='./data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl',
                        help='相机参数pkl文件路径')
    parser.add_argument('--occ_data_root', default='./data/surroundocc/samples',
                        help='surroundocc数据目录，包含.npy文件')
    parser.add_argument('--output_root', default='./rendered_gt',
                        help='输出图像目录')
    parser.add_argument('--fixed_scale', type=float, default=0.3,
                        help='高斯球的固定尺度（米）')
    parser.add_argument('--scene', default=None,
                        help='指定场景token（不指定则处理所有场景）')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='每个场景最多处理的帧数（默认处理所有帧）')
    parser.add_argument('--rgb_root', default='./data/nuscenes',
                        help='原始RGB图像的根目录（默认./data/nuscenes）')
    args = parser.parse_args()
    
    # 检查pkl文件是否存在
    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"Camera pkl file not found: {args.pkl_path}")
    
    # 加载pkl
    infos = load_pkl(args.pkl_path)
    #print(f"Loaded {len(infos)} scenes")
    
    # 遍历场景
    for scene_token, frames in tqdm(infos.items()):
        if args.scene and scene_token != args.scene:
            continue
        # 限制每个场景的帧数
        if args.max_frames is not None:
            frames = frames[:args.max_frames]
        for frame_idx, frame in enumerate(frames):
            # 创建输出目录：output_root/scene_token/frame_idx
            output_dir = os.path.join(args.output_root, scene_token, f'frame_{frame_idx:03d}')
            
            render_one_frame(frame, args.occ_data_root, output_dir, args.fixed_scale, rgb_root=args.rgb_root)
            

if __name__ == '__main__':
    main()