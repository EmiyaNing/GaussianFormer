import torch
import numpy as np
import sys
import os

# 添加路径以便导入lidar_processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lidar_processor import LidarVoxelProcessor

def simple_test():
    """简单测试LidarVoxelProcessor的基本功能"""
    print("=== 简单测试LiDAR处理器 ===")
    
    # 配置参数
    pc_range = [-50, -50, -5, 50, 50, 3]  # 点云范围
    voxel_size = 0.5  # 体素大小
    
    # 初始化处理器
    processor = LidarVoxelProcessor(
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_points_per_voxel=20,
        num_sweeps=5  # 增加到5帧融合
    )
    
    print(f"处理器初始化完成")
    
    # 生成模拟点云数据
    torch.manual_seed(42)  # 固定随机种子以便重现
    
    # 生成更多的点进行测试
    num_points = 2000  # 增加到2000个点
    points = torch.rand(num_points, 3) * torch.tensor([100, 100, 8]) + torch.tensor([-50, -50, -5])
    
    # 创建当前帧位姿（单位矩阵）
    current_pose = torch.eye(4)
    
    print(f"模拟点云形状: {points.shape}")
    print(f"点云范围: X[{points[:,0].min():.1f}, {points[:,0].max():.1f}], "
          f"Y[{points[:,1].min():.1f}, {points[:,1].max():.1f}], "
          f"Z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
    
    # 测试1: 坐标系变换
    print("\n=== 测试坐标系变换 ===")
    test_points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    
    # 创建变换矩阵：平移(1, 1, 1)
    source_pose = torch.eye(4)
    target_pose = torch.eye(4)
    target_pose[:3, 3] = torch.tensor([1.0, 1.0, 1.0])
    
    transformed_points = processor.transform_points_between_frames(
        test_points, source_pose, target_pose
    )
    
    print(f"原始点: {test_points.tolist()}")
    print(f"变换后点: {transformed_points.tolist()}")
    
    # 验证变换正确性（应该平移(-1, -1, -1)）
    expected_points = test_points - torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(transformed_points, expected_points, atol=1e-6)
    print("✅ 坐标系变换测试通过!")
    
    # 测试2: 体素化
    print("\n=== 测试体素化 ===")
    voxel_centers, point_counts, voxel_coords = processor.voxelize(points)
    
    print(f"非空体素数量: {len(voxel_centers)}")
    print(f"体素中心形状: {voxel_centers.shape}")
    print(f"体素密度形状: {point_counts.shape}")
    print(f"体素坐标形状: {voxel_coords.shape}")
    
    # 验证体素化结果合理性
    assert len(voxel_centers) > 0, "应该至少有一个非空体素"
    assert torch.all(point_counts > 0), "每个体素应该至少包含一个点"
    assert torch.all(voxel_centers[:, 0] >= pc_range[0]), "体素中心应在指定范围内"
    assert torch.all(voxel_centers[:, 0] <= pc_range[3]), "体素中心应在指定范围内"
    
    # 显示前几个体素的信息
    print("\n前5个体素信息:")
    for i in range(min(5, len(voxel_centers))):
        print(f"体素 {i}: 中心={voxel_centers[i].tolist()}, "
              f"点数={point_counts[i]}")
    
    # 测试3: 多帧融合（使用更多帧和点）
    print("\n=== 测试多帧融合 ===")
    
    # 创建多个模拟的前一帧点云
    previous_points_list = []
    previous_poses_list = []
    
    # 创建4个前一帧（加上当前帧共5帧）
    for i in range(4):
        # 创建不同的变换矩阵
        transform = torch.eye(4)
        transform[0, 3] = -0.5 * (i + 1)  # 在x轴上移动
        transform[1, 3] = 0.1 * (i + 1)   # 在y轴上移动
        
        # 变换点云（使用不同的随机点云）
        prev_points = torch.rand(num_points, 3) * torch.tensor([100, 100, 8]) + torch.tensor([-50, -50, -5])
        points_homo = torch.cat([prev_points, torch.ones(prev_points.shape[0], 1)], dim=1)
        prev_points_transformed = (transform @ points_homo.T).T[:, :3]
        
        previous_points_list.append(prev_points_transformed)
        previous_poses_list.append(transform)
    
    # 打印融合前各帧的点数
    print("融合前各帧点数:")
    print(f"当前帧: {points.shape[0]} 点")
    for i, prev_points in enumerate(previous_points_list):
        print(f"前一帧 {i+1}: {prev_points.shape[0]} 点")
    
    # 测试多帧融合
    fused_points = processor.fuse_multisweep_lidar(
        points, current_pose, previous_points_list, previous_poses_list
    )
    
    print(f"\n融合后点云形状: {fused_points.shape}")
    print(f"融合后点数: {fused_points.shape[0]}")
    
    # 验证融合结果
    expected_total_points = points.shape[0] + sum([p.shape[0] for p in previous_points_list[:processor.num_sweeps-1]])
    assert fused_points.shape[0] == expected_total_points, f"融合后点云点数不正确"
    print("✅ 多帧融合测试通过!")
    
    # 测试4: 完整前向传播
    print("\n=== 测试完整前向传播 ===")
    results = processor(
        current_lidar=points,
        current_pose=current_pose,
        previous_lidars=previous_points_list,
        previous_poses=previous_poses_list
    )
    
    print("处理结果:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # 验证结果完整性
    required_keys = ['voxel_centers', 'point_counts', 'voxel_coords', 'num_voxels']
    for key in required_keys:
        assert key in results, f"结果中应包含{key}"
    
    print("\n✅ 所有基本功能测试通过! LiDAR处理器工作正常")
    
    # 高斯初始化参数建议
    print("\n=== 高斯初始化参数建议 ===")
    print(f"初始高斯数量: {results['num_voxels']}")
    print("体素几何中心 → 高斯均值 (mean)")
    print("体素尺寸 → 高斯尺度 (scale)")
    print("体素点密度 → 高斯不透明度 (opacity)")

if __name__ == "__main__":
    simple_test()
