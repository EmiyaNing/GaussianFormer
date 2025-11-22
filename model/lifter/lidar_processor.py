import torch
import torch.nn as nn
import numpy as np

class MeanVFE(nn.Module):
    def __init__(self, num_point_features, **kwargs):
        super().__init__()
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict


class LidarVoxelProcessor(nn.Module):
    def __init__(self, pc_range, lidar_voxel_size, max_points_per_voxel=20, num_sweeps=3):
        super().__init__()
        self.register_buffer('pc_range', torch.tensor(pc_range))
        self.lidar_voxel_size = lidar_voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.num_sweeps = num_sweeps
        
        
    def transform_points_between_frames(self, points, source_pose, target_pose):
        """将点云从源坐标系转换到目标坐标系"""
        transform = torch.inverse(target_pose) @ source_pose
        
        points_homo = torch.cat([
            points[:, :3],  # 只转换坐标，保留强度等信息
            torch.ones(points.shape[0], 1, device=points.device)
        ], dim=1)
        points_transformed = (transform @ points_homo.T).T
        points_transformed = torch.cat([points_transformed[:, :3], points[:, 3:]], dim=1)
        return points_transformed
    

    def fuse_multisweep_lidar(self, current_points, current_pose, previous_lidars=None, previous_poses=None):
        """融合多帧LiDAR点云"""
        fused_points = [current_points]
        
        for i, (prev_points, prev_pose) in enumerate(zip(previous_lidars, previous_poses)):
            if i >= self.num_sweeps - 1:  # 包括当前帧，所以减1
                break
                    
            if prev_points is not None and len(prev_points) > 0:
                prev_points_transformed = self.transform_points_between_frames(
                    prev_points, prev_pose, current_pose)
                fused_points.append(prev_points_transformed)
        
        if len(fused_points) > 1:
            fused_points = torch.cat(fused_points, dim=0)
        else:
            fused_points = fused_points[0] if len(fused_points) == 1 else torch.empty((0, 3))
            
        return fused_points
    
    
    def voxelize(self, points):

        mask = (points[:, 0] >= self.pc_range[0]) & (points[:, 0] <= self.pc_range[3]) & \
               (points[:, 1] >= self.pc_range[1]) & (points[:, 1] <= self.pc_range[4]) & \
               (points[:, 2] >= self.pc_range[2]) & (points[:, 2] <= self.pc_range[5])
        points = points[mask]
        
        voxel_coords = ((points - self.pc_range[:3]) / self.lidar_voxel_size).floor().int()
        
        # 非空体素
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_coords, dim=0, return_inverse=True, return_counts=True
        )
        
        voxel_centers = (unique_voxels.float() + 0.5) * self.lidar_voxel_size + self.pc_range[:3]
        return voxel_centers, counts
    

    def forward(self, current_lidar, current_pose, previous_lidars=None, previous_poses=None):
        # if previous_lidars is not None and previous_poses is not None:
        #     fused_points = self.fuse_multisweep_lidar(
        #         current_lidar, current_pose, previous_lidars, previous_poses
        #     )
        # else:
        #     fused_points = current_lidar
        
        fused_points = current_lidar
        
        voxel_centers,  point_counts = self.voxelize(fused_points)
        
        return {
            'voxel_centers': voxel_centers,           # 体素几何中心 [N, 3] (当前帧坐标系)
            'point_counts': point_counts,             # 每个体素中的点数 [N]
            'num_voxels': len(voxel_centers)          # 非空体素数量
        }