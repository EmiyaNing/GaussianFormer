import torch, torch.nn as nn
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from .lidar_processor import LidarVoxelProcessor


@MODELS.register_module()
class GaussianVoxelLifterOnly(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        pts_init=False,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=0.5,
        occ_resolution=[200, 200, 16],
        empty_label=17,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pts_init = pts_init
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        assert not (pts_init and anchor_grad)
        
        self.include_opa = include_opa
        self.xyz_act     = xyz_activation
        self.scale_act   = scale_activation
        self.semantics   = semantics
        self.semantic_dim= semantic_dim



        self.pc_range   = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution

        self.lidar_processor = LidarVoxelProcessor(
            pc_range=pc_range,
            lidar_voxel_size=voxel_size,
            max_points_per_voxel=20,
            num_sweeps=1
        )

        self.num_anchor = num_anchor

        self.instance_feature = nn.Parameter(
            torch.zeros([self.num_anchor, self.embed_dims]),
            requires_grad=feat_grad,
        )

    def init_weights(self):
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def process_lidar_data(self, metas):
        batch_size = len(metas['lidar_points'])
        device     = next(self.parameters()).device
        self.lidar_processor.to(device)
        all_voxel_centers = []
        all_point_counts  = []


        for i in range(batch_size):
            current_lidar = metas['lidar_points'][i]
            current_pose  = metas['lidar_pose'][i]
            current_lidar = current_lidar[:, :3]

            lidar_result = self.lidar_processor(
                current_lidar=current_lidar,
                current_pose=current_pose,
                previous_lidars=None,
                previous_poses=None
            )

            all_voxel_centers.append(lidar_result['voxel_centers'])
            all_point_counts.append(lidar_result['point_counts'])
        
        return all_voxel_centers, all_point_counts

    def init_anchors_from_lidar(self, voxel_centers_list, point_counts_list, batch_size):
        anchors_list         = []
        actual_anchor_counts = []

        for b in range(batch_size):
            voxel_centers = voxel_centers_list[b]
            point_counts  = point_counts_list[b]

            num_voxels    = len(voxel_centers)
            num_anchor_act= min(num_voxels, self.num_anchor)
            actual_anchor_counts.append(num_anchor_act)

            if num_voxels > self.num_anchor:
                weights = point_counts.float() / point_counts.sum()
                indices = torch.multinomial(weights, self.num_anchor, replacement=False)
                selected_centers = voxel_centers[indices]
                selected_counts  = point_counts[indices]
            else:
                selected_centers = voxel_centers
                selected_counts = point_counts

            pc_range_tensor = torch.tensor(self.pc_range, device=selected_centers.device)
            normalized_xyz  = (selected_centers - pc_range_tensor[:3]) / (pc_range_tensor[3:] - pc_range_tensor[:3])

            if self.xyz_act == 'sigmoid':
                normalized_xyz = safe_inverse_sigmoid(normalized_xyz)

            scale = torch.ones(num_anchor_act, 3, device=selected_centers.device) * self.voxel_size
            if self.scale_act == 'sigmoid':
                scale = safe_inverse_sigmoid(scale)

            rots = torch.zeros(num_anchor_act, 4, device=selected_centers.device)
            rots[:, 0] = 1
            
            # 初始化不透明度（基于点云密度）
            if self.include_opa:
                density = selected_counts.float() / selected_counts.max().clamp(min=1)
                opacity = safe_inverse_sigmoid(density.unsqueeze(1))
            else:
                opacity = torch.zeros((num_anchor_act, 0), device=selected_centers.device)
            

            if self.semantics:
                semantic_dim = self.semantic_dim
                semantic = torch.randn(num_anchor_act, semantic_dim, device=selected_centers.device)
            else:
                semantic = torch.zeros((num_anchor_act, 0), device=selected_centers.device)
            

            anchor = torch.cat([normalized_xyz, scale, rots, opacity, semantic], dim=-1)
            anchors_list.append(anchor)
        
        return anchors_list, actual_anchor_counts





    def forward(self, ms_img_feats, metas, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        voxel_centers_list, point_counts_list   = self.process_lidar_data(metas)
        lidar_anchors_list, actual_anchor_counts = self.init_anchors_from_lidar(
            voxel_centers_list, point_counts_list, batch_size
        )

        anchors_list = []
        features_list= []

            
        for b, (lidar_anchors, actual_count) in enumerate(zip(lidar_anchors_list, actual_anchor_counts)):
            final_anchors = lidar_anchors
            current_features = self.instance_feature[:actual_count, :]
            
            anchors_list.append(final_anchors)
            
            features_list.append(current_features)
        
        # 堆叠成batch
        anchor = torch.stack(anchors_list)
        instance_feature = torch.stack(features_list)
        


        return {
            'rep_features': instance_feature,
            'representation': anchor,
            #'anchor_init': self.anchor.clone()
        }