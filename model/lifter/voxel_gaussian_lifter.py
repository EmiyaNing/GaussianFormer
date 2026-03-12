import torch, torch.nn as nn

import torch.nn.functional as F

from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from .lidar_processor import LidarVoxelProcessor
from .spconv_voxelize import VoxelGeneratorWrapper
from .spconv_backbone import VoxelResBackBone8x, MeanVFE

@MODELS.register_module()
class GaussianVoxelLearnear(BaseLifter):
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

        #if include_opa:
        #    opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        #else:
        #    opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        #semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)


        self.pc_range   = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution

        self.lidar_processor = VoxelGeneratorWrapper(
            vsize_xyz=[voxel_size / 8, voxel_size / 8, voxel_size / 8],
            coors_range_xyz=pc_range,
            num_point_features=4,
            max_num_points_per_voxel=5,
            max_num_voxels=1600000
        )

        self.lidar_backbone  = VoxelResBackBone8x(4, embed_dims, [128, 1600, 1600])
        self.lidar_vfe       = MeanVFE(num_point_features = 4)

        self.scale_learner   = nn.Linear(embed_dims, 3)
        self.rot_learner     = nn.Linear(embed_dims, 3)
        if self.semantics:
            self.semantic_learner= nn.Linear(embed_dims, semantic_dim)

        if self.include_opa:
            self.opa_learner = nn.Linear(embed_dims, 1)



    def init_weights(self):
        #self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        #if self.instance_feature.requires_grad:
        #    torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)
        pass

    def process_lidar_data(self, metas):
        batch_size = len(metas['lidar_points'])
        device     = next(self.parameters()).device
        voxel_dict = dict()
        voxel_dict['all_voxels']       = []
        voxel_dict['all_num_ponts']    = []
        voxel_dict['all_voxel_coords'] = []
        #self.lidar_processor.to(device)


        for i in range(batch_size):
            current_lidar = metas['lidar_points'][i]

            voxels, coords, num_points = self.lidar_processor.generate(current_lidar.cpu().numpy())
            voxels = torch.from_numpy(voxels).to(device)
            coords = torch.from_numpy(coords).to(device)
            num_points = torch.from_numpy(num_points).to(device)
            bs_indices = torch.ones([coords.shape[0], 1], device=device) * i
            coords = torch.cat([bs_indices, coords], dim=-1)
            voxel_dict['all_voxels'].append(voxels)
            voxel_dict['all_num_ponts'].append(num_points)
            voxel_dict['all_voxel_coords'].append(coords)
        
        voxel_dict['voxels'] = torch.cat(voxel_dict['all_voxels'])
        voxel_dict['voxel_num_points'] = torch.cat(voxel_dict['all_num_ponts'])
        voxel_dict['voxel_coords'] = torch.cat(voxel_dict['all_voxel_coords'])

        return voxel_dict


    def decode_anchors_from_voxel(self, voxel_indices, voxel_features, spatial_shape, batch_size):
        anchors_list = []
        features_list= []
        device       = voxel_indices.device
        
        spatial_shape= torch.tensor(spatial_shape[::-1], device=device)

        for b in range(batch_size):
            bs_mask   = voxel_indices[:, 0] == b
            cur_voxel = voxel_indices[bs_mask][:, 1:].flip(dims=[-1])
            cur_feats = voxel_features[bs_mask]


            normalized_xyz = cur_voxel / spatial_shape
            if self.xyz_act == 'sigmoid':
                normalized_xyz = safe_inverse_sigmoid(normalized_xyz)
       

            scales = F.sigmoid(self.scale_learner(cur_feats))
            if self.scale_act == 'sigmoid':
                scales = safe_inverse_sigmoid(scales)
            
            rots_learner = F.sigmoid(self.rot_learner(cur_feats))
            rots_padding = torch.zeros([rots_learner.shape[0], 1], device=rots_learner.device, dtype=rots_learner.dtype)
            rots = torch.cat([rots_padding, rots_learner],dim=-1)

            if self.include_opa:
                density = self.opa_learner(cur_feats)
                opacity = nn.functional.sigmoid(density)
            else:
                opacity = torch.zeros((self.num_anchor, 0), device=device)

            if self.semantics:
                semantic = self.semantic_learner(cur_feats)
            else:
                semantic = torch.zeros((self.num_anchor, 0), device=device)
            


            anchor = torch.cat([normalized_xyz, scales, rots, opacity, semantic], dim=-1)
            anchors_list.append(anchor)
            features_list.append(cur_feats)

        return anchors_list, features_list


    def decode_only_ctr_features(self, voxel_indices, voxel_features, spatial_shape, batch_size):
        xyz_list     = []
        feature_list = []
        bs_masks      = []
        device       = voxel_indices.device
        
        spatial_shape= torch.tensor(spatial_shape[::-1], device=device)

        for b in range(batch_size):
            bs_mask   = voxel_indices[:, 0] == b
            cur_voxel = voxel_indices[bs_mask][:, 1:].flip(dims=[-1])
            cur_feats = voxel_features[bs_mask]

            normalized_xyz = cur_voxel / spatial_shape
            if self.xyz_act == 'sigmoid':
                normalized_xyz = safe_inverse_sigmoid(normalized_xyz)
            xyz_list.append(normalized_xyz)
            feature_list.append(cur_feats)
            bs_masks.append(bs_mask)

        return xyz_list, feature_list, bs_masks



    def forward(self, imgs, metas, **kwargs):
        batch_size = imgs.shape[0]
        voxel_dict = self.process_lidar_data(metas)
        voxel_dict['batch_size'] = batch_size
        voxel_dict = self.lidar_vfe(voxel_dict)
        voxel_dict = self.lidar_backbone(voxel_dict)
        feat_voxel = voxel_dict['out_tensor']
        voxel_ctrs = feat_voxel.indices
        voxel_feat = feat_voxel.features
        spatial_shape = feat_voxel.spatial_shape
        
        anchors_list, features_list = self.decode_anchors_from_voxel(
            voxel_ctrs, voxel_feat, spatial_shape, batch_size
        )

        multi_voxel = voxel_dict['multi_scale_3d_features']
        stride_2 = multi_voxel['x_conv2']
        anchor_stride2, feature_stride2, bs_mask2 = self.decode_only_ctr_features(stride_2.indices, stride_2.features, stride_2.spatial_shape, batch_size)

        stride_4 = multi_voxel['x_conv3']
        anchor_stride4, feature_stride4, bs_mask4 = self.decode_only_ctr_features(stride_4.indices, stride_4.features, stride_4.spatial_shape, batch_size)

        stride_8 = multi_voxel['x_conv4']
        anchor_stride8, feature_stride8, bs_mask8 = self.decode_only_ctr_features(stride_8.indices, stride_8.features, stride_8.spatial_shape, batch_size)

        multi_stride_features = dict(
            stride2 = dict(
                center=anchor_stride2,
                feature=feature_stride2,
                bs_mask=bs_mask2
            ),
            stride4 = dict(
                center=anchor_stride4,
                feature=feature_stride4,
                bs_mask=bs_mask4
            ),
            stride8 = dict(
                center=anchor_stride8,
                feature=feature_stride8,
                bs_mask=bs_mask8
            )
        )

        # 堆叠成batch
        anchor = torch.stack(anchors_list)
        instance_feature = torch.stack(features_list)
        


        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor.clone(),
            'multi_stride_features': multi_stride_features
        }