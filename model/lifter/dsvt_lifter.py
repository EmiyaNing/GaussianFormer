import torch, torch.nn as nn

import torch.nn.functional as F

from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from .spconv_voxelize import VoxelGeneratorWrapper
from .spconv_backbone import MeanVFE
from .dsvt import DSVT

@MODELS.register_module()
class GaussianDSVT(BaseLifter):
    def __init__(
        self,
        embed_dims,
        anchor_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        pts_init=False,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=0.5,
        occ_resolution=[200, 200, 16],
        dsvt_info=dict(
            input_layer=dict(
                sparse_shape = [200, 200, 16],
                downsample_stride = [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                d_model = [192, 192, 192, 192],
                set_info = [[48, 1], [48, 1], [48, 1], [48, 1]],
                window_shape = [[12, 12, 16], [12, 12, 16], [12, 12, 16], [12, 12, 16]],
                hybrid_factor = [2, 2, 1], # x, y, z
                shifts_list = [[[0, 0, 0], [6, 6, 0]], [[0, 0, 0], [6, 6, 0]], [[0, 0, 0], [6, 6, 0]], [[0, 0, 0], [6, 6, 0]]],
                normalize_pos = False),
            block_name = ['DSVTBlock','DSVTBlock','DSVTBlock','DSVTBlock'],
            set_info = [[48, 1], [48, 1], [48, 1], [48, 1]],
            d_model = [192, 192, 192, 192],
            nhead = [8, 8, 8, 8],
            dim_feedforward = [384, 384, 384, 384],
            dropout = 0.0,
            activation = 'gelu',
            reduction_type = 'attention',
            output_shape = [200, 200, 16],
            conv_out_channel = 128,
            use_checkpoint = False,
        ),
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pts_init = pts_init
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.dsvt_info = dsvt_info
        assert not (pts_init and anchor_grad)
        
        self.include_opa = include_opa
        self.xyz_act     = xyz_activation
        self.scale_act   = scale_activation
        self.semantics   = semantics
        self.semantic_dim= semantic_dim


        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0



        self.pc_range   = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution

        self.lidar_processor = VoxelGeneratorWrapper(
            vsize_xyz=[voxel_size, voxel_size, voxel_size],
            coors_range_xyz=pc_range,
            num_point_features=4,
            max_num_points_per_voxel=5,
            max_num_voxels=1600000
        )
        self.lidar_backbone  = DSVT(dsvt_info)
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
        voxel_ctrs = voxel_dict['voxel_coords']
        voxel_feat = voxel_dict['voxel_features']
        spatial_shape = self.dsvt_info['output_shape']
        
        anchors_list, features_list = self.decode_anchors_from_voxel(
            voxel_ctrs, voxel_feat, spatial_shape, batch_size
        )

    
        # 堆叠成batch
        anchor = torch.stack(anchors_list)
        instance_feature = torch.stack(features_list)
        


        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': anchor.clone(),
        }