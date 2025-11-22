import torch, torch.nn as nn
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from .lidar_processor import LidarVoxelProcessor
from .spconv_voxelize import VoxelGeneratorWrapper
from .spconv_backbone import VoxelBackBone8x, MeanVFE

@MODELS.register_module()
class GaussianVoxelSPLifter(BaseLifter):
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
        
        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if xyz_activation == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)
            
        scale = torch.rand_like(xyz)
        if scale_activation == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1
        self.include_opa = include_opa
        self.xyz_act     = xyz_activation
        self.scale_act   = scale_activation
        self.semantics   = semantics
        self.semantic_dim= semantic_dim

        if include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)

        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

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

        self.lidar_backbone  = VoxelBackBone8x(4, embed_dims, [128, 1600, 1600])
        self.lidar_vfe       = MeanVFE(num_point_features = 4)

        self.scale_learner   = nn.Linear(embed_dims, 3)
        self.rot_learner     = nn.Linear(embed_dims, 4)
        if self.semantics:
            self.semantic_learner= nn.Linear(embed_dims, semantic_dim)

        if self.include_opa:
            self.opa_learner = nn.Linear(embed_dims, 1)


        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )

    def init_weights(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

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

            num_voxels= cur_voxel.shape[0]
            if num_voxels > self.num_anchor:
                # filter extra voxels
                cur_voxel = cur_voxel[:self.num_anchor]
                cur_feats = cur_feats[:self.num_anchor]
                normalized_xyz = cur_voxel / spatial_shape
                if self.xyz_act == 'sigmoid':
                    normalized_xyz = safe_inverse_sigmoid(normalized_xyz)
            else:
                # pad voxel and feats
                pad_count    = self.num_anchor - num_voxels
                act_norm_xyz = cur_voxel / spatial_shape
                if self.xyz_act == 'sigmoid':
                    act_norm_xyz = safe_inverse_sigmoid(act_norm_xyz)

                pad_xyz      = torch.rand([pad_count, 3], device=device)
                pad_norm_xyz = safe_inverse_sigmoid(pad_xyz)
                normalized_xyz = torch.cat([act_norm_xyz, pad_norm_xyz], dim=0)

                pad_features = torch.randn(pad_count, cur_feats.shape[-1], device=device)
                cur_feats    = torch.cat([cur_feats, pad_features], dim=0)
            #import pdb
            #pdb.set_trace()


            scales = torch.ones(self.num_anchor, 3, device=device) * self.voxel_size
            if self.scale_act == 'sigmoid':
                scales = safe_inverse_sigmoid(scales)
            
            rots = torch.zeros(self.num_anchor, 4, device=device)
            rots[:, 0] = 1

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






    def forward(self, ms_img_feats, metas, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        voxel_dict               = self.process_lidar_data(metas)
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

       
        # 堆叠成batch
        anchor = torch.stack(anchors_list)
        instance_feature = torch.stack(features_list)
        


        return {
            'rep_features': instance_feature,
            'representation': anchor,
            'anchor_init': self.anchor.clone()
        }