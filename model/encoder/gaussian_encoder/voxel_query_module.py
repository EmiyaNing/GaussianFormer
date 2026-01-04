import torch, torch.nn as nn
import frnn

import torch.nn.functional as F


from mmseg.registry import MODELS

@MODELS.register_module()
class VoxelQueryModule(nn.Module):
    '''
        This module should be used to integrate gaussian ball's feature tensor with multi-stride 
        voxel points.
        Args:
            pc_range: the point range of current dataset.
            embedding-channels: the embedding_channel of our voxel backbone.
            backbone_channels: the multi-stride feature channels outputed by voxel backbone.
            mlp_channels: the channel's count of feature aggregation networks.
    '''
    def __init__(self,
                 pc_range,
                 embedding_channels,
                 backbone_channels,
                 mlp_channels):
        super(VoxelQueryModule, self).__init__()
        self.pc_range = pc_range
        self.backbone_channels = backbone_channels
        #self.mlp_channels = mlp_channels
        self.embedding_channels= embedding_channels
        self.mlp_in = nn.ModuleList()
        self.mlp_pos= nn.ModuleList()
        self.mlp_out= nn.ModuleList()

        cated_channels = 0
        for idx in range(len(mlp_channels)):
            input_ch = backbone_channels[idx]
            mlp_ch   = mlp_channels[idx]

            cur_in = nn.Sequential(
                nn.Conv2d(input_ch, mlp_ch[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mlp_ch[0]),
                nn.GELU(),
                nn.Conv2d(mlp_ch[0], mlp_ch[1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mlp_ch[1]),
                nn.GELU(),
            )

            cur_pos= nn.Sequential(
                nn.Conv2d(3, mlp_ch[1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mlp_ch[1]),
                nn.GELU(),
            )

            cur_out= nn.Sequential(
                nn.Conv2d(mlp_ch[1], mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm2d(mlp_ch[1]),
                nn.GELU()
            )
            cated_channels = cated_channels + mlp_ch[1]
            self.mlp_in.append(cur_in)
            self.mlp_pos.append(cur_pos)
            self.mlp_out.append(cur_out)
        
        self.embed_mapping = nn.Sequential(
            nn.Conv1d(cated_channels, self.embedding_channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm1d(self.embedding_channels)
        )

    def scale_ctr_size(self, normalize_ctr):
        normalize_ctr[:, 0] = normalize_ctr[:, 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        normalize_ctr[:, 1] = normalize_ctr[:, 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        normalize_ctr[:, 2] = normalize_ctr[:, 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        return normalize_ctr


    def forward(self,
                gaussian,
                instance_features,
                multi_stride_features):
        '''
        This function use the gaussian's mean as center to perform the multi-stride voxel ball query.
        Each center of the gaussian's center will search 27 voxel point from stride2, 4, 8 voxel space, respectively.
        
        Example:
            Given one sence gaussian's center with shape [num_gs, 3]
            Given multi-scale feature with {[num_voxel2, 3] + [num_voxel2, ch2]}
                                           {[num_voxel4, 3] + [num_voxel4, ch4]}
                                           {[num_voxel8, 3] + [num_voxel8, ch8]}
            This function will use frnn.frnn_grid_points and frnn.frnn_gather to get {[1, num_gs, 27, 3], [1, num_gs, 27, ch2]}
                                                                                     {[1, num_gs, 27, 3], [1, num_gs, 27, ch4]}
                                                                                     {[1, num_gs, 27, 3], [1, num_gs, 27, ch8]}  
            Then, multiple conv2d layer consisted layer will perform feature aggregation from this tensors as voxelrcnn style.
            Finally, all conv2d feature will be added with original instance_features and get final new_instance_features.
        :param self: classes
        :param gaussian: gaussian predictions
        :param instance_features: instance features of each gaussian predictions
        :param multi_stride_features: multi-stride voxel features.
        Returns:
            new_instance_features
        '''
        ctr_2x, fet_2x = multi_stride_features['stride2']['center'], multi_stride_features['stride2']['feature'] # 2x down, 32 channels
        ctr_4x, fet_4x = multi_stride_features['stride4']['center'], multi_stride_features['stride4']['feature'] # 4x down, 64 channels
        ctr_8x, fet_8x = multi_stride_features['stride8']['center'], multi_stride_features['stride8']['feature'] # 8x down, 128 channels
        bs_mask2, bs_mask4, bs_mask8 = multi_stride_features['stride2']['bs_mask'], \
                                       multi_stride_features['stride4']['bs_mask'], \
                                       multi_stride_features['stride8']['bs_mask']

        means = gaussian.means
        sizes = gaussian.scales

        batch_size = means.shape[0]
        queried_features = []

        for i in range(batch_size):
            cur_ctr_2x = self.scale_ctr_size(ctr_2x[i])
            cur_ctr_4x = self.scale_ctr_size(ctr_4x[i])
            cur_ctr_8x = self.scale_ctr_size(ctr_8x[i])

            cur_fet_2x, cur_fet_4x, cur_fet_8x = fet_2x[i], fet_4x[i], fet_8x[i]
            
            cur_means = means[i].unsqueeze(0) # 1, num_gs, 3
            cur_sizes = sizes[i]
            cur_instance_features = instance_features[i]

            cur_ctr_2x = cur_ctr_2x[bs_mask2[i]].unsqueeze(0) # 1, num_voxe12, 3
            cur_ctr_4x = cur_ctr_4x[bs_mask4[i]].unsqueeze(0) # 1, num_voxel4, 3
            cur_ctr_8x = cur_ctr_8x[bs_mask8[i]].unsqueeze(0) # 1, num_voxel8, 3
            cur_fet_2x = cur_fet_2x[bs_mask2[i]].unsqueeze(0) # 1, num_voxel2, ch2
            cur_fet_4x = cur_fet_4x[bs_mask4[i]].unsqueeze(0) # 1, num_voxel4, ch4
            cur_fet_8x = cur_fet_8x[bs_mask8[i]].unsqueeze(0) # 1, num_voxel8, ch8

            _, idx2, nn_2x, _ = frnn.frnn_grid_points(
                cur_means, cur_ctr_2x, K=27, r=2, return_nn=True
            ) 
            # idx2 with shape 1, num_gs, 27
            # nn_2x with shape 1, num_gs, 27, 3

            gathered_2x_feat = frnn.frnn_gather(cur_fet_2x, idx2) # 1, Num_gs, 27, ch2
            feat2x_in = self.mlp_in[0](gathered_2x_feat.permute(0, 3, 1, 2)) # 1, mlp[0][0], num_gs, 27
            feat2x_pos= self.mlp_pos[0](nn_2x.permute(0, 3, 1, 2)) # 1, mlp[0][0], num_gs, 27
            feat2x = feat2x_in + feat2x_pos
            feat2x = self.mlp_out[0](feat2x) # 1, mlp[0][1], num_gs, 27
            feat2x = feat2x.sum(-1).permute(0, 2, 1) # 1, num_gs, mlp[0][1]

            
            _, idx4, nn_4x, _ = frnn.frnn_grid_points(
                cur_means, cur_ctr_4x, K=27, r=2, return_nn=True
            )
            gathered_4x_feat = frnn.frnn_gather(cur_fet_4x, idx4) # 1, Num_gs, 27, ch4
            feat4x_in = self.mlp_in[1](gathered_4x_feat.permute(0, 3, 1, 2)) # 1, mlp[1][0], num_gs, 27
            feat4x_pos= self.mlp_pos[1](nn_4x.permute(0, 3, 1, 2)) # 1, mlp[1][0], num_gs, 27        
            feat4x = feat4x_in + feat4x_pos  
            feat4x = self.mlp_out[1](feat4x) # 1, mlp[1][1], num_gs, 27
            feat4x = feat4x.sum(-1).permute(0, 2, 1) # 1, num_gs, mlp[1][1]

            _, idx8, nn_8x, _ = frnn.frnn_grid_points(
                cur_means, cur_ctr_8x, K=27, r=2, return_nn=True
            )
            gathered_8x_feat = frnn.frnn_gather(cur_fet_8x, idx8) # 1, num_gs, 27, ch8
            feat8x_in = self.mlp_in[2](gathered_8x_feat.permute(0, 3, 1, 2)) # 1, mlp[2][0], num_gs, 27        
            feat8x_pos= self.mlp_pos[2](nn_8x.permute(0, 3, 1, 2)) # 1, mlp[2][0], num_gs, 27
            feat8x = feat8x_in + feat8x_pos 
            feat8x = self.mlp_out[2](feat8x) # 1, mlp[2][1], num_gs, 27
            feat8x = feat8x.sum(-1).permute(0, 2, 1) # 1, num_gs, mlp[2][1]

            cated_features = torch.cat([feat2x, feat4x, feat8x], dim=-1) # 1, num_gs, (mlp[0][1] + mlp[1][1] + mlp[2][1])
            cated_features = self.embed_mapping(cated_features.permute(0, 2, 1)).permute(0, 2, 1).squeeze(0) # 1, num_gs, 256

            cur_instance_features = cur_instance_features + cated_features
            queried_features.append(cur_instance_features)

        new_instance_features = torch.stack(queried_features, dim=0)
        return new_instance_features


