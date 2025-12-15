import torch, torch.nn as nn
import frnn

import torch.nn.functional as F


from mmseg.registry import MODELS

@MODELS.register_module()
class VoxelQueryModule(nn.Module):
    '''
        This module should be used to integrate gaussian ball's feature tensor with multi-stride 
        voxel points.
    '''
    def __init__(self,
                 pc_range,
                 backbone_channels,
                 mlp_channels):
        super(VoxelQueryModule, self).__init__()
        self.pc_range = pc_range
        self.backbone_channels = backbone_channels
        #self.mlp_channels = mlp_channels

        self.mlp_in = nn.ModuleList()
        self.mlp_pos= nn.ModuleList()
        self.mlp_out= nn.ModuleList()

        for idx in range(len(mlp_channels)):
            input_ch = backbone_channels[idx]
            mlp_ch   = mlp_channels[idx]

            cur_in = nn.Sequential(
                nn.Conv1d(input_ch, mlp_ch[0], kernel_size=1, stride=1),
                nn.BatchNorm1d(mlp_ch[0]),
                nn.GELU(),
                nn.Conv1d(mlp_ch[0], mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm1d(mlp_ch[1])
            )

            cur_pos= nn.Sequential(
                nn.Conv2d(3, mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm2d(mlp_ch[2])
            )

            cur_out= nn.Sequential(
                nn.Conv1d(mlp_ch[1], mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm1d(mlp_ch[1]),
                nn.GELU()
            )
            self.mlp_in.append(cur_in)
            self.mlp_pos.append(cur_pos)
            self.mlp_out.append(cur_out)

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
        This function use the gaussian's mean and size to search voxel query feature from multi-stride voxel features.
        :param self: classes
        :param gaussian: gaussian predictions
        :param instance_features: instance features of each gaussian predictions
        :param multi_stride_features: multi-stride voxel features.
        '''
        ctr_2x, fet_2x = multi_stride_features['stride2']['center'], multi_stride_features['stride2']['feature'] # 2x down, 32 channels
        ctr_4x, fet_4x = multi_stride_features['stride4']['center'], multi_stride_features['stride4']['feature'] # 4x down, 64 channels
        ctr_8x, fet_8x = multi_stride_features['stride8']['center'], multi_stride_features['stride8']['feature'] # 8x down, 128 channels
        bs_mask2, bs_mask4, bs_mask8 = multi_stride_features['stride2']['bs_mask'], \
                                       multi_stride_features['stride4']['bs_mask'], \
                                       multi_stride_features['stride8']['bs_mask']

        ctr_2x = self.scale_ctr_size(ctr_2x)
        ctr_4x = self.scale_ctr_size(ctr_4x)
        ctr_8x = self.scale_ctr_size(ctr_8x)


        means = gaussian.means
        sizes = gaussian.size

        batch_size = means.shape[0]

        for i in range(batch_size):
            cur_means = means[i]
            cur_sizes = sizes[i]

            cur_ctr_2x = ctr_2x[bs_mask2[i]]
            cur_ctr_4x = ctr_4x[bs_mask4[i]]
            cur_ctr_8x = ctr_8x[bs_mask8[i]]
            cur_fet_2x = fet_2x[bs_mask2[i]]
            cur_fet_4x = fet_4x[bs_mask4[i]]
            cur_fet_8x = fet_8x[bs_mask8[i]]

            _, idx2, nn_2x, _ = frnn.frnn_grid_points(
                cur_means, cur_ctr_2x, K=27, r=2
            ) 
            gathered_2x_feat = frnn.frnn_gather(cur_fet_2x, idx2) # Num_gs, 27, 32
            feat2x_in = self.mlp_in[0](gathered_2x_feat)
            feat2x_pos= self.mlp_pos[0](nn_2x)
            feat2x = feat2x_in + feat2x_pos
            feat2x = self.mlp_out[0](feat2x)

            _, idx4, nn_4x, _ = frnn.frnn_grid_points(
                cur_means, cur_ctr_4x, K=27, r=2
            )
            gathered_4x_feat = frnn.frnn_gather(cur_fet_4x, idx4) # Num_gs, 27, 64
            feat4x_in = self.mpl_in[1](gathered_4x_feat)
            feat4x_pos= self.mlp_pos[1](nn_4x)
            feat4x = feat4x_in + feat4x_pos
            feat4x = self.mlp_out[1](feat4x)

            _, idx8, nn_8x, _ = frnn.frnn_grid_points(
                cur_means, cur_ctr_8x, K=27, r=2
            )
            gathered_8x_feat = frnn.frnn_gather(cur_fet_8x, idx8)
            feat8x_in = self.mlp_in[2](gathered_8x_feat)
            feat8x_pos= self.mlp_pos[2](gathered_8x_feat)
            feat8x = feat8x_in + feat8x_pos
            feat8x = self.mlp_out[2](feat8x)

            


        pass


