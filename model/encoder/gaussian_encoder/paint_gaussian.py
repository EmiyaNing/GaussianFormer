import torch

import torch.nn.functional as F

from mmengine.registry import MODELS
from mmengine.model import BaseModule

from .utils import GaussianPrediction

rgb_means = torch.tensor([123.675, 116.28, 103.53]).cuda()
rgb_stds  = torch.tensor([58.395, 57.12, 57.375]).cuda()

@MODELS.register_module()
class GaussianPainter(BaseModule):
    '''
    This module project the gaussian into image plane and get correspond rgb colors.
    The rgb colors of each gaussian will be saved as a extra attribute of each gaussian.
    The rgb colors will be updated during training by other modules.
    '''
    def __init__(self):
        super().__init__()

    def forward(self,
                img,
                mask_img,
                gaussians,
                metas):
        '''
        img      with shape B, N, 3, H, W
        mask_img with shape B, N, 1, H, W
        gaussians is the class GaussianPrediction 
        metas    contain the lidar to img projection matrix
        '''
        means       = gaussians.means
        B, num_pts,_= means.shape 
        pts_extend = torch.cat(
            [means, torch.ones_like(means[..., :1])], dim=-1
        )
        projection_mat = metas['projection_mat']
        image_wh  = metas['image_wh']


        points_2d = torch.matmul(
            projection_mat[:, :, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        # current points_2d with shape B, N, num_pts, 2
        points_2d = points_2d / image_wh[:, :, None]
        # current mask with shape B, N, num_pts
        mask = (depth > 1e-5) & (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1) & \
                                (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1)

        # current imgs with shape B, N, 3, H, W
        img = img.permute(0, 1, 3, 4, 2)
        img = img * rgb_stds + rgb_means
        img = img.clamp(0, 255)
        img = img.permute(0, 1, 4, 2, 3)
        valid_imgs = img.float() * mask_img
        
        B, N, C, H, W = valid_imgs.shape
        valid_imgs = valid_imgs.reshape(B * N, C, H, W)
        points_2d  = points_2d.reshape(B * N, num_pts, 2)
        mask = mask.reshape(B * N, -1)


        # sample color from valid_imgs
        grid = points_2d * 2 - 1  # 转换到 [-1, 1] 范围，适应 grid_sample
        grid = grid.unsqueeze(1)  # (B*N, 1, num_pts, 2)
        colors = F.grid_sample(valid_imgs.float(), grid, align_corners=False)  # (B*N, C, 1, num_pts)
        colors = colors.squeeze(2).permute(0, 2, 1)  # (B*N, num_pts, C)
        # normalize the colors from [0, 255] to [0, 1]
        colors = colors / 255.0

        # 重塑为 (B, N, num_pts, C)
        colors = colors.reshape(B, N, num_pts, C)

        new_gaussian = GaussianPrediction(
            means=gaussians.means,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
            opacities=gaussians.opacities,
            semantics=gaussians.semantics,
            colors=colors,
        )
        return new_gaussian