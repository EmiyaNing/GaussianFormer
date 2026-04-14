import torch
import gsplat

import torch.nn.functional as F

from mmengine.registry import MODELS
from mmengine.model import BaseModule

rgb_means = torch.tensor([123.675, 116.28, 103.53]).cuda()
rgb_stds  = torch.tensor([58.395, 57.12, 57.375]).cuda()

@MODELS.register_module()
class GaussianPhoto(BaseModule):
    '''
    This module project the gaussian into image plane and get correspond rgb colors.
    Then, this module render surround view image from these gaussians.
    The surround view image will be used to calculate the photometric supervise loss.
    '''
    def __init__(self, scale_idx=4):
        super().__init__()
        self.scale_idx = scale_idx


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
        quats       = gaussians.rotations
        scales      = gaussians.scales
        opacities   = gaussians.opacities
        B, num_pts,_= means.shape 
        pts_extend = torch.cat(
            [means, torch.ones_like(means[..., :1])], dim=-1
        )
        projection_mat = metas['projection_mat']
        image_wh  = metas['image_wh']

        background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=means.device)

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
        
        
        # 从 metas 中获取渲染所需的相机参数
        # 使用 projection_mat 作为 lidar2cam，intrinsic 作为 camera_intrinsic
        lidar2cam = metas['lidar2cam']  # (B, N, 4, 4)
        camera_intrinsic = metas['intrinsic']  # (B, N, 3, 3)
        # 图像尺寸
        W = image_wh[..., 0].int()
        H = image_wh[..., 1].int()
        # 使用第一个批次和第一个视图的尺寸作为渲染尺寸（假设所有视图尺寸相同）
        render_height = H[0, 0].item() // self.scale_idx
        render_width = W[0, 0].item() // self.scale_idx
        
        # 准备渲染输出列表
        rendered_list = []
        
        # 遍历每个视图
        for view_idx in range(N):
            # 获取当前视图的相机参数，并增加一个相机维度 C=1
            view_lidar2cam = lidar2cam[:, view_idx:view_idx+1]  # (B, 1, 4, 4)
            view_K = camera_intrinsic[:, view_idx:view_idx+1]   # (B, 1, 3, 3)
            view_colors = colors[:, view_idx]                   # (B, num_pts, 3)
            view_masks  = mask[view_idx].unsqueeze(0)
            view_colors = view_colors[view_masks].unsqueeze(0)
            view_means  = means[view_masks].unsqueeze(0)
            view_quats  = quats[view_masks].unsqueeze(0)
            view_scales = scales[view_masks].unsqueeze(0)
            view_opacity= opacities[view_masks].unsqueeze(0).squeeze(-1)
            
            # 调用 gsplat.rasterization，支持批次维度 B 和相机维度 1
            render, _, _ = gsplat.rasterization(
                view_means,                     # (B, num_pts, 3)
                view_quats,                     # (B, num_pts, 4)
                view_scales,                    # (B, num_pts, 3)
                view_opacity,                 # (B, num_pts)
                view_colors,               # (B, num_pts, 3)
                view_lidar2cam.float(),            # (B, 1, 4, 4)
                view_K.float(),                    # (B, 1, 3, 3)
                render_width,
                render_height,
                backgrounds=background_color
            )
            # render 形状 (B, H, W, 3)
            rendered_list.append(render)
        # 将列表转换为张量 (B, N, H, W, 3)
        rendered = torch.cat(rendered_list, dim=1)  # (B, N, H, W, 3)
        # 调整通道顺序以匹配输入 img 的格式 (B, N, 3, H, W)
        rendered = rendered.permute(0, 1, 4, 2, 3)
    
        return rendered
