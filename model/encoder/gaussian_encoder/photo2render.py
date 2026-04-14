import torch
import gsplat
import torch.nn as nn

import torch.nn.functional as F

from mmcv.cnn import Scale
from mmengine.registry import MODELS
from mmengine.model import BaseModule

from .utils import linear_relu_ln, GaussianPrediction

rgb_means = torch.tensor([123.675, 116.28, 103.53]).cuda()
rgb_stds  = torch.tensor([58.395, 57.12, 57.375]).cuda()

@MODELS.register_module()
class Photo2Render(BaseModule):
    '''
    This module first use the instance features to predict the truth rgb color value of each gaussian.
    Then, the predicted rgb color will add original gaussian's color to get the final color attribute of each gaussian.
    Finally, this module use the gaussian's color attribute to render surround view image.
    The surround view image will be used to calculate the photometric supervise loss.
    '''
    def __init__(self, embed_dims=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.rgb_modify = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, 3),
            Scale([1.0] * 3))
        
    def forward(self,
                instance_features,
                gaussians,
                metas):
        '''
        instance_features with shape B, num_gaussians, embed_dims
        gaussians is the class GaussianPrediction
        '''
        ori_colors = gaussians.colors
        means      = gaussians.means    
        quats      = gaussians.rotations
        scales     = gaussians.scales
        opacities  = gaussians.opacities
        color_modify = self.rgb_modify(instance_features)
        colors = (ori_colors + color_modify)
        

        background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=means.device)

        lidar2cam = metas['lidar2cam']  # (B, N, 4, 4)
        camera_intrinsic = metas['intrinsic']  # (B, N, 3, 3)
        image_wh  = metas['image_wh']
        # 图像尺寸
        W = image_wh[..., 0].int()
        H = image_wh[..., 1].int()
        # 使用第一个批次和第一个视图的尺寸作为渲染尺寸（假设所有视图尺寸相同）
        render_height = H[0, 0].item() // 4
        render_width = W[0, 0].item() // 4
        
        B, N, _, _ = lidar2cam.shape
        # 准备渲染输出列表
        rendered_list = []
        
        # 遍历每个视图
        for view_idx in range(N):
            # 获取当前视图的相机参数，并增加一个相机维度 C=1
            view_lidar2cam = lidar2cam[:, view_idx:view_idx+1]  # (B, 1, 4, 4)
            view_K = camera_intrinsic[:, view_idx:view_idx+1]   # (B, 1, 3, 3)
         
            # 调用 gsplat.rasterization，支持批次维度 B 和相机维度 1
            render, _, _ = gsplat.rasterization(
                means,                     # (B, num_pts, 3)
                quats,                     # (B, num_pts, 4)
                scales,                    # (B, num_pts, 3)
                opacities.squeeze(-1),     # (B, num_pts)
                colors,                    # (B, num_pts, 3)
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
        new_gaussian = GaussianPrediction(
            means=gaussians.means,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
            opacities=gaussians.opacities,
            semantics=gaussians.semantics,
            colors=colors,
        )

        return new_gaussian, rendered