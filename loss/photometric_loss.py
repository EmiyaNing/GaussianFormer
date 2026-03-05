import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmseg.models.losses import DiceLoss
from kornia.losses import SSIMLoss

from . import OPENOCC_LOSS
from .base_loss import BaseLoss

@OPENOCC_LOSS.register_module()
class RenderLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        lambda_idx=0.2,
        input_dict=None
    ):
        super().__init__()
        self.weight = weight
        self.lambda_idx = lambda_idx
        if input_dict is None:
            self.input_dict = {
                'render_imgs' : 'render_imgs',
                'imgs': 'imgs'

            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_gaussianrender
        self.ssim_loss_fn = SSIMLoss(window_size=11, reduction='mean')


    def loss_gaussianrender(self, render_imgs, imgs):
        '''
        render_imgs: list of rendered image tensors, each image tensor with shape B, N, C, H, W.
        imgs:        with shape B, N, C, H, W
        '''
        # this function will calculate the l1 loss and D-SSIM loss between imgs and each image tensor.
        # the l1 loss will use weight (1 - self.lambda_idx), while D-SSIM loss will use the weight self.lambda_idx
        # a problem is that current render_imgs and imgs all normalized by means: 123.675, 116.28, 103.53] and stds:[58.395, 57.12, 57.375]
        # so before calcualte the D-SSIM loss, we should consider the de-normalize process.
        # 如果 render_imgs 是单个张量，转换为列表

        if isinstance(render_imgs, torch.Tensor):
            render_imgs = [render_imgs]

        # 定义 ImageNet 归一化参数
        device = render_imgs[0].device
        means = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 1, 3, 1, 1)
        stds = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 1, 3, 1, 1)

        # 导入 SSIMLoss
        
        

        total_loss = 0.0
        for render in render_imgs:
            # 反归一化
            render_denorm = render * stds + means
            imgs_denorm = imgs * stds + means
            # 缩放到 [0, 1]
            render_denorm = render_denorm / 255.0
            imgs_denorm = imgs_denorm / 255.0

            # 可选：裁剪到有效范围，防止极端值
            render_denorm = torch.clamp(render_denorm, 0.0, 1.0)
            imgs_denorm = torch.clamp(imgs_denorm, 0.0, 1.0)

            # L1 损失
            l1_loss = torch.abs(render_denorm - imgs_denorm).mean()

            # D-SSIM 损失
            # 输入形状应为 (B*N, C, H, W)
            B, N, C, H, W = render.shape
            render_flat = render_denorm.view(B * N, C, H, W)
            imgs_flat = imgs_denorm.view(B * N, C, H, W)
            dssim_loss = 1 - self.ssim_loss_fn(render_flat, imgs_flat)

            # 加权组合
            loss = (1 - self.lambda_idx) * l1_loss + self.lambda_idx * dssim_loss
            total_loss += loss

        # 平均所有渲染图像
        total_loss = total_loss / len(render_imgs)
        return total_loss
        