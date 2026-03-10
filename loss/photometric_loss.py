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
                'mask_img': 'mask_img',
                'imgs': 'imgs'

            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_gaussianrender
        self.ssim_loss_fn = SSIMLoss(window_size=11, reduction='mean')


    def loss_gaussianrender(self, render_imgs, mask_img, imgs):
        '''
        In current version, the input render_imgs is denormalized imgs, which value range is [0, 1],
        while the imgs is normalized imgs, which value is [-2.0, 2.0].
        To correctly calculate the loss value, we need to de-normalize the input imgs.
        Then, we use a pre-calculated mask to tell model which image region is needed to consider in current input scenarios.
        Args:
            render_imgs: list of rendered image tensors, each image tensor with shape B, N, C, H, W. 
            mask_img:    with shape B, N, 1, H, W.
            imgs:        with shape B, N, C, H, W.
        '''
  
        means = torch.tensor([123.675, 116.28, 103.53], device=imgs.device).view(1, 1, 3, 1, 1)
        stds = torch.tensor([58.395, 57.12, 57.375], device=imgs.device).view(1, 1, 3, 1, 1)

        if isinstance(render_imgs, torch.Tensor):
            render_imgs = [render_imgs]

        imgs_denorm = imgs.float() * stds + means
        imgs_denorm = imgs_denorm.clamp(0, 255)
        # project the imgs value sapce from [0, 255] to [0, 1]
        # and mask the image region needs to be considered.
        imgs_denorm = (imgs_denorm * mask_img) / 255.0        
        
        

        total_loss = 0.0
        for render in render_imgs:
            # L1 loss
            l1_loss = torch.abs(render - imgs_denorm).mean()

            # D-SSIM loss
            # Input shape (B*N, C, H, W)
            B, N, C, H, W = render.shape
            render_flat = render.view(B * N, C, H, W)
            imgs_flat = imgs_denorm.view(B * N, C, H, W)
            dssim_loss = 1 - self.ssim_loss_fn(render_flat, imgs_flat)

            # weighted add function
            loss = (1 - self.lambda_idx) * l1_loss + self.lambda_idx * dssim_loss
            total_loss += loss

        # average all input images.
        total_loss = total_loss / len(render_imgs)
        return total_loss
        