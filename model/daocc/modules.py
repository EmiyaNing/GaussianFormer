"""DAOcc-specific modules ported from ZYang2077/DAOcc.

The math and tensor layouts follow official commit
6846432bbc1eff9d2a601900d21f1031df170a8c.  Only the MMCV-1.x registry and
mixed-precision wrappers have been replaced.
"""

from typing import List, Sequence

import einops
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmdet.models.backbones import ResNet
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import nn


@MODELS.register_module()
class DAOccResNetReLU6(ResNet):
    """ResNet image backbone with bounded ReLU6 activations.

    MMDetection 3.0's ResNet hard-codes ``nn.ReLU`` in the stem and residual
    blocks and does not expose an ``act_cfg`` argument.  Replacing the
    parameter-free activation modules after construction preserves the module
    names and pretrained ResNet state-dict compatibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_relu(self)

    @classmethod
    def _replace_relu(cls, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU6(inplace=child.inplace))
            else:
                cls._replace_relu(child)


def daocc_reference_points(x_range, y_range, z_range, size, device='cpu'):
    """Return voxel centres in official ``(z, y, x)`` flattening order."""
    xs, ys, zs = size
    x = torch.linspace(
        x_range[0] + (x_range[1] - x_range[0]) / xs / 2,
        x_range[1] - (x_range[1] - x_range[0]) / xs / 2, xs, device=device)
    y = torch.linspace(
        y_range[0] + (y_range[1] - y_range[0]) / ys / 2,
        y_range[1] - (y_range[1] - y_range[0]) / ys / 2, ys, device=device)
    z = torch.linspace(
        z_range[0] + (z_range[1] - z_range[0]) / zs / 2,
        z_range[1] - (z_range[1] - z_range[0]) / zs / 2, zs, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


@MODELS.register_module()
class DAOccGeneralizedLSSFPN(BaseModule):
    def __init__(self, in_channels, out_channels, num_outs=3, start_level=0,
                 end_level=-1, no_norm_on_lateral=False, conv_cfg=None,
                 norm_cfg=dict(type='BN2d'), act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = list(in_channels)
        self.start_level = start_level
        self.backbone_end_level = len(in_channels) - 1 if end_level == -1 else end_level
        self.upsample_cfg = upsample_cfg.copy()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(start_level, self.backbone_end_level):
            cat_channels = in_channels[i] + (
                in_channels[i + 1]
                if i == self.backbone_end_level - 1 else out_channels)
            self.lateral_convs.append(ConvModule(
                cat_channels, out_channels, 1, conv_cfg=conv_cfg,
                norm_cfg=None if no_norm_on_lateral else norm_cfg,
                act_cfg=act_cfg, inplace=False))
            self.fpn_convs.append(ConvModule(
                out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = list(inputs)
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                **self.upsample_cfg)
            laterals[i] = self.fpn_convs[i](
                self.lateral_convs[i](torch.cat((laterals[i], up), dim=1)))
        return tuple(laterals[:-1])


@MODELS.register_module()
class DAOccBEVTransform(BaseModule):
    """Official SurroundOcc image-to-BEV projection and camera averaging."""

    def __init__(self, x, y, z, xs, ys, zs, input_size,
                 in_channels=256, out_channels=128, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.volume_size = (int(xs), int(ys), int(zs))
        self.input_size = tuple(input_size)
        self.transfer_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.register_buffer(
            'ref_3d',
            daocc_reference_points(x, y, z, self.volume_size),
            persistent=True)

    def _project(self, camera2lidar, intrinsics, img_aug, lidar_aug):
        # Geometry is intentionally FP32 even when the convolutional path uses AMP.
        camera2lidar = camera2lidar.float()
        intrinsics = intrinsics.float()
        img_aug = img_aug.float()
        lidar_aug = lidar_aug.float()
        b, n = camera2lidar.shape[:2]
        q = self.ref_3d.shape[0]
        ref = self.ref_3d.float().view(1, q, 3)
        ref = ref - lidar_aug[:, None, :3, 3]
        ref = torch.linalg.inv(lidar_aug[:, :3, :3])[:, None] @ ref[..., None]
        ref = ref.squeeze(-1)[:, None].expand(b, n, q, 3)
        ref = ref - camera2lidar[:, :, None, :3, 3]
        rotation = camera2lidar[:, :, :3, :3]
        cam = rotation.transpose(-1, -2)[:, :, None] @ ref[..., None]
        cam = cam.squeeze(-1)
        uvw = intrinsics[:, :, None, :3, :3] @ cam[..., None]
        uvw = uvw.squeeze(-1)
        eps = 1e-5
        visible = uvw[..., 2:3] > eps
        uv = uvw[..., :2] / uvw[..., 2:3].clamp_min(eps)
        uv = img_aug[:, :, None, :2, :2] @ uv[..., None]
        uv = uv.squeeze(-1) + img_aug[:, :, None, :2, 3]
        h, w = self.input_size
        uv[..., 0] /= w
        uv[..., 1] /= h
        visible &= (uv[..., 0:1] > 0) & (uv[..., 0:1] < 1)
        visible &= (uv[..., 1:2] > 0) & (uv[..., 1:2] < 1)
        return uv, visible

    def forward(self, image_features, camera2lidar, camera_intrinsics,
                img_aug_matrix, lidar_aug_matrix):
        # The official module decorates the complete forward with
        # ``@force_fp32()``.  Keep transfer_conv, grid_sample, camera
        # scatter-add and averaging in FP32 as well as the geometry.
        with torch.autocast(
                device_type=image_features.device.type, enabled=False):
            return self._forward_float32(
                image_features.float(),
                camera2lidar,
                camera_intrinsics,
                img_aug_matrix,
                lidar_aug_matrix)

    def _forward_float32(self, image_features, camera2lidar,
                         camera_intrinsics, img_aug_matrix,
                         lidar_aug_matrix):
        b, n, c, h, w = image_features.shape
        x = self.transfer_conv(image_features.reshape(b * n, c, h, w))
        c = x.shape[1]
        uv, visible = self._project(
            camera2lidar, camera_intrinsics,
            img_aug_matrix, lidar_aug_matrix)

        # Match the official BEVTransform rebatch implementation: compact each
        # camera's visible voxels to max_len, sample all B*N views in one
        # grid_sample call, then scatter-add them back to the dense volume.
        # This avoids constructing six independent GridSampler autograd graphs.
        visible_indices = []
        for bi in range(b):
            per_batch = []
            for ci in range(n):
                per_batch.append(
                    visible[bi, ci, :, 0].nonzero(
                        as_tuple=False).flatten())
            visible_indices.append(per_batch)

        max_len = max(
            (index.numel()
             for per_batch in visible_indices for index in per_batch),
            default=0)
        if max_len == 0:
            volume = x.new_zeros((b, self.ref_3d.shape[0], c))
        else:
            rebatch_grid = uv.new_zeros((b, n, max_len, 1, 2))
            for bi, per_batch in enumerate(visible_indices):
                for ci, index in enumerate(per_batch):
                    length = index.numel()
                    if length:
                        rebatch_grid[bi, ci, :length, 0] = uv[bi, ci, index]
            rebatch_grid = rebatch_grid.mul(2).sub(1)
            rebatch_grid = rebatch_grid.reshape(
                b * n, max_len, 1, 2)
            sampled = F.grid_sample(
                x, rebatch_grid, mode='bilinear',
                padding_mode='zeros', align_corners=False)
            sampled = einops.rearrange(
                sampled, '(b n) c q 1 -> b n q c', b=b, n=n)

            volume = x.new_zeros((b, self.ref_3d.shape[0], c))
            counts = x.new_zeros((b, self.ref_3d.shape[0], 1))
            for bi, per_batch in enumerate(visible_indices):
                for ci, index in enumerate(per_batch):
                    length = index.numel()
                    if length:
                        volume[bi, index] += sampled[bi, ci, :length]
                        counts[bi, index] += 1
            volume = volume / counts.clamp_min(1)
        xs, ys, zs = self.volume_size
        return einops.rearrange(
            volume, 'b (z y x) c -> b (z c) x y', z=zs, y=ys, x=xs)


@MODELS.register_module()
class DAOccConvFuser(nn.Sequential):
    def __init__(self, in_channels: Sequence[int], out_channels: int):
        self.in_channels = list(in_channels)
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, inputs: List[torch.Tensor]):
        assert len(inputs) == len(self.in_channels)
        return super().forward(torch.cat(inputs, dim=1))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
            if in_channels != out_channels or stride != 1 else nn.Identity())

    def forward(self, x):
        identity = self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


@MODELS.register_module()
class DAOccCustomResNet(BaseModule):
    def __init__(self, numC_input, num_layer=(2, 2, 2),
                 num_channels=(128, 256, 512), stride=(1, 1, 2), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        layers, current = [], numC_input
        for blocks, channels, stage_stride in zip(num_layer, num_channels, stride):
            stage = [_ResidualBlock(current, channels, stage_stride)]
            stage += [_ResidualBlock(channels, channels) for _ in range(blocks - 1)]
            layers.append(nn.Sequential(*stage))
            current = channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


@MODELS.register_module()
class DAOccFPN(BaseModule):
    def __init__(self, in_channels=640, out_channels=512, scale_factor=2,
                 extra_upsample=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        assert extra_upsample is None, 'Official SurroundOcc config has no extra upsample'

    def forward(self, feats):
        return self.conv(torch.cat((feats[0], self.up(feats[2])), dim=1))


@MODELS.register_module()
class DAOccCoordinateSample(BaseModule):
    def __init__(self, point_range=(-50., 50., -50., 50., 0., 0.),
                 point_num=(200, 200, 1),
                 lidar_point_range=(-54., 54., -54., 54.),
                 in_dim=512, out_dim=128, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # Official helper interprets the first pair as y and second as x.
        ref = daocc_reference_points(
            point_range[2:4], point_range[0:2], point_range[4:6],
            (point_num[1], point_num[0], point_num[2]))
        self.register_buffer('ref_points', ref, persistent=True)
        self.output_h, self.output_w = point_num[:2]
        self.y_min, self.y_max, self.x_min, self.x_max = lidar_point_range
        self.transfer_conv = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        # Match the official CrossCoordinateSample ``@force_fp32()`` boundary:
        # both the channel projection and coordinate grid_sample run in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_float32(x.float())

    def _forward_float32(self, x):
        x = self.transfer_conv(x)
        ref = self.ref_points[:, :2].view(1, -1, 1, 2).expand(x.shape[0], -1, -1, -1)
        ref = ref.clone()
        ref[..., 0] = (ref[..., 0] - self.x_min) / (self.x_max - self.x_min)
        ref[..., 1] = (ref[..., 1] - self.y_min) / (self.y_max - self.y_min)
        sampled = F.grid_sample(
            x, ref.mul(2).sub(1), mode='bilinear',
            padding_mode='zeros', align_corners=False)
        return einops.rearrange(
            sampled.squeeze(-1), 'b c (h w) -> b c h w',
            h=self.output_h, w=self.output_w)


@MODELS.register_module()
class DAOccHead(BaseModule):
    def __init__(self, in_dim=128, out_dim=128, Dz=16, num_classes=17,
                 drop_free=True, free_label=0, keep_free_ratio=0.2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dz = Dz
        self.num_classes = num_classes
        self.drop_free = drop_free
        self.free_label = free_label
        self.keep_free_ratio = keep_free_ratio
        self.final_conv = ConvModule(in_dim, out_dim, 3, padding=1, bias=True)
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.Softplus(),
            nn.Linear(out_dim * 2, num_classes * Dz))

    def forward(self, features):
        features = einops.rearrange(features, 'b c w h -> b c h w')
        logits = self.final_conv(features).permute(0, 3, 2, 1)
        b, x, y, _ = logits.shape
        return self.predictor(logits).view(
            b, x, y, self.dz, self.num_classes)

    def loss(self, logits, target):
        target = target.long().reshape(-1)
        logits = logits.reshape(-1, self.num_classes)

        # Reproduce DAOcc's custom CrossEntropyLoss exactly: ignored voxels
        # have zero element-wise loss, while the drop-free normalization mask
        # still treats every non-free label (including 255) as retained.
        per_voxel_loss = F.cross_entropy(
            logits, target, reduction='none', ignore_index=255)
        if self.drop_free:
            free = target == self.free_label
            keep_free = torch.rand_like(target.float()) <= self.keep_free_ratio
            remain = (free & keep_free) | (~free)
            avg_factor = remain.sum()
            per_voxel_loss = per_voxel_loss * remain
        else:
            avg_factor = logits.new_tensor(target.numel())
        return per_voxel_loss.sum() / avg_factor.clamp_min(1)
