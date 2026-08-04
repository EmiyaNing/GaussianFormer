"""DAOcc multi-modal segmentor for the current GaussianFormer runner."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from mmcv.ops import Voxelization
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmseg.registry import MODELS
from torch import nn

from model.daocc import (
    DAOccBEVTransform,
    DAOccConvFuser,
    DAOccCoordinateSample,
    DAOccCustomResNet,
    DAOccFPN,
    DAOccGeneralizedLSSFPN,
    DAOccHead,
)


@MODELS.register_module()
class DAOccSegmentor(BaseModule):
    """Strict DAOcc architecture with a GaussianFormer-compatible forward API."""

    def __init__(
        self,
        img_backbone,
        lidar_encoder,
        detection_head,
        point_cloud_range=(-54., -54., -5., 54., 54., 3.),
        voxel_size=(0.075, 0.075, 0.2),
        max_num_points=10,
        max_voxels=(120000, 160000),
        detection_loss_weight=0.01,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = DAOccGeneralizedLSSFPN(
            in_channels=[512, 1024, 2048], out_channels=256,
            num_outs=3, start_level=0,
            norm_cfg=dict(type='BN2d', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            upsample_cfg=dict(mode='bilinear', align_corners=False))
        self.view_transform = DAOccBEVTransform(
            x=(-54., 54.), y=(-54., 54.), z=(-5., 3.),
            xs=180, ys=180, zs=10, input_size=(256, 704),
            in_channels=256, out_channels=128)
        self.voxelizer = Voxelization(
            voxel_size=list(voxel_size),
            point_cloud_range=list(point_cloud_range),
            max_num_points=max_num_points,
            max_voxels=max_voxels)

        # Importing all mmdet3d models initializes its optional sparse backends.
        # Keep this local so dataset/config tools remain usable without a GPU.
        import mmdet3d.models  # noqa: F401
        from mmdet3d.registry import MODELS as MMDET3D_MODELS
        self.lidar_encoder = MMDET3D_MODELS.build(lidar_encoder)
        self.detection_head = MMDET3D_MODELS.build(detection_head)

        self.fuser = DAOccConvFuser([1280, 256], 512)
        self.decoder_backbone = DAOccCustomResNet(
            numC_input=512, num_channels=(128, 256, 512),
            stride=(1, 1, 2))
        self.decoder_neck = DAOccFPN(
            in_channels=640, out_channels=512, scale_factor=2,
            extra_upsample=None)
        self.coordinate_sample = DAOccCoordinateSample(
            point_range=(-50., 50., -50., 50., 0., 0.),
            point_num=(200, 200, 1),
            lidar_point_range=(-54., 54., -54., 54.),
            in_dim=512, out_dim=128)
        self.occ_head = DAOccHead(
            in_dim=128, out_dim=128, Dz=16, num_classes=17,
            drop_free=True, free_label=0, keep_free_ratio=0.2)
        self.detection_loss_weight = detection_loss_weight

    def _camera_features(self, imgs, metas):
        b, n, c, h, w = imgs.shape
        features = self.img_backbone(imgs.reshape(b * n, c, h, w))
        features = self.img_neck(list(features)[1:4])
        image_feature = features[0].reshape(
            b, n, features[0].shape[1], features[0].shape[2], features[0].shape[3])
        return self.view_transform(
            image_feature,
            metas['camera2lidar'],
            metas['camera_intrinsics'],
            metas['img_aug_matrix'],
            metas['lidar_aug_matrix'])

    def _voxelize(self, points: List[torch.Tensor]):
        voxel_features, coordinates, sizes = [], [], []
        for batch_index, sample_points in enumerate(points):
            voxels, coors, num_points = self.voxelizer(sample_points)
            voxel_features.append(voxels.sum(1) / num_points[:, None].type_as(voxels))
            coordinates.append(F.pad(
                coors, (1, 0), mode='constant', value=batch_index))
            sizes.append(num_points)
        return (
            torch.cat(voxel_features, dim=0),
            torch.cat(coordinates, dim=0),
            torch.cat(sizes, dim=0))

    def _lidar_features(self, points):
        features, coordinates, _ = self._voxelize(points)
        return self.lidar_encoder(
            features, coordinates, batch_size=len(points))

    @staticmethod
    def _detection_instances(metas):
        instances = []
        for boxes, labels in zip(
                metas['gt_bboxes_3d'], metas['gt_labels_3d']):
            instance = InstanceData()
            instance.bboxes_3d = boxes
            instance.labels_3d = labels
            instances.append(instance)
        return instances

    @staticmethod
    def _loss_float32(value):
        """Recursively cast floating predictions to FP32 for stable losses.

        This restores the behavior of the official DAOcc CenterHead
        ``force_fp32(apply_to=('preds_dicts',))`` decorator without moving the
        memory-intensive backbone and head forward passes out of AMP.
        """
        if torch.is_tensor(value):
            return value.float() if value.is_floating_point() else value
        if isinstance(value, dict):
            return {
                key: DAOccSegmentor._loss_float32(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [DAOccSegmentor._loss_float32(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DAOccSegmentor._loss_float32(item) for item in value)
        return value

    def forward(self, imgs=None, metas=None, **kwargs) -> Dict[str, torch.Tensor]:
        points = metas['lidar_points']
        camera_bev = self._camera_features(imgs, metas)
        lidar_bev = self._lidar_features(points)
        fused = self.fuser([camera_bev, lidar_bev])
        decoded = self.decoder_neck(self.decoder_backbone(fused))
        occ_features = self.coordinate_sample(decoded)
        pred_occ = self.occ_head(occ_features)

        outputs = dict(
            pred_occ=pred_occ,
            final_occ=pred_occ.argmax(dim=-1).reshape(pred_occ.shape[0], -1),
            sampled_label=metas['occ_label'].reshape(
                metas['occ_label'].shape[0], -1),
            occ_cam_mask=metas['occ_cam_mask'])
        if 'occ_label' in metas:
            # Cross entropy is numerically sensitive to FP16 logits.  Explicit
            # conversion is required because disabling autocast alone does not
            # change tensors that were already produced in FP16.
            with torch.autocast(
                    device_type=pred_occ.device.type, enabled=False):
                loss_occ = self.occ_head.loss(
                    pred_occ.float(), metas['occ_label'])
            outputs['loss_occ'] = loss_occ
            outputs['loss_total'] = loss_occ
        if self.training:
            detection_predictions = self.detection_head([decoded])
            # Match the official CenterHead force_fp32 loss boundary while
            # retaining AMP for its convolutional forward path.
            with torch.autocast(
                    device_type=decoded.device.type, enabled=False):
                detection_losses = self.detection_head.loss_by_feat(
                    self._loss_float32(detection_predictions),
                    self._detection_instances(metas))
            loss_det = sum(
                value if torch.is_tensor(value) else sum(value)
                for value in detection_losses.values())
            outputs.update(
                loss_det=loss_det,
                loss_total=loss_occ + self.detection_loss_weight * loss_det,
                detection_losses=detection_losses)
        return outputs
