"""Fast v5 with a DINOv3 ConvNeXt-Small backbone and the camera mask enabled.

The backbone loads its DINOv3 weights itself through ``img_backbone.load_path``
(``ckpts/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth``; its 344 tensors
match the module's own keys). The inherited global ``load_from`` is therefore
disabled: the RayDN ResNet checkpoint shares no keys with ConvNeXt and only
produced a large incompatibility report.

Camera mask: applied to both training and evaluation, mirroring the earlier
no-mask recipe. ``train_mask_type='camera'`` makes ``occ_loss_mask`` equal to
``mask_camera`` (~24.5 % of the grid), and ``occ3d_eval_mask='camera'`` makes
``Metric_mIoU`` score the same camera-visible voxels. The no-mask twin is
``img_voxel_react_occ3d_fast_v5_dinov3_nomask.py``.

Everything else is inherited from ``img_voxel_react_occ3d_fast_v5.py``
(256 x 704 images, tiled React renderer, BF16 image path, four workers,
``OccupancyLoss``), with two deliberate changes. First, the LiDAR branch is
temporal: ``num_lidar_history=9`` fuses the current keyframe scan with the nine
previous ones, so each sample carries ten LiDAR frames; only keyframes carry
``LIDAR_TOP`` in ``*_sweeps_occ.pkl``, so history steps are keyframe scans
(~0.5 s apart) and scene starts simply provide fewer frames. Second,
``image_frozen_bn`` is turned off because ConvNeXt has no BatchNorm and no
``norm_eval`` attribute, which the fixed-BN helper requires; LayerNorm needs
no frozen statistics anyway. The backbone is frozen; set
``freeze_img_backbone=False`` to fine-tune it.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5.py']

img_backbone_path = 'ckpts/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'
# ConvNeXt cannot consume the inherited RayDN ResNet checkpoint.
load_from = None

# Ten LiDAR frames in total: current keyframe scan + nine history scans.
num_lidar_history = 9

# ---- camera mask for training: occ_loss_mask == mask_camera ----
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline in (train_pipeline, test_pipeline):
    for transform in pipeline:
        if transform['type'] == 'LoadOccupancyOcc3D':
            transform['train_mask_type'] = 'camera'
train_dataset_config = dict(
    pipeline=train_pipeline, num_lidar_history=num_lidar_history)
val_dataset_config = dict(
    pipeline=test_pipeline, num_lidar_history=num_lidar_history)

# ---- camera mask for evaluation: mIoU over camera-visible voxels ----
occ3d_eval_mask = 'camera'

# ---- DINOv3 ConvNeXt-Small image backbone ----
embed_dims = 128
model = dict(
    image_frozen_bn=False,
    freeze_img_backbone=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ConvNeXt',
        # ConvNeXt-Small geometry; the module default (3/3/9/3) is Tiny and
        # would silently keep only the first nine stage-2 blocks.
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        load_path=img_backbone_path),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[96, 192, 384, 768],
    ),
)
