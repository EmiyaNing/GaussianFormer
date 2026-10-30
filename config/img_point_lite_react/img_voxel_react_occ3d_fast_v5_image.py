"""Image BF16/channels_last + fixed BN, with FP32 geometry and losses.

This changes image arithmetic. Validate full training mIoU before adopting.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v4.py']
amp = False  # Image autocast is local; geometry, neck, decoder and loss are FP32.
model = dict(image_backbone_bf16=True, image_channels_last=True,
             image_frozen_bn=True)
