"""Uniform 101-frame subset of the official Occ3D eval.

Same protocol as ``..._eval_camera_mask.py`` (camera-mask mIoU + RayIoU), but
only every 60th validation frame (6019 frames -> 101).  Useful to check the
metric plumbing and to get a stable +/-1 mIoU indication in about one minute
instead of the ~65 min full evaluation.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v5_lidar10f_img4f_eval_camera_mask.py']

val_dataset_config = dict(vis_indices=list(range(0, 6019, 60)))
print_freq = 20
