"""Official Occ3D evaluation for the four-frame (img4f) perceiver.

Two official protocols are enabled at once:

* ``occ3d_eval_mask = 'camera'`` — before the 18-class mIoU is accumulated,
  both the prediction and the ground truth are filtered with the ``mask_camera``
  visibility mask of ``labels.npz``. This is the Occ3D camera-mask protocol,
  the counterpart of the ``occ3d_eval_mask='none'`` number reported for this
  perceiver during training.
* ``occ3d_ray_iou = True`` — additionally report the official RayIoU from the
  LiDAR ray pattern cast at the scene's keyframe LiDAR origins, at 1 m / 2 m /
  4 m tolerances. RayIoU uses the unmasked prediction and ground truth; it is
  a ray-level metric, not the voxel ``mIoU`` and not the binary ``iou2``.

``LoadOcc3DRayOrigins`` appends the per-sample ``ray_origins`` ``[T, 3]``
(ego frame, at most eight keyframe origins) that the RayIoU accumulator needs.
The transform only reads ``history_context`` metadata, so it leaves the image,
LiDAR and occupancy pipeline untouched.

Example:

```bash
python eval.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_lidar10f_img4f_eval_camera_mask.py \
  --work-dir out/react_img4f_eval_camera_mask
```

``--resume-from`` can still point at another img4f or world checkpoint and
takes precedence over ``load_from``.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v5_lidar10f_img4f.py']

occ3d_eval_mask = 'camera'
eval_mask_flag = True
occ3d_ray_iou = True

# Evaluate the locally trained four-frame perceiver by default. ``strict_load``
# requires exactly the 639 perception tensors of that checkpoint.
load_from = 'ckpts/img_voxel_react_occ3d_fast_v5_lidar10f_img4f.pth'
strict_load = True

# RayIoU needs the keyframe LiDAR origins of each scene, in the current ego frame.
val_pipeline = list(_base_.val_dataset_config['pipeline'])
val_pipeline.append(dict(type='LoadOcc3DRayOrigins', max_origins=8, range_limit=39.0))
val_dataset_config = dict(
    pipeline=val_pipeline,
    return_keys=list(_base_.val_dataset_config['return_keys']) + ['ray_origins'])
