"""Four-frame (img4f) perceiver with the Occ3D camera mask in train AND eval, five epochs.

Architecture, data (ten LiDAR frames, current + three history images, 704x256),
optimizer and warm start are inherited unchanged from
``img_voxel_react_occ3d_fast_v5_lidar10f_img4f.py``. Only the mask protocol and
the epoch budget differ:

* Training: ``LoadOccupancyOcc3D(train_mask_type='camera')`` makes
  ``occ_loss_mask`` equal to ``labels.npz``'s ``mask_camera`` (~24.5 % of the
  ``200 x 200 x 16`` grid), so ``OccupancyLoss`` supervises only the voxels the
  six cameras can actually see. The inherited recipe supervises the full grid
  (``train_mask_type='none'``). The validation pipeline is flipped too so the
  validation loss uses the same voxel set; the metrics read ``occ_cam_mask``
  directly and do not depend on this switch.
* Evaluation: ``occ3d_eval_mask='camera'``. The in-training ``MeanIoU``
  (``train.py``) and ``Metric_mIoU(use_image_mask=True)`` (``eval.py``) both
  filter prediction and ground truth with ``mask_camera`` before accumulating
  the 18-class confusion matrix -- the official Occ3D camera-mask protocol. The
  inherited img4f default is ``'none'`` (full grid), which yields a different,
  lower mIoU.

Five epochs: this recipe trains with ``onecyclelr=False``, so the cosine
schedule is built from ``len(train_loader) * max_epochs``; ``max_epochs=5``
therefore also shrinks the LR schedule to these five epochs. One checkpoint per
epoch is written (``epoch_1.pth`` ... ``epoch_5.pth``) and validation runs after
every epoch, so intermediate epochs can be compared.

Warm start: ``load_from`` is inherited from the img4f recipe
(``ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth``; only the two small temporal
bias vectors are new). To instead fine-tune the already trained no-mask img4f
checkpoint under the camera mask, set
``train_load_from = 'ckpts/img_voxel_react_occ3d_fast_v5_lidar10f_img4f.pth'``.

Example:

```bash
python train.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_lidar10f_img4f_camera_mask_5e.py \
  --work-dir out/react_img4f_camera_mask_5e
```

The resulting checkpoint is evaluated with the dedicated camera-mask entry point
(it already sets ``occ3d_eval_mask='camera'`` and additionally reports RayIoU):

```bash
python eval.py \
  --py-config config/img_point_lite_react/img_voxel_react_occ3d_fast_v5_lidar10f_img4f_eval_camera_mask.py \
  --work-dir out/react_img4f_camera_mask_5e_eval \
  --resume-from out/react_img4f_camera_mask_5e/latest.pth
```
"""
_base_ = ['./img_voxel_react_occ3d_fast_v5_lidar10f_img4f.py']

# ---- camera mask for training: occ_loss_mask == labels.npz mask_camera ----
# Both pipelines are updated so that the validation loss uses the same voxel
# set as training. Rebuild the dataset configs explicitly because the inherited
# ones capture the pipeline lists built by the parent config.
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline in (train_pipeline, test_pipeline):
    for transform in pipeline:
        if transform['type'] == 'LoadOccupancyOcc3D':
            transform['train_mask_type'] = 'camera'

train_dataset_config = dict(
    pipeline=train_pipeline,
    return_keys=list(_base_.train_dataset_config['return_keys']))
val_dataset_config = dict(
    pipeline=test_pipeline,
    return_keys=list(_base_.val_dataset_config['return_keys']))

# ---- camera mask for evaluation: in-training MeanIoU and eval.py mIoU ----
# 'camera' filters prediction and GT with mask_camera. An unset key is NOT
# equivalent to 'camera' or to 'none': it falls back to eval_mask_flag, and the
# inherited img4f default is 'none'.
occ3d_eval_mask = 'camera'
eval_mask_flag = True

# ---- schedule: five epochs ----
max_epochs = 5
eval_every_epochs = 1
