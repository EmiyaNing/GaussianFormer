"""World-model evaluation with the Occ3D camera mask on every horizon.

The training recipe (``img_voxel_react_occ3d_world_v1_lidar10f.py``) scores the
full voxel grid (``occ3d_eval_mask='none'``).  This snapshot keeps the model,
data, checkpoint and six-future-step settings identical and only switches the
React World metric to the per-frame Occ3D ``mask_camera``:

* current frame (0s): ``occ_cam_mask`` from the current ``labels.npz``;
* 1s / 2s / 3s: the camera mask of the 2nd / 4th / 6th future target frame
  (``future_occ_cam_masks``).  Each mask lives in its target frame's native
  future-ego grid, i.e. exactly the grid of the corresponding
  ``future_occ_grids`` entry produced by the ground-truth relative-pose
  readout, so no coordinate transform is involved.

Planning L2 is mask-independent and is reported unchanged.  Collision is an
ST-P3/UniAD open-loop metric: the planned ego rectangle (1.85 x 4.084 m) is
tested against the *unmasked* GT future occupancy of vehicle/pedestrian classes
at 1s/2s/3s.  The metric prints 0s/1s/2s/3s mIoU, binary occupancy IoU, occupied
precision/recall, ``*_finite_classes``, the pooled averages, planning L2 and
``collision/1s|2s|3s`` (plus ``collision_valid/*`` over GT-safe samples and
``collision_gt/*`` as a sanity check).

Best future-prediction epoch of the completed local 24-epoch run
(``out/img_voxel_world_v1``): epoch 19 has the best 1s/2s/3s mIoU
(28.12/25.20/22.77, avg 25.36); epoch 23 has the best future IoU
(46.26/43.00/39.83) with the same 3s mIoU.  Evaluate epoch 19 first:

python eval.py \
  --py-config config/sagformer_world_model/img_voxel_react_occ3d_world_v1_eval_camera_mask.py \
  --resume-from out/img_voxel_world_v1/epoch_19.pth \
  --work-dir out/img_voxel_world_v1_eval_camera_mask

The checkpoint must be a complete React World checkpoint (perception +
``world_decoder.*``); the original perception checkpoint is only needed at
model construction time (``perception_checkpoint`` in the parent config).
"""

_base_ = ['./img_voxel_react_occ3d_world_v1_lidar10f.py']

# Applies to the current frame and to each future target's own camera mask.
# The sibling training config keeps 'none' so its history is unchanged.
occ3d_eval_mask = 'camera'
react_world_eval = True

# ST-P3/UniAD open-loop collision rate at 1s/2s/3s: the planned ego rectangle
# (1.85 x 4.084 m, +0.5 m rear-axle offset) tested against the *unmasked* GT
# future occupancy of vehicle/pedestrian classes {2,3,4,5,6,7,9,10}.  The
# camera mask is only used for the occupancy score, never for collision.
react_world_collision = 'agent'
