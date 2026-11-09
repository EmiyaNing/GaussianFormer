"""Four-sample smoke check for the camera-mask world evaluation.

One sample per GPU on a four-GPU machine: it exercises the future-label loading,
the per-frame camera masks, the metric reduction and the planning readout
without waiting for the full 4220-sample validation split.  The resulting
numbers are not a performance measurement.
"""

_base_ = ['./img_voxel_react_occ3d_world_v1_eval_camera_mask.py']

# Raw keyframe indices away from scene starts where the current sensors and the
# six future occupancy targets exist. 136/264 match the regular training smoke;
# 392/500 add two more distinct scenes so a four-GPU run has one sample per rank.
val_dataset_config = dict(vis_indices=[136, 264, 392, 500])
val_loader = dict(num_workers=0)
print_freq = 1
