"""Full-val protocol check for A5 Gaussian cap=0 (plan T3.1 anchor reproduction).

``plans/sparse_render_plans/P3_fair_end_to_end_experiments.md`` T3.1 makes two
hard requirements that this config exists to satisfy:

1. The published SparseWorldStrict anchors must be reproduced **before** the main
   experiments, "otherwise the A0 baseline is not trustworthy": 0/2/4/6 horizon,
   no camera/lidar mask, Avg mIoU **13.20**, Avg occupied IoU **22.03**.
2. Camera mask / no-mask, epoch count, image resolution and temporal frame count
   must never be mixed.

Every P1 result in this project was produced on a *reduced* val split, and
``tools/make_reduced_split.py`` keeps ``scene_order[:N]`` -- the first N scenes
**in sorted order**, not a random sample.  For val that is 4 scenes / 161
keyframes / 113 evaluated samples out of 150 scenes / 6019 keyframes.  So the
anchors were never checked, and the reduced subset is not representative.

This config changes **only** the val imageset, back to the full split.  The
model, checkpoint, resolution and frame count are untouched, so a difference
from the reduced-split number is purely a protocol/data-coverage effect.

Kept out of ``p1_eval.json`` on purpose: mixing full-val and reduced-split rows
in one table is exactly the protocol mixing T3.1 forbids.
"""
_base_ = ['./p1_a5_gaussian.py']

val_dataset_config = dict(imageset='data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl')
