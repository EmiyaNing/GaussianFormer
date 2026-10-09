"""Full-val protocol check for A4 (Gaussian kernel, fixed isotropic geometry).

Same rationale as ``p4_fullval_a0.py``: only the val imageset changes, back to
the full 150-scene split, so the row can sit next to the reduced-split A4 result
without mixing protocols in one table (plan T3.1).
"""
_base_ = ['./p1_a4_fixed_gaussian.py']

val_dataset_config = dict(imageset='data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl')
