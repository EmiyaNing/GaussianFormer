"""Same training/model recipe as the main config; evaluate without a mask."""

_base_ = ['./opusv1_voxel_fusion_t_r50_704x256_8f_occ3d.py']

occ3d_eval_mask = 'none'
