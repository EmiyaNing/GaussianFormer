"""Five-epoch ablation: camera-visible occupancy supervision, full OPUS set."""
_base_ = ['./opusv1_gaussian_softplus.py']

loss_input_convertion = dict(occ_mask='occ_cam_mask')
