"""Full Occ3D five-epoch OPUSv1 Gaussian sparse-render experiment."""
_base_ = ['../mini_occ3d_4096_1024/opusv1_gaussian_softplus_learned_bias.py']

train_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl')
val_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl')

max_epochs = 5
eval_every_epochs = 1
seed = 928
