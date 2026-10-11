"""Full Occ3D, ten epochs from pretrained backbone, without cuDNN search."""
_base_ = ['../full_occ3d_5e/opusv1_gaussian_softplus_learned_bias.py']

max_epochs = 10
eval_every_epochs = 1
cudnn_benchmark = False
cuda_memory_log_interval = 50
seed = 928
