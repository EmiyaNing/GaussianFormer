"""Mini Occ3D: temporal tiny, full-grid supervision and evaluation, no camera mask."""
_base_ = ['../full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py']

max_epochs = 5
eval_every_epochs = 1
occ3d_eval_mask = 'none'
eval_mask_flag = False
loss_input_convertion = dict(occ_mask='occ_loss_mask')

# OPUSSetLoss also reads visibility to weight GT-to-prediction distances.
# With visibility absent it uses every nonempty GT as eligible for weighting.
# Keep the original annotation intact, but never expose its camera mask to
# either the model or the loss. Both pipelines already use train_mask_type=none.
occ3d_return_keys = [key for key in _base_.occ3d_return_keys if key != 'occ_cam_mask']
train_dataset_config = dict(imageset='data/mini_occ3d_4096_1024/train.pkl',
                            return_keys=occ3d_return_keys)
val_dataset_config = dict(imageset='data/mini_occ3d_4096_1024/val.pkl',
                          return_keys=occ3d_return_keys)
