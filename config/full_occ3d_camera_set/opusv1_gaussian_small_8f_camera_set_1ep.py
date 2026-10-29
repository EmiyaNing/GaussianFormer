"""Fine-tune small for one full epoch with camera-filtered Set targets.

All six Set stages retain only camera-visible foreground GT before both KNN
directions and classification-target assignment. Predicted GS are not filtered
by camera visibility. Occupancy supervision already uses the same camera mask.

Single-GPU launch (gradient accumulation is a train.py CLI option):
CUDA_VISIBLE_DEVICES=0 python3.10 -u train.py \
    --py-config config/full_occ3d_camera_set/opusv1_gaussian_small_8f_camera_set_1ep.py \
    --work-dir work_dirs/small_camera_set_full_1ep --gradient-accumulation 2
"""
_base_ = [
    '../full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_small_8f.py',
]

# Start a new fine-tuning schedule from the trained model parameters.
load_from = 'ckpts/opus_gaussian_8f_small.pth'
train_load_from = load_from
train_load_revise_keys = []
train_load_strict = True
strict_load = True
max_epochs = 1
eval_every_epochs = 1
occ3d_eval_mask = 'camera'
optimizer = dict(optimizer=dict(lr=2e-5))
warmup_iters = 400
warmup_lr_init = 2e-6
min_lr_ratio = 0.1
gradient_accumulation = 2

# Reuse the existing domain-aware implementation of the original Set loss.
# Setting both domains is necessary: gt_domain applies to stages 1--5, while
# final_gt_domain applies to stage 6. Keep all original term weights.
loss = _base_.loss
loss.loss_cfgs[0].type = 'CenterSupportDiagnosticSetLoss'
loss.loss_cfgs[0].gt_domain = 'camera'
loss.loss_cfgs[0].final_gt_domain = 'camera'
loss.loss_cfgs[0].final_geometry_weight = 1.0
loss.loss_cfgs[0].final_classification_weight = 1.0

# Make the Occupancy CE/Lovasz input explicit, as in the parent configuration.
loss_input_convertion = dict(occ_mask='occ_cam_mask')
