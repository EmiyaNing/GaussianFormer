"""1.5x final candidates; occupancy-trained query scores; camera-filtered Set."""
_base_ = ['../full_occ3d_camera_set/opusv1_gaussian_small_8f_camera_set_1ep.py']

custom_imports = dict(imports=[
    'model.encoder.opus_v1_gaussian_budget_encoder',
    'model.head.opus_v1_gaussian_budget_head',
    'loss.gaussian_budget_loss',
], allow_failed_imports=False)

model = dict(
    encoder=dict(type='OPUSV1GaussianBudgetEncoder',
        num_refines=[1, 4, 8, 16, 32, 96], keep_budget=76800,
        budget_total_iters=28130, budget_temperature_start=.5,
        budget_temperature_end=.15, importance_hidden_dims=64),
    head=dict(type='OPUSV1GaussianBudgetHead'))

budget_source_checkpoint = 'ckpts/opus_gaussian_8f_small.pth'
budget_source_children = 64
budget_extra_offset_m = .02
budget_extra_opacity_logit_shift = -1.0
load_from = 'work_dirs/small_query_budget_camera_full_1ep/initial_model.pth'
train_load_from = load_from
train_load_revise_keys = []
train_load_strict = True
strict_load = True

loss = _base_.loss
loss.loss_cfgs.append(dict(type='GaussianBudgetLoss', binary_weight=.1,
    total_iters=28130, warmup_fraction=.1, ramp_fraction=.2))
loss_input_convertion = dict(allocation='allocation')

optimizer = dict(paramwise_cfg=dict(custom_keys={
    'encoder.importance_branch': dict(lr_mult=10.0, decay_mult=0.0)}))
max_epochs = 1
eval_every_epochs = 1
gradient_accumulation = 2
print_freq = 50
fail_on_nonfinite_grad = True
