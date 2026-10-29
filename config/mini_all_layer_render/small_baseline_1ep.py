"""Matched original Small: six Set losses, final-stage occupancy only."""
_base_ = ['../full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_small_8f.py']
custom_imports = dict(imports=['loss.all_layer_occupancy_loss',
    'model.head.opus_all_layer_experiment_head'], allow_failed_imports=False)
model = dict(head=dict(type='OPUSAllLayerExperimentHead', render_checkpoint=True))
load_from = 'ckpts/opus_gaussian_8f_small.pth'
train_dataset_config = dict(imageset='work_dirs/all_layer_render_small_mini_v1/annotations/train_4090.pkl')
val_dataset_config = dict(imageset='data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl')
max_epochs = 1
seed = 928
gradient_accumulation = 2
optimizer = dict(optimizer=dict(lr=2e-5))
finetune_warmup_updates = 50
finetune_min_ratio = .1
save_every_updates = 128
expected_train_frames = 4090
expected_train_scenes = 698
expected_val_frames = 6019
expected_val_scenes = 150
