"""Full Occ3D, OPUS-V1 temporal Gaussian + camera mask + learned bias.

Size schedules match OPUS paper Table 7 (arXiv:2409.09350v2).
T/L source revision: fdb17339f40168ea0ceda9615863e3f5d0c3f5ce.
Current + seven history frames. All eight frames retain backbone gradients.
"""
_base_ = ['../full_occ3d_5e/opusv1_gaussian_softplus_learned_bias.py']

num_frames = 8
max_epochs = 30
eval_every_epochs = 1
seed = 928
batch_size = 1
gradient_accumulation = 8
cudnn_benchmark = False
cuda_memory_log_interval = 50
amp = False  # BF16 image backbone only; FPN, decoder, Gaussian head/loss FP32.
fail_on_nonfinite_grad = True
grad_max_norm = 35
warmup_iters = 4000  # 500 effective batch-8 updates on this single-GPU runner.
warmup_lr_init = 2e-4 / 3
warmup_ratio = 1.0 / 3
min_lr_ratio = 1e-3
scheduler_step_before_optimizer = True
load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
train_load_revise_keys = [('backbone.', 'img_backbone.')]

pc_range = [-40., -40., -1., 40., 40., 5.4]
occ3d_path = 'data/occ3d/gts'
input_shape = (704, 256)
data_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704),
                     bot_pct_lim=(0., 0.), rot_lim=(0., 0.),
                     H=900, W=1600, rand_flip=True)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOPUSV1ImageHistory', num_history=7, num_cams=6,
         test_mode=False, interval=(4, 8), to_float32=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=0.4, model_coord='ego', train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=48),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOPUSV1ImageHistory', num_history=7, num_cams=6,
         test_mode=True, interval=(4, 8), to_float32=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=0.4, model_coord='ego', train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=48),
]
train_dataset_config = dict(data_aug_conf=data_aug_conf, pipeline=train_pipeline)
val_dataset_config = dict(data_aug_conf=data_aug_conf, pipeline=test_pipeline)
train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)
model = dict(type='OPUSTemporalSegmentor', image_chunk_size=6,
             image_checkpoint=True, image_amp_dtype='bfloat16',
             # Chunk checkpoints already cover the backbone. Avoid mmseg's
             # reentrant block checkpoints on parameters shared across chunks.
             img_backbone=dict(with_cp=False),
             encoder=dict(type='OPUSV1GaussianTemporalEncoder',
                          num_frames=8, decoder_checkpoint=True))
