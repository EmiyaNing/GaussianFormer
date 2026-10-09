"""Five-epoch ablation: camera mask and trainable rendered empty bias."""
_base_ = ['./opusv1_gaussian_softplus_camera_mask.py']

model = dict(head=dict(type='OPUSV1GaussianTrainableBiasHead',
                       empty_logit_offset=0.0, initial_empty_bias=0.0))
optimizer = dict(paramwise_cfg=dict(custom_keys={
    'head.empty_bias': dict(lr_mult=25.0, decay_mult=0.0),
}))
