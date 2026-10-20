"""Full Occ3D, eight-frame OPUS with GF2 probability rendering.

Reuse the established temporal/data/optimizer recipe. Keep the OPUS Set
objective, but consume rendered probabilities directly in NLL and Lovasz.
No experiment tracing, empty Gaussian, or learned empty-logit offset.
"""
_base_ = ['../full_occ3d_oups_guassian/_base_8f.py']

custom_imports = dict(
    imports=['model.head.opus_gf2_probability_head'],
    allow_failed_imports=False)

model = dict(head=dict(
    _delete_=True,
    type='OPUSGF2ProbabilityHead',
    apply_loss_type='fixed_5',
    max_render_scale=0.40,
    num_classes=18,
    empty_label=17,
    with_empty=False,
    dataset_type='nusc',
    cuda_kwargs=dict(scale_multiplier=3, H=200, W=200, D=16,
                     pc_min=[-40., -40., -1.], grid_size=0.4)))

# The inherited list contains the original OPUSSetLoss and OccupancyLoss.
# False selects probability NLL as well as bypassing softmax before Lovasz.
loss = _base_.loss
loss.loss_cfgs[1].lovasz_use_softmax = False

# Remove the old head.empty_bias learning-rate multiplier entirely.
optimizer = dict(paramwise_cfg=dict(custom_keys=dict(
    _delete_=True,
    img_backbone=dict(lr_mult=0.1),
    sampling_offset=dict(lr_mult=0.1))))

# Occupancy supervision/evaluation use the camera mask; Set targets remain
# all nonempty GT, with the inherited camera-dependent distance weighting.
loss_input_convertion = dict(occ_mask='occ_cam_mask')
occ3d_eval_mask = 'camera'
