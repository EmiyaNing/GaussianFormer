"""Five-epoch ablation: camera mask and true-class-versus-empty margin."""
_base_ = ['./opusv1_gaussian_softplus_camera_mask.py']

loss = dict(type='MultiLossWithMargin',
            margin_cfg=dict(type='OccupancyClassAwareMarginLoss',
                            empty_label=17, weight=1.0))
