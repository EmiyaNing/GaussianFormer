"""Five-epoch ablation: direct occupied-vs-empty margin supervision."""
_base_ = ['./opusv1_gaussian_sparse_render.py']

loss = dict(
    type='MultiLossWithMargin',
    margin_cfg=dict(type='OccupancyMarginLoss', empty_label=17,
                    bce_weight=1.0, dice_weight=1.0, weight=1.0,
                    max_positive_weight=50.0),
)
model = dict(head=dict(empty_logit_offset=0.0))
