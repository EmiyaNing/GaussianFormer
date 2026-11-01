"""Keep parameter-group learning-rate ratios during timm scheduling."""


def prepare_proportional_lr_groups(optimizer, reference_lr):
    """Use timm's lr_scale for the whole curve, including warm-up and floor.

    MMEngine has already applied lr_mult to each group's initial learning rate.
    Normalize the scheduler's base rates, then let timm multiply every scheduled
    value by that group's original ratio. Call before constructing the scheduler.
    """
    if reference_lr <= 0:
        raise ValueError('The reference learning rate must be positive')
    for group in optimizer.param_groups:
        group['lr_scale'] = group['lr'] / reference_lr
        group['lr'] = reference_lr
        group['initial_lr'] = reference_lr
