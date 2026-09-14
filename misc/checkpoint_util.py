def refine_load_from_sd(sd):
    for k in list(sd.keys()):
        if 'img_neck.' in k or 'lifter.anchor' in k:
            del sd[k]
    return sd


def select_model_state_dict(checkpoint, prefer_ema=False):
    """Select model weights while remaining compatible with legacy files."""
    if not isinstance(checkpoint, dict):
        return checkpoint, 'checkpoint'
    if prefer_ema and checkpoint.get('ema_state_dict') is not None:
        return checkpoint['ema_state_dict'], 'ema_state_dict'
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict'], 'state_dict'
    return checkpoint, 'checkpoint'
