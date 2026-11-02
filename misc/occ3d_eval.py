"""Shared evaluation-mask policy for training validation and standalone eval."""

MASK_NAMES = ("none", "camera", "lidar", "nonempty")


def get_occ3d_eval_mask_name(cfg):
    mask_name = cfg.get('occ3d_eval_mask')
    if mask_name is None:
        # Retain explicitly supplied legacy flags; absence means full volume.
        mask_name = 'camera' if cfg.get('eval_mask_flag', False) else 'none'
    if mask_name not in MASK_NAMES:
        raise ValueError(f'Unsupported Occ3D eval mask: {mask_name!r}; choose from {MASK_NAMES}')
    return mask_name


def configure_occ3d_eval_mask(cfg, override=None):
    """Apply a CLI override and record the effective policy in resolved config."""
    if override is not None:
        cfg['occ3d_eval_mask'] = override
    mask_name = get_occ3d_eval_mask_name(cfg)
    cfg['occ3d_eval_mask'] = mask_name
    return mask_name


def add_occ3d_eval_mask_argument(parser):
    parser.add_argument(
        '--occ3d-eval-mask', choices=MASK_NAMES, default=None,
        help='Occ3D evaluation mask; overrides config (default: none unless configured explicitly)')


def select_occ3d_eval_mask(metas, idx, cfg, expected_numel=None):
    mask_name = get_occ3d_eval_mask_name(cfg)
    if mask_name == 'none':
        return None
    key = {'camera': 'occ_cam_mask', 'lidar': 'occ_lidar_mask',
           'nonempty': 'occ_nonempty_mask'}[mask_name]
    mask = metas.get(key)
    if mask is None and mask_name == 'nonempty':
        mask = metas.get('occ_mask')
    if mask is None:
        raise ValueError(f'Occ3D eval mask {mask_name!r} was requested, but {key!r} is missing')
    # Raw local-rank metadata keeps the leading batch dimension, including
    # tail batches. Do not index globally gathered/head-adapted masks.
    if mask.dim() >= 2:
        if idx >= mask.shape[0]:
            raise IndexError(f'Occ3D mask batch index {idx} is outside local shape {tuple(mask.shape)}')
        mask = mask[idx]
    mask = mask.reshape(-1).bool()
    if expected_numel is not None and mask.numel() != expected_numel:
        raise ValueError(
            'Occ3D evaluation mask/prediction size mismatch on one local rank: '
            f'mask={mask.numel()}, prediction={expected_numel}. '
            'Check that occ masks retain [B, X, Y, Z] or [B, V] layout.')
    return mask


def add_occ3d_eval_batch(metric, prediction, target, metas, idx, cfg):
    """Accumulate one local sample into the unmasked NumPy Occ3D metric."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if prediction.numel() != target.numel():
        raise ValueError('Occ3D prediction and target must contain the same number of voxels')
    mask = select_occ3d_eval_mask(metas, idx, cfg, expected_numel=prediction.numel())
    if mask is not None:
        prediction, target = prediction[mask], target[mask]
    metric.add_batch(prediction.cpu().numpy(), target.cpu().numpy(), None, None)
