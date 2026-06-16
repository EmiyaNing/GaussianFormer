import numpy as np


ADAPTIVE_GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float32)
ADAPTIVE_RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ADAPTIVE_BLUE = np.array([0.0, 0.25, 1.0], dtype=np.float32)


def get_adaptive_gaussian_colors(pred, scales, seed=42):
    """Choose Gaussian colors from semantic labels and raw mean scale."""
    pred = np.asarray(pred, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float32)

    if pred.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    mean_size = scales.mean(axis=-1)
    out_colors = np.tile(ADAPTIVE_GRAY[None, :], (len(pred), 1))

    static_group = np.isin(pred, [14, 13, 10, 11])
    out_colors[static_group & (mean_size <= 0.4)] = ADAPTIVE_RED

    mixed_group = np.isin(pred, [1, 2, 4, 7, 8, 9])
    mixed_small = mixed_group & (mean_size <= 0.3)
    mixed_large = mixed_group & (mean_size > 0.3)
    out_colors[mixed_small] = ADAPTIVE_RED

    mixed_large_indices = np.where(mixed_large)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(mixed_large_indices)
    blue_count = len(mixed_large_indices) // 2
    out_colors[mixed_large_indices[:blue_count]] = ADAPTIVE_BLUE
    out_colors[mixed_large_indices[blue_count:]] = ADAPTIVE_GRAY

    return out_colors
