import numpy as np


ADAPTIVE_GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float32)
ADAPTIVE_RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ADAPTIVE_BLUE = np.array([0.0, 0.25, 1.0], dtype=np.float32)


def get_adaptive_gaussian_colors(pred, scales, opacities=None, seed=42):
    """Choose Gaussian colors by randomly splitting high-opacity Gaussians."""
    pred = np.asarray(pred, dtype=np.int64)

    if pred.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if opacities is None:
        opacities = np.ones(len(pred), dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)

    out_colors = np.tile(ADAPTIVE_GRAY[None, :], (len(pred), 1))

    high_opacity_indices = np.where(opacities > 0.7)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(high_opacity_indices)
    red_count = len(high_opacity_indices) // 2
    out_colors[high_opacity_indices[:red_count]] = ADAPTIVE_RED
    out_colors[high_opacity_indices[red_count:]] = ADAPTIVE_BLUE

    return out_colors
