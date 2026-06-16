import numpy as np


ADAPTIVE_GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float32)
ADAPTIVE_RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ADAPTIVE_BLUE = np.array([0.0, 0.25, 1.0], dtype=np.float32)


def get_adaptive_gaussian_colors(pred, scales, opacities=None, seed=42):
    """Choose Gaussian colors from opacity and raw mean scale."""
    pred = np.asarray(pred, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.float32)

    if pred.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if opacities is None:
        opacities = np.ones(len(pred), dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)

    mean_size = scales.mean(axis=-1)
    out_colors = np.tile(ADAPTIVE_GRAY[None, :], (len(pred), 1))

    high_opacity = opacities > 0.7
    out_colors[high_opacity & (mean_size > 0.35)] = ADAPTIVE_RED
    out_colors[high_opacity & (mean_size <= 0.35)] = ADAPTIVE_BLUE

    return out_colors
