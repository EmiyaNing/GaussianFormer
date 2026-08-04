"""共享的点云场景熵、体素 DECI 与组合熵计算。"""

import numpy as np


DEFAULT_VOXEL_SIZE = (0.5, 0.5, 0.5)
DEFAULT_DECI_BATCH_SIZE = 65536


def compute_deci_values_batched(
    voxels,
    num_points,
    voxel_size=DEFAULT_VOXEL_SIZE,
    batch_size=DEFAULT_DECI_BATCH_SIZE,
):
    """批量计算所有非空体素的 DECI，保持原逐体素公式。"""
    voxel_count = len(num_points)
    values = np.zeros(voxel_count, dtype=np.float64)
    if voxel_count == 0:
        return values

    # 现有 DECI 公式使用标量 vsize；当前配置是各向同性体素。
    vsize = float(voxel_size[0])
    if not np.allclose(voxel_size, vsize):
        raise ValueError('DECI currently requires isotropic voxel_size.')

    max_points = voxels.shape[1]
    point_indices = np.arange(max_points)[None, :]

    for start in range(0, voxel_count, batch_size):
        end = min(start + batch_size, voxel_count)
        batch_counts = np.asarray(num_points[start:end], dtype=np.int64)
        valid_indices = np.flatnonzero(batch_counts > 1)
        if len(valid_indices) == 0:
            continue
        counts = batch_counts[valid_indices]

        # DECI(count<=1) 恒为0，只为有效体素构造协方差和执行 eigvalsh。
        # 保持输入 dtype，以贴近原 float32 逐体素计算路径。
        points = np.asarray(
            voxels[start:end, :, :3][valid_indices]
        ) / vsize
        mask = point_indices < counts[:, None]
        points = points * mask[..., None]
        means = points.sum(axis=1) / np.maximum(counts[:, None], 1)
        centered = (points - means[:, None, :]) * mask[..., None]
        covariance = np.einsum(
            'bpi,bpj->bij', centered, centered, optimize=True
        ) / np.maximum(counts - 1, 1)[:, None, None]

        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sort(np.abs(eigenvalues), axis=1)[:, ::-1]
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        ranks = np.sum(
            eigenvalues > 0.01 * eigenvalues[:, :1], axis=1
        ).astype(np.int64)
        ranks = np.minimum(ranks, np.minimum(counts - 1, 3))
        normalized = eigenvalues / eigenvalues.sum(axis=1, keepdims=True)

        products = np.ones(len(valid_indices), dtype=np.float64)
        for rank in (1, 2, 3):
            selected = ranks == rank
            if np.any(selected):
                products[selected] = np.prod(
                    normalized[selected, :rank], axis=1
                )

        constants = np.power(2.0 * np.pi * np.e, ranks)
        entropy = 0.5 * np.log(constants * products + 1.0)
        valid_entropy = (ranks > 0) & (entropy > 0)
        batch_values = np.zeros(len(valid_indices), dtype=np.float64)
        batch_values[valid_entropy] = 1.0 / entropy[valid_entropy]
        values[start + valid_indices] = batch_values

    return values


def compute_composite_entropy(
    points,
    voxel_generator,
    voxel_size=DEFAULT_VOXEL_SIZE,
    scene_weight=0.4,
    local_weight=0.6,
    deci_batch_size=DEFAULT_DECI_BATCH_SIZE,
    deci_topk=64,
):
    """一次体素化计算 H、Top-K DECI mean 与组合熵。"""
    if deci_topk < 1:
        raise ValueError('deci_topk must be at least 1.')
    if points.shape[0] == 0:
        return dict(H=0.0, deci_max=0.0, deci_topk_mean=0.0,
                    local_entropy=0.0, combined=0.0,
                    voxel_count=0, point_count=0)

    voxels, coordinates, num_points = voxel_generator.generate(points)
    if len(num_points) == 0:
        return dict(H=0.0, deci_max=0.0, deci_topk_mean=0.0,
                    local_entropy=0.0, combined=0.0,
                    voxel_count=0, point_count=0)

    counts = np.asarray(num_points, dtype=np.int64)
    total_points = int(counts.sum())
    probabilities = counts.astype(np.float64) / total_points
    scene_entropy = float(-np.sum(probabilities * np.log(probabilities)))
    deci_values = compute_deci_values_batched(
        voxels,
        counts,
        voxel_size=voxel_size,
        batch_size=deci_batch_size,
    )
    deci_max = float(np.max(deci_values)) if len(deci_values) else 0.0
    actual_topk = min(int(deci_topk), len(deci_values))
    if actual_topk > 0:
        topk_values = np.partition(
            deci_values, len(deci_values) - actual_topk
        )[-actual_topk:]
        deci_topk_mean = float(np.mean(topk_values))
    else:
        deci_topk_mean = 0.0
    combined = scene_weight * scene_entropy + local_weight * deci_topk_mean

    return dict(
        H=scene_entropy,
        deci_max=deci_max,
        deci_topk_mean=deci_topk_mean,
        local_entropy=deci_topk_mean,
        combined=float(combined),
        voxel_count=int(len(coordinates)),
        point_count=total_points,
    )
