"""Gaussian Statistic Reporter —— 结果格式化输出。"""

import os
import json
from typing import Dict


def report_statistics(stats: Dict, logger, work_dir: str) -> None:
    """将统计结果同时输出到终端日志和 JSON 文件。

    stats: aggregator.finalize() 返回的完整统计 dict
    logger: MMLogger 实例
    work_dir: 输出目录
    """
    if 'error' in stats:
        logger.error(f"Statistics error: {stats['error']}")
        return

    # ── 终端打印 ──────────────────────────────────────────
    logger.info('=' * 60)
    logger.info('====  Gaussian Statistic Results  ====')
    logger.info('=' * 60)
    logger.info(f"  Frames processed:  {stats['num_frames']}")
    logger.info(f"  Total Gaussians:     {stats['num_gaussians']}")
    logger.info(f"  Coverage Threshold τ: {stats.get('cov_threshold', 'unknown')}")
    logger.info(f"  Coverage Scope:       {stats.get('coverage_voxel_scope', 'unknown')}")
    scale_sanity = stats.get('scale_sanity', {})
    if scale_sanity:
        logger.info(f"  Config Scale Range:   {scale_sanity.get('configured_scale_range', 'unknown')}")
        logger.info(
            f"  Observed s_hat Range: "
            f"[{scale_sanity.get('observed_min_s_hat', 0):.6f}, "
            f"{scale_sanity.get('observed_max_s_hat', 0):.6f}]")
    coverage_debug = stats.get('coverage_debug', {})
    if coverage_debug and not coverage_debug.get('coverage_counter_consistent', True):
        logger.warning(
            'Coverage counter consistency check failed: '
            f"bins={coverage_debug.get('coverage_from_bins', 0):.8f}, "
            f"scalar={coverage_debug.get('coverage_from_scalar', 0):.8f}, "
            f"diff={coverage_debug.get('coverage_counter_abs_diff', 0):.8e}")
    logger.info('─' * 60)

    # 基础指标
    logger.info(f"  Mean Scale:          {stats['mean_scale']:.6f}")
    logger.info(f"  Mean AR:             {stats['anisotropy_ratio']['mean_ar']:.6f}")
    logger.info(f"  Near-Spherical Ratio: {stats['near_spherical_ratio']:.6f}")
    logger.info(f"  LIGR:                {stats['ligr']:.6f}")
    logger.info(f"  Mean Coverage:       {stats['mean_coverage']:.6f}")
    # Purity 系列（兼容新旧字段）
    if 'mean_purity_valid' in stats:
        logger.info(f"  Mean Purity(valid):    {stats['mean_purity_valid']:.6f}")
        logger.info(f"  Mean Purity(penalized): {stats['mean_purity_penalized']:.6f}")
        logger.info(f"  Unused Gaussian Ratio: {stats['unused_gaussian_ratio']:.6f}")
    else:
        logger.info(f"  Mean Purity:         {stats['mean_purity']:.6f}")

    logger.info('─' * 60)
    logger.info('  [Scale Percentiles]')
    for k, v in stats['scale_percentiles'].items():
        logger.info(f"    {k}: {v:.6f}")

    logger.info('─' * 60)
    logger.info('  [Scale Volume]')
    for k, v in stats['scale_volume'].items():
        logger.info(f"    {k}: {v:.6f}")

    logger.info('─' * 60)
    logger.info('  [Anisotropy Ratio]')
    for k, v in stats['anisotropy_ratio'].items():
        logger.info(f"    {k}: {v:.6f}")

    # Category-wise
    logger.info('─' * 60)
    logger.info('  [Category-wise Statistics]')
    has_ratio = 'ratio' in next(iter(stats['category_stats'].values()))
    if has_ratio:
        logger.info(f"    {'Class':>6s} | {'Count':>8s} | {'Ratio':>8s} | {'MeanScale':>10s} | {'MeanAR':>8s} | {'NSR':>8s} | {'Purity':>8s}")
        logger.info('    ' + '-' * 75)
        for c, v in sorted(stats['category_stats'].items()):
            logger.info(
                f"    {int(c):>6d} | {v['count']:>8d} | "
                f"{v['ratio']:>8.6f} | {v['mean_scale']:>10.6f} | {v['mean_ar']:>8.4f} | "
                f"{v['near_spherical_ratio']:>8.4f} | {v['mean_purity']:>8.4f}"
            )
    else:
        logger.info(f"    {'Class':>6s} | {'Count':>8s} | {'MeanScale':>10s} | {'MeanAR':>8s} | {'NSR':>8s} | {'Purity':>8s}")
        logger.info('    ' + '-' * 65)
        for c, v in sorted(stats['category_stats'].items()):
            logger.info(
                f"    {int(c):>6d} | {v['count']:>8d} | "
                f"{v['mean_scale']:>10.6f} | {v['mean_ar']:>8.4f} | "
                f"{v['near_spherical_ratio']:>8.4f} | {v['mean_purity']:>8.4f}"
            )

    # Distance-wise
    logger.info('─' * 60)
    logger.info('  [Distance-wise Scale/Shape]')
    logger.info(f"    {'Bin':>12s} | {'Count':>8s} | {'MeanScale':>10s} | {'MeanAR':>8s} | {'NSR':>8s}")
    logger.info('    ' + '-' * 58)
    for label, v in stats['distance_stats'].items():
        logger.info(
            f"    {label:>12s} | {v['count']:>8d} | "
            f"{v['mean_scale']:>10.6f} | {v['mean_ar']:>8.4f} | "
            f"{v['near_spherical_ratio']:>8.4f}"
        )

    # Distance-wise Coverage
    scope_label = stats.get('coverage_voxel_scope', 'unknown')
    logger.info('─' * 60)
    logger.info(f'  [Distance-wise {scope_label.capitalize()} Coverage]')
    for label, v in stats['distancewise_coverage'].items():
        logger.info(f"    {label:>12s}: {v:.4f}")

    logger.info('=' * 60)

    # ── JSON 文件写入 ─────────────────────────────────────
    json_path = os.path.join(work_dir, 'gaussian_statistic_result.json')
    serializable = _make_serializable(stats)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f'Statistics JSON saved to: {json_path}')


def _make_serializable(obj):
    """递归将不可序列化类型（如 numpy 类型）转为 Python 原生类型。"""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
