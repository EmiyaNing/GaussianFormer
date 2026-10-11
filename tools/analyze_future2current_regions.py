#!/usr/bin/env python3
"""Post-hoc dynamic/static analysis of saved Future2Current confusion matrices.

No model is loaded and no training or GPU evaluation is performed. Histograms
follow the Occ3D convention: rows are GT classes, columns are predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free',
)
DYNAMIC = (2, 3, 4, 5, 6, 7, 9, 10)
STATIC = (1, 8, 11, 12, 13, 14, 15, 16)
GROUPS = {'dynamic': DYNAMIC, 'static': STATIC}
HORIZONS = ('1s', '2s', '3s')
MEASURES = ('macro_iou', 'gt_exact_accuracy', 'group_iou')


def grouped_metrics(hist):
    """Return percentage metrics for (..., 18, 18) GT-by-pred histograms."""
    hist = np.asarray(hist, dtype=np.float64)
    if hist.shape[-2:] != (18, 18):
        raise ValueError(f'Expected 18x18 histograms, got {hist.shape}')
    if not np.isfinite(hist).all() or (hist < 0).any():
        raise ValueError('Histogram must contain nonnegative finite counts')
    diagonal = np.diagonal(hist, axis1=-2, axis2=-1)
    union = hist.sum(axis=-1) + hist.sum(axis=-2) - diagonal
    iou = np.full_like(union, np.nan, dtype=np.float64)
    np.divide(100.0 * diagonal, union, out=iou, where=union > 0)
    result = {'class_iou': iou}
    for name, ids in GROUPS.items():
        selected = iou[..., ids]
        valid = np.isfinite(selected)
        macro = np.divide(np.nansum(selected, axis=-1), valid.sum(axis=-1),
                          out=np.full(selected.shape[:-1], np.nan),
                          where=valid.sum(axis=-1) > 0)
        gt_count = hist[..., ids, :].sum(axis=(-1, -2))
        exact_count = diagonal[..., ids].sum(axis=-1)
        accuracy = np.divide(100.0 * exact_count, gt_count,
                             out=np.full_like(gt_count, np.nan), where=gt_count > 0)
        pred_count = hist[..., :, ids].sum(axis=(-1, -2))
        group_intersection = hist[..., ids, :][..., :, ids].sum(axis=(-1, -2))
        group_union = gt_count + pred_count - group_intersection
        group_iou = np.divide(100.0 * group_intersection, group_union,
                              out=np.full_like(group_union, np.nan),
                              where=group_union > 0)
        result[name] = dict(macro_iou=macro, gt_exact_accuracy=accuracy,
                            group_iou=group_iou, gt_voxels=gt_count)
    return result


def load_hist(path, samples):
    with np.load(path) as data:
        hist = np.array(data['hist'], dtype=np.float64)
    if hist.shape != (samples, 3, 18, 18):
        raise ValueError(f'{path}: expected {(samples, 3, 18, 18)}, got {hist.shape}')
    return hist


def scene_hist(hist, scenes, scene_ids):
    output = np.zeros((len(scene_ids), 3, 18, 18), dtype=np.float64)
    positions = {scene: i for i, scene in enumerate(scene_ids)}
    for sample, scene in enumerate(scenes):
        output[positions[scene]] += hist[sample]
    return output


def compact_metrics(hist):
    metrics = grouped_metrics(hist)
    output = {}
    for h, label in enumerate(HORIZONS):
        output[label] = {}
        for group in GROUPS:
            output[label][group] = {
                measure: float(metrics[group][measure][h])
                for measure in MEASURES
            }
            output[label][group]['gt_voxels'] = int(metrics[group]['gt_voxels'][h])
        output[label]['other_iou'] = float(metrics['class_iou'][h, 0])
        output[label]['class_iou'] = {
            CLASS_NAMES[i]: float(metrics['class_iou'][h, i])
            for i in range(17)
        }
    return output


def bootstrap_metrics(hist, scenes, scene_ids, counts):
    grouped = scene_hist(hist, scenes, scene_ids)
    sampled = np.tensordot(counts, grouped, axes=(1, 0))
    metrics = grouped_metrics(sampled)
    return {group: {measure: metrics[group][measure] for measure in MEASURES}
            for group in GROUPS}


def mean_seed_metrics(values):
    output = {}
    for horizon in HORIZONS:
        output[horizon] = {}
        for group in GROUPS:
            output[horizon][group] = {
                measure: float(np.mean([row[horizon][group][measure] for row in values]))
                for measure in MEASURES
            }
            output[horizon][group]['gt_voxels'] = values[0][horizon][group]['gt_voxels']
        output[horizon]['other_iou'] = float(np.mean(
            [row[horizon]['other_iou'] for row in values]))
        output[horizon]['class_iou'] = {
            name: float(np.mean([row[horizon]['class_iou'][name] for row in values]))
            for name in CLASS_NAMES[:-1]
        }
    return output


def mean_seed_bootstrap(values):
    return {group: {measure: np.mean([v[group][measure] for v in values], axis=0)
                    for measure in MEASURES}
            for group in GROUPS}


def difference(left, right, left_boot, right_boot):
    output = {}
    for h, label in enumerate(HORIZONS):
        output[label] = {}
        for group in GROUPS:
            output[label][group] = {}
            for measure in MEASURES:
                point = left[label][group][measure] - right[label][group][measure]
                samples = left_boot[group][measure][:, h] - right_boot[group][measure][:, h]
                lower, upper = np.nanpercentile(samples, [2.5, 97.5])
                output[label][group][measure] = {
                    'delta_pp': float(point),
                    'ci95_pp': [float(lower), float(upper)],
                }
    return output


def analyze(results_path, output_path, repeats=1000, seed=20260929):
    results_path = Path(results_path)
    root = results_path.parent
    results = json.loads(results_path.read_text(encoding='utf8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    scenes = manifest['val']['scenes']
    samples = len(scenes)
    scene_ids = sorted(set(scenes))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(scene_ids), np.full(len(scene_ids), 1 / len(scene_ids)),
                             size=repeats)

    def from_row_path(row_path):
        path = Path(row_path)
        if not path.is_absolute() and not path.exists():
            path = results_path.resolve().parents[3] / path
        hist = load_hist(path, samples)
        return compact_metrics(hist.sum(axis=0)), bootstrap_metrics(hist, scenes, scene_ids, counts)

    before, before_boot = from_row_path(results['oracle']['rows']['student'])
    teacher, teacher_boot = from_row_path(results['oracle']['rows']['teacher'])
    arms = {}
    boot_by_arm = {}
    for arm, runs in results['arms'].items():
        metrics, boots = zip(*(from_row_path(run['rows']['student']) for run in runs))
        arms[arm] = mean_seed_metrics(metrics)
        boot_by_arm[arm] = mean_seed_bootstrap(boots)

    controls = {}
    for arm in arms:
        if arm == 'task_only':
            continue
        controls[arm] = ('adapter_task' if arm.startswith('adapter_') and
                         arm != 'adapter_task' else 'task_only')
    matched_controls = {
        'raw_align': 'raw_shuffled_matched',
        'adapter_align': 'adapter_shuffled_matched',
        'semantic_kd_3s': 'semantic_kd_3s_shuffled_matched',
    }
    report = {
        'status': 'complete',
        'source': str(results_path),
        'samples': samples,
        'validation_scenes': len(scene_ids),
        'bootstrap_repeats': repeats,
        'bootstrap_seed': seed,
        'class_names': CLASS_NAMES,
        'groups': {name: [CLASS_NAMES[i] for i in ids] for name, ids in GROUPS.items()},
        'excluded_from_groups': ['others', 'free'],
        'metric_definitions': {
            'macro_iou': 'Mean of per-class IoU over the 8 classes in this group.',
            'gt_exact_accuracy': 'On GT voxels in this group, percent assigned the exact GT class.',
            'group_iou': 'Binary IoU of membership in this group; within-group class confusion counts as correct.',
        },
        'before': before,
        'teacher': teacher,
        'arms': arms,
        'delta_teacher_minus_before': difference(teacher, before, teacher_boot, before_boot),
        'delta_after_minus_before': {
            arm: difference(metrics, before, boot_by_arm[arm], before_boot)
            for arm, metrics in arms.items()
        },
        'control_for_arm': controls,
        'delta_after_minus_control': {
            arm: difference(arms[arm], arms[control], boot_by_arm[arm], boot_by_arm[control])
            for arm, control in controls.items()
        },
        'delta_correct_minus_shuffled_matched': {
            arm: difference(arms[arm], arms[control], boot_by_arm[arm], boot_by_arm[control])
            for arm, control in matched_controls.items()
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('out/future2current/feature_identity_2048x512/results.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('out/future2current/feature_identity_2048x512/dynamic_static.json'))
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    args = parser.parse_args()
    report = analyze(args.results, args.output, args.bootstrap_repeats)
    print(f"Saved {args.output}: {len(report['arms'])} arms, "
          f"{report['samples']} samples, {report['validation_scenes']} scenes")


if __name__ == '__main__':
    main()
