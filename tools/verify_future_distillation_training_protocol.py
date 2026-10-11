#!/usr/bin/env python3
"""Read-only numerical checks, with a separate machine-readable audit artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_future2current_regions import grouped_metrics


def verify(root):
    root = Path(root)
    report = json.loads((root / 'results.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    count = len(manifest['val']['indices'])
    errors, checked = [], []
    if set(manifest['train']['scenes']) & set(manifest['val']['scenes']):
        errors.append('Training and validation scenes overlap')

    def validate(label, row, path):
        with np.load(path) as data:
            hist = data['hist']
            intersection, union = data['intersection'], data['union']
        if hist.shape != (count, 3, 18, 18):
            errors.append(f'{label}: histogram shape {hist.shape}')
            return
        if not np.isfinite(hist).all() or (hist < 0).any():
            errors.append(f'{label}: invalid histogram counts')
            return
        summed = hist.sum(0)
        diagonal = np.diagonal(summed, axis1=-2, axis2=-1)
        denominator = summed.sum(-1) + summed.sum(-2) - diagonal
        with np.errstate(divide='ignore', invalid='ignore'):
            miou = np.nanmean(100 * diagonal[:, :17] / denominator[:, :17], -1)
        occ = 100 * intersection.sum(0) / np.maximum(union.sum(0), 1)
        groups = grouped_metrics(summed)
        for i, h in enumerate(('1s', '2s', '3s')):
            comparisons = [('mIoU', miou[i], row['metrics'][h]['mIoU']),
                           ('OccIoU', occ[i], row['metrics'][h]['OccIoU'])]
            for g in ('dynamic', 'static'):
                comparisons.extend((f'{g}/{metric}', groups[g][key][i], row['groups'][h][g][metric])
                                   for metric, key in [('mIoU', 'macro_iou'), ('IoU', 'group_iou')])
            for name, actual, recorded in comparisons:
                if not np.isclose(actual, recorded, rtol=0, atol=1e-8, equal_nan=True):
                    errors.append(f'{label}/{h}/{name}: raw {actual}, report {recorded}')
        checked.append(label)

    validate('baseline', report['baseline'], root / 'baseline.npz')
    for step, row in report.get('bn_only', {}).get('steps', {}).items():
        validate(f'bn_only/{step}', row, root / f'bn_only_step{step}.npz')
    for label, arm in report['arms'].items():
        for step, row in arm.get('steps', {}).items():
            validate(f'{label}/{step}', row, row['rows'])
            for h in ('1s', '2s', '3s'):
                for metric in ('mIoU', 'OccIoU'):
                    expected = row['metrics'][h][metric] - report['baseline']['metrics'][h][metric]
                    if not np.isclose(expected, row['delta_vs_baseline_pp'][h][metric], atol=1e-8, rtol=0):
                        errors.append(f'{label}/{step}/{h}/{metric}: baseline delta differs')
                for group in ('dynamic', 'static'):
                    for metric in ('mIoU', 'IoU'):
                        expected = row['groups'][h][group][metric] - report['baseline']['groups'][h][group][metric]
                        actual = row['group_delta_vs_baseline_pp'][h][group][metric]
                        if not np.isclose(expected, actual, atol=1e-8, rtol=0):
                            errors.append(f'{label}/{step}/{h}/{group}/{metric}: group baseline delta differs')
        if arm.get('status') == 'complete' and arm['fixed_bn'] and not arm.get('bn_unchanged'):
            errors.append(f'{label}: fixed BN buffers changed')
    if report.get('bn_only', {}).get('status') == 'complete' and not report['bn_only'].get('parameters_unchanged'):
        errors.append('BN-only changed learnable parameters')

    reference = Path('out/future_distillation_causal/20260930/s1_baseline.npz')
    with np.load(root / 'baseline.npz') as actual, np.load(reference) as original:
        for field in ('hist', 'intersection', 'union'):
            if not np.array_equal(actual[field], original[field][:count]):
                errors.append(f'Untouched original baseline {field} differs')
    teacher_manifest = json.loads(Path('out/future2current/predictable_full_finetune_4096x1024/manifest.json').read_text())
    for field in ('indices', 'scenes'):
        if manifest['val'][field] != teacher_manifest['val'][field][:count]:
            errors.append(f'Teacher validation {field} differs')
    if manifest['checkpoint_sha256'] != teacher_manifest['checkpoint_sha256']:
        errors.append('Teacher checkpoint identity differs')
    alignment_path = root / 'alignment_audit.json'
    alignment = json.loads(alignment_path.read_text()) if alignment_path.exists() else None
    if alignment:
        for label, row in alignment['cases'].items():
            if [(v['index'], v['scene']) for v in row['rows']] != list(zip(manifest['val']['indices'], manifest['val']['scenes'])):
                errors.append(f'{label}: alignment sample identity differs')
        baseline = alignment['cases'].get('baseline')
        if baseline and not np.isclose(baseline['fixed_feature_loss'], baseline['moving_feature_loss'], rtol=0, atol=1e-6):
            errors.append('Initial fixed and moving feature losses differ')
    result = dict(status='passed' if not errors else 'failed', checked_native_results=len(checked),
                  cases=checked, errors=errors, training_status=report['status'],
                  alignment_status=alignment['status'] if alignment else 'not_available',
                  scope='Raw native-head occupancy counts and declared identity/BN checks; not proof of object matching or independent causal confirmation.',
                  results_sha256=hashlib.sha256((root / 'results.json').read_bytes()).hexdigest())
    target = root / 'verification.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print('[protocol verification]', result['status'], len(checked), errors)
    if errors:
        raise RuntimeError('Native report verification failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('out/future_distillation_causal/20261001_training_protocol'))
    verify(parser.parse_args().root)
