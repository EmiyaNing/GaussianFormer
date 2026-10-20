"""Scene-paired uncertainty for local interventions; not training seed variance."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.center_support_common import write_json


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    result = dict(n_scenes=len(values), mean=float(values.mean()))
    if len(values) < 2:
        result['ci95'] = None
        return result
    rng = np.random.default_rng(928)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(-1)
    result['ci95'] = np.quantile(samples, [.025, .975]).tolist()
    return result


def summarize(data):
    if data.get('status') != 'complete':
        return dict(status=data.get('status'), reason='Only complete real diagnostics are summarized')
    collected = defaultdict(list)
    cosine = defaultdict(list)
    for scene in data['pairs']:
        for comparison, groups in scene['gradients'].items():
            for group, stats in groups.items():
                if group == 'primitive_strata':
                    continue
                if stats['cosine'] is not None:
                    cosine[comparison + ' / ' + group].append(stats['cosine'])
        for branch in scene['interventions']:
            key = (branch['branch'], branch['radius'])
            collected[key].append(branch)
    branches = []
    for (name, radius), observations in sorted(collected.items()):
        # Self-loss decreases are necessary to label a step a useful intervention.
        def self_delta(row):
            d = row['delta_losses']
            if name.endswith('geometry_only'):
                domain = name.split('/')[0]
                return d[domain + '/pred_to_gt'] + d[domain + '/gt_to_pred']
            if name.endswith('classification_only'):
                return d[name.split('/')[0] + '/classification']
            return None
        own = [self_delta(row) for row in observations]
        same_ce = [row['delta_losses']['voxel_ce'] for row in observations]
        same_voxel = [row['delta_losses']['voxel_ce'] + row['delta_losses']['voxel_lovasz'] for row in observations]
        held_ce = [row['delta_heldout_losses']['voxel_ce'] for row in observations]
        held_voxel = [row['delta_heldout_losses']['voxel_ce'] + row['delta_heldout_losses']['voxel_lovasz'] for row in observations]
        branches.append(dict(branch=name, radius=radius, same_batch_ce=interval(same_ce),
            same_batch_voxel=interval(same_voxel), heldout_ce=interval(held_ce),
            heldout_voxel=interval(held_voxel),
            self_loss=interval(own) if own and own[0] is not None else None,
            own_down_field_up_scenes=sum(a is not None and a < -1e-7 and b > 1e-7
                                        for a, b in zip(own, same_voxel)),
            zero_gradient_scenes=sum(row['gradient_norm'] <= 1e-12 for row in observations)))
    gradient_summaries = {}
    for key, values in cosine.items():
        gradient_summaries[key] = dict(**interval(values), negative_fraction=float((np.asarray(values) < 0).mean()))
    return dict(status='complete', diagnostic_protocol=data['protocol'],
        uncertainty='10,000 paired scene bootstrap draws; no training seed uncertainty',
        gradient_summaries=gradient_summaries, branches=branches,
        verdict='Interpret after camera alignment, radius sensitivity and A4 training; no automatic novelty claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    write_json(args.out, summarize(json.loads(Path(args.input).read_text())))
    print(args.out)


if __name__ == '__main__':
    main()
