"""Post-hoc diagnostics only: reuse frozen heads, samples and validation counts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from tools.fit_empty_decision_optimization import encoded, load_artifact, outcome

RUN = ROOT / 'work_dirs/empty_decision_optimization'
MODEL = 'prob_support_seed928'


def main():
    torch.set_num_threads(8)
    path = RUN / 'models' / f'{MODEL}.pt'
    record, model = load_artifact(path, device='cpu')
    result = {
        'purpose': 'Post-hoc failure analysis; no fitting or operating-point selection',
        'model': MODEL,
        'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'abs(native margin)<=2; q=current foreground winner correct; e=GT empty',
        'training_estimator': 'Existing inverse-inclusion weights; 15 equal-width ECE bins',
        'cohorts': {},
    }
    for split in ['fit', 'tune']:
        with np.load(RUN / f'samples_{split}.npz') as z:
            data = {k: z[k] for k in z.files}
        with torch.no_grad():
            probabilities = (model(encoded(data, record['scale'], device='cpu')) /
                             record['temperature']).softmax(-1).numpy()
        target = outcome(data['gt'], data['fg'])
        weight = data['weight']
        rows = {}
        for index, region in [(0, 'camera'), (1, 'camera_complement'), (-1, 'all')]:
            keep = data['region'] == index if index >= 0 else np.ones(len(target), bool)
            w, y, p = weight[keep], target[keep], probabilities[keep]
            row = {
                'sample_rows': int(keep.sum()),
                'population_fraction': float(w.sum() / weight.sum()),
                'NLL': float(np.average(-np.log(p[np.arange(len(y)), y].clip(1e-12)), weights=w)),
            }
            for event, name in [(0, 'q'), (1, 'e')]:
                score, truth = p[:, event], y == event
                bins = np.minimum((score * 15).astype(int), 14)
                ece = sum(abs(float((w[bins == b] * (score[bins == b] - truth[bins == b])).sum()))
                          for b in range(15)) / w.sum()
                row[name] = {
                    'mean_prediction': float(np.average(score, weights=w)),
                    'prevalence': float(np.average(truth, weights=w)),
                    'ECE': float(ece),
                    'AP': float(average_precision_score(truth, score, sample_weight=w)),
                }
            rows[region] = row
        result['cohorts'][split] = rows
        if split == 'fit':
            result['fit_raw_gt_counts'] = np.bincount(data['gt'], minlength=18).tolist()

    validation = json.loads((RUN / 'validation/summary.json').read_text())
    raw = np.asarray(validation['raw_calibration'][MODEL])
    rows = {}
    for index, region in [(0, 'camera'), (1, 'camera_complement'), (-1, 'all')]:
        stats = raw[index] if index >= 0 else raw.sum(axis=0)
        n = stats[0]
        row = {'candidate_voxels': int(n), 'population_fraction': float(n / raw[:, 0].sum()),
               'NLL': float(stats[1] / n)}
        for event, name in [(0, 'q'), (1, 'e')]:
            bins = stats[3 + event * 45:3 + (event + 1) * 45].reshape(15, 3)
            assert np.isclose(bins[:, 0].sum(), n)
            row[name] = {
                'mean_prediction': float(bins[:, 1].sum() / n),
                'prevalence': float(bins[:, 2].sum() / n),
                'ECE': float(np.abs(bins[:, 1] - bins[:, 2]).sum() / n),
            }
        rows[region] = row
    result['cohorts']['validation_full_census'] = rows
    result['notes'] = [
        'Fit/tune are weighted sample estimates; validation values use all candidate voxels.',
        'Fit and tune scenes were both used to train the frozen backbone.',
        'Subgroup errors alone do not identify whether inputs, capacity or optimization are limiting.',
        'All ECE and prevalence values are fractions; multiply by 100 for percentages.',
    ]
    output = RUN / 'posthoc_failure_diagnostics.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for split, rows in result['cohorts'].items():
        for region, row in rows.items():
            print(split, region, 'fraction', round(row['population_fraction'], 4),
                  'q_pred/true/ECE', [round(row['q'][k] * 100, 4) for k in
                                      ['mean_prediction', 'prevalence', 'ECE']])
    print(output)


if __name__ == '__main__':
    main()
