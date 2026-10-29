"""Paired factorial contrasts from existing full-scene confusion counters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/gaussian_geometry_causal'
ARMS = ('standard', 'no_amplitude', 'no_geometry', 'no_both')


def miou(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)[..., :17]
    union = cm.sum(-1)[..., :17] + cm.sum(-2)[..., :17] - tp
    return 100 * np.divide(tp, union, out=np.zeros_like(tp, dtype=float),
                            where=union > 0).mean(-1)


def contrasts(scores):
    s, a, g, both = (scores[n] for n in ARMS)
    return {
        'geometry_effect_amplitude_available': s - g,
        'geometry_effect_amplitude_blocked': a - both,
        'factorial_interaction': s - a - g + both,
    }


def main():
    result = {'scope': 'Early underfit branches only; these are path-intervention '
                        'contrasts, not mediation percentages or seed replication',
              'units': 'mIoU percentage points', 'bootstrap_repetitions': 1000,
              'bootstrap_seed': 928, 'splits': {}}
    for split, expected in (('train', 700), ('val', 150)):
        matrices = {}
        names = None
        for arm in ARMS:
            paths = sorted((RUN / 'eval' / arm / split).glob('*.npz'))
            assert len(paths) == expected
            scene_names = [p.stem for p in paths]
            if names is None:
                names = scene_names
            else:
                assert names == scene_names
            values = []
            for p in paths:
                with np.load(p) as data:
                    values.append(data['confusion'].copy())
            matrices[arm] = np.stack(values)
        weights = np.random.default_rng(928).multinomial(
            expected, np.full(expected, 1 / expected), size=1000)
        points = contrasts({arm: miou(cm.sum(0)) for arm, cm in matrices.items()})
        samples = contrasts({arm: miou((weights @ cm.reshape(expected, -1))
                           .reshape(-1, 18, 18)) for arm, cm in matrices.items()})
        result['splits'][split] = {
            key: {'estimate': float(points[key]),
                  'scene_bootstrap_95': np.quantile(value, [.025, .975]).tolist()}
            for key, value in samples.items()}
    (RUN / 'factorial_interaction.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
