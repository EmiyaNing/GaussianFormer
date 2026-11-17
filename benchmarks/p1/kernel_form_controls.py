"""The kernel-form control (A4): fixed isotropic geometry vs the other two.

Plan T1.1 lists A4 -- the Gaussian kernel with **fixed isotropic** scale and
opacity -- and P1-4 recorded it as "not run".  It is the missing cell that
separates two explanations for A5 finishing last among the adapters:

    A4 vs A2   the kernel FORM (Gaussian distance weighting vs uniform mean)
    A5 vs A4   the learnable GEOMETRY (per-point scale/opacity vs frozen)

Two things this script is careful about, because both have bitten this project:

* **Only the second contrast is single-factor.**  A4 keeps A5's support-box
  (shell) neighbour selection, so A4 vs A2 differs in *both* the weighting and
  the selection -- the same confound P1-6 documented for A5 vs A2.  The output
  says so instead of presenting it as a clean kernel comparison.
* **The class-group split is the discriminator.**  A5's deficit is concentrated
  on thin classes (P1-6 §4.2).  If A4 -- same kernel shape, frozen geometry --
  does *not* show that deficit, the harm comes from the *learned* geometry
  rather than from the Gaussian form.

    python3.10 benchmarks/p1/kernel_form_controls.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

VARIANTS = {
    'A0 baseline': 'a0_base',
    'A2 uniform mean (kNN-8)': 'a2_knn_mean',
    'A4 fixed isotropic Gaussian': 'a4_fixed_gaussian',
    'A5 learnable Gaussian': 'a5_cap0',
}
CONTRASTS = (
    ('A5 - A4', 'a5_cap0', 'a4_fixed_gaussian',
     'learnable vs frozen geometry (single factor: only freeze_geometry differs)'),
    ('A4 - A2', 'a4_fixed_gaussian', 'a2_knn_mean',
     'kernel form AND selection differ (A4 keeps the support-box shell) -> confounded'),
    ('A5 - A2', 'a5_cap0', 'a2_knn_mean',
     'reference contrast; also confounded (P1-6 §4.2.1)'),
)


def geometry_verdict(a4_thin_penalty, a5_thin_penalty, tolerance=0.05):
    """Attribute the thin-class deficit to the learned geometry or the kernel form.

    ``a4_thin_penalty`` / ``a5_thin_penalty`` are each adapter's thin-class delta
    against A2.  If the frozen-geometry kernel keeps most of the gain A5 loses,
    the harm is the *learned* geometry; if it shows the same penalty, the harm
    travels with the kernel form; otherwise neither attribution is supported.
    """
    if a4_thin_penalty is None or a5_thin_penalty is None:
        return 'INCOMPLETE'
    if abs(a4_thin_penalty) < abs(a5_thin_penalty) / 2:
        return 'geometry-is-the-harm'
    if abs(a4_thin_penalty - a5_thin_penalty) <= tolerance:
        return 'form-is-the-harm'
    return 'mixed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p1_kernel_form_controls.json')
    args = parser.parse_args(argv)

    from benchmarks.p1 import per_class_analysis as classes  # noqa: PLC0415
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415

    recorded = {row['variant']: row['metrics']
                for row in json.loads((ROOT / args.eval).read_text(encoding='utf-8'))
                if row.get('metrics')}
    available = {label: name for label, name in VARIANTS.items()
                 if name in recorded}
    missing = [label for label, name in VARIANTS.items() if name not in recorded]
    print(f'available: {list(available)}')
    print(f'missing (queued): {missing or "none"}')

    names = list(available.values())
    logs = load(names)
    table, _ = comparable_for(logs, names) if logs else ({}, {})
    fixed = {name: table[name]['avg'] for name in table}

    keys = None
    for name in names:
        if name in logs:
            variety = classes.finite_keys(logs[name], classes.CLASSES[:17])
            keys = variety if keys is None else (keys & variety)
    small = {key for key in (keys or ()) if key[1] in classes.SMALL}
    large = {key for key in (keys or ()) if key[1] in classes.LARGE}

    def group(name, subset):
        if name not in logs or not subset:
            return None
        return classes.group_mean(logs[name], classes.CLASSES[:17], keys=subset)

    print()
    print('| variant | recorded mIoU | fixed mIoU | thin | structural |')
    print('| --- | ---: | ---: | ---: | ---: |')
    rows = []
    for label, name in available.items():
        thin = group(name, small)
        structural = group(name, large)
        rows.append({'label': label, 'variant': name,
                     'recorded_mIoU': recorded[name].get('avg/mIoU'),
                     'fixed_mIoU': (round(fixed[name], 4) if name in fixed else None),
                     'thin': (round(thin, 3) if thin is not None else None),
                     'structural': (round(structural, 3)
                                    if structural is not None else None)})
        print(f"| {label} | {recorded[name].get('avg/mIoU')} | "
              f"{'—' if name not in fixed else f'{fixed[name]:.2f}'} | "
              f"{'—' if thin is None else f'{thin:.2f}'} | "
              f"{'—' if structural is None else f'{structural:.2f}'} |")

    print()
    contrasts = {}
    for label, left, right, note in CONTRASTS:
        if left in fixed and right in fixed:
            delta = fixed[left] - fixed[right]
            thin = None
            if group(left, small) is not None and group(right, small) is not None:
                thin = group(left, small) - group(right, small)
            contrasts[label] = {'delta': round(delta, 4),
                                'thin_delta': (round(thin, 4)
                                               if thin is not None else None),
                                'single_factor': 'single factor' in note}
            print(f'  {label:10s} {delta:+.3f} mIoU'
                  f'{"" if thin is None else f"   (thin classes {thin:+.3f})"}'
                  f'   [{note}]')
        else:
            print(f'  {label:10s} — (missing)   [{note}]')

    # The discriminator: does the frozen-geometry kernel still hurt thin classes?
    verdict = 'INCOMPLETE'
    if 'A4 fixed isotropic Gaussian' in available:
        a4 = group('a4_fixed_gaussian', small)
        a2 = group('a2_knn_mean', small)
        a5 = group('a5_cap0', small)
        if a4 is not None and a2 is not None:
            a4_penalty = a4 - a2
            a5_penalty = (a5 - a2) if a5 is not None else None
            print(f'\nthin-class penalty vs A2: A4 (frozen) {a4_penalty:+.3f}'
                  f'{"" if a5_penalty is None else f", A5 (learnable) {a5_penalty:+.3f}"}')
            verdict = geometry_verdict(a4_penalty, a5_penalty)
            if verdict == 'geometry-is-the-harm':
                print('VERDICT: the frozen-geometry kernel keeps most of the '
                      'thin-class gain that A5 loses -> the harm is the LEARNED '
                      'geometry, not the Gaussian form.')
            elif verdict == 'form-is-the-harm':
                print('VERDICT: A4 shows the same thin-class penalty as A5 -> the '
                      'harm travels with the kernel FORM, not with learnability.')
            else:
                print('VERDICT: A4 sits between -> report both contrasts and do not '
                      'attribute the deficit to one factor.')
    if verdict == 'INCOMPLETE':
        print('\nVERDICT: A4 has not run, so "kernel form adds nothing" and '
              '"the kernel works but was parameterised wrong" remain '
              'indistinguishable -- which is the question plan T1.1 asks.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/kernel_form_controls.py',
        'variants': VARIANTS, 'available': list(available), 'missing': missing,
        'rows': rows, 'contrasts': contrasts, 'verdict': verdict,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
