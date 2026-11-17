"""The receptive-field-matched controls, decided by the pre-registered table.

Report P1-3 §5 left a falsifiable prediction for two queued controls, and P1-8 is
the report that must settle it:

    A2''  k = 59   matches A5's EFFECTIVE field (ESS 59.3) -> isolates weighting
    A2'   k = 215  matches A5's NOMINAL field (~215)      -> tests whether a
                                                             uniform mean survives
                                                             a 3.6x larger field

The pre-registered reading (P1-3 §5), in terms of the k=8 control and A5:

    A2'' ~= A2-k8        -> neighbour count is irrelevant; the learned scale
                            growth is wasted                       (stronger refutation)
    A2'' >> A5           -> a uniform mean over a large field is better
                                                                   (stronger refutation)
    A2'' << A2-k8 and
    A5 ~= A2-k8          -> the uniform mean collapses at large fields while the
                            Gaussian does not                       (partial exculpation)
    in between           -> read the k trend                        (undecided)

A ceiling is printed alongside: the best uniform approximation to A5's weighting
differs by ~0.147 relative L2 (P4-ESS), so no k-NN control can reproduce A5's
operator exactly.

    python3.10 benchmarks/p1/receptive_field_controls.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.common_paths import display  # noqa: E402

REFERENCE = 'a5_cap0'          # A5, the Gaussian adapter
SMALL_K = 'a2_knn_mean'        # A2, k = 8
EFFECTIVE_MATCH = 'a2_knn_k59'  # A2'', k = 59
NOMINAL_MATCH = 'a2_knn_k215'  # A2', k = 215
# Noise the comparison has to beat: the seed spread measured in P1-6.
SEED_SPREAD = 0.054


def classify(delta_effective_vs_small, delta_gaussian_vs_small,
             delta_effective_vs_gaussian, tolerance):
    """Map the three deltas onto the pre-registered branches.

    ``tolerance`` is the seed spread -- differences inside it are called equal.
    """
    if delta_effective_vs_gaussian is None:
        return 'INCOMPLETE'
    equal = lambda value: abs(value) <= tolerance  # noqa: E731
    if equal(delta_effective_vs_small):
        return 'size-irrelevant'
    if delta_effective_vs_gaussian > tolerance:
        return 'uniform-better'
    if (delta_effective_vs_small < -tolerance
            and equal(delta_gaussian_vs_small)):
        return 'gaussian-needed'
    return 'intermediate'


BRANCH_TEXT = {
    'size-irrelevant': ('neighbour count is irrelevant at this scale; the learned '
                        'scale growth buys nothing -> STRONGER refutation'),
    'uniform-better': ('a uniform mean over the matched field is BETTER than the '
                       'Gaussian -> STRONGER refutation'),
    'gaussian-needed': ('the uniform mean collapses at the matched field while the '
                        'Gaussian holds -> PARTIAL EXCULPATION (the Gaussian buys '
                        'field size, not accuracy)'),
    'intermediate': ('falls between the branches -> read the k trend before '
                     'concluding anything'),
    'INCOMPLETE': 'a control is missing',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', default='benchmarks/results/p1_eval.json')
    parser.add_argument('--ess', default='benchmarks/results/p4_ess.json')
    parser.add_argument('--json',
                        default='benchmarks/results/p1_receptive_field_controls.json')
    parser.add_argument('--tolerance', type=float, default=SEED_SPREAD)
    args = parser.parse_args(argv)

    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415

    wanted = [REFERENCE, SMALL_K, EFFECTIVE_MATCH, NOMINAL_MATCH]
    present_names = [row['variant'] for row in json.loads(
        (ROOT / args.eval).read_text(encoding='utf-8')) if row.get('metrics')]
    missing = [name for name in wanted if name not in present_names]
    available = [name for name in wanted if name in present_names]
    logs = load(available)
    table, _ = comparable_for(logs, available) if logs else ({}, {})
    fixed = {name: table[name]['avg'] for name in table}

    print(f'available: {available}')
    print(f'missing (queued): {missing or "none"}')
    print(f'tolerance (seed spread from P1-6): {args.tolerance:.3f} mIoU')
    print()
    print('| variant | k | fixed-divisor mIoU |')
    print('| --- | ---: | ---: |')
    for label, name, k in (('A5 Gaussian', REFERENCE, 'all in box'),
                           ('A2 KNN', SMALL_K, 8),
                           ("A2'' effective match", EFFECTIVE_MATCH, 59),
                           ("A2' nominal match", NOMINAL_MATCH, 215)):
        value = fixed.get(name)
        print(f'| {label} | {k} | '
              f'{"—" if value is None else f"{value:.3f}"} |')

    def delta(a, b):
        if a in fixed and b in fixed:
            return fixed[a] - fixed[b]
        return None

    d_eff_small = delta(EFFECTIVE_MATCH, SMALL_K)
    d_gauss_small = delta(REFERENCE, SMALL_K)
    d_eff_gauss = delta(EFFECTIVE_MATCH, REFERENCE)
    d_nom_gauss = delta(NOMINAL_MATCH, REFERENCE)
    d_nom_small = delta(NOMINAL_MATCH, SMALL_K)

    print()
    for label, value in (("A2'' - A2(k=8)", d_eff_small),
                         ('A5 - A2(k=8)', d_gauss_small),
                         ("A2'' - A5", d_eff_gauss),
                         ("A2' - A5 (nominal match)", d_nom_gauss),
                         ("A2' - A2(k=8)", d_nom_small)):
        print(f'  {label:26s} '
              f'{"—" if value is None else f"{value:+.3f} mIoU"}')

    branch = classify(d_eff_small, d_gauss_small, d_eff_gauss, args.tolerance)
    print(f'\nA2\'\' branch: {branch} -- {BRANCH_TEXT[branch]}')
    if d_nom_gauss is not None:
        if abs(d_nom_gauss) <= args.tolerance:
            print("A2' branch: the uniform mean holds at the nominal field too "
                  '(no exculpation).')
        elif d_nom_gauss > args.tolerance:
            print("A2' branch: the uniform mean gets BETTER with a larger field "
                  '(no exculpation).')
        else:
            print("A2' branch: the uniform mean degrades with a larger field; "
                  'compare the k trend before concluding.')

    # The ceiling: even a perfect uniform mean is a different operator.
    ceiling = None
    ess_path = ROOT / args.ess
    if ess_path.exists():
        payload = json.loads(ess_path.read_text(encoding='utf-8'))
        rows = payload.get('rows') or []
        if rows:
            approx = rows[0].get('uniform_approx_error') or {}
            ceiling = approx.get('64')
    if ceiling is not None:
        print(f'\nCeiling: the best uniform approximation differs from A5\'s '
              f'weighting by {ceiling:.3f} relative L2 (P4-ESS, k=64), so no k-NN '
              f'control can replicate A5 exactly.')

    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        'script': 'benchmarks/p1/receptive_field_controls.py',
        'fixed_divisor_mIoU': fixed, 'missing': missing,
        'tolerance': args.tolerance, 'branch': branch,
        'deltas': {'effective_minus_small': d_eff_small,
                   'gaussian_minus_small': d_gauss_small,
                   'effective_minus_gaussian': d_eff_gauss,
                   'nominal_minus_gaussian': d_nom_gauss,
                   'nominal_minus_small': d_nom_small},
        'uniform_approximation_gap': ceiling,
    }, indent=1), encoding='utf-8')
    print(f'\nwrote {display(target)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
