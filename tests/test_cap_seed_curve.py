"""Tests for the cap-cost curve's seed dimension (P1-11).

The curve is the project's clearest efficiency/accuracy trade-off statement
("discard 31.5% of the weight mass for -0.06 mIoU"), and it was measured on a
single checkpoint.  These tests pin the two properties that make the seed
extension meaningful:

* the cost is **paired within a seed** -- each cap is compared against the same
  seed's uncapped number, so a seed that happens to score higher does not look
  like a cap benefit;
* mixing seeds is detected: if some seed pays and another gains, the summary must
  say the sign is not stable rather than averaging the two into a comfortable
  middle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.cap_seed_curve import (  # noqa: E402
    BASELINE, CAPS, SEEDS, cap_curve, summarise, variant_name)


def test_cap_cost_is_paired_within_a_seed():
    """A seed whose overall level is higher must not look like a cap benefit."""
    values = {
        'a5_cap0': 14.0, 'a5_cap8': 13.0, 'a5_cap32': 13.5,
        'a5_cap64': 13.9, 'a5_cap128': 14.0,
        'a5_s43': 15.0, 'a5_cap8_s43': 14.5, 'a5_cap32_s43': 14.9,
        'a5_cap64_s43': 14.95, 'a5_cap128_s43': 15.0,
    }
    rows = cap_curve(values)
    by_key = {(seed, cap): cost for seed, cap, _, _, cost in rows}
    assert by_key[('s42', 8)] == pytest.approx(-1.0)
    assert by_key[('s43', 8)] == pytest.approx(-0.5)
    # If the levels were mixed (s43 minus s42's baseline) the second cost would
    # read +0.5 -- a cap "benefit" that is really just a better seed.
    assert by_key[('s43', 8)] < 0
    assert by_key[('s42', 128)] == pytest.approx(0.0)
    assert by_key[('s43', 128)] == pytest.approx(0.0)
    # A seed with no values contributes no rows.
    assert all(row[0] != 's44' for row in rows)


def test_summary_flags_a_sign_flip_instead_of_averaging_it_away():
    values = {
        'a5_cap0': 14.0, 'a5_cap8': 13.0,
        'a5_s43': 15.0, 'a5_cap8_s43': 15.1,          # this seed *gains*
        'a5_s44': 16.0, 'a5_cap8_s44': 15.2,
    }
    summary = summarise(cap_curve(values))
    info = summary[8]
    assert info['seeds'] == 3
    assert info['costs'] == pytest.approx([-1.0, 0.1, -0.8])
    assert not info['all_negative'] and not info['all_positive']
    # The mean (-0.567) is a real number but must not be read as "the cost":
    # the sign is not stable, which is the finding.
    assert info['mean'] == pytest.approx(-0.5667, abs=1e-3)
    assert info['spread'] == pytest.approx(1.1)


def test_variant_names_match_the_evaluation_registry():
    """The analysis and the eval registry must not drift apart.

    ``cap_seed_curve`` builds ``a5_cap{cap}_{seed}`` names; if the registry ever
    names them differently the script would silently report "missing" for points
    that were actually evaluated.
    """
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.eval_matrix import VARIANTS  # noqa: PLC0415

    # The seed-42 checkpoint lives at out/p1_a5_gaussian (it predates the seed
    # directory convention), the repeats at out/a5_s43 and out/a5_s44.
    checkpoints = {'s42': 'out/p1_a5_gaussian/epoch_40.pth',
                   's43': 'out/a5_s43/epoch_40.pth',
                   's44': 'out/a5_s44/epoch_40.pth'}
    for cap in CAPS:
        for seed in SEEDS:
            name = variant_name(cap, seed)
            assert name in VARIANTS, f'{name} missing from the eval registry'
            config, checkpoint = VARIANTS[name]
            assert config == f'config/opus/ablations/p1_a5_cap{cap}.py'
            assert checkpoint == checkpoints[seed], (name, checkpoint)
    for seed, name in BASELINE.items():
        assert name in VARIANTS
        assert VARIANTS[name][1] == checkpoints[seed], (name, seed)


def test_seed42_costs_reproduce_the_published_curve_when_available():
    """The single-seed points must still match P1-1 after the refactor."""
    path = ROOT / 'benchmarks/results/p1_cap_seed_curve.json'
    if not path.exists():
        pytest.skip('cap seed curve artifact not present')
    import json
    payload = json.loads(path.read_text(encoding='utf-8'))
    values = payload['values']
    if not all(name in values for name in ('a5_cap0', 'a5_cap8', 'a5_cap32',
                                           'a5_cap64', 'a5_cap128')):
        pytest.skip('seed-42 cap points not present')
    rows = cap_curve(values)
    costs = {cap: cost for seed, cap, _, _, cost in rows if seed == 's42'}
    assert costs[8] == pytest.approx(-0.267, abs=0.01)
    assert costs[32] == pytest.approx(-0.053, abs=0.01)
    assert costs[64] == pytest.approx(-0.016, abs=0.01)
    assert costs[128] == pytest.approx(-0.001, abs=0.01)
    # The reports quote these costs in the rounded forms they use: P1-6 gives
    # the sequence ("cap=8 掉 0.26", "cap=32 只掉 0.05") and P1-1 states the
    # headline trade-off ("cap=32 用 −0.06 mIoU 换来 20.4x").
    p1_6 = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md'
            ).read_text(encoding='utf-8')
    p1_1 = (ROOT / 'analysis/sparse_render_result/P1-1_gaussian_vs_baseline_and_cap.md'
            ).read_text(encoding='utf-8')
    assert '0.26' in p1_6 and '0.05' in p1_6, 'P1-6 lost the cap cost sequence'
    assert '−0.06' in p1_1, 'P1-1 lost the cap=32 headline cost'
