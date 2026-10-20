"""Hand-computed tests for the P1 multi-seed statistics.

The P1-6 verdict rests entirely on ``paired_statistics``, so its three
ingredients are pinned here against values computed by hand:

* the t interval (the normal ``2 x SE`` approximation is 2.15x too narrow at
  n = 3, which is exactly the seed count the plan asks for);
* the two-sided sign test and the fact that it cannot reach p < 0.05 below
  n = 6 agreeing pairs;
* the normal-approximation seed count needed to resolve an observed effect.

These are cheap tests for an expensive conclusion: getting them wrong would
silently turn "no resolvable difference" into a published claim.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.summarize_seeds import MIN_SIGN_N, T_975, paired_statistics  # noqa: E402


def test_mean_std_and_se_for_a_hand_computed_triple():
    # mean 0.2; sample sd = sqrt((0.01 + 0 + 0.01) / 2) = 0.1
    stats = paired_statistics([0.1, 0.2, 0.3])
    assert stats['mean'] == pytest.approx(0.2, abs=1e-12)
    assert stats['std'] == pytest.approx(0.1, abs=1e-12)
    assert stats['se'] == pytest.approx(0.1 / math.sqrt(3), abs=1e-12)


def test_t_interval_is_wider_than_two_standard_errors():
    """df = 2 gives t = 4.303, so the t interval is 2.15x the naive one."""
    stats = paired_statistics([0.1, 0.2, 0.3])
    assert stats['t_critical'] == 4.303
    assert stats['ci_low'] == pytest.approx(0.2 - 4.303 * 0.1 / math.sqrt(3), abs=1e-12)
    assert stats['ci_high'] == pytest.approx(0.2 + 4.303 * 0.1 / math.sqrt(3), abs=1e-12)
    # The interval straddles zero, so this triple is not evidence of a direction.
    assert stats['ci_low'] < 0 < stats['ci_high']
    # And it is strictly wider than the 2 x SE interval.
    assert (stats['ci_high'] - stats['ci_low']) > 4 * stats['se']


def test_sign_test_all_pairs_agree_at_three_seeds():
    # 2 * C(3,3) / 2**3 = 2/8 = 0.25: a unanimous direction still is not
    # significant at n = 3.  This is the plan's ">= 3 seeds" ceiling.
    stats = paired_statistics([0.1, 0.2, 0.3])
    assert stats['positives'] == 3
    assert stats['sign_p_two_sided'] == pytest.approx(0.25, abs=1e-12)


def test_sign_test_with_a_split_direction_is_uninformative():
    # 2 positives out of 3: tail = C(3,2) + C(3,3) = 4, p = min(1, 8/8) = 1.
    stats = paired_statistics([-0.1, 0.2, 0.3])
    assert stats['positives'] == 2
    assert stats['sign_p_two_sided'] == pytest.approx(1.0, abs=1e-12)


def test_min_sign_n_is_the_smallest_n_that_can_reach_significance():
    assert MIN_SIGN_N == 6
    # 2 * (1/2)**5 = 0.0625 is still above 0.05; 2 * (1/2)**6 = 0.03125 is not.
    assert 2 * 0.5 ** (MIN_SIGN_N - 1) > 0.05
    assert 2 * 0.5 ** MIN_SIGN_N < 0.05


def test_seeds_needed_uses_the_observed_sd_and_effect():
    # For symmetric triples [a-d, a, a+d] the mean is a and the sd is d, so the
    # requirement is (2d/a)**2.  Here d = 0.1 and a = 0.2 -> (1)**2 = 1.
    stats = paired_statistics([0.1, 0.2, 0.3])
    assert stats['seeds_needed_normal_approx'] == 1
    # Same spread (d = 0.1) but a 10x smaller effect (a = 0.02) -> 100 seeds.
    # This is the case a single-seed or three-seed study silently overclaims.
    stats = paired_statistics([-0.08, 0.02, 0.12])
    assert stats['std'] == pytest.approx(0.1, abs=1e-12)
    assert stats['mean'] == pytest.approx(0.02, abs=1e-12)
    assert stats['seeds_needed_normal_approx'] == 100


def test_two_pairs_use_the_df_one_multiplier():
    # 2.15x wider than the normal approximation at df = 1, t = 12.706.
    stats = paired_statistics([0.0, 0.4])
    assert T_975[1] == 12.706
    assert stats['t_critical'] == 12.706
    assert stats['mean'] == pytest.approx(0.2, abs=1e-12)
    assert stats['std'] == pytest.approx(0.2 * math.sqrt(2), abs=1e-12)


def test_single_pair_has_no_interval_or_spread():
    stats = paired_statistics([0.05])
    assert stats['n'] == 1
    assert stats['std'] != stats['std']          # NaN: no spread from one pair
    assert stats['se'] != stats['se']
    assert stats['sign_p_two_sided'] == pytest.approx(1.0, abs=1e-12)


def test_seed_aggregation_prefers_the_fixed_divisor():
    """The aggregate must not quote the artifact-inflated recorded mIoU.

    Recorded ``avg/mIoU`` uses ``nanmean(semantic[:17])``, whose divisor depends
    on which classes are finite in that run.  For the A5 seeds that inflated the
    spread from the real 0.05 to 0.38 and made A5 look 0.11 worse than A2 instead
    of 0.01 (P1-5, third instance).  ``seed_aggregation`` therefore prefers the
    fixed-divisor value and only falls back when a log is unavailable.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.p1.summarize_p1 import seed_aggregation

    rows = [
        {'variant': 'a5_cap0', 'metrics': {'avg/mIoU': 14.32}},
        {'variant': 'a5_s43', 'metrics': {'avg/mIoU': 14.16}},
        {'variant': 'a5_s44', 'metrics': {'avg/mIoU': 13.94}},
        {'variant': 'a2_knn_mean', 'metrics': {'avg/mIoU': 14.37}},
        {'variant': 'a2_s43', 'metrics': {'avg/mIoU': 14.13}},
    ]
    fixed = {'a5_cap0': 14.317, 'a5_s43': 14.365, 'a5_s44': 14.370,
             'a2_knn_mean': 14.375, 'a2_s43': 14.354}
    groups = {'a5': ['a5_cap0', 'a5_s43', 'a5_s44'],
              'a2': ['a2_knn_mean', 'a2_s43', 'a2_s44']}
    aggregated = seed_aggregation(rows, fixed, groups)

    by_label = {label: (count, mean, spread, values)
                for label, count, mean, spread, values in aggregated}
    a5_count, a5_mean, a5_spread, a5_values = by_label['a5']
    a2_count, a2_mean, a2_spread, a2_values = by_label['a2']
    assert a5_count == 3 and a2_count == 2
    assert a5_values == [14.317, 14.365, 14.370]
    assert a5_mean == pytest.approx(14.351, abs=1e-3)
    assert a5_spread == pytest.approx(0.053, abs=1e-3)
    # The recorded spread would have been 0.38 -- 7x larger.
    assert a5_spread < 0.1
    assert a2_mean == pytest.approx(14.3645, abs=1e-3)
    assert a2_spread == pytest.approx(0.021, abs=1e-3)
    # And the headline gap is ~0, not the recorded -0.11.
    assert abs(a5_mean - a2_mean) < 0.02

    # Without a fixed value for a member, the recorded one is the fallback.
    fallback = seed_aggregation(rows, {'a5_cap0': 14.317, 'a5_s43': 14.365},
                                {'a5': ['a5_cap0', 'a5_s43', 'a5_s44']})
    assert fallback[0][4] == [14.317, 14.365, 13.94]


def test_horizon_cell_flags_a_divisor_mismatch():
    """The per-horizon table must show both values where the artifact bites.

    A5 seed 44's 4 s is recorded 13.36 but 14.25 on a fixed divisor (+0.89), and
    A2 seed 43's 4 s is 13.38 vs 14.27.  Printing only the recorded value is how
    the artifact gets quoted as model behaviour.
    """
    from benchmarks.p1.summarize_p1 import horizon_cell
    assert horizon_cell(13.36, 14.254, 19.79) == '13.36⚠️14.25 / 19.79'
    assert horizon_cell(12.54, 13.373, 19.43) == '12.54⚠️13.37 / 19.43'
    assert horizon_cell(13.38, 14.273, 19.80) == '13.38⚠️14.27 / 19.80'
    # Within the threshold the two agree to rounding: one number, no flag.
    assert horizon_cell(14.85, 14.855, 21.48) == '14.85 / 21.48'
    # Missing fixed value (no log) degrades to the recorded number.
    assert horizon_cell(14.32, None, 19.81) == '14.32 / 19.81'


def test_partial_log_set_is_reported_as_a_mixed_convention(tmp_path, capsys):
    """A row without its eval log must be flagged, not silently mixed in.

    The seed statistics recompute mIoU on a fixed class divisor from the per-class
    eval logs.  If a variant has a recorded row but its log is gone, that variant
    keeps the *recorded* value while the others get the fixed one -- and a paired
    difference then mixes two conventions.  A simulated seed without a log
    produced a spurious +0.34 mIoU delta that way, so the loader now names the
    fallbacks.  A variant that has no row at all (not yet evaluated) is a
    different situation and must not trigger this warning.
    """
    import copy
    import json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.p1.summarize_seeds import load_rows  # noqa: PLC0415

    real = json.loads((Path(__file__).resolve().parents[1]
                       / 'benchmarks/results/p1_eval.json').read_text(encoding='utf-8'))
    if not real:
        pytest.skip('p1_eval.json is empty')

    # Every real seed now has an eval log, so point GROUPS at a variant name that
    # has a row but no log -- that is the situation the warning exists for.
    import benchmarks.p1.summarize_seeds as summarize_seeds
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(summarize_seeds.GROUPS['A2 KNN-mean'], 44, 'a2_ghost_seed')
    fake = copy.deepcopy(next(row for row in real if row['variant'] == 'a2_s43'))
    fake['variant'] = 'a2_ghost_seed'
    with_row = tmp_path / 'with_row.json'
    with_row.write_text(json.dumps(real + [fake]), encoding='utf-8')
    load_rows(with_row)
    output = capsys.readouterr().out
    assert '仍使用**记录值**' in output, output[:400]
    assert 'a2_ghost_seed' in output

    without_row = tmp_path / 'without_row.json'
    without_row.write_text(json.dumps(real), encoding='utf-8')
    load_rows(without_row)
    output = capsys.readouterr().out
    assert '仍使用**记录值**' not in output, (
        'a not-yet-evaluated seed must not be reported as a mixed convention')
    monkeypatch.undo()
