"""Tests for the per-class small-object analysis.

The first version of this analysis averaged whatever (horizon, class) pairs
happened to be finite in each variant.  Because ``motorcycle`` is NaN in several
runs, that compares *different class sets* between variants and manufactured a
sign flip in the small-object delta -- a second, independent instance of the
failure class documented in P1-5 (the ``nanmean`` divisor), one level up.

These tests pin the discipline that fixes it: group means must be taken over the
intersection of pairs finite in **all** compared variants, and the resulting
small-object delta must be negative for both seeds.  If someone reverts to
per-variant finite sets, the sign flips and these tests fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.common_class_miou import CLASSES, HORIZONS  # noqa: E402
from benchmarks.p1.per_class_analysis import (  # noqa: E402
    LARGE, SMALL, VARIANTS, finite_keys, group_mean)
from benchmarks.p1.common_class_miou import load  # noqa: E402


def test_class_groups_partition_the_scored_seventeen():
    scored = set(CLASSES[:17])
    assert set(SMALL) | set(LARGE) == scored
    assert not (set(SMALL) & set(LARGE))


def test_group_mean_uses_only_the_requested_keys():
    values = {f'{h}s/{name}': 1.0 for h in HORIZONS for name in SMALL}
    values['0s/bicycle'] = 5.0
    # Restricting to a single pair must return that pair's value exactly.
    assert group_mean(values, SMALL, {(0, 'bicycle')}) == pytest.approx(5.0)
    # Using every pair must average them.
    keys = finite_keys(values, SMALL)
    assert len(keys) == len(HORIZONS) * len(SMALL)
    expected = (5.0 + (len(keys) - 1) * 1.0) / len(keys)
    assert group_mean(values, SMALL, keys) == pytest.approx(expected)


def test_intersection_set_is_smaller_when_a_class_is_missing():
    """The whole point: a NaN in one variant shrinks the shared set."""
    full = {f'{h}s/{name}': 1.0 for h in HORIZONS for name in SMALL}
    gapped = dict(full)
    for horizon in (2, 4, 6):
        gapped[f'{horizon}s/motorcycle'] = float('nan')
    assert finite_keys(full, SMALL) > finite_keys(gapped, SMALL)
    shared = finite_keys(full, SMALL) & finite_keys(gapped, SMALL)
    assert all(name != 'motorcycle' or horizon == 0 for horizon, name in shared)


@pytest.mark.skipif(not (ROOT / 'out/p1_eval_a5_s43/eval.log').exists(),
                    reason='seed-43 eval logs not present')
def test_small_object_delta_is_negative_in_both_seeds():
    """A5 must be worse on small objects in both seeds, with the shared set.

    This is the report's revised verdict.  It is computed here from the logs, so
    it fails both if the data changes and if the estimator regresses.
    """
    data = load(list(VARIANTS.values()))
    present = [name for name in VARIANTS.values() if name in data]
    assert len(present) >= 5, 'not enough variants with logs'
    small_keys = set.intersection(*(finite_keys(data[n], SMALL) for n in present))
    assert small_keys, 'the shared small-object set is empty'
    deltas = []
    for a5, a2 in (('a5_cap0', 'a2_knn_mean'), ('a5_s43', 'a2_s43')):
        assert a5 in data and a2 in data
        deltas.append(group_mean(data[a5], SMALL, small_keys)
                      - group_mean(data[a2], SMALL, small_keys))
    assert all(delta < 0 for delta in deltas), (
        f'A5 should be worse on small objects in both seeds, got {deltas}')
    # And the effect must be larger than the seed-to-seed wobble of the metric
    # itself, otherwise "worse" would be overclaiming.
    assert all(abs(delta) > 0.05 for delta in deltas), deltas


@pytest.mark.skipif(not (ROOT / 'out/p1_eval_a5_cap8/eval.log').exists(),
                    reason='cap-sweep logs not present')
def test_matched_k_comparison_refutes_the_receptive_field_explanation():
    """At k = 8 the Gaussian adapter is still worse on thin classes than KNN-8.

    This is what refutes the "bigger field smooths thin structures away" reading
    of the adapter ordering: if field size were the cause, capping A5 to 8
    neighbours should recover the thin-class IoU, and it does the opposite
    (6.42 -> 6.22), while A2 at the same k scores 6.63.
    """
    data = load(['a5_cap0', 'a5_cap8', 'a5_cap32', 'a2_knn_mean'])
    if len(data) < 4:
        pytest.skip('cap sweep logs not present')
    keys = set.intersection(*(finite_keys(data[n], SMALL) for n in data))
    small = {n: group_mean(data[n], SMALL, keys) for n in data}
    # Matched neighbour count, different weighting: A2 must be clearly better.
    assert small['a2_knn_mean'] - small['a5_cap8'] > 0.3, small
    # And shrinking the field must NOT improve the thin-class score, which is
    # what kills the over-smoothing explanation.
    assert small['a5_cap8'] < small['a5_cap0'], small
    assert small['a5_cap32'] - small['a5_cap0'] < 0.1, small


@pytest.mark.skipif(not (ROOT / 'benchmarks/results/p4_selection_overlap.json').exists(),
                    reason='selection-overlap measurement not present')
def test_matched_k_is_not_matched_neighbours():
    """Pins why the matched-k comparison cannot be read as a weighting test.

    A5's cap selects by cell shells (unordered within a shell) and A2's selects
    the Euclidean nearest; measured on real points their sets overlap by only
    ~0.18 and A5's are ~2.5x farther.  If this ever became high-overlap the
    attribution in P1-6 would need revisiting, so the number is asserted rather
    than left in prose.
    """
    payload = json.loads(
        (ROOT / 'benchmarks/results/p4_selection_overlap.json').read_text(encoding='utf-8'))
    assert payload['k'] == 8
    assert payload['mean_selection_overlap'] < 0.3, payload
    assert payload['mean_distance_shell_m'] > 2 * payload['mean_distance_knn_m'], payload
    body = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md').read_text(encoding='utf-8')
    assert '0.177' in body and '2.5' in body
    assert '未解决' in body, 'the report must state what remains unresolved'


def test_small_object_deficit_holds_for_every_a5_seed():
    """The seed dimension of the small-object result.

    P1-6's avg-mIoU paired difference flips sign between seeds, but the
    small-object deficit does not: with A5 seed 44 added, all three A5 seeds sit
    below both A2 seeds (6.42 / 6.43 / 6.52 vs 6.55 / 6.63, mean gap -0.14, about
    3x the 0.05 seed SD).  This is the plan's threshold-3 alternative path
    ("a clear boundary/small-object gain") failing with a sign that is stable
    across seeds, which is a stronger statement than the report's original
    "worse in both seeds".

    Scope: the intersection over exactly these five variants, which yields the
    same 17 small (horizon, class) pairs the P1-6 table uses.
    """
    names = ['a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43', 'a2_s44']
    data = load(names)
    if not all(name in data for name in names):
        pytest.skip('per-variant eval logs not present')
    keys = None
    for name in names:
        variant_keys = finite_keys(data[name], CLASSES[:17])
        keys = variant_keys if keys is None else (keys & variant_keys)
    small = {key for key in keys if key[1] in SMALL}
    large = {key for key in keys if key[1] in LARGE}
    assert (len(small), len(large)) == (17, 45), (len(small), len(large))

    a5 = [group_mean(data[name], CLASSES[:17], keys=small)
          for name in ('a5_cap0', 'a5_s43', 'a5_s44')]
    a2 = [group_mean(data[name], CLASSES[:17], keys=small)
          for name in ('a2_knn_mean', 'a2_s43', 'a2_s44')]
    assert a5 == [pytest.approx(6.42, abs=0.005), pytest.approx(6.43, abs=0.005),
                  pytest.approx(6.52, abs=0.005)]
    assert a2 == [pytest.approx(6.63, abs=0.005), pytest.approx(6.55, abs=0.005),
                  pytest.approx(6.64, abs=0.005)]
    # The sign never flips: every A5 seed is below every A2 seed.
    assert max(a5) < min(a2)
    gap = sum(a5) / 3 - sum(a2) / 3
    assert gap == pytest.approx(-0.15, abs=0.02)

    # The structural-class difference stays inside the noise.
    large_gap = (sum(group_mean(data[name], CLASSES[:17], keys=large)
                     for name in ('a5_cap0', 'a5_s43', 'a5_s44')) / 3
                 - sum(group_mean(data[name], CLASSES[:17], keys=large)
                       for name in ('a2_knn_mean', 'a2_s43')) / 2)
    assert large_gap == pytest.approx(0.04, abs=0.02)

    text = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md').read_text(
        encoding='utf-8')
    for token in ('6.46', '6.61', '−0.15', '6.52', '6.64'):
        assert token in text, f'P1-6 lost the small-object token {token!r}'


def test_small_object_deficit_has_a_future_horizon_structure():
    """The pooled deficit hides a time structure worth pinning.

    Broken down by horizon, the small-object gap is +0.08 at 0 s and then
    -0.19 / -0.27 / -0.22 at 2/4/6 s: A5 is *better* on the current frame and
    worse on every future one, and the sign never flips within any horizon (all
    three A5 seeds on one side of both A2 seeds).  The per-horizon magnitudes are
    only ~2-3 SE, so the pooled 17-pair estimate stays the robust one -- this
    test pins that the report says both.
    """
    import statistics as stats
    names = ['a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43', 'a2_s44']
    data = load(names)
    if not all(name in data for name in names):
        pytest.skip('per-variant eval logs not present')
    keys = None
    for name in names:
        variant_keys = finite_keys(data[name], CLASSES[:17])
        keys = variant_keys if keys is None else (keys & variant_keys)
    small = {key for key in keys if key[1] in SMALL}

    expected = {0: (5, 7.22, 7.16, 0.06), 2: (4, 7.47, 7.69, -0.21),
                4: (4, 5.99, 6.31, -0.32), 6: (4, 4.95, 5.14, -0.20)}
    for horizon, (pairs, want_a5, want_a2, want_gap) in expected.items():
        subset = {key for key in small if key[0] == horizon}
        assert len(subset) == pairs, (horizon, len(subset))
        values = lambda name: group_mean(data[name], CLASSES[:17], keys=subset)
        a5 = [values(name) for name in ('a5_cap0', 'a5_s43', 'a5_s44')]
        a2 = [values(name) for name in ('a2_knn_mean', 'a2_s43', 'a2_s44')]
        assert stats.mean(a5) == pytest.approx(want_a5, abs=0.01), horizon
        assert stats.mean(a2) == pytest.approx(want_a2, abs=0.01), horizon
        gap = stats.mean(a5) - stats.mean(a2)
        assert gap == pytest.approx(want_gap, abs=0.02), horizon
        # The future horizons are two-sided separated; the current frame is NOT
        # any more, now that A2 has three seeds (A5 min 7.17 < A2 max 7.20).
        if horizon == 0:
            assert not (min(a5) > max(a2)), horizon
        else:
            assert max(a5) < min(a2), horizon

    text = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md').read_text(
        encoding='utf-8')
    for token in ('+0.06', '−0.21', '−0.32', '−0.20', '未来帧'):
        assert token in text, f'P1-6 lost the horizon-structure token {token!r}'


def test_structural_classes_show_no_a5_difference_at_any_horizon():
    """The other half of the decomposition: A5 vs A2 on structural classes.

    The small-object deficit is horizon-structured (-0.19/-0.27/-0.22 at
    2/4/6 s); on the 45 structural (horizon, class) pairs the two adapters are
    within +0.02..+0.05 at every horizon.  So the macro-average tie is "large
    objects even, thin objects behind", not "no difference anywhere" -- and any
    claim that A5's kernel helps has to name a class group where it does.
    """
    import statistics as stats
    names = ['a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43', 'a2_s44']
    data = load(names)
    if not all(name in data for name in names):
        pytest.skip('per-variant eval logs not present')
    keys = None
    for name in names:
        variant_keys = finite_keys(data[name], CLASSES[:17])
        keys = variant_keys if keys is None else (keys & variant_keys)
    large = {key for key in keys if key[1] in LARGE}
    assert len(large) == 45

    expected = {0: (12, 18.05, 18.03, 0.02), 2: (11, 17.69, 17.63, 0.06),
                4: (11, 17.24, 17.20, 0.04), 6: (11, 16.39, 16.34, 0.05)}
    for horizon, (pairs, want_a5, want_a2, want_gap) in expected.items():
        subset = {key for key in large if key[0] == horizon}
        assert len(subset) == pairs, (horizon, len(subset))
        values = lambda name: group_mean(data[name], CLASSES[:17], keys=subset)
        a5 = [values(name) for name in ('a5_cap0', 'a5_s43', 'a5_s44')]
        a2 = [values(name) for name in ('a2_knn_mean', 'a2_s43', 'a2_s44')]
        assert stats.mean(a5) == pytest.approx(want_a5, abs=0.01), horizon
        assert stats.mean(a2) == pytest.approx(want_a2, abs=0.01), horizon
        gap = stats.mean(a5) - stats.mean(a2)
        assert gap == pytest.approx(want_gap, abs=0.02), horizon
        # Never a usable A5 advantage, never a deficit either.
        assert -0.05 <= gap <= 0.10, (horizon, gap)

    text = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md').read_text(
        encoding='utf-8')
    assert '结构类' in text and '全部差异都落在细长类' in text


def test_every_adapters_gain_over_a0_is_concentrated_in_thin_classes():
    """Where each adapter's gain over A0 actually comes from.

    All four adapters gain almost exclusively on the 17 thin-class pairs
    (81-104% of the two-group total) while the 45 structural pairs stay flat, and
    A5 keeps only ~65% of the thin-class gain A2/A3 get.  That reframes the
    report's "A5 is worse on small objects": it is *less of a gain over A0*, not
    a loss relative to the baseline, and it is the only place A5 leads
    (structural classes, +0.06, inside noise).
    """
    import statistics as stats
    names = ['a0_base', 'a1_pointwise', 'a2_knn_mean', 'a2_s43', 'a2_s44',
             'a3_graph_attention', 'a5_cap0', 'a5_s43', 'a5_s44']
    data = load(names)
    if not all(name in data for name in names):
        pytest.skip('per-variant eval logs not present')
    keys = None
    for name in names:
        variant_keys = finite_keys(data[name], CLASSES[:17])
        keys = variant_keys if keys is None else (keys & variant_keys)
    small = {key for key in keys if key[1] in SMALL}
    large = {key for key in keys if key[1] in LARGE}
    assert (len(small), len(large)) == (17, 45)

    def mean_of(members, subset):
        return stats.mean([group_mean(data[name], CLASSES[:17], keys=subset)
                           for name in members])

    a0_small = group_mean(data['a0_base'], CLASSES[:17], keys=small)
    a0_large = group_mean(data['a0_base'], CLASSES[:17], keys=large)
    assert a0_small == pytest.approx(6.20, abs=0.01)
    assert a0_large == pytest.approx(17.30, abs=0.01)

    groups = {'A1': ['a1_pointwise'],
              'A2': ['a2_knn_mean', 'a2_s43', 'a2_s44'],
              'A3': ['a3_graph_attention'],
              'A5': ['a5_cap0', 'a5_s43', 'a5_s44']}
    gains = {}
    for label, members in groups.items():
        small_gain = mean_of(members, small) - a0_small
        large_gain = mean_of(members, large) - a0_large
        gains[label] = (small_gain, large_gain)
        # Gains are overwhelmingly on thin classes, structural ones stay flat.
        assert small_gain > 0.2, (label, small_gain)
        assert abs(large_gain) < 0.1, (label, large_gain)
        assert small_gain + large_gain > 0, label
    assert gains['A1'][0] == pytest.approx(0.45, abs=0.02)
    assert gains['A2'][0] == pytest.approx(0.41, abs=0.02)
    assert gains['A3'][0] == pytest.approx(0.40, abs=0.02)
    assert gains['A5'][0] == pytest.approx(0.26, abs=0.02)
    # A5 keeps only about two thirds of the thin-class gain A2 gets.
    assert gains['A5'][0] / gains['A2'][0] == pytest.approx(0.63, abs=0.05)
    # And it is the only adapter that does not lose ground on structural classes.
    assert gains['A5'][1] > max(gains['A1'][1], gains['A2'][1], gains['A3'][1])

    text = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md').read_text(
        encoding='utf-8')
    for token in ('+0.45', '+0.41', '+0.26', '+0.06', '的约 **63%**'):
        assert token in text, f'P1-6 lost the gain-decomposition token {token!r}'


def test_density_cap_damage_is_not_selective_unlike_the_gaussian_kernel():
    """Two different damage mechanisms, separated by the class-group frame.

    The Gaussian kernel is selective: -0.14 on thin classes, +0.04 on structural
    ones.  A tight cap is not: at cap=8 the structural group loses *more*
    (-0.29 vs the uncapped run) than the thin group (-0.20) and drops below the
    A0 baseline, which is what under-sampling the neighbourhood looks like.  The
    distinction decides what to fix: restoring cap=32 recovers the thin-class
    gain, while only changing the kernel recovers what the kernel removes.
    """
    import statistics as stats
    names = ['a0_base', 'a5_cap0', 'a5_cap128', 'a5_cap64', 'a5_cap32', 'a5_cap8']
    data = load(names)
    if not all(name in data for name in names):
        pytest.skip('per-variant eval logs not present')
    keys = None
    for name in names:
        variant_keys = finite_keys(data[name], CLASSES[:17])
        keys = variant_keys if keys is None else (keys & variant_keys)
    small = {key for key in keys if key[1] in SMALL}
    large = {key for key in keys if key[1] in LARGE}
    assert (len(small), len(large)) == (17, 45)

    def pair(name):
        return (group_mean(data[name], CLASSES[:17], keys=small),
                group_mean(data[name], CLASSES[:17], keys=large))

    a0_small, a0_large = pair('a0_base')
    uncapped_small, uncapped_large = pair('a5_cap0')
    assert (a0_small, a0_large) == pytest.approx((6.20, 17.30), abs=0.01)
    assert (uncapped_small, uncapped_large) == pytest.approx((6.42, 17.32), abs=0.01)

    cap8_small, cap8_large = pair('a5_cap8')
    assert cap8_small == pytest.approx(6.22, abs=0.01)
    assert cap8_large == pytest.approx(17.04, abs=0.01)
    thin_cost = cap8_small - uncapped_small
    structural_cost = cap8_large - uncapped_large
    assert thin_cost == pytest.approx(-0.20, abs=0.02)
    assert structural_cost == pytest.approx(-0.29, abs=0.02)
    # Non-selective: the structural group pays at least as much as the thin one.
    assert structural_cost <= thin_cost
    # ... and cap=8 falls below the baseline on structural classes.
    assert cap8_large < a0_large
    # By contrast the Gaussian kernel's cost is selective (P1-6 4.2).
    assert abs(uncapped_large - a0_large) < 0.05

    # cap=32 keeps the thin-class gain and pays little structurally.
    cap32_small, cap32_large = pair('a5_cap32')
    assert cap32_small - a0_small == pytest.approx(0.25, abs=0.02)
    assert cap32_large - uncapped_large == pytest.approx(-0.08, abs=0.02)

    text = (ROOT / 'analysis/sparse_render_result/P1-6_multi_seed.md').read_text(
        encoding='utf-8')
    for token in ('不是选择性的', '−0.29', '−0.26', '+0.25'):
        assert token in text, f'P1-6 lost the cap-damage token {token!r}'


def test_eval_log_path_convention_matches_between_writer_and_loader():
    """The loader must find what the evaluation actually wrote.

    ``eval_matrix`` writes ``out/p1_eval_<variant>/eval.log`` and
    ``common_class_miou.load`` reads exactly that.  A mismatch would not crash --
    the loader simply returns fewer variants, and every fixed-divisor analysis
    (cap curve, seed aggregation, class groups) would quietly drop points.  The
    end-to-end half of this test is the part that matters.
    """
    import inspect
    from benchmarks.p1 import common_class_miou, eval_matrix

    loader_source = inspect.getsource(common_class_miou.load)
    assert "'p1_eval_'" in loader_source or 'p1_eval_' in loader_source
    assert 'eval.log' in loader_source
    writer_source = inspect.getsource(eval_matrix)
    assert 'p1_eval_' in writer_source and 'eval.log' in writer_source

    # End to end: the variants this session relies on must all be discoverable.
    names = ['a0_base', 'a1_pointwise', 'a2_knn_mean', 'a2_s43',
             'a3_graph_attention', 'a5_cap0', 'a5_s43', 'a5_s44']
    found = common_class_miou.load(names)
    missing = sorted(set(names) - set(found))
    assert not missing, f'loader cannot find evaluated variants: {missing}'
    # And the new cap-seed variants are absent only because they have not run.
    for name in ('a5_cap8_s43', 'a5_cap128_s44'):
        path = ROOT / 'out' / f'p1_eval_{name}' / 'eval.log'
        assert (name in found) == path.exists(), name


def test_per_class_artifact_covers_every_seed_and_derives_its_verdict():
    """The artifact must span all three seeds and not hardcode "both seeds".

    Seed 44 was added to ``VARIANTS`` only after verifying that the intersection
    set (17 thin + 45 structural pairs) and every group mean stay bit-identical,
    so extending coverage cannot shift the report's numbers.  The verdict text
    used to say "in both seeds"; it is now derived from the pairs actually used.
    """
    import json as json_module
    path = ROOT / 'benchmarks/results/p1_per_class.json'
    if not path.exists():
        pytest.skip('per-class artifact not present')
    payload = json_module.loads(path.read_text(encoding='utf-8'))
    paired = payload['paired']
    assert set(paired) == {'seed 42', 'seed 43', 'seed 44'}, sorted(paired)
    assert payload['small_object_seeds'] == 3
    assert payload['small_object_sign_repeats'] is True
    small = [paired[key]['small'] for key in ('seed 42', 'seed 43', 'seed 44')]
    assert all(value < 0 for value in small), small
    assert small[0] == pytest.approx(-0.21, abs=0.01)
    assert small[1] == pytest.approx(-0.12, abs=0.01)
    assert small[2] == pytest.approx(-0.13, abs=0.01)
    # The fixed sets cover the scored 17 classes exactly, split into the two groups.
    assert len(payload['fixed_sets']['small_pairs']) == 17
    assert len(payload['fixed_sets']['large_pairs']) == 45
