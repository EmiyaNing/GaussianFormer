"""Tests for the aggregation-family comparison (report P1-4).

P1-4's conclusion -- the gain comes from adding a residual branch, not from the
aggregation form -- rested on A2's uniform mean while the max control trained.
This script exists to put the whole family at one fixed-divisor scope, so the
tests pin the two things that could quietly misstate it:

* the neighbour labels (A0 has no adapter at all, and labelling it "8 nearest"
  would misread the baseline as a kNN aggregator);
* the verdict, which must require BOTH a tight control spread and the Gaussian
  adapter not leading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.aggregation_family import (  # noqa: E402
    FAMILY, family_verdict, neighbour_label)

CONTROLS = ('a1_pointwise', 'a2_knn_mean', 'a2_knn_max', 'a3_graph_attention')


def test_neighbour_labels_do_not_call_the_baseline_an_aggregator():
    assert neighbour_label('a0_base') == '— (frozen baseline)'
    assert neighbour_label('a1_pointwise') == 'none (no aggregation)'
    assert '8 nearest' in neighbour_label('a2_knn_max')
    assert 'support box' in neighbour_label('a5_cap0')
    assert neighbour_label('something_else') == 'unknown'
    # Every family member but A0 aggregates something, and A0 must not claim to.
    for name in FAMILY.values():
        label = neighbour_label(name)
        assert label != 'unknown', name
        if name == 'a0_base':
            assert 'frozen' in label
        else:
            assert 'frozen' not in label


def test_verdict_requires_a_tight_spread_and_no_gaussian_lead():
    tight = {'a1_pointwise': 14.35, 'a2_knn_mean': 14.37, 'a2_knn_max': 14.36,
             'a3_graph_attention': 14.37, 'a5_cap0': 14.32}
    verdict, spread, margin = family_verdict(tight, CONTROLS, 'a5_cap0')
    assert verdict == 'CONFIRMED'
    assert spread == pytest.approx(0.02, abs=1e-9)
    assert margin == pytest.approx(-0.05, abs=1e-9)

    # A Gaussian that leads the controls is not the P1-4 story.
    leading = dict(tight, a5_cap0=14.60)
    verdict, _, margin = family_verdict(leading, CONTROLS, 'a5_cap0')
    assert verdict == 'REVIEW' and margin > 0.05

    # A wide control spread means the aggregation form does matter here.
    wide = dict(tight, a2_knn_max=14.80)
    verdict, spread, _ = family_verdict(wide, CONTROLS, 'a5_cap0')
    assert verdict == 'REVIEW' and spread > 0.4

    # Missing pieces never produce a confident verdict.
    assert family_verdict({}, CONTROLS, 'a5_cap0')[0] == 'INCOMPLETE'
    assert family_verdict({'a1_pointwise': 14.3}, CONTROLS, 'a5_cap0')[0] == \
        'INCOMPLETE'


def test_family_covers_the_controls_the_report_names():
    labels = set(FAMILY)
    assert 'A2 KNN max' in labels, 'the max-pooling control must stay in the family'
    assert 'A5 Gaussian' in labels and 'A1 point-wise' in labels
    assert len(FAMILY) == 6


def test_the_delivered_family_artifact_is_complete_and_matches_p1_4():
    """P1-4's §2 table quotes these rows; re-derive them from the artifact.

    The report's headline is *stronger* than when it was first written: the
    max-pooling control (which landed last) turned out to be the family's best
    adapter, so the Gaussian kernel is the worst of five rather than of four.
    That is exactly the kind of statement that must not live only in prose --
    the numbers below are what P1-4 §2 prints.
    """
    import json
    path = ROOT / 'benchmarks/results/p1_aggregation_family.json'
    if not path.exists():
        pytest.skip('p1_aggregation_family.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['missing'] == [], 'the family must be complete to quote it'
    assert payload['verdict'] == 'CONFIRMED'
    rows = {row['variant']: row for row in payload['rows']}
    assert set(rows) == set(FAMILY.values())
    assert rows['a2_knn_max']['recorded_mIoU'] == pytest.approx(14.40, abs=0.005)
    assert rows['a2_knn_max']['fixed_mIoU'] == pytest.approx(14.401, abs=0.0005)
    assert rows['a2_knn_max']['thin'] == pytest.approx(6.727, abs=0.0005)
    assert rows['a2_knn_max']['structural'] == pytest.approx(17.322, abs=0.0005)
    # The max control is the best adapter, and the Gaussian is the worst.
    adapters = {name: row['fixed_mIoU'] for name, row in rows.items()
                if name != 'a0_base'}
    assert max(adapters, key=adapters.get) == 'a2_knn_max'
    assert min(adapters, key=adapters.get) == 'a5_cap0'
    # Thin-class ordering is the load-bearing per-class detail in P1-4 §2.
    thin = {name: row['thin'] for name, row in rows.items()}
    assert min(adapters, key=lambda n: thin[n]) == 'a5_cap0'
    assert thin['a5_cap0'] == pytest.approx(6.418, abs=0.001)

    text = (ROOT / 'analysis/sparse_render/results/P1-4_full_control_matrix.md'
            ).read_text(encoding='utf-8')
    for token in ('14.401', '6.73', '0.084', 'A2 KNN-8 最大值'):
        assert token in text, f'P1-4 lost {token!r}'
