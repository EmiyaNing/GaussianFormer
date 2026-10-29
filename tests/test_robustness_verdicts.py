"""Tests for the robustness probe's pre-registered verdict logic.

``cell_size`` is only a candidate-lookup structure, so cell variants are a
*consistency check*: identical metrics are expected, and a difference above the
seed noise is a bug rather than evidence of robustness.  ``support_sigma``
changes the support box itself, so a delta there is real model behaviour.  The
probe prints this reading next to its numbers; these tests pin the classification
so the wording cannot silently invert.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p4.robustness_probe import delta_verdicts  # noqa: E402


def row(name, value):
    return {'variant': name, 'metrics': {'avg/mIoU': value}}


def test_cell_variants_are_a_consistency_check_not_a_robustness_result():
    rows = [row('baseline', 14.32), row('cell0p9', 14.32), row('cell3p6', 14.31),
            row('support2', 13.90), row('support4', 14.10)]
    verdicts = delta_verdicts(rows, 14.32)
    worst, verdict, count = verdicts['cell']
    assert count == 2
    assert worst == pytest.approx(0.01, abs=1e-9)
    assert verdict == 'CONSISTENT'
    # Support deltas are real, and the verdict says so rather than judging them
    # against the cell tolerance: the largest is |14.32 - 13.90| = 0.42.
    worst, verdict, count = verdicts['support']
    assert count == 2
    assert worst == pytest.approx(0.42, abs=1e-9)
    assert verdict == 'REAL DELTA'


def test_a_material_cell_difference_is_flagged_as_a_defect():
    rows = [row('baseline', 14.32), row('cell0p9', 14.00)]
    worst, verdict, _ = delta_verdicts(rows, 14.32)['cell']
    assert worst == pytest.approx(0.32, abs=1e-9)
    assert verdict == 'DIFFERS', (
        'a cell variant that moved by 0.32 mIoU must be reported as a defect, '
        'not as robustness')


def test_axes_without_rows_are_omitted():
    assert delta_verdicts([row('baseline', 14.3)], 14.3) == {}
    assert set(delta_verdicts([row('cell0p9', 14.3)], 14.3)) == {'cell'}


def test_a_delta_inside_the_repeat_floor_is_not_called_a_detected_effect():
    """``noise`` qualifies the verdict instead of letting 'REAL DELTA' stand.

    The support axis is nominal: its wording means "a difference here would be
    real model behaviour".  Printed without qualification it reads as "a
    difference was detected", which the data does not support -- every measured
    delta is inside the same-config repeat floor.  Passing the floor must add the
    qualifier, and must not change the axis classification itself.
    """
    from benchmarks.p4.robustness_probe import SAME_CONFIG_FLOOR, delta_verdicts

    assert SAME_CONFIG_FLOOR == pytest.approx(0.03)
    rows = [row('baseline', 14.32), row('cell0p9', 14.32), row('support2', 14.31),
            row('support4', 14.32)]
    quiet = delta_verdicts(rows, 14.32, noise=SAME_CONFIG_FLOOR)
    assert quiet['support'][0] == pytest.approx(0.01, abs=1e-9)
    assert quiet['support'][1].startswith('REAL DELTA')
    assert 'no effect resolved' in quiet['support'][1]
    assert 'no effect resolved' in quiet['cell'][1]
    # Without the floor the historical wording is unchanged (other callers and
    # the existing tests rely on it).
    plain = delta_verdicts(rows, 14.32)
    assert plain['support'][1] == 'REAL DELTA'
    assert plain['cell'][1] == 'CONSISTENT'
    # A delta above the floor keeps the unqualified reading.
    loud = delta_verdicts([row('baseline', 14.32), row('support4', 14.90)],
                          14.32, noise=SAME_CONFIG_FLOOR)
    assert loud['support'][1] == 'REAL DELTA'


def test_accepted_degree_sweep_numbers_are_re_derived():
    """P4-8 section 3.1 quotes this sweep; re-derive it rather than trust it.

    Three properties are load-bearing: (a) the candidate count is a step function
    of ``k``, so on the support axis it is *literally constant* while the accepted
    degree moves 6.35x; (b) one candidate count maps to a 25x+ spread of real
    work, so a candidate-keyed cost model carries an irreducible error band; (c)
    the candidates never under-count, because the candidate set is a superset.
    The acceptance column must NOT be cited for cell invariance -- the helper
    takes no ``cell_size`` -- and this test pins that more directly: the same
    accepted value must appear for every cell at a fixed (scale, support).
    """
    path = ROOT / 'benchmarks/results/p4_accepted_degree.json'
    if not path.exists():
        pytest.skip('p4_accepted_degree.json not present')
    rows = json.loads(path.read_text(encoding='utf-8'))['rows']
    assert len(rows) == 22

    def find(dist, cell=None, scale=None, support=None):
        for row in rows:
            if row['dist'] != dist:
                continue
            if cell is not None and row['cell'] != cell:
                continue
            if scale is not None and row['scale'] != scale:
                continue
            if support is not None and row['support'] != support:
                continue
            return row
        raise AssertionError((dist, cell, scale, support))

    # (a) support axis: candidates identical, accepted varies.
    sup = [find('uniform', cell=1.8, scale=0.35, support=s)
           for s in (2.0, 3.0, 4.0)]
    assert len({row['candidate_degree_mean'] for row in sup}) == 1
    assert sup[0]['candidate_degree_mean'] == pytest.approx(226.486, abs=0.001)
    accepted = [row['accepted_degree_mean'] for row in sup]
    assert accepted == pytest.approx([5.820, 16.654, 36.971], abs=0.001)
    assert max(accepted) / min(accepted) == pytest.approx(6.35, abs=0.01)

    # (b) one candidate count -> a wide range of real work.
    same = [row for row in rows if row['dist'] == 'uniform'
            and row['candidate_degree_mean'] == pytest.approx(226.486, abs=0.001)]
    # Six rows: the base point is repeated once per axis, plus scale 0.1 and
    # the two other support values.
    assert len(same) == 6
    spread = (max(r['accepted_degree_mean'] for r in same)
              / min(r['accepted_degree_mean'] for r in same))
    assert spread == pytest.approx(25.8, abs=0.2), spread

    # (c) never an under-count; worst over-count is at the low-density end.
    ratios = [row['accepted_to_candidate'] for row in rows]
    assert max(ratios) < 1.0
    assert min(ratios) == pytest.approx(1 / 158.2, abs=1e-4)
    # The float32 boundary effect: scale 0.6 and 0.8 share one candidate count.
    assert (find('uniform', cell=1.8, scale=0.6, support=3.0)
            ['candidate_degree_mean']
            == find('uniform', cell=1.8, scale=0.8, support=3.0)
            ['candidate_degree_mean'])
    # Acceptance does not depend on cell (by construction) -- so it cannot be
    # evidence for the invariant; the report must say where the real evidence is.
    cell_rows = [find(row_dist, cell=c, scale=0.35, support=3.0)
                 for row_dist in ('uniform', 'clustered')
                 for c in (0.6, 1.2, 1.8, 2.4)]
    for row_dist in ('uniform', 'clustered'):
        values = {r['accepted_degree_mean'] for r in cell_rows
                  if r['dist'] == row_dist}
        assert len(values) == 1
    readme = (ROOT / 'analysis/sparse_render/results/README.md'
              ).read_text(encoding='utf-8')
    # The authoritative summary must carry the corrected scope too: candidates are
    # a step function of k, not "decided by max_scale alone".
    for token in ('1.50×–158×', '台阶函数', '6.35×', '25.8×'):
        assert token in readme, f'README summary lost {token!r}'
    report_text = (ROOT / 'analysis/sparse_render/results/P4-8_robustness.md'
                   ).read_text(encoding='utf-8')
    for token in ('3.1', '226.49', '158.2', '构造使然', '25.8'):
        assert token in report_text, f'P4-8 lost {token!r}'


# ---------------------------------------------------------------------------
# The GPU-free reparse path and the delivered artifact
# ---------------------------------------------------------------------------

CELL_VARIANTS = ('cell0p9', 'cell3p6')
SUPPORT_VARIANTS = ('support2', 'support4')


def test_reparse_rebuilds_rows_from_stored_logs_without_a_gpu(tmp_path):
    """``--reparse`` exists because the JSON is not the only copy of the result.

    An accidental re-invocation overwrote the JSON with a single row while the
    GPU was already busy with the next training stage.  Re-running would have
    meant waiting for the queue (or OOM-ing the training run), so the probe grew
    a reparse mode that rebuilds every row from ``out/p4_rob_<name>/eval.log``.
    It must refuse a partial rebuild rather than write a short artifact.
    """
    from benchmarks.p4.robustness_probe import VARIANTS, reparse_rows

    log = ('09/15 00:47:02 - mmengine - INFO - EVAL Iter 110\n'
           "09/15 00:47:02 - mmengine - INFO - {'num_samples': 113, "
           "'0s/mIoU': 14.85, 'avg/mIoU': 14.32, 'avg/boundaryIoU': 14.63}\n")
    for name in VARIANTS:
        path = tmp_path / 'out' / f'p4_rob_{name}' / 'eval.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(log, encoding='utf-8')

    rows = reparse_rows(list(VARIANTS), root=tmp_path)
    assert [r['variant'] for r in rows] == list(VARIANTS)
    for r in rows:
        assert r['metrics']['avg/mIoU'] == 14.32
        assert r['returncode'] is None, 'a reparse must not invent a return code'
        assert r['reparsed_from'].endswith(f'p4_rob_{r["variant"]}/eval.log')

    # A missing log is a hard failure: a partial artifact would look complete.
    (tmp_path / 'out' / 'p4_rob_support4' / 'eval.log').unlink()
    with pytest.raises(SystemExit, match='support4'):
        reparse_rows(list(VARIANTS), root=tmp_path)

    # An unknown name is rejected rather than silently skipped.
    with pytest.raises(SystemExit, match='unknown variant'):
        reparse_rows(['nope'], root=tmp_path)


def test_delivered_robustness_artifact_has_all_five_rows():
    """The delivered table must be complete and reproducible from its logs."""
    import json
    from benchmarks.p4.robustness_probe import VARIANTS, reparse_rows

    path = ROOT / 'benchmarks/results/p4_robustness.json'
    if not path.exists():
        pytest.skip('p4_robustness.json not present')
    stored = {r['variant']: r['metrics'] for r in
              json.loads(path.read_text(encoding='utf-8'))}
    assert set(stored) == set(VARIANTS)
    # Every row must agree with its own eval.log: that is what makes the table
    # auditable, and it is exactly the property the reparse path was written to
    # restore after the JSON was clobbered.
    reparsed = {r['variant']: r['metrics'] for r in reparse_rows(list(VARIANTS))}
    assert reparsed == stored
    for name, metrics in stored.items():
        assert metrics.get('avg/mIoU') is not None, name
        assert metrics.get('num_samples') == 113, name


def test_every_knob_delta_is_inside_the_same_config_repeatability():
    """The accuracy-side claim: no cliff, but also no resolved effect.

    Two evaluations of the *same* configuration with the *same* command line
    disagree by 0.01 avg / 0.02 per horizon (P1's ``a5_cap0``, the probe's first
    baseline, and the re-measured baseline).  Every cell/support delta must sit
    inside that floor -- otherwise the report's "no cliff" wording would be
    hiding a resolved effect.  The same-config pair is also asserted here so the
    floor cannot be quietly dropped from the report.
    """
    import json
    first_run = ROOT / 'benchmarks/results/p4_robustness_first_run_stdout.log'
    if not first_run.exists():
        pytest.skip('the first-run stdout was not preserved')
    text = first_run.read_text(encoding='utf-8')
    # The first run's own table: baseline 14.31 with the variants at 14.31-14.32.
    assert '| baseline | 14.31 | +0.00 |' in text
    for token in ('| cell0p9 | 14.32 | +0.01 |', '| cell3p6 | 14.32 | +0.01 |',
                  '| support2 | 14.31 | +0.00 |', '| support4 | 14.32 | +0.01 |'):
        assert token in text, token

    stored = {r['variant']: r['metrics'] for r in json.loads(
        (ROOT / 'benchmarks/results/p4_robustness.json').read_text('utf-8'))}
    p1 = {r['variant']: r['metrics'] for r in json.loads(
        (ROOT / 'benchmarks/results/p1_eval.json').read_text('utf-8'))}
    horizons = (0, 2, 4, 6)
    base = stored['baseline']
    # Same config, two different processes -> the noise floor.
    repeat = max(abs(p1['a5_cap0'][f'{h}s/mIoU'] - base[f'{h}s/mIoU'])
                 for h in horizons)
    assert repeat == pytest.approx(0.02, abs=1e-9), repeat

    # Every knob delta, against both the re-measured and the first-run baseline.
    for name in CELL_VARIANTS + SUPPORT_VARIANTS:
        worst = max(abs(stored[name][f'{h}s/mIoU'] - base[f'{h}s/mIoU'])
                    for h in horizons)
        assert worst <= 0.02 + 1e-9, (name, worst)
        assert abs(stored[name]['avg/mIoU'] - base['avg/mIoU']) <= 0.01 + 1e-9

    # The strongest cell-axis statement the data supports: bitwise equality of
    # two different bin edges.
    assert stored['cell0p9'] == stored['cell3p6']
    text = (ROOT / 'analysis/sparse_render/results/P4-8_robustness.md'
            ).read_text(encoding='utf-8')
    for token in ('逐位相同', '8.1', '14.31', '14.93', '0.03'):
        assert token in text, f'P4-8 lost {token!r}'
