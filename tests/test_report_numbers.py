"""Audit load-bearing numbers in the P1-P4 reports against their sources.

Manual auditing of this report set has already found six real defects that no
test would have caught:

* a README headline that contradicted itself (two "efficiency summary" blocks,
  the stale one claiming the training-memory premise *held*);
* P4-2's self-check table marking the 2.61x training-memory claim as
  "unaffected" when it had been refuted;
* P3-2/P3-3 quoting cross-session training step times as ratios;
* P4-1 labelling candidate degrees as accepted degrees (a 2.8x overstatement);
* P1-3 asserting a seed spread that turned out to be a metric-divisor artifact;
* an "A4 not run" row whose stated reason was cost, not capability.

Each of those is a *number or claim drifting away from its recorded source*.
This module makes the numeric part mechanical: every assertion below re-reads
the artifact and requires the report to contain the formatted value.  It is
deliberately a transcription check, not a semantic one -- it cannot tell whether
a claim is well-founded, only that the digits on the page are the digits that
were measured.

Sources:
* ``benchmarks/results/p1_eval.json``      -- P1 control matrix (reduced split)
* ``benchmarks/results/p0_matrix_sweeps.jsonl`` -- cell/scale/support sweeps
* ``benchmarks/results/p3_train_mem_*.json``    -- training memory probes
* ``benchmarks/results/p4_block_vs_naive.json`` -- block vs naive dense G2V
* ``benchmarks/results/p4_tile_sweep.json``     -- analytic tile bound
* ``benchmarks/results/p4_ess.json``            -- effective sample size
* ``out/p1_eval_*/eval.log``                    -- per-class IoU (divisor check)
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'analysis/sparse_render/results'
RESULTS = ROOT / 'benchmarks/results'


def report(name: str) -> str:
    path = REPORTS / name
    assert path.exists(), f'missing report {name}'
    return path.read_text(encoding='utf-8')


NUMBER_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def numbers_in(text: str) -> list[float]:
    """Every numeric token in a report, commas removed.

    Assertions here are transcription checks, so they must not depend on how a
    value was rounded for display: a report may write 3.866 where 3.87 is the
    2-dp rendering.  Scanning the numbers and comparing with a tolerance is both
    robust to that and stronger, because it fails if the recorded value is
    absent at any plausible precision.
    """
    return [float(token.replace(',', '')) for token in NUMBER_RE.findall(text)]


def assert_value_in(text: str, value: float, tol: float, label: str) -> None:
    found = [n for n in numbers_in(text) if abs(n - value) <= tol]
    assert found, f'{label}: recorded value {value} (tol {tol}) not found in the report'


def p1_metrics() -> dict:
    rows = json.loads((RESULTS / 'p1_eval.json').read_text(encoding='utf-8'))
    return {row['variant']: row.get('metrics') or {} for row in rows}


def sweep(group: str, op: str = 'g2q') -> list[dict]:
    path = RESULTS / 'p0_matrix_sweeps.jsonl'
    if not path.exists():
        pytest.skip('p0 sweep results not present')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
            and json.loads(line).get('group') == group
            and json.loads(line).get('op') == op]


def train_mem(name: str) -> dict:
    path = RESULTS / f'p3_train_mem_{name}.json'
    if not path.exists():
        pytest.skip(f'{path.name} not present')
    return json.loads(path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# P1 control matrix: the numbers P1-1/P1-3/P1-4 all rest on
# ---------------------------------------------------------------------------

P1_TABLE = {
    'a0_base': 14.23, 'a1_pointwise': 14.35, 'a2_knn_mean': 14.37,
    'a2_knn_max': 14.40, 'a3_graph_attention': 14.37, 'a5_cap0': 14.32,
    'a5_cap8': 14.05,
    'a5_cap32': 14.26, 'a5_cap64': 14.30, 'a5_cap128': 14.32,
}


def test_p1_control_matrix_numbers_match_the_recording():
    metrics = p1_metrics()
    text = report('P1-1_gaussian_vs_baseline_and_cap.md') + \
        report('P1-3_aggregation_is_not_the_source.md') + \
        report('P1-4_full_control_matrix.md')
    for variant, expected in P1_TABLE.items():
        recorded = metrics.get(variant, {}).get('avg/mIoU')
        assert recorded == pytest.approx(expected, abs=0.005), (
            f'{variant}: recording changed ({recorded} vs {expected})')
        assert f'{expected:.2f}' in text, (
            f'{variant}={expected:.2f} is recorded but absent from the reports')


def test_same_weight_eval_repeats_differ_per_class_but_not_in_divisor():
    """The evaluation is not bit-reproducible, and the reason is NOT the divisor.

    Two evaluations of the same frozen checkpoint with the same command line
    disagree.  The tempting explanation -- P1-5's NaN/0.00 divisor flip -- is
    wrong here, and the reports say so explicitly: the *finite-class counts are
    identical* at every horizon while the per-class IoU values move (bicycle
    14.76 -> 14.44 at 2 s).  This re-derives both halves from the eval logs, so
    the distinction cannot rot back into a hand-wave.
    """
    classes = ['others', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable_surface',
               'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
    logs = (ROOT / 'out/p1_eval_a5_cap0/eval.log',
            ROOT / 'out/p4_rob_baseline/eval.log')
    for log in logs:
        if not log.exists():
            pytest.skip(f'{log} not present')

    def per_class(path):
        text = path.read_text(encoding='utf-8', errors='replace')
        table = {}
        for horizon in (0, 2, 4, 6):
            values = {}
            for name in classes:
                match = re.search(
                    rf"'{horizon}s/{re.escape(name)}':\s*([-+]?\d+\.?\d*|nan)",
                    text)
                assert match, f'{path.name}: no {horizon}s/{name}'
                values[name] = None if match.group(1) == 'nan' else float(
                    match.group(1))
            table[horizon] = values
        return table

    first, second = (per_class(log) for log in logs)
    # Same weights: the two commands are identical apart from --work-dir.
    for log in logs:
        head = log.read_text(encoding='utf-8', errors='replace')
        assert "resume_from='out/p1_a5_gaussian/epoch_40.pth'" in head
    expected = {0: 9, 2: 9, 4: 8, 6: 9}
    for horizon in (0, 2, 4, 6):
        finite_a = [n for n in classes if first[horizon][n] is not None]
        finite_b = [n for n in classes if second[horizon][n] is not None]
        assert finite_a == finite_b, (horizon, set(finite_a) ^ set(finite_b))
        assert len(finite_a) == 17 if horizon == 0 else len(finite_a) == 15
        differing = [n for n in classes
                     if first[horizon][n] != second[horizon][n]]
        assert len(differing) == expected[horizon], (horizon, differing)

    worst = max(abs(first[2][n] - second[2][n]) for n in classes
                if first[2][n] is not None and second[2][n] is not None)
    assert worst == pytest.approx(0.32, abs=0.005), worst

    p4_8 = report('P4-8_robustness.md')
    for token in ('不是**', '0.32', 'bicycle', '有限类别数', '假说'):
        assert token in p4_8, f'P4-8 lost {token!r}'
    # The authoritative summary must carry this floor too: it constrains every
    # small difference the directory reports.
    readme = report('README.md')
    for token in ('评测侧本底', '0.1–0.3', '不涉及分母'):
        assert token in readme, f'README summary lost {token!r}'
    p1_5 = report('P1-5_metric_divisor_artifact.md')
    for token in ('同一份冻结权重', '14.31', '14.32', '17/15/15/15'):
        assert token in p1_5, f'P1-5 lost {token!r}'
    # The retracted hypothesis must not survive anywhere in P1-5.
    assert '同一份权重、同一批 113 个场景' not in p1_5


def test_p1_7_a4_kernel_form_numbers():
    """Re-derive P1-7's contrasts from the artifacts, not from its prose.

    A4 is the control that separates "the Gaussian kernel form is useless" from
    "the learned geometry was the mistake".  Its numbers are load-bearing for the
    *scope* of P1's negative claim, so they are recomputed here: the fixed-divisor
    values, the three contrasts, the class-group split, and the verdict branch.
    """
    path = ROOT / 'benchmarks/results/p1_kernel_form_controls.json'
    if not path.exists():
        pytest.skip('p1_kernel_form_controls.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['missing'] == []
    rows = {row['variant']: row for row in payload['rows']}
    assert rows['a4_fixed_gaussian']['fixed_mIoU'] == pytest.approx(14.361, abs=0.0005)
    assert rows['a4_fixed_gaussian']['recorded_mIoU'] == pytest.approx(14.36, abs=0.005)
    contrasts = payload['contrasts']
    assert contrasts['A5 - A4']['delta'] == pytest.approx(-0.0441, abs=0.0005)
    assert contrasts['A5 - A4']['thin_delta'] == pytest.approx(-0.1071, abs=0.0005)
    assert contrasts['A5 - A4']['single_factor'] is True
    assert contrasts['A4 - A2']['delta'] == pytest.approx(-0.0136, abs=0.0005)
    assert contrasts['A4 - A2']['single_factor'] is False, (
        'A4 keeps the shell selection, so A4-A2 is confounded')
    assert contrasts['A5 - A2']['thin_delta'] == pytest.approx(-0.2112, abs=0.0005)
    assert payload['verdict'] == 'geometry-is-the-harm'
    # The pre-registered branch is met only just: |A4 thin| < |A5 thin| / 2.
    a4_thin, a5_thin = (abs(contrasts['A4 - A2']['thin_delta']),
                        abs(contrasts['A5 - A2']['thin_delta']))
    assert a4_thin < a5_thin / 2
    margin = (a5_thin / 2 - a4_thin) / (a5_thin / 2)
    assert margin < 0.02, 'if this ever widens, the caveat can be relaxed'
    # A4 is the best of the four on structural classes and the worst nowhere
    # except thin classes.
    assert rows['a4_fixed_gaussian']['structural'] == max(
        row['structural'] for row in rows.values())
    # A5 stays the lowest of *all* adapters once A4 is included.
    fixed = {name: row['fixed_mIoU'] for name, row in rows.items()}
    assert fixed['a5_cap0'] < fixed['a4_fixed_gaussian']
    assert min(fixed.values()) == fixed['a0_base'], 'A0 is the frozen baseline'
    adapters = {k: v for k, v in fixed.items() if k != 'a0_base'}
    assert min(adapters, key=adapters.get) == 'a5_cap0'

    # A4 is smaller than A5 while being no worse: the frozen heads drop 512
    # trainable parameters.  Read that from the training log, not from prose.
    import re
    logs = sorted((ROOT / 'out/a4_fixed_gaussian').glob('*.log'))
    if logs:
        found = re.search(r'Number of params: (\d+)',
                          logs[-1].read_text(encoding='utf-8', errors='replace'))
        assert found, 'the A4 training log must record its parameter count'
        assert int(found.group(1)) == 179989
        assert 180501 - int(found.group(1)) == 512
    text = report('P1-7_a4_fixed_geometry.md')
    for token in ('14.361', '−0.044', '0.107', 'geometry-is-the-harm', '1.4%',
                  '0.405', '只过线', '179,989', '少 512'):
        assert token in text, f'P1-7 lost {token!r}'


def test_p1_7_field_size_confound_is_measured():
    """P1-7 discloses a field-size confound; it must be a measurement.

    ``freeze_geometry`` also changes the scale the model settles on, so A4
    aggregates a smaller neighbourhood than A5 -- and P1-6 showed smaller
    neighbourhoods hurt thin classes.  That confound runs *against* P1-7's
    conclusion, so the report quantifies it instead of estimating it.  This
    re-derives the ratio from the probe artifact.
    """
    path = ROOT / 'benchmarks/results/p1_a4_field_size.json'
    if not path.exists():
        pytest.skip('p1_a4_field_size.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['points'] == 138124
    rows = {row['label'].split()[0]: row for row in payload['rows']}
    assert rows['A4']['scale'] == pytest.approx(0.4051, abs=0.0005)
    assert rows['A5']['scale'] == pytest.approx(0.5381, abs=0.0005)
    assert rows['A4']['accepted_degree_mean'] == pytest.approx(545.6, abs=0.2)
    assert rows['A5']['accepted_degree_mean'] == pytest.approx(941.0, abs=0.2)
    assert payload['half_width_ratio'] == pytest.approx(0.753, abs=0.002)
    assert payload['accepted_ratio_a5_over_a4'] == pytest.approx(1.72, abs=0.02)
    # Sub-cubic: the real cloud's neighbour density grows slower than the box
    # volume, so the cube-scaling bound (2.34x) over-predicts.
    assert payload['accepted_ratio_a5_over_a4'] < 1 / payload['half_width_ratio'] ** 3
    text = report('P1-7_a4_fixed_geometry.md')
    for token in ('545.6', '941.0', '1.72×', '4 个样本拼接', '次立方'):
        assert token in text, f'P1-7 lost {token!r}'
    # The confound is bounded by P4-8's support sweep at A5's own weights: a -33%
    # box costs <=0.01 mIoU, so A4's -24.7% box cannot explain +0.044.
    rob = {row['variant']: row['metrics'] for row in json.loads(
        (ROOT / 'benchmarks/results/p4_robustness.json').read_text('utf-8'))}
    assert rob['support2']['avg/mIoU'] == pytest.approx(14.31, abs=0.005)
    assert rob['baseline']['avg/mIoU'] == pytest.approx(14.32, abs=0.005)
    assert abs(rob['support2']['avg/mIoU']
               - rob['baseline']['avg/mIoU']) <= 0.03
    for token in ('−33% / 0 / +33%', '场大小解释不了'):
        assert token in text, f'P1-7 lost {token!r}'


def test_p1_7_a4_geometry_is_global_not_frozen():
    """A4 removes *per-point* geometry; it does not freeze the geometry.

    ``freeze_geometry`` sets ``requires_grad_(False)`` on the two heads' zero
    weights and leaves the biases trainable, so the shared scale/opacity are still
    learned.  Three documents (the model comment, the A4 config and P1-4) said
    "pins scale = 0.35 / opacity = 0.02", which the checkpoint refutes.  This
    re-reads the checkpoint so the corrected statement cannot silently regress.
    """
    import math
    import torch

    ckpt = ROOT / 'out/a4_fixed_gaussian/epoch_40.pth'
    if not ckpt.exists():
        pytest.skip('A4 checkpoint not present')
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = payload.get('state_dict', payload)
    weight = [v.float() for k, v in state.items() if k.endswith('scale_head.weight')]
    assert weight, 'no scale_head.weight in the checkpoint'
    assert float(weight[0].abs().max()) == 0.0, (
        'the frozen weight must be exactly zero -- that is what removes the '
        'per-point term')
    bias = [v.float() for k, v in state.items() if k.endswith('scale_head.bias')][0]
    opacity_bias = [v.float() for k, v in state.items()
                    if k.endswith('opacity_head.bias')][0]
    assert bias.abs().max() > 0.1, (
        'the bias must have moved from its zero init, i.e. the geometry was '
        'learned, not frozen')
    lo, hi = 0.1, 0.6
    scales = sorted((lo + (hi - lo) * torch.sigmoid(bias)).tolist())
    # Order-agnostic: the per-axis values are the *set* {0.4033, 0.4051, 0.4069}.
    assert scales == pytest.approx([0.403, 0.405, 0.407], abs=0.002), scales
    opacity = float(torch.sigmoid(opacity_bias).item())
    assert opacity == pytest.approx(0.0186, abs=0.0005), opacity
    # And the prose must say so.
    p1_7 = report('P1-7_a4_fixed_geometry.md')
    for token in ('偏置仍可训练', '{0.403, 0.405, 0.407}', '0.0186', '逐点'):
        assert token in p1_7, f'P1-7 lost {token!r}'
    p1_4 = report('P1-4_full_control_matrix.md')
    assert '**不是** 0.35/0.02' in p1_4
    # The full-val stage labels A4 too; its row is written last, so the label
    # must not reintroduce "fixed isotropic" (which the checkpoint refutes).
    from benchmarks.p1.fullval_anchor import VARIANTS
    label = VARIANTS['a4_fixed_gaussian'][2]
    assert 'global' in label and 'fixed isotropic' not in label, label
    model_src = (ROOT / 'model/head/point_anchored_semantic_gaussian.py'
                 ).read_text(encoding='utf-8')
    assert 'pins\n            # scale = initial_scale' not in model_src
    assert 'biases\n            # remain trainable' in model_src or \
        'biases' in model_src.split('freeze_geometry')[-1][:900]


def test_p1_4_reconciles_the_recorded_and_fixed_divisor_orderings():
    """The control matrix must not rest on the recorded (drifting) divisor.

    P1-4's §1 table shows recorded ``avg/mIoU`` values, but every cross-run
    comparison in this project uses the fixed-class-set recomputation (P1-5 §4).
    The report therefore has to state that both orderings agree -- otherwise a
    reader who knows about the divisor artifact cannot tell whether the FAIL
    verdict survives it.  The claim is re-derived here from the artifact rather
    than trusted.
    """
    path = RESULTS / 'p1_common_class_miou.json'
    if not path.exists():
        pytest.skip('p1_common_class_miou.json not present')
    rows = json.loads(path.read_text(encoding='utf-8'))['rows']
    fixed = {name: rows[name]['comparable_avg_mIoU'] for name in
             ('a0_base', 'a1_pointwise', 'a2_knn_mean', 'a3_graph_attention',
              'a5_cap0')}
    adapters = ('a1_pointwise', 'a2_knn_mean', 'a3_graph_attention', 'a5_cap0')
    # A5 is the lowest of the four adapters under the fixed divisor...
    assert fixed['a5_cap0'] == min(fixed[name] for name in adapters)
    next_lowest = min(fixed[name] for name in adapters if name != 'a5_cap0')
    assert next_lowest - fixed['a5_cap0'] == pytest.approx(0.030, abs=0.001)
    # ... and A2 minus A5 is the +0.058 the report quotes.
    assert fixed['a2_knn_mean'] - fixed['a5_cap0'] == pytest.approx(0.058, abs=0.001)

    text = report('P1-4_full_control_matrix.md')
    for token in ('固定类别集重算', '14.229', '14.347', '14.375', '14.367',
                  '14.317', '+0.058'):
        assert token in text, f'P1-4 lost the reconciliation token {token!r}'


def test_boundary_iou_row_is_still_reported():
    metrics = p1_metrics()
    text = report('P4-7_quality_diagnostics.md')
    for variant in ('a0_base', 'a2_knn_mean', 'a5_cap0'):
        value = metrics[variant]['avg/boundaryIoU']
        assert f'{value:.2f}' in text, f'{variant} boundary IoU {value} missing'


# ---------------------------------------------------------------------------
# P1-5: the divisor artifact must stay auditable by hand
# ---------------------------------------------------------------------------

def test_divisor_artifact_arithmetic_is_reproducible_from_the_logs():
    """Recompute both 6 s mIoUs from the per-class IoU of the seed pair."""
    classes = ['others', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable_surface',
               'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
    results = {}
    for variant in ('a5_cap0', 'a5_s43'):
        log = (ROOT / 'out' / f'p1_eval_{variant}' / 'eval.log')
        if not log.exists():
            pytest.skip(f'{log} not present')
        text = log.read_text(encoding='utf-8', errors='replace')
        values = []
        for name in classes:
            match = re.search(rf"'6s/{re.escape(name)}':\s*([-+]?\d+\.?\d*|nan)", text)
            assert match, f'{variant}: no 6s/{name} in log'
            if match.group(1) != 'nan':
                values.append(float(match.group(1)))
        results[variant] = (sum(values) / len(values), len(values))

    (mean42, n42), (mean43, n43) = results['a5_cap0'], results['a5_s43']
    # The divisor differs: 15 finite classes for seed 42, 16 for seed 43, which
    # is the entire 0.75 mIoU gap P1-5 documents.
    assert (n42, n43) == (15, 16)
    assert round(mean42, 2) == pytest.approx(13.28, abs=0.005)
    assert round(mean43, 2) == pytest.approx(12.53, abs=0.005)
    text = report('P1-5_metric_divisor_artifact.md')
    for token in ('199.21/15', '200.50/16', '0.75', '13.28', '12.53', '14.365'):
        assert token in text, f'P1-5 lost the token {token!r}'


def test_the_divisor_artifact_disappears_on_the_full_val_split():
    """P1-5 2.2's boundary: with 37x more samples the divisor is uniform.

    The artifact needs a rare class that never appears in *any* evaluated sample;
    on the full val split every class is finite at every horizon, so the recorded
    value equals the fixed-divisor one there.  That is the honest boundary of
    P1-5's claim -- and it must not be read as "the fixed-divisor discipline was
    unnecessary", which is why the report says both things.
    """
    path = ROOT / 'benchmarks/results/p1_fullval.json'
    if not path.exists():
        pytest.skip('p1_fullval.json not present')
    rows = {r['variant']: r.get('metrics') or {} for r in
            json.loads(path.read_text(encoding='utf-8'))}
    base = rows.get('a0_base') or {}
    if base.get('avg/mIoU_divisor_min') is None:
        pytest.skip('the full-val row predates the divisor keys')
    assert base['num_samples'] >= 4000
    assert base['avg/mIoU_divisor_min'] == 17
    assert base['avg/mIoU_divisor_max'] == 17
    assert base['avg/mIoU_divisor_uniform_across_horizons'] is True
    # The reduced split is where the artifact lives: the finite-class count varies
    # by horizon there (17/15/15/15), and the two re-evaluated rows carry the
    # explicit non-uniform flag.
    reduced = {r['variant']: r.get('metrics') or {} for r in
               json.loads((ROOT / 'benchmarks/results/p1_eval.json'
                           ).read_text(encoding='utf-8'))}
    flagged = [m for m in reduced.values()
               if 'avg/mIoU_divisor_min' in m]
    assert flagged, 'a reduced row must carry the divisor keys'
    for metrics in flagged:
        assert metrics['avg/mIoU_divisor_uniform_across_horizons'] is False
        assert (metrics['avg/mIoU_divisor_min']
                < metrics['avg/mIoU_divisor_max'])
    ccm = json.loads((ROOT / 'benchmarks/results/p1_common_class_miou.json'
                      ).read_text(encoding='utf-8'))['rows']
    assert ccm['a5_cap0']['finite_classes'] == {'0': 17, '2': 15, '4': 15, '6': 15}
    assert ccm['a5_s43']['finite_classes'] == {'0': 17, '2': 15, '4': 15, '6': 16}
    text = report('P1-5_metric_divisor_artifact.md')
    for token in ('2.2', 'divisor_min', '4,219', '不能替代固定除数'):
        assert token in text, f'P1-5 lost {token!r}'
    # The authoritative summary must carry the anchor outcome, because the old
    # text ("never checked") became false the moment the full-val stage ran.
    readme = report('README.md')
    for token in ('14.47 / 23.08', '+1.27', '+1.05', 'neither', '3.6×'):
        assert token in readme, f'README ⑧ lost {token!r}'
    assert '从未复核' not in readme


def test_p1_common_class_script_fixes_the_divisor():
    path = RESULTS / 'p1_common_class_miou.json'
    if not path.exists():
        pytest.skip('p1_common_class_miou.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    row = payload['rows']['a5_s43']
    assert row['recorded_avg_mIoU'] == pytest.approx(14.16, abs=0.005)
    assert row['comparable_avg_mIoU'] == pytest.approx(14.365, abs=0.005)
    assert '14.36' in report('P1-5_metric_divisor_artifact.md')


# ---------------------------------------------------------------------------
# P4-8: the cost-side robustness pivots
# ---------------------------------------------------------------------------

def test_p4_8_cell_sweep_matches_the_recorded_sweep():
    rows = {(r['dist'], r['cell_size']): r for r in sweep('cell')}
    if not rows:
        pytest.skip('cell sweep missing')
    text = report('P4-8_robustness.md')
    for cell, expected in ((0.6, 1.809), (1.2, 1.702), (1.8, 2.597), (2.4, 3.839)):
        recorded = rows[('uniform', cell)]['forward']['p50_ms']
        assert recorded == pytest.approx(expected, abs=0.002), (
            f'cell {cell} uniform latency changed: {recorded}')
        assert_value_in(text, expected, 0.001, f'cell {cell} uniform')
    for cell, expected in ((0.6, 46.120), (2.4, 51.761)):
        recorded = rows[('clustered', cell)]['forward']['p50_ms']
        assert recorded == pytest.approx(expected, abs=0.002)
        assert_value_in(text, expected, 0.001, f'cell {cell} clustered')


def test_p4_8_support_axis_shows_the_candidate_count_is_blind():
    """The pivot of P4-8 section 3: identical candidates, different latency."""
    rows = {r['support_sigma']: r for r in sweep('support')
            if r['dist'] == 'uniform'}
    if len(rows) < 3:
        pytest.skip('support sweep missing')
    candidates = {r['pairs']['candidate_pairs'] for r in rows.values()}
    assert len(candidates) == 1, (
        f'the support axis is supposed to leave candidates identical, got {candidates}')
    latencies = [rows[s]['forward']['p50_ms'] for s in (2.0, 3.0, 4.0)]
    assert latencies == sorted(latencies), 'latency should grow with support'
    text = report('P4-8_robustness.md')
    for value in latencies:
        assert_value_in(text, value, 0.001, f'support latency {value}')
    assert '17.39 M' in text or '17,394,142' in text


def test_p4_8_scale_sweep_extremes_are_quoted():
    rows = {(r['dist'], r['scale']): r for r in sweep('scale')}
    if not rows:
        pytest.skip('scale sweep missing')
    best = rows[('uniform', 0.1)]['forward']['p50_ms']
    worst = rows[('clustered', 0.8)]['forward']['p50_ms']
    assert best == pytest.approx(1.325, abs=0.002)
    assert worst == pytest.approx(100.024, abs=0.05)
    text = report('P4-8_robustness.md')
    assert_value_in(text, best, 0.001, 'scale 0.1 uniform')
    assert_value_in(text, worst, 0.01, 'scale 0.8 clustered')
    # 75x spread and the reachability caveat (the model's scale_range caps at 0.6)
    assert '75' in text and '0.6' in text


# ---------------------------------------------------------------------------
# Training memory: P4-4's whole claim
# ---------------------------------------------------------------------------

def test_p4_4_training_memory_pair_matches_the_probes():
    dense = train_mem('p3_dense_g2v_block_fullhorizon')
    sparse = train_mem('sparse_cap0_fullhorizon')
    dense_mib = max(r['peak_alloc_mib'] for r in dense['rows'])
    sparse_mib = max(r['peak_alloc_mib'] for r in sparse['rows'])
    assert dense_mib == pytest.approx(2226.92, abs=0.01)
    assert sparse_mib == pytest.approx(2210.85, abs=0.01)
    text = report('P4-4_training_memory_final.md')
    for value in (2226.92, 2210.85, 5761.27):
        assert_value_in(text, value, 0.01, f'P4-4 memory {value}')
    assert '2.61' in text


def test_fullval_arms_are_the_same_models_on_the_full_split():
    """The split comparison is only valid if nothing else changed.

    P1-9 compares every variant's reduced-split row against a full-val row and
    asks whether the *sample set* moved the numbers.  That reading requires the
    two rows to be the **same weights** evaluated on a different split, so this
    pins both halves: the checkpoint is identical, and the full-val config is the
    reduced config's child whose only change is the imageset (back to the full
    pickle).  A stray edit here would silently turn the comparison into a
    different-model comparison.
    """
    import re
    from benchmarks.p1.fullval_anchor import VARIANTS

    reduced = {row['variant']: row for row in json.loads(
        (ROOT / 'benchmarks/results/p1_eval.json').read_text(encoding='utf-8'))}
    full_split = 'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'
    for name, (config, checkpoint, _) in VARIANTS.items():
        row = reduced.get(name)
        if row is None:
            continue
        assert row['checkpoint'] == checkpoint, (
            f'{name}: the full-val arm must reuse the reduced row\'s weights')
        text = (ROOT / config).read_text(encoding='utf-8')
        assert f"imageset='{full_split}'" in text, f'{name}: not the full split'
        base = re.search(r"_base_ = \[['\"]([^'\"]+)['\"]\]", text)
        assert base, f'{name}: no _base_'
        base = base.group(1).lstrip('./')
        assert row['config'].endswith(base), (name, row['config'], base)
    # A0's arm must use the *released* checkpoint, not a trained adapter.
    assert VARIANTS['a0_base'][1] == 'ckpts/sparseworld_strict_epoch_56.pth'


def test_sparse_cap0_time_instability_is_documented_where_it_is_quoted():
    """303.0 vs 669.1 ms for the same config: P3-2 and P4-4 must both say so."""
    a5 = train_mem('a5_fullhorizon')
    cap0 = train_mem('sparse_cap0_fullhorizon')
    assert a5['rows'][0]['p50_ms'] == pytest.approx(669.118, abs=0.01)
    assert cap0['rows'][0]['p50_ms'] == pytest.approx(303.035, abs=0.01)
    for name in ('P3-2_training_cost_probe.md', 'P4-4_training_memory_final.md',
                 'P3-3_dense_vs_sparse.md'):
        text = report(name)
        assert '303.0' in text or '303.035' in text, (
            f'{name} quotes training step time without the instability caveat')


# ---------------------------------------------------------------------------
# Block G2V and the tile bound
# ---------------------------------------------------------------------------

def test_p4_3_block_speedup_matches_the_benchmark():
    path = RESULTS / 'p4_block_vs_naive.json'
    if not path.exists():
        pytest.skip('p4_block_vs_naive.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = {r['operator']: r for r in payload['rows']}
    assert payload['speedup_p50'] == pytest.approx(8.141, abs=0.001)
    assert rows['naive_localagg']['p50_ms'] == pytest.approx(9.379, abs=0.001)
    assert rows['block_4x4x4']['p50_ms'] == pytest.approx(1.152, abs=0.001)
    assert rows['naive_localagg']['peak_alloc_mib'] == pytest.approx(607.7, abs=0.05)
    assert rows['block_4x4x4']['peak_alloc_mib'] == pytest.approx(95.0, abs=0.05)
    text = report('P4-3_block_g2v_measured.md')
    for value in (8.141, 9.379, 1.152, 607.7, 95.0):
        assert_value_in(text, value, 0.005, f'P4-3 value {value}')


def test_tile_bound_numbers_match_the_sweep():
    path = RESULTS / 'p4_tile_sweep.json'
    if not path.exists():
        pytest.skip('p4_tile_sweep.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    bounds = {row['tile_edge']: row for row in payload['bounds']}
    assert bounds[2]['bound_ratio'] == pytest.approx(5.918, abs=0.001)
    assert bounds[4]['bound_ratio'] == pytest.approx(28.945, abs=0.001)
    assert bounds[8]['bound_ratio'] == pytest.approx(116.855, abs=0.001)
    assert bounds[4]['alu_inflation'] == pytest.approx(2.211, abs=0.001)
    assert bounds[4]['L_voxel_loads'] == 20_303_505
    text = report('P4-8_robustness.md') + report('P4-2_block_g2v_bound.md')
    # Ratios are quoted at 2 dp in the report, so the tolerance must exceed half
    # a unit in that place; 116.855 renders as 116.85 and lands exactly on the
    # 0.005 boundary, so a hair more is needed to stay off it.
    for value in (5.918, 28.945, 116.855, 2.211):
        assert_value_in(text, value, 0.006, f'tile bound {value}')
    assert_value_in(text, 20_303_505, 0.5, 'tile L')


# ---------------------------------------------------------------------------
# Effective sample size: the receptive-field confound
# ---------------------------------------------------------------------------

def test_ess_numbers_match_and_are_quoted_with_the_confound():
    path = RESULTS / 'p4_ess.json'
    if not path.exists():
        pytest.skip('p4_ess.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    row = next(r for r in payload['rows']
               if r['scale'] == 0.53 and r['include_self'])
    assert row['accepted_mean'] == pytest.approx(215.2, abs=0.05)
    assert row['ess_mean'] == pytest.approx(59.25, abs=0.01)
    assert row['ess_p50'] == pytest.approx(38.55, abs=0.01)
    assert row['ess_over_accepted'] == pytest.approx(0.2776, abs=0.001)
    assert row['corr_ess_accepted'] > 0.9
    text = report('P1-3_aggregation_is_not_the_source.md')
    for value, tol in ((215.2, 0.05), (59.3, 0.05), (38.6, 0.05),
                       (0.278, 0.001), (0.938, 0.001)):
        assert_value_in(text, value, tol, f'P1-3 ESS {value}')
    # The cap-cost predictor, and the interpretation that a third of the weight
    # mass is discardable: -0.06 mIoU at a 68.5% retained share.
    approx = {int(k): v for k, v in row['uniform_approx_error'].items()
              if not k.startswith('_')}
    assert min(approx, key=approx.get) == 64, 'the optimal uniform k moved'
    assert approx[64] == pytest.approx(0.147, abs=0.001)
    assert_value_in(text, 0.147, 0.001, 'best uniform approximation error')
    shares = row['nearest_weight_share']
    assert float(shares['32']) == pytest.approx(0.685, abs=0.005)
    assert float(shares['8']) == pytest.approx(0.331, abs=0.005)
    assert_value_in(text, 0.685, 0.002, 'cap=32 retained share')
    assert '68.5' in text and '−0.06' in text


def test_receptive_field_controls_are_registered_as_variants():
    """Both matched controls must exist as configs and be wired into the harness."""
    matrix = (ROOT / 'benchmarks/p1/eval_matrix.py').read_text(encoding='utf-8')
    labels = (ROOT / 'benchmarks/p1/summarize_p1.py').read_text(encoding='utf-8')
    for name, k in (('a2_knn_k59', 59), ('a2_knn_k215', 215)):
        config = ROOT / 'config/opus/ablations' / f'p1_{name}.py'
        assert config.exists(), f'{config} missing'
        assert f'num_neighbors={k}' in config.read_text(encoding='utf-8')
        assert name in matrix and name in labels


# ---------------------------------------------------------------------------
# Experiment metadata (plan README section 4.2)
# ---------------------------------------------------------------------------

def test_metadata_table_is_in_the_readme_and_matches_the_configs():
    from benchmarks.p1.experiment_metadata import EXPERIMENTS, describe  # noqa: PLC0415
    text = report('README.md')
    for token in ('ResNet-50', '256x704', '720×48 = 34,560',
                  '无 camera/lidar mask'):
        assert token in text, f'metadata table lost {token!r}'
    for name in EXPERIMENTS:
        row = describe(name, EXPERIMENTS[name])
        assert row['backbone'] == 'ResNet-50'
        assert row['input_size'] == '256x704'
        assert row['frames'] == 5
        assert row['queries'] == 720
        # A0 has no adapter, so it has no children and its point count IS the
        # query count -- printing "720 x 0 = 0" would look like a broken config.
        assert row['points'] == 720 * row['children_effective']
        if row['children_effective'] == 1:
            assert row['adapter'].startswith('('), (
                f'{name}: only the adapter-less baseline may have one child')


def test_deep_adapters_do_not_advertise_an_inert_k():
    """A5/A4/dense override forward(), so num_neighbors does not apply to them.

    The inherited config value is 8, and the reports used to imply A5 aggregates
    8 neighbours -- the exact opposite of what it does (every point in its
    support box, 215.2 accepted on average).  This pins the fix.
    """
    from benchmarks.p1.experiment_metadata import EXPERIMENTS, describe  # noqa: PLC0415
    for name in ('a5_cap0', 'a4_fixed_gaussian', 'dense_block'):
        row = describe(name, EXPERIMENTS[name])
        assert row['adapter_neighbours'] == 8, 'inherited value should still be 8'
        assert not row['adapter'].endswith('KNNResidualAggregator'), (
            f'{name} is not a KNN adapter, so its k must not be reported')
    knn = describe('a2_knn_mean', EXPERIMENTS['a2_knn_mean'])
    assert knn['adapter'].endswith('KNNResidualAggregator')
    assert knn['adapter_neighbours'] == 8


# ---------------------------------------------------------------------------
# P1-6: the multi-seed paired comparison
# ---------------------------------------------------------------------------

def test_p1_6_multi_seed_numbers_match_the_aggregate():
    """P1-6's three paired seeds; pin every number it quotes.

    Both arms now have three seeds, so ``missing_seeds`` must be empty and there
    must be three paired differences.  The paired differences still do not share
    a sign -- that is the report's verdict -- and the required-seed count is now
    derived from the smaller 3-seed SD and the smaller effect size.
    """
    path = RESULTS / 'p1_seeds.json'
    if not path.exists():
        pytest.skip('p1_seeds.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    text = report('P1-6_multi_seed.md')

    assert payload['groups']['A5 Gaussian']['mIoU'] == [
        pytest.approx(14.317, abs=0.005), pytest.approx(14.365, abs=0.005),
        pytest.approx(14.371, abs=0.005)]
    assert payload['groups']['A2 KNN-mean']['mIoU'] == [
        pytest.approx(14.375, abs=0.005), pytest.approx(14.354, abs=0.005),
        pytest.approx(14.367, abs=0.005)]
    assert payload['groups']['A5 Gaussian']['missing_seeds'] == []
    assert payload['groups']['A2 KNN-mean']['missing_seeds'] == []
    for token in ('14.317', '14.365', '14.371', '14.375', '14.354', '14.367'):
        assert_value_in(text, float(token), 0.001, f'P1-6 seed value {token}')
    assert_value_in(text, payload['groups']['A5 Gaussian']['mIoU_mean'], 0.01,
                    'P1-6 A5 three-seed mean')
    assert_value_in(text, payload['groups']['A2 KNN-mean']['mIoU_mean'], 0.01,
                    'P1-6 A2 three-seed mean')

    # Three paired differences, and they still disagree in sign: that is the
    # verdict, and it is what the pre-registration said would settle it.
    diffs = payload['paired_differences']
    assert len(diffs) == 3, diffs
    assert diffs[0] == pytest.approx(-0.058, abs=0.001)
    assert diffs[1] == pytest.approx(0.011, abs=0.001)
    assert diffs[2] == pytest.approx(0.003, abs=0.001)
    assert min(diffs) < 0 < max(diffs), 'the sign disagreement is the verdict'
    assert payload['sign_p_two_sided'] == pytest.approx(1.0, abs=1e-9)
    # The required-n is a function of BOTH the SD and the effect size, and both
    # shrank; 28 (not the 2-seed 18) is what the current estimate implies.
    assert payload['seeds_needed_normal_approx'] == 28
    for token in ('−0.058', '+0.011', '+0.000', '28'):
        assert token in text, f'P1-6 lost {token!r}'


def test_p1_6_records_the_second_divisor_instance():
    """A2 seed 43's artifact is at 4 s, A5 seed 43's at 6 s: both must be stated."""
    path = RESULTS / 'p1_common_class_miou.json'
    if not path.exists():
        pytest.skip('p1_common_class_miou.json not present')
    rows = json.loads(path.read_text(encoding='utf-8'))['rows']
    assert rows['a5_s43']['finite_classes']['6'] == 16
    assert rows['a2_s43']['finite_classes']['4'] == 16
    assert rows['a5_s43']['comparable_avg_mIoU'] == pytest.approx(14.365, abs=0.005)
    assert rows['a2_s43']['comparable_avg_mIoU'] == pytest.approx(14.354, abs=0.005)
    for name in ('P1-5_metric_divisor_artifact.md', 'P1-6_multi_seed.md'):
        text = report(name)
        assert_value_in(text, 0.205, 0.001, f'{name}: A5 artifact')
        assert_value_in(text, 0.224, 0.001, f'{name}: A2 artifact')
        assert '16' in text, f'{name} must state the 16-class divisor'


def test_p1_6_is_indexed_and_declares_its_completion_status():
    text = report('README.md')
    assert 'P1-6_multi_seed.md' in text, 'P1-6 missing from the index'
    body = report('P1-6_multi_seed.md')
    # The provisional status used to be "seed 44 is still training"; now both arms
    # have three seeds and the pre-registered branch has been settled, so the
    # report must say so rather than still claiming work is in flight.
    assert '正在训练' not in body, (
        'P1-6 still claims a seed is training')
    assert '各 3 个 seed' in body or '各 3 seed' in body, (
        'P1-6 must state that both arms now have three seeds')
    assert '2:1' in body and 'p=1.000' in body.replace('$', '').replace(' ', ''), (
        'P1-6 must record the settled pre-registered branch')


def test_fixed_divisor_values_are_stable_across_comparison_scopes():
    """The comparable class set is computed per comparison; it must at least be
    consistent here, or every report quoting it would need to name its scope."""
    import sys
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.common_class_miou import comparable_for, load
    names = ['a0_base', 'a1_pointwise', 'a2_knn_mean', 'a3_graph_attention',
             'a5_cap0', 'a5_s43', 'a2_s43']
    data = load(names)
    if len(data) < 5:
        pytest.skip('per-class logs not present')
    scopes = {
        'control': ['a0_base', 'a1_pointwise', 'a2_knn_mean',
                    'a3_graph_attention', 'a5_cap0'],
        'multi-seed': names,
    }
    results = {}
    for label, group in scopes.items():
        rows, common = comparable_for(data, group)
        results[label] = (rows, {h: len(v) for h, v in common.items()})
    sizes = {label: value[1] for label, value in results.items()}
    assert sizes['control'] == sizes['multi-seed'], (
        f'the class set now depends on the comparison scope ({sizes}); every '
        f'report quoting a fixed-divisor value must then name its scope')
    for variant in ('a5_cap0', 'a2_knn_mean'):
        a = results['control'][0][variant]['avg']
        b = results['multi-seed'][0][variant]['avg']
        assert a == pytest.approx(b, abs=1e-9), (
            f'{variant}: fixed-divisor value differs by scope ({a} vs {b})')


def test_p1_6_a0_asymmetry_numbers():
    """The A0 comparison has one random arm, not two; pin the reported figures."""
    rows = json.loads((RESULTS / 'p1_common_class_miou.json').read_text(encoding='utf-8'))['rows']
    seeds = json.loads((RESULTS / 'p1_seeds.json').read_text(encoding='utf-8'))
    a0 = rows['a0_base']['comparable_avg_mIoU']
    a5 = seeds['groups']['A5 Gaussian']['mIoU']
    a2 = seeds['groups']['A2 KNN-mean']['mIoU']
    import statistics as stats
    a5_mean, a2_mean = stats.mean(a5), stats.mean(a2)
    sd5 = stats.stdev(a5)
    se = sd5 / len(a5) ** 0.5
    assert a0 == pytest.approx(14.229, abs=0.005)
    assert a5_mean - a0 == pytest.approx(0.122, abs=0.002)
    assert a2_mean - a0 == pytest.approx(0.136, abs=0.002)
    assert (a5_mean - a0) / se == pytest.approx(7.2, abs=0.3)
    # Each row must use its OWN arm's standard error.  The original table gave
    # the A2 row A5's SE ("5.6 SE", then 12.8 with two seeds); with three seeds
    # A2's own SE is ~0.006, i.e. ~22 SE.
    se2 = stats.stdev(a2) / len(a2) ** 0.5
    assert (a2_mean - a0) / se2 == pytest.approx(22, abs=1.0)
    text = report('P1-6_multi_seed.md')
    for token in ('+0.122', '+0.136', '7.2', '22', '4.303'):
        assert token in text, f'P1-6 lost the A0-asymmetry token {token!r}'
    assert '从未重新训练' in text, (
        'P1-6 must state WHY A0 carries no training-seed variance')


def test_pending_report_contract_is_indexed_and_accurate():
    """The queue feeds a written contract; every claimed trigger must be real.

    The contract exists because the objective requires a report per experiment
    and the queue runs unattended: if an artifact lands and its report is never
    written, the gap would be invisible.  This checks that the contract names
    reports that exist (or are explicitly new) and that the trigger paths it
    references are the ones the queue scripts actually write.
    """
    text = report('README.md')
    assert '待出报告清单' in text
    for name in ('P1-7_a4_fixed_geometry.md', 'P1-8_receptive_field_controls.md',
                 'P1-9_fullval_anchor.md', 'P4-10_tile_size_sweep.md'):
        assert name in text, f'contract lost {name}'
    # Each trigger must be the path a queue script really produces, so the
    # contract cannot drift into describing files nothing writes.
    # A trigger may be written by the shell stage or by a Python entry point it
    # invokes (e.g. p1_fullval.json comes from fullval_anchor.py's --json
    # default), so search both.
    producers = list((ROOT / 'tools').glob('run_*.sh')) + \
        list((ROOT / 'benchmarks').rglob('*.py'))
    scripts = '\n'.join(path.read_text(encoding='utf-8') for path in producers)
    for trigger in ('out/a4_fixed_gaussian', 'out/a2_knn_k215',
                    'out/a2_knn_k59', 'p1_fullval.json',
                    'p1_persample', 'real_points_grouped'):
        assert trigger in text, f'contract lost trigger {trigger}'
        assert trigger in scripts, (
            f'contract names {trigger} but no queue script writes it')


def test_average_precision_is_definition_gated_in_the_parser():
    """A log without the pooled flag must not contribute an average.

    The recorded a2_s43 log was produced by the older, nanmean-based evaluator.
    Without gating, ``--reparse`` silently restores that superseded value into the
    results file -- which is exactly what happened before this test existed.
    """
    from benchmarks.p1.eval_matrix import parse_metrics
    old_log = "mmengine - INFO - {'avg/occupiedPrecision': 27.15, " \
              "'avg/occupiedRecall': 43.82, 'avg/mIoU': 14.13}"
    metrics = parse_metrics(old_log)
    assert 'avg/occupiedPrecision' not in metrics
    assert 'avg/occupiedRecall' not in metrics
    assert metrics['avg/mIoU'] == pytest.approx(14.13)

    new_log = "mmengine - INFO - {'avg/occupiedPrecision': 27.15, " \
              "'avg/occupiedRecall': 43.82, " \
              "'avg/occupiedPrecision_pooled': True, 'avg/mIoU': 14.13}"
    metrics = parse_metrics(new_log)
    assert metrics['avg/occupiedPrecision'] == pytest.approx(27.15)
    assert metrics['avg/occupiedPrecision_pooled'] is True


def test_paired_horizon_probe_supersedes_the_unpaired_ratio():
    """Re-derive P3-2 3.2/3.3 from the paired artifact.

    Two claims changed: the horizon ratio is 2.15x, not 5.4x (the unpaired
    numerator was inflated by contention -- in the paired run the 4-horizon point
    is 244.7 ms, not 669.1 ms), and the epoch-8 horizon count is 4, not 6 (the
    model's own training-mode formula).  Both must be argued from an artifact,
    and the models' formula is re-computed here so the label cannot drift again.
    """
    import math
    path = ROOT / 'benchmarks/results/p3_train_mem_horizon_paired.json'
    if not path.exists():
        pytest.skip('p3_train_mem_horizon_paired.json not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['epochs'] == [0, 8] and payload['interleaved'] is True
    rows = {(r['set_epoch'], r['max_neighbours']): r for r in payload['rows']}
    assert set(rows) == {(0, 0), (0, 32), (8, 0), (8, 32)}
    for row in rows.values():
        assert row['steps'] == 40, 'the pre-registered sample size is 40'

    base0, base1 = rows[(0, 0)]['p50_ms'], rows[(8, 0)]['p50_ms']
    assert base0 == pytest.approx(113.874, abs=0.001)
    assert base1 == pytest.approx(244.712, abs=0.001)
    horizon_ratio = base1 / base0
    assert horizon_ratio == pytest.approx(2.149, abs=0.001)
    assert horizon_ratio < 5.4 / 1.5, 'the paired run rules out the old 5.4x'

    def sigma(row):
        return math.log(row['p95_ms'] / row['p50_ms']) / 1.645

    def ci(ratio, *arms):
        se = math.sqrt(sum(sigma(a) ** 2 / a['steps'] for a in arms))
        margin = 2.023 * se  # t_{0.975, df=39}
        return ratio * math.exp(-margin), ratio * math.exp(margin)

    low, high = ci(horizon_ratio, rows[(0, 0)], rows[(8, 0)])
    assert (low, high) == pytest.approx((1.79, 2.58), abs=0.02)
    assert low > 1.0, 'the horizon effect must stay resolved'

    # The cap speedup grows with the horizon count.
    cap_ratio_1 = rows[(0, 0)]['p50_ms'] / rows[(0, 32)]['p50_ms']
    cap_ratio_4 = rows[(8, 0)]['p50_ms'] / rows[(8, 32)]['p50_ms']
    assert cap_ratio_1 == pytest.approx(1.060, abs=0.001)
    assert cap_ratio_4 == pytest.approx(1.426, abs=0.001)
    assert ci(cap_ratio_4, rows[(8, 0)], rows[(8, 32)])[0] == pytest.approx(
        1.19, abs=0.02)
    # The cap also removes the tail: cap=0's p95/p50 is 2.5x at 4 horizons while
    # cap=32's is ~1.0.
    tail = rows[(8, 0)]['p95_ms'] / rows[(8, 0)]['p50_ms']
    assert tail == pytest.approx(2.50, abs=0.01)
    assert rows[(8, 32)]['p95_ms'] / rows[(8, 32)]['p50_ms'] < 1.05
    # Peak memory is horizon-invariant *and* cap-invariant to ~0.06%.
    assert rows[(0, 0)]['peak_alloc_mib'] == rows[(8, 0)]['peak_alloc_mib']
    assert rows[(0, 32)]['peak_alloc_mib'] == rows[(8, 32)]['peak_alloc_mib']
    assert abs(rows[(8, 32)]['peak_alloc_mib']
               - rows[(8, 0)]['peak_alloc_mib']) < 2.0

    # The horizon count at epoch 8 comes from the model, not from a comment.
    assert '4' == str(max(1, min(8 - 5 + 1, 6)))
    text = report('P3-2_training_cost_probe.md')
    for token in ('2.15×', '1.79, 2.58', '1.43×', '1.19, 1.70', '修正 2',
                  '244.712', '不是 6'):
        assert token in text, f'P3-2 lost {token!r}'
    # The superseded value may only survive *flagged as superseded* -- every
    # line that still mentions it must say so, in the report and in the README.
    markers = ('已作废', '此前记', '修正', '未配对', '早期', '旧分子', '→')
    for name in ('P4-4_training_memory_final.md', 'P4-5_end_to_end_final.md',
                 'P4-6_cost_decomposition_failed.md', 'README.md',
                 'P3-2_training_cost_probe.md'):
        for lineno, line in enumerate(report(name).splitlines(), 1):
            if '5.4×' in line or '5.4x' in line:
                assert any(marker in line for marker in markers), (
                    f'{name}:{lineno} quotes 5.4x without marking it superseded')


def test_horizon_probe_has_enough_iterations_to_resolve_the_cap_effect():
    """Pin the sample size of the paired horizon probe to its own power analysis.

    The probe exists to settle two numbers that cross-session measurements made
    unusable: the 5.4x horizon ratio and the ~1.15x density-cap effect.  At
    ``--iters 12`` the recorded per-step spread gives a 95% margin *larger* than
    the cap effect, so the probe would have reproduced the ambiguity instead of
    resolving it.  This computes the requirement from the recorded spread and
    asserts the stage meets it, so lowering the value back would fail here.
    """
    import math
    import re

    spread = train_mem('a5_fullhorizon')
    row = spread['rows'][0]
    p50, p95 = row['p50_ms'], row['p95_ms']
    sigma = math.log(p95 / p50) / 1.645
    cap0 = row['p50_ms']
    cap32 = spread['rows'][1]['p50_ms']
    effect = abs(1 - cap32 / cap0)

    def margin(n):
        return math.exp(1.96 * sigma / math.sqrt(n)) - 1

    assert margin(12) > effect, 'the recorded spread should make 12 insufficient'
    required = next(n for n in range(5, 200) if margin(n) <= effect)
    assert required <= 40, required

    text = (ROOT / 'tools/run_p3_horizon_paired.sh').read_text(encoding='utf-8')
    match = re.search(r'--iters\s+(\d+)', text)
    assert match, 'the stage must declare --iters'
    declared = int(match.group(1))
    assert declared >= required, (
        f'--iters {declared} cannot resolve the {effect:.1%} cap effect '
        f'(needs >= {required} at the recorded spread)')


def test_no_comment_inside_a_continued_shell_command():
    """A ``#`` inside a ``\\``-continued command splits it silently.

    This happened: a rationale comment was inserted between ``--iters`` and
    ``--json`` in the horizon probe stage.  ``bash -n`` still passed -- the result
    is two syntactically valid commands -- but the probe ran without ``--json``
    and the trailing arguments became a "command not found", so the stage would
    have failed after hours of waiting.  This checks every queue script for the
    pattern and, for the probe specifically, that its ``--json`` argument is
    still part of the same command.
    """
    import re

    offenders = []
    for path in sorted((ROOT / 'tools').glob('run_*.sh')):
        lines = path.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(lines):
            if line.rstrip().endswith('\\') and index + 1 < len(lines):
                nxt = lines[index + 1].lstrip()
                if nxt.startswith('#'):
                    offenders.append(f'{path.name}:{index + 1}')
    assert not offenders, f'comment inside a continued command: {offenders}'

    text = (ROOT / 'tools/run_p3_horizon_paired.sh').read_text(encoding='utf-8')
    joined = re.sub(r'\\\n', ' ', text)
    command = next(line for line in joined.splitlines()
                   if 'profile_training.py' in line)
    assert '--json' in command and '--iters 40' in command, command


# --------------------------------------------------------------------------
# cell_size invariance artifact (P4-8 section 2.1.1)
# --------------------------------------------------------------------------

def cell_invariance() -> dict:
    path = RESULTS / 'p4_cell_invariance.json'
    if not path.exists():
        pytest.skip('cell invariance artifact not present')
    return json.loads(path.read_text(encoding='utf-8'))


def test_cell_invariance_artifact_has_no_missing_neighbours():
    """Every cell size must admit exactly the brute-force set.

    The report claims cell_size is a lookup structure rather than a model
    hyper-parameter, so the artifact must show zero dropped and zero spurious
    neighbours for every cell in every row -- both under uniform scales (a weak
    probe: every neighbour sits at the same relative distance) and under
    jittered scales (max_scale 1.26, where the window has to grow to k=7 to keep
    covering the box).
    """
    payload = cell_invariance()
    assert payload['rows'], 'artifact has no rows'
    for row in payload['rows']:
        for cell, info in row['cells'].items():
            assert info['missing_total'] == 0, (row['scale'], row['scale_jitter'],
                                                cell, info['missing_total'])
            assert info['extra_total'] == 0, (row['scale'], row['scale_jitter'],
                                              cell, info['extra_total'])
            assert info['mismatched_queries'] == 0
            assert abs(info['accepted_mean']
                       - row['brute_force_accepted_mean']) < 0.01, (cell, info)
            assert info['window_covers_box'], (cell, info)


def test_window_radius_grows_as_the_cell_shrinks():
    """Pin *why* the invariant holds, as an artifact-level assertion."""
    for row in cell_invariance()['rows']:
        ordered = sorted(((float(c), i) for c, i in row['cells'].items()))
        ks = [i['k'] for _, i in ordered]
        assert ks == sorted(ks, reverse=True), (row['scale'], ks)
        for cell, info in ordered:
            assert info['k'] * cell + 1e-9 >= row['support_sigma'] * row['max_scale']


def test_cell_invariance_cross_validates_the_ess_artifact():
    """Same workpoint as p4_ess.json => the accepted degrees must agree.

    ``benchmarks/p4/effective_sample_size.py`` recorded ``accepted_mean`` at
    G=34560, scale 0.53, support 3.0.  The invariance script re-derives the same
    quantity by a different route (explicit cell-window enumeration plus a
    brute-force re-count), so agreement is a real cross-check of both.
    """
    payload = cell_invariance()
    match = [r for r in payload['rows']
             if r['scale_jitter'] == 0.0 and payload['gaussians'] == 34560]
    assert match, 'artifact lacks the uniform-scale 34560-Gaussian row'
    ess_path = RESULTS / 'p4_ess.json'
    if not ess_path.exists():
        pytest.skip('p4_ess.json not present')
    ess_rows = json.loads(ess_path.read_text(encoding='utf-8'))['rows']
    peers = [r for r in ess_rows
             if r['gaussians'] == payload['gaussians'] and r['support_sigma'] == 3.0
             and abs(r['scale'] - match[0]['scale_requested']) < 1e-9]
    assert peers, 'p4_ess.json lacks the matching workpoint'
    assert abs(match[0]['brute_force_accepted_mean']
               - peers[0]['accepted_mean']) < 1e-3, (match[0], peers[0])


def test_cell_invariance_headline_is_transcribed_and_the_stale_value_is_gone():
    """The report and README must quote the artifact, not the old 246.2."""
    payload = cell_invariance()
    row = [r for r in payload['rows'] if r['scale_jitter'] == 0.0][0]
    value = row['brute_force_accepted_mean']
    for name in ('P4-8_robustness.md', 'README.md'):
        text = report(name)
        assert_value_in(text, value, 0.05, f'{name} accepted degree')
        assert '246.2' not in text, (
            f'{name} still quotes 246.2, which had no recorded workpoint or '
            f'artifact; the artifact value is {value}')


# --------------------------------------------------------------------------
# Preflight: the pending-report contract is checked mechanically
# --------------------------------------------------------------------------

def test_report_contract_check_flags_a_trigger_nobody_writes():
    """Falsification test: the preflight check must not be vacuous.

    ``check_report_contract`` matches on the artifact basename, so a renamed
    output must be reported even though its parent directory (which occurs in
    dozens of scripts) does.  An earlier version fell back to the parent
    directory and therefore accepted any ``benchmarks/results/*`` token --
    passing while checking nothing.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import check_report_contract  # noqa: PLC0415
    corpus = ('python3.10 benchmarks/p4/robustness_probe.py '
              '--json benchmarks/results/p4_robustness.json\n')
    assert check_report_contract(
        ['benchmarks/results/p4_robustness.json'], corpus) == []
    assert check_report_contract(
        ['benchmarks/results/p4_robustness_v2.json'], corpus
    ) == ['benchmarks/results/p4_robustness_v2.json'], (
        'a renamed artifact must be reported; matching on the parent directory '
        'would hide it')
    # Glob and directory-shaped tokens keep their path form.
    assert check_report_contract(
        ['out/p1_persample/*.json'], 'DSH_EVAL_PER_SAMPLE=out/p1_persample\n') == []
    assert check_report_contract(
        ['out/p1_persample/*.json'], 'DSH_EVAL_PER_SAMPLE=out/other\n'
    ) == ['out/p1_persample/*.json']
    assert check_report_contract(['out/a5_s44'], '--work-dir out/a5_s44\n') == []


def test_every_promised_trigger_has_a_writer():
    """All 15 contract triggers, not the six the older test spot-checks."""
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import (check_report_contract,  # noqa: PLC0415
                                    report_contract_triggers)
    triggers = report_contract_triggers()
    assert len(triggers) >= 12, f'contract parsing found only {triggers}'
    unowned = check_report_contract(triggers)
    assert unowned == [], f'contract promises artifacts nothing writes: {unowned}'


def test_cap_selection_is_cell_dependent_so_the_invariant_stops_at_the_cap():
    """The admitted-set invariant must not be extended to capped runs.

    With ``max_neighbours > 0`` the operator keeps the first ``cap`` neighbours
    found by visiting cell shells in increasing Chebyshev radius, so the
    *selected* subset inherits the bin size.  The artifact records the measured
    overlap; this test fails if someone "simplifies" the story back to
    "cell_size never matters" (for example by quoting the admitted-set table
    next to a capped accuracy comparison).
    """
    payload = cell_invariance()
    assert 'cap_selection' in payload, (
        'artifact lacks the cap-selection scan; regenerate with --cap-scan')
    scan = payload['cap_selection']
    assert scan['points'] >= 1000, scan
    worst = 1.0
    for cap, entry in scan['caps'].items():
        assert entry['cells'], cap
        for cell, info in entry['cells'].items():
            assert info['mean_jaccard'] < 0.5, (
                f'cap={cap} cell={cell}: mean Jaccard {info["mean_jaccard"]} -- '
                f'if capped selection really is cell-independent the P4-8 '
                f'caveat needs re-deriving')
            worst = min(worst, info['mean_jaccard'])
    assert worst < 0.2, worst
    text = report('P4-8_robustness.md')
    assert '不能' in text and 'max_neighbours' in text
    assert_value_in(text, worst, 0.02, 'P4-8 worst capped-selection Jaccard')


def test_contract_parser_ignores_producer_scripts():
    """A contract cell may name the script next to the artifact it produces.

    Scripts are producers, never triggers.  An earlier parser special-cased
    ``benchmarks/p1/`` and therefore flagged the new
    ``tools/run_p4_cap_cell_probe.sh`` as an artifact nobody writes -- a false
    positive from the preflight check itself.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import report_contract_triggers  # noqa: PLC0415
    triggers = report_contract_triggers()
    assert triggers, 'contract parser found no triggers'
    assert not any(t.endswith(('.sh', '.py')) for t in triggers), triggers
    assert 'benchmarks/results/p1_fullval.json' in triggers
    assert 'p1_eval.json' in triggers


# --------------------------------------------------------------------------
# Evaluation protocol premises (P1 reduced split vs full val)
# --------------------------------------------------------------------------

def test_reduced_split_is_the_lexicographic_prefix_and_scales_37x():
    """Pin the two premises every P1/P4 comparison rests on.

    1. The reduced split really is the **lexicographically first four scene
       tokens** of the full val set -- the reports call it "the first 4 scenes by
       dictionary order, not a random subset", and any comparison against the
       published anchor depends on that being what was evaluated.
    2. The reduced/full scale factor: 161 vs 6,019 keyframes (37.4x), which is
       what makes the recorded 113 evaluated samples extrapolate to ~4,200 on
       full val -- the number P1-9's two pre-registered branches are written
       against.  The full count also equals the 6,019 occupancy labels the data
       integrity check verifies, so the two independent counts agree.
    """
    import pickle
    full_path = ROOT / 'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'
    red_path = ROOT / 'data/reduced_splits/val_reduced.pkl'
    if not (full_path.exists() and red_path.exists()):
        pytest.skip('nuScenes val infos not present')
    full = pickle.load(full_path.open('rb'))['infos']
    reduced = pickle.load(red_path.open('rb'))['infos']

    full_scenes = sorted(full)
    assert len(full_scenes) == 150, len(full_scenes)
    assert sorted(reduced) == full_scenes[:4], (
        'the reduced split is not the lexicographic prefix of the full val '
        'scene list; every "first 4 scenes, non-random" claim would be wrong')

    def keyframes(infos):
        return sum(1 for frames in infos.values() for frame in frames
                   if frame.get('is_key_frame'))

    kf_full, kf_reduced = keyframes(full), keyframes(reduced)
    assert (kf_full, kf_reduced) == (6019, 161), (kf_full, kf_reduced)
    ratio = kf_full / kf_reduced
    assert 37.0 <= ratio < 38.0, ratio

    recorded = [r for r in json.loads(
        (RESULTS / 'p1_eval.json').read_text(encoding='utf-8'))
        if (r.get('metrics') or {}).get('num_samples')]
    assert recorded, 'no recorded sample count'
    samples = {r['metrics']['num_samples'] for r in recorded}
    assert samples == {113}, samples
    implied = 113 * ratio
    assert 4200 <= implied <= 4300, implied

    text = report('README.md')
    assert '37×' in text or '37x' in text, 'README lost the 37x scale factor'
    assert '4200' in text, 'README lost the ~4200 full-val sample estimate'


def test_cited_artifact_check_is_sensitive_and_the_repo_is_clean():
    """A typo inside an inline code span is not a markdown link.

    The KaTeX/link checker cannot see it, so a report could cite
    ``benchmarks/results/p4_tile_sweep.json`` when the file is actually named
    something else, and no test would notice.  ``check_cited_artifacts`` closes
    that: a cited ``benchmarks/results/*`` path must exist, or be a trigger the
    pending-report contract promises is still coming.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import (check_cited_artifacts,  # noqa: PLC0415
                                    cited_artifact_paths,
                                    report_contract_triggers)

    # A synthetic pending token, not a real artifact: this block used to name
    # p1_fullval.json, which stopped being absent the moment the full-val stage
    # ran -- an assertion about "currently missing" rots by design.
    promised = ['benchmarks/results/__pending_placeholder.json']
    # Falsification: an absent, unpromised citation must be reported.
    missing, unpromised = check_cited_artifacts(
        ['benchmarks/results/p4_tile_sweep_v2.json'], promised)
    assert missing == ['benchmarks/results/p4_tile_sweep_v2.json']
    assert unpromised == ['benchmarks/results/p4_tile_sweep_v2.json']
    # A promised-but-absent artifact is expected, not a defect.
    missing, unpromised = check_cited_artifacts(
        ['benchmarks/results/__pending_placeholder.json'], promised)
    assert missing == ['benchmarks/results/__pending_placeholder.json']
    assert unpromised == []
    # An existing artifact is never flagged.
    missing, unpromised = check_cited_artifacts(
        ['benchmarks/results/p1_eval.json'], promised)
    assert (missing, unpromised) == ([], [])

    # And the repository state: every absent citation must be a queued trigger.
    # The missing set is not hardcoded -- new experiments add promised artifacts
    # (the tile holdout did), and freezing the list here would only force edits.
    # What must hold is the invariant: absent <=> promised.  An earlier version
    # also asserted that three specific artifacts were *currently* missing; that
    # rotted the moment the queue produced one of them (p4_robustness.json), which
    # is not a defect in the reports but in the assertion, so it now asserts the
    # invariant per token instead: a cited-but-absent artifact must be promised.
    cited = cited_artifact_paths()
    assert len(cited) >= 20, cited
    triggers = report_contract_triggers()
    missing, unpromised = check_cited_artifacts(cited, triggers)
    assert unpromised == [], f'unexplained absent citations: {unpromised}'
    promised_names = {token.rsplit('/', 1)[-1] for token in triggers}
    assert all(path.rsplit('/', 1)[-1] in promised_names for path in missing), missing
    for token in ('p1_fullval.json', 'p4_robustness.json',
                  'real_points_grouped.pt'):
        absent = [path for path in missing if path.endswith(token)]
        delivered = any(path.endswith(token) for path in cited) and not absent
        assert absent or delivered, (
            f'{token} is neither cited-and-absent nor cited-and-delivered')


def test_section_references_resolve_to_real_headings():
    """``[report](file.md) §N`` must point at a heading the target actually has.

    The link checker validates the file, not the section.  These reports are read
    as an argument, so "see P4-7 §4.3" when P4-7 only has §4 is a citation the
    reader cannot follow -- a real defect found when this check was written, and
    the reason it exists.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import check_section_references  # noqa: PLC0415

    total, dangling = check_section_references()
    assert total >= 40, total
    assert dangling == [], f'dangling section references: {dangling}'

    # Falsification: a section the target does not have must be reported.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'src.md'
        target = Path(directory) / 'target.md'
        source.write_text('see [x](target.md) §9.9\n', encoding='utf-8')
        target.write_text('# t\n\n## 1 first\n\n## 9 ninth\n', encoding='utf-8')
        total, dangling = check_section_references(Path(directory))
        assert total == 1
        assert dangling == [('src.md', 'target.md', '9.9', 'no such section')]
        # ... but a heading that does exist resolves.
        source.write_text('see [x](target.md) §9\n', encoding='utf-8')
        total, dangling = check_section_references(Path(directory))
        assert (total, dangling) == (1, [])


def test_cited_script_paths_exist():
    """Scripts cited in the reports must exist, bar documented exceptions.

    ``check_cited_artifacts`` only looks at ``benchmarks/results/*`` data files.
    The reports also tell the reader which *script* to run
    (``tools/run_p1_a4.sh``), and that is the path most likely to be renamed
    while the prose stays put -- the reader then has a reproduction step that
    cannot be followed.  The first run of this check found two such citations
    (``misc/occ3d_ray_metrics.py`` and ``benchmarks/baselines/block_g2v.py``),
    both of them intentional: the README quotes the plan's expected path and
    marks it absent/relocated on the same line.
    """
    import tempfile
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import check_script_references  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / 'r.md'
        report.write_text(
            'run `tools/run_p1_a4.sh` then `benchmarks/p1/ghost_script.py`\n',
            encoding='utf-8')
        total, missing = check_script_references(Path(directory))
        assert total == 2
        assert missing == [('r.md:1', 'benchmarks/p1/ghost_script.py')]

        # A line that *documents* the path as absent or relocated is a record of
        # a discrepancy, not a reproduction step, and must not be flagged.
        report.write_text(
            '| expected | `misc/gone_metrics.py` | **`misc/gone_metrics.py` 不存在** |\n'
            '| planned | `benchmarks/baselines/block_g2v.py` | delivered elsewhere, '
            '**已交付，路径不同** |\n',
            encoding='utf-8')
        total, missing = check_script_references(Path(directory))
        assert (total, missing) == (3, [])

        # A bare "or" is *not* a suppression marker: it must not be able to hide
        # a renamed script sitting next to a valid one.
        report.write_text('run `tools/run_p1_a4.sh` or `tools/ghost.sh`\n',
                          encoding='utf-8')
        total, missing = check_script_references(Path(directory))
        assert missing == [('r.md:1', 'tools/ghost.sh')]

    # And the repository state: every cited script path must resolve.
    total, missing = check_script_references()
    assert total >= 200, total
    assert missing == [], f'dangling script citations: {missing}'


def test_the_two_p1_designs_are_not_overstated_as_factorials():
    """A 2x2 needs four cells; both P1 "2x2"s have three.

    The README's queue row described A2 / A2-gauss / A5-cap8 as "constituting the
    selection x weighting 2x2", while the same file's summary section and P1-6
    correctly said "three cells of a 2x2".  A reader citing the row would take the
    design for a full factorial and read the missing interaction as zero.  The A4
    comparison has the same shape (A2 -> A4 -> A5 is a chain), and the loose
    "missing cell of the 2x2" wording lived in the config and stage comments too.
    """
    import re
    files = [ROOT / 'analysis/sparse_render/results/README.md',
             ROOT / 'analysis/sparse_render/results/P1-6_multi_seed.md',
             ROOT / 'analysis/sparse_render/results/P1-4_full_control_matrix.md',
             ROOT / 'config/opus/ablations/p1_a4_fixed_gaussian.py',
             ROOT / 'benchmarks/p1/eval_matrix.py']
    text = '\n'.join(path.read_text(encoding='utf-8') for path in files)
    # The overstating phrasings must be gone.
    assert not re.search(r'构成[^。\n]*2×2', text), 'still claims a complete 2x2'
    assert 'missing cell of the P1 2x2' not in text
    assert 'one empty cell of the P1 2x2' not in text
    # ... and the qualified statements must be present where the design is named.
    readme = report('README.md')
    summary = [line for line in readme.splitlines()
               if '完整 2×2' in line and '第四格' in line]
    assert summary, 'the summary must name the missing cell'
    assert '不是' in summary[0]
    row = [line for line in readme.splitlines()
           if line.startswith('| P1 权重对照 A2-gauss')]
    assert row, 'the A2-gauss queue row'
    assert '三格' in row[0] and '第四格' in row[0], row[0]
    p1_6 = report('P1-6_multi_seed.md')
    assert '2×2（选取 × 权重）中的三个格子' in p1_6


def test_every_written_artifact_has_a_promised_report():
    """The objective requires a report per experiment; check coverage.

    ``check_report_contract`` proves the forward direction (each promised
    artifact has a writer).  Nothing proved the reverse, so a script could write
    a result file that the artifact plan never mentions -- an experiment with no
    write-up.  The first run found two classes of gap: a queue-produced JSON
    nobody had promised (``p4_real_workload_grouped.json``) and nine analysis
    artifacts, including a *second* output of the tile-holdout script whose other
    output was already promised.  All are now cited by a report or a contract row.
    """
    import tempfile
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import (INTERMEDIATE_ARTIFACTS,  # noqa: PLC0415
                                    analysis_artifacts,
                                    check_artifact_coverage,
                                    queue_stage_artifacts)

    queue = queue_stage_artifacts()
    analysis = analysis_artifacts()
    assert len(queue) >= 13, queue
    assert len(analysis) >= 25, len(analysis)
    # Every format a stage writes must be visible to the check.  A derived summary
    # table (.md) and a raw stage log (.log) both escaped the first pattern, which
    # is how four tables and one log stayed unpromised.
    assert 'benchmarks/results/p1_table_a4.md' in queue
    assert 'benchmarks/results/p1_table_a2k215.md' in queue
    assert 'benchmarks/results/p4_tile_sweep.log' in queue
    assert check_artifact_coverage() == {}
    # The exemption list must stay justified and must not swallow real results.
    assert set(INTERMEDIATE_ARTIFACTS) <= set(queue) | set(analysis)
    for token, reason in INTERMEDIATE_ARTIFACTS.items():
        assert 'consumed' in reason or 'input' in reason, (token, reason)
    # A script that writes two artifacts must have both covered: this is the
    # shape of the gap the check found, so pin the instance.
    assert 'benchmarks/results/p4_tile_holdout.json' in analysis
    assert 'benchmarks/results/p4_tile_sweep_holdout.json' in queue

    # Falsification: an artifact mentioned nowhere must be reported.
    with tempfile.TemporaryDirectory() as directory:
        readme = Path(directory) / 'README.md'
        readme.write_text('nothing relevant here\n', encoding='utf-8')
        uncovered = check_artifact_coverage(readme, reports_dir=directory)
        assert 'benchmarks/results/p4_tile_sweep_holdout.json' in uncovered
        assert 'benchmarks/results/p1_receptive_field_controls.json' in uncovered
        # ... and a basename mention anywhere in the corpus covers it.
        readme.write_text('see p4_tile_sweep_holdout.json for details\n',
                          encoding='utf-8')
        uncovered = check_artifact_coverage(readme, reports_dir=directory)
        assert 'benchmarks/results/p4_tile_sweep_holdout.json' not in uncovered


def test_guard_audit_ignores_processes_that_vanished():
    """The audit must not classify a reaped pid as hazardous.

    ``pgrep`` and the process snapshot are taken at different instants, so a
    training stage cycling its dataloader workers could make a wrapper look like
    it has no pinned child.  That produced a transient ``AUDIT FAILED`` on a live
    tree (observed 2026-09-14), and a false FAIL in an unattended pipeline is
    worse than a missing line because it teaches the operator to ignore the
    check.  ``vanished`` skips such pids instead of classifying them.
    """
    import os
    sys.path.insert(0, str(ROOT / 'tools'))
    from queue_guard_audit import snapshot_processes, vanished  # noqa: PLC0415

    assert not vanished(str(os.getpid())), 'a live pid must not look vanished'
    # A pid far above any plausible allocation cannot be running.
    assert vanished('999999999')
    snapshot = snapshot_processes()
    assert str(os.getpid()) in snapshot
    assert len(snapshot) > 1


def test_p1_5_third_divisor_instance_is_transcribed_from_the_logs():
    """Re-derive P1-5 4.5.1's table instead of trusting the transcription.

    The third artifact instance (A5 seed 44) carries the largest numbers in the
    whole report set -- a +0.43 average artifact built from +0.89 and +0.83 at
    two horizons -- and it is the instance that changes the headline from
    "A5 is 0.11 worse" to "A5 and A2 are 0.01 apart".  Every value in that table
    is recomputed here from the eval logs, in the same comparison scope the
    report uses (the fixed class set depends on which variants are compared,
    see P1-5 4.6), and then required to appear on the page.
    """
    names = ['a0_base', 'a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43']
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415
    logs = load(names)
    if not logs:
        pytest.skip('per-variant eval logs not present')
    table, common = comparable_for(logs, names)
    assert {horizon: len(classes) for horizon, classes in common.items()} == \
        {0: 17, 2: 15, 4: 15, 6: 15}, common

    recorded = {}
    recorded_horizon = {}
    for row in json.loads((RESULTS / 'p1_eval.json').read_text(encoding='utf-8')):
        if row['variant'] not in names:
            continue
        metrics = row.get('metrics') or {}
        recorded[row['variant']] = metrics.get('avg/mIoU')
        recorded_horizon[row['variant']] = [metrics.get(f'{h}s/mIoU')
                                            for h in (0, 2, 4, 6)]
    for name in names:
        assert recorded.get(name) is not None, name
        assert all(v is not None for v in recorded_horizon[name]), name

    fixed = {name: table[name]['avg'] for name in names}
    per_horizon = {name: table[name]['per_horizon'] for name in names}

    # The numbers the report quotes for the third instance.
    assert fixed['a5_s44'] == pytest.approx(14.3706, abs=5e-4)
    assert recorded['a5_s44'] == pytest.approx(13.94, abs=0.005)
    assert fixed['a5_s44'] - recorded['a5_s44'] == pytest.approx(0.43, abs=0.01)
    assert (per_horizon['a5_s44'][2] - recorded_horizon['a5_s44'][2]
            == pytest.approx(0.89, abs=0.01))
    assert (per_horizon['a5_s44'][3] - recorded_horizon['a5_s44'][3]
            == pytest.approx(0.83, abs=0.01))
    # ... and for the two earlier instances the table repeats.
    assert fixed['a5_s43'] - recorded['a5_s43'] == pytest.approx(0.205, abs=0.01)
    assert (per_horizon['a5_s43'][3] - recorded_horizon['a5_s43'][3]
            == pytest.approx(0.84, abs=0.01))
    assert fixed['a2_s43'] - recorded['a2_s43'] == pytest.approx(0.224, abs=0.01)
    assert (per_horizon['a2_s43'][2] - recorded_horizon['a2_s43'][2]
            == pytest.approx(0.89, abs=0.01))

    text = report('P1-5_metric_divisor_artifact.md')
    for name in ('a5_s44', 'a5_s43', 'a2_s43'):
        assert_value_in(text, fixed[name], 0.01, f'P1-5 fixed {name}')
        assert_value_in(text, recorded[name], 0.01, f'P1-5 recorded {name}')
    assert_value_in(text, fixed['a5_s44'] - recorded['a5_s44'], 0.01,
                    'P1-5 seed-44 average artifact')
    assert_value_in(text, per_horizon['a5_s44'][2], 0.01, 'P1-5 4s fixed (44)')
    assert_value_in(text, per_horizon['a5_s44'][3], 0.01, 'P1-5 6s fixed (44)')

    # The three-seed aggregation the report now states.
    seeds = ['a5_cap0', 'a5_s43', 'a5_s44']
    values = [fixed[name] for name in seeds]
    assert max(values) - min(values) == pytest.approx(0.053, abs=0.005)
    a2_values = [fixed['a2_knn_mean'], fixed['a2_s43']]
    assert max(a2_values) - min(a2_values) == pytest.approx(0.021, abs=0.005)
    gap = sum(values) / 3 - sum(a2_values) / 2
    assert gap == pytest.approx(-0.01, abs=0.01)
    assert_value_in(text, max(values) - min(values), 0.005, 'P1-5 A5 spread')
    assert_value_in(text, abs(gap), 0.01, 'P1-5 A5-A2 gap')


def test_boundary_signal_seed_dimension_is_transcribed():
    """P4-7 3.5: the boundary lead is sign-stable but sits inside seed noise.

    The boundary gain over the overall gain is the only monotone signal in the P1
    matrix, and until seed 43/44 landed it rested on one seed.  Two independent
    facts matter and must both be on the page:

    * the sign never flips -- all three A5 seeds beat both A2 seeds;
    * the magnitude (+0.045) is *smaller than A5's own seed spread* (0.070), so
      it stays a diagnostic rather than a usable result.  Contrast the
      small-object deficit in P1-6, which is also sign-stable but ~3x its seed SD.
    """
    import statistics as stats
    rows = {row['variant']: (row.get('metrics') or {})
            for row in json.loads((RESULTS / 'p1_eval.json').read_text(encoding='utf-8'))}
    required = ('a0_base', 'a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43')
    if not all(name in rows for name in required):
        pytest.skip('seed variants not all present')
    seeds = ('a5_cap0', 'a5_s43', 'a5_s44')
    a2_seeds = ('a2_knn_mean', 'a2_s43', 'a2_s44')
    a5 = [rows[name]['avg/boundaryIoU'] for name in seeds]
    a2 = [rows[name]['avg/boundaryIoU'] for name in a2_seeds]
    assert a5 == pytest.approx([14.63, 14.70, 14.65], abs=0.005)
    assert a2 == pytest.approx([14.61, 14.62, 14.59], abs=0.005)
    assert min(a5) > max(a2), 'the sign flip would invalidate the P4-7 claim'
    a5_mean, a2_mean = stats.mean(a5), stats.mean(a2)
    spread = max(a5) - min(a5)
    assert a5_mean == pytest.approx(14.660, abs=0.005)
    assert a2_mean == pytest.approx(14.607, abs=0.005)
    assert spread == pytest.approx(0.070, abs=0.005)
    assert (max(a2) - min(a2)) == pytest.approx(0.030, abs=0.005)
    gap = a5_mean - a2_mean
    assert gap == pytest.approx(0.053, abs=0.005)
    assert gap < spread, (
        'the whole point of P4-7 3.5 is that the boundary lead is smaller than '
        'the seed spread; if that changed, the section must be rewritten')
    # The A0-relative form of the same signal: boundary gain > overall gain.
    # The mIoU side MUST use fixed divisors.  Averaging the *recorded* values over
    # the three seeds gives 14.14 vs A0's 14.23 -- i.e. -0.09, a sign flip of the
    # project's main positive claim, caused entirely by two of the three seeds
    # having an extra finite class (P1-5 4.5.1).
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415
    names = list(required)
    table, _ = comparable_for(load(names), names)
    fixed = {name: table[name]['avg'] for name in names if name in table}
    assert len(fixed) == len(required), sorted(set(required) - set(fixed))
    a0_boundary = rows['a0_base']['avg/boundaryIoU']
    a0_miou = fixed['a0_base']
    a5_miou = stats.mean([fixed[name] for name in ('a5_cap0', 'a5_s43', 'a5_s44')])
    assert a5_mean - a0_boundary == pytest.approx(0.34, abs=0.01)
    assert a5_miou - a0_miou == pytest.approx(0.12, abs=0.01)
    assert (a5_mean - a0_boundary) > (a5_miou - a0_miou)
    recorded_miou = stats.mean([rows[name]['avg/mIoU']
                                for name in ('a5_cap0', 'a5_s43', 'a5_s44')])
    assert recorded_miou - rows['a0_base']['avg/mIoU'] == pytest.approx(-0.09, abs=0.02), (
        'the recorded-value sign flip is the reason this test uses fixed values')

    # Per-horizon: the signal is horizon-LOCALISED.  A5 trails at 0/2 s and leads
    # at 4/6 s with a seed-stable sign at every horizon, so the average (+0.045)
    # is small only because the two cancel.
    per_horizon = {}
    for horizon in (0, 2, 4, 6):
        key = f'{horizon}s/boundaryIoU'
        per_horizon[horizon] = (
            [rows[name][key] for name in seeds],
            [rows[name][key] for name in a2_seeds])
    assert per_horizon[0][0] == pytest.approx([16.74, 16.75, 16.72], abs=0.005)
    assert per_horizon[0][1] == pytest.approx([16.78, 16.78, 16.77], abs=0.005)
    assert per_horizon[4][0] == pytest.approx([13.97, 14.02, 13.95], abs=0.005)
    assert per_horizon[4][1] == pytest.approx([13.80, 13.84, 13.79], abs=0.005)
    assert per_horizon[6][0] == pytest.approx([13.56, 13.69, 13.56], abs=0.005)
    assert per_horizon[6][1] == pytest.approx([13.48, 13.48, 13.42], abs=0.005)
    gaps = {}
    for horizon, (left, right) in per_horizon.items():
        gaps[horizon] = stats.mean(left) - stats.mean(right)
        # Sign stability within every horizon.
        if horizon in (0, 2):
            assert max(left) < min(right), horizon
        else:
            assert min(left) > max(right), horizon
    assert gaps[0] == pytest.approx(-0.040, abs=0.01)
    assert gaps[2] == pytest.approx(-0.070, abs=0.01)
    assert gaps[4] == pytest.approx(0.170, abs=0.01)
    assert gaps[6] == pytest.approx(0.143, abs=0.01)
    # At 4 s the lead exceeds A5's own seed spread; the horizon average does not.
    a5_spread_4s = max(per_horizon[4][0]) - min(per_horizon[4][0])
    assert gaps[4] > a5_spread_4s
    assert gaps[4] + gaps[6] + gaps[0] + gaps[2] == pytest.approx(4 * gap, abs=0.02)

    text = report('P4-7_quality_diagnostics.md')
    for token in ('14.660', '14.607', '0.070', '+0.053', '+0.34', '3.5',
                  '+0.170', '+0.143', '−0.040', '−0.070', 'horizon 局部化'):
        assert token in text, f'P4-7 lost the boundary seed token {token!r}'


def test_per_horizon_seed_spread_inflation_is_transcribed():
    """P1-5 4.5.1 item 5: the artifact inflates the per-horizon seed spread 8-23x.

    Even a reader who has accepted "use fixed divisors" can be misled by a
    per-horizon table, because only the horizons with a class-presence flip are
    affected: 0 s and 2 s agree between the two conventions to 0.05, while 4 s
    and 6 s differ by up to 0.90.  The numbers here are re-derived, and the
    report must carry both the inflated and the true spread.
    """
    names = ['a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43']
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415
    logs = load(names)
    if not logs:
        pytest.skip('per-variant eval logs not present')
    table, _ = comparable_for(logs, names)
    recorded = {row['variant']: row.get('metrics') or {}
                for row in json.loads((RESULTS / 'p1_eval.json').read_text(encoding='utf-8'))}

    def spreads(variants):
        out = {}
        for idx, horizon in enumerate((0, 2, 4, 6)):
            rec = [recorded[name][f'{horizon}s/mIoU'] for name in variants]
            cmp_ = [table[name]['per_horizon'][idx] for name in variants]
            out[horizon] = (max(rec) - min(rec), max(cmp_) - min(cmp_))
        return out

    a5 = spreads(['a5_cap0', 'a5_s43', 'a5_s44'])
    a2 = spreads(['a2_knn_mean', 'a2_s43'])
    # NB: the dict is keyed by the horizon in seconds, not by position.
    assert a5[0] == pytest.approx((0.02, 0.01), abs=0.011)
    assert a5[2] == pytest.approx((0.05, 0.05), abs=0.011)
    assert a5[4] == pytest.approx((0.94, 0.12), abs=0.011)
    assert a5[6] == pytest.approx((0.75, 0.09), abs=0.011)
    assert a2[4] == pytest.approx((0.94, 0.04), abs=0.011)
    # The unaffected horizons agree between conventions; the hit ones do not.
    for spreads_ in (a5, a2):
        for horizon in (0, 2):
            assert spreads_[horizon][0] - spreads_[horizon][1] < 0.05, horizon
        assert spreads_[4][0] / max(spreads_[4][1], 0.01) > 5
    assert a5[6][0] / max(a5[6][1], 0.01) > 5

    text = report('P1-5_metric_divisor_artifact.md')
    for token in ('0.94', '0.75', '0.12', '23×', '8×'):
        assert token in text, f'P1-5 lost the inflation token {token!r}'


def test_avg_miou_sign_separation_is_only_at_horizon_zero():
    """Only the 0 s lead is separated on both sides by the seeds.

    P1-6 says "A5 leads only at 0 s".  With three A5 seeds and two A2 seeds that
    become checkable per horizon: at 0 s every A5 seed beats every A2 seed, while
    at 2/4/6 s the two arms overlap, because the mean gaps there (0.013-0.055)
    are smaller than A5's own seed spread (up to 0.125).  The distinction matters
    because it is what makes the small-object deficit (-0.14, pooled) the only
    place where A5 is *stably* behind.
    """
    import statistics as stats
    names = ['a5_cap0', 'a5_s43', 'a5_s44', 'a2_knn_mean', 'a2_s43']
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.common_class_miou import comparable_for, load  # noqa: PLC0415
    logs = load(names)
    if not logs:
        pytest.skip('per-variant eval logs not present')
    table, _ = comparable_for(logs, names)

    separated = {}
    gaps = {}
    for index, horizon in enumerate((0, 2, 4, 6)):
        a5 = [table[name]['per_horizon'][index]
              for name in ('a5_cap0', 'a5_s43', 'a5_s44')]
        a2 = [table[name]['per_horizon'][index]
              for name in ('a2_knn_mean', 'a2_s43')]
        separated[horizon] = min(a5) > max(a2) or max(a5) < min(a2)
        gaps[horizon] = stats.mean(a5) - stats.mean(a2)
    assert separated == {0: True, 2: False, 4: False, 6: False}, separated
    assert gaps[0] == pytest.approx(0.038, abs=0.005)
    assert gaps[4] == pytest.approx(-0.055, abs=0.005)
    # Each unseparated horizon's gap is smaller than A5's own spread there.
    for index, horizon in ((1, 2), (2, 4), (3, 6)):
        a5 = [table[name]['per_horizon'][index]
              for name in ('a5_cap0', 'a5_s43', 'a5_s44')]
        assert abs(gaps[horizon]) < (max(a5) - min(a5)), horizon

    text = report('P1-6_multi_seed.md')
    for token in ('14.855', '14.868', '14.816', '两侧可分'):
        assert token in text, f'P1-6 lost the separation token {token!r}'


def test_summarizer_precision_column_is_definition_gated():
    """The occupied P/R column must show '-' rather than a stale number.

    Only runs that carry the pooled-count flag have a valid average; the others
    (evaluated before the pooled definition landed) must print '-'.  A column
    that silently displayed the old ``nanmean``-based value is exactly the
    failure P1-5 documented.

    The gate is keyed on the *flag in the log*, not on a variant name: the first
    version of this test asserted that only "seed44" rows could carry a value,
    which broke as soon as a newly trained control (A2-max) was evaluated with
    the pooled definition.  Which variants have the flag is read from the eval
    record here, so the gate is checked rather than the roster.
    """
    import subprocess
    from benchmarks.p1.summarize_p1 import VARIANT_LABELS
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p1/summarize_p1.py',
         'benchmarks/results/p1_eval.json'],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-400:]
    # Only the FIRST table: the per-horizon table below it is full of "x / y"
    # cells that would otherwise be read as occupancy pairs.
    main_table = completed.stdout.split('### 每时域明细')[0]
    # NB: no '/' requirement -- a row whose pooled flag is absent has no slash
    # anywhere, and filtering on '/' would hide exactly the rows under test.
    rows = [line for line in main_table.splitlines()
            if line.startswith('| ') and '---' not in line
            and 'avg/' not in line]
    assert rows, 'summarizer produced no table rows'

    record = json.loads((RESULTS / 'p1_eval.json').read_text(encoding='utf-8'))
    pooled = {VARIANT_LABELS.get(row['variant'], row['variant'])
              for row in record
              if row['metrics'].get('avg/occupiedPrecision') is not None}
    unpooled = {VARIANT_LABELS.get(row['variant'], row['variant'])
                for row in record
                if row['metrics'].get('avg/occupiedPrecision') is None}
    assert pooled and unpooled, (pooled, unpooled)

    def label_of(line):
        return line.split(' | ')[0].removeprefix('| ').replace(' ⚠️', '')

    valued = {label_of(line) for line in rows if ' | - | ' not in line}
    dashed = {label_of(line) for line in rows if ' | - | ' in line}
    assert valued == pooled, f'valued rows {valued} != flagged variants {pooled}'
    assert dashed >= unpooled, f'rows without the flag must show "-": {unpooled - dashed}'
    assert '27.10 / 43.86' in completed.stdout, 'seed-44 pooled pair missing'
    assert '定义门控' in completed.stdout


def test_contract_parser_skips_unverifiable_expansion_tokens():
    """Brace/shell-expanded names cannot be matched against a script corpus.

    A row may promise ``p1_bootstrap_ci_s{42,43,44}.json``, but the stage builds
    that name from a loop variable, so no literal exists to find.  Reporting such
    a token as "nobody writes it" is a false positive; skipping it is honest as
    long as the row keeps a verifiable sibling (the combined artifact or the
    directory), which is why the parser excludes these and the row still carries
    ``benchmarks/results/p1_bootstrap_ci.json``.
    """
    sys.path.insert(0, str(ROOT / 'tools'))
    from pipeline_preflight import (check_report_contract,  # noqa: PLC0415
                                    report_contract_triggers)
    triggers = report_contract_triggers()
    assert triggers, 'contract parser found no triggers'
    assert not any(marker in token for token in triggers
                   for marker in ('{', '}', '$')), triggers
    assert 'benchmarks/results/p1_bootstrap_ci.json' in triggers
    # A brace token handed in directly is skipped rather than flagged.
    assert check_report_contract(['benchmarks/results/thing_s{1,2}.json'],
                                 'irrelevant corpus') == []
    # ... while a plain missing one is still reported.
    assert check_report_contract(['benchmarks/results/thing_s42.json'],
                                 'irrelevant corpus') == [
        'benchmarks/results/thing_s42.json']


def test_horizon_probe_power_analysis_is_recomputable_from_the_artifact():
    """Why the paired horizon probe uses ``--iters 40``, recomputed end to end.

    Both inputs of the power calculation live in one artifact, so the claim is
    self-contained:

    * the per-step spread ``sigma = ln(p95/p50) / 1.645`` for the 6-horizon
      configuration (cap=0 row: 669.12 / 958.93 ms -> sigma = 0.219);
    * the effect the probe exists to resolve: cap=32 vs cap=0 on the same
      artifact, 584.19 / 669.12 -> a 12.7% reduction.

    The 95% margin on a mean is ``c * sigma / sqrt(n)``.  With ``c`` = the normal
    1.96 the n=12 margin is 12.4%, i.e. *just below* the 12.7% effect -- the
    decision to use 40 iterations therefore rests on the t convention
    (t(0.975, 11) = 2.201 -> 13.9%), where n=12 genuinely cannot resolve it.  The
    script's comment quotes 13.2%, which matches neither convention; the numbers
    below are the reproducible ones.
    """
    path = RESULTS / 'p3_train_mem_a5_fullhorizon.json'
    if not path.exists():
        pytest.skip('horizon training artifact not present')
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = {int(row['max_neighbours']): row for row in payload['rows']}
    if 0 not in rows or 32 not in rows:
        pytest.skip('artifact lacks the cap=0 / cap=32 pair')

    uncapped, capped = rows[0], rows[32]
    assert uncapped['steps'] == capped['steps'] == 6
    sigma = math.log(uncapped['p95_ms'] / uncapped['p50_ms']) / 1.645
    assert uncapped['p95_ms'] / uncapped['p50_ms'] == pytest.approx(1.433, abs=0.002)
    assert sigma == pytest.approx(0.219, abs=0.002)

    effect = 1.0 - capped['p50_ms'] / uncapped['p50_ms']
    assert effect == pytest.approx(0.127, abs=0.002)

    def margin(n, critical):
        return critical * sigma / math.sqrt(n)

    # The normal convention at n=12 is marginal (margin just under the effect).
    assert margin(12, 1.96) == pytest.approx(0.124, abs=0.003)
    assert margin(12, 1.96) < effect, 'the z-margin at n=12 is marginally too small'
    # The t convention is the one that decides it: n=12 cannot resolve, n=40 can.
    assert margin(12, 2.201) == pytest.approx(0.139, abs=0.003)
    assert margin(12, 2.201) > effect
    assert margin(40, 2.023) == pytest.approx(0.070, abs=0.003)
    assert margin(40, 2.023) < effect / 1.5, 'n=40 must have real headroom over the effect'


def test_fullval_anchor_verdict_distinguishes_the_two_preregistered_branches():
    """The three-way branch logic behind P1-9's verdict.

    Branch 1: full val reproduces the published anchors (13.20 / 22.03) -> the
    earlier gap was the reduced subset.  Branch 2: full val stays at the
    *reduced-split* level (14.23 / 20.17) -> the protocol differs, not the sample
    set; note the observed shape is mIoU HIGHER and IoU LOWER, i.e. both
    directions disagree.  Anything else is 'neither' and must not be reported as
    either branch.
    """
    sys.path.insert(0, str(ROOT))
    from benchmarks.p1.fullval_anchor import anchor_verdict  # noqa: PLC0415

    reduced = {'avg/mIoU': 14.23, 'avg/IoU': 20.17}
    assert anchor_verdict({'avg/mIoU': 13.18, 'avg/IoU': 22.00}, reduced) == 'reproduces'
    assert anchor_verdict({'avg/mIoU': 14.25, 'avg/IoU': 20.15}, reduced) == 'protocol'
    # Away from both: neither branch.
    assert anchor_verdict({'avg/mIoU': 13.80, 'avg/IoU': 21.00}, reduced) == 'neither'
    # Missing metrics never claims a branch.
    assert anchor_verdict({}, reduced) == 'neither'
    assert anchor_verdict({'avg/mIoU': 14.2, 'avg/IoU': 20.2}, {}) == 'neither'

    # The contract must describe branch 2 the way the data actually looks, not as
    # "mIoU matches the anchor while IoU is low".
    text = report('README.md')
    assert 'P1-9_fullval_anchor.md' in text
    # Check the CONTRACT ROW itself: an earlier version described branch 2 as
    # "mIoU matches the published value while IoU is low", which contradicts the
    # measured reduced-split shape (mIoU HIGHER, IoU LOWER).
    row = next((line for line in text.splitlines()
                if 'P1-9_fullval_anchor.md' in line), '')
    assert row, 'README lost the P1-9 contract row'
    assert 'mIoU 与发布值一致' not in row, (
        'the P1-9 contract still describes branch 2 as an mIoU match')
    assert '两个方向都不一致' in row, row[:200]


def test_readme_boundary_average_is_quoted_consistently():
    """The summary block and the retraction log must not disagree.

    The README's "final conclusion" section states the boundary average (+0.053
    with three seeds per arm) and, a few lines below, a retraction entry repeats
    it.  An earlier edit updated one of them to +0.045 and the other to +0.053 --
    a contradiction inside the same section, which is exactly the kind of drift a
    reader cannot resolve.  Both must carry the value that P4-7 reports.
    """
    text = report('README.md')
    # Two places state it: the ③a claim and the retraction entry below it.  Both
    # must carry the three-seed value; the two-seed +0.045 must be gone.  (A
    # regex over every "平均" figure would also catch unrelated averages, e.g.
    # P4-7's single-seed +0.02, so the invariant is stated on the exact tokens.)
    assert text.count('+0.053') >= 2, (
        'both the summary claim and the retraction entry must quote the '
        'three-seed boundary average')
    assert '+0.045' not in text, 'a two-seed boundary average is still quoted'

    p4_7 = report('P4-7_quality_diagnostics.md')
    assert '+0.053' in p4_7, 'P4-7 must carry the three-seed average'
    assert '0.045' not in p4_7, (
        'P4-7 still carries the two-seed average somewhere')
