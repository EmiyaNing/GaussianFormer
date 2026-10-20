"""Tests for per-sample evaluation dumps and the bootstrap analysis.

These close a limitation three reports shared: the strict metric aggregated every
scene into one confusion matrix, so no confidence interval was possible.  What
must hold:

* the evaluator writes one 18x18 semantic and 2x2 binary matrix per (scene,
  horizon), and those sum exactly to its own aggregate -- otherwise the dump
  describes a different quantity than the reported number;
* the per-scene mIoU recomputation matches the aggregate on a single-scene dump,
  which is the only case where the two are directly comparable;
* a degenerate (constant) input gives a zero-width bootstrap interval, so the
  resampling itself is not manufacturing spread;
* a paired bootstrap on identical samples cancels the shared component;
* a scene-level (cluster) bootstrap is wider than the sample-level one exactly
  when samples inside a scene are correlated -- the correction for treating 113
  correlated samples as 113 independent draws.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p1.bootstrap_ci import (  # noqa: E402
    bootstrap_ci, cluster_bootstrap_ci, load_scores, paired_bootstrap_ci,
    paired_cluster_bootstrap_ci, per_sample_miou)
from misc.sparseworld_strict_eval import SparseWorldStrictMetric  # noqa: E402

FREE = 17


def build_scene(shape, prediction_occupied, target_occupied):
    prediction = torch.full(shape, FREE, dtype=torch.long)
    target = torch.full(shape, FREE, dtype=torch.long)
    for index in prediction_occupied:
        prediction.view(-1)[index] = 0
    for index in target_occupied:
        target.view(-1)[index] = 0
    return prediction, target


def feed(metric, prediction, target, horizons=6):
    result = {'final_occ_grid': prediction.unsqueeze(0),
              'future_occ_grids': [prediction.unsqueeze(0)] * horizons}
    metas = {'occ_label': target.unsqueeze(0),
             'future_occ_labels': target.unsqueeze(0).unsqueeze(0).expand(
                 1, horizons, *prediction.shape).contiguous()}
    metric.add_batch(result, metas)


def test_per_sample_dump_sums_to_the_aggregate(tmp_path):
    dump = tmp_path / 'per_sample.json'
    metric = SparseWorldStrictMetric(boundary_stride=0, per_sample_path=dump)
    scenes = [((4, 4, 4), [0, 2], [0, 1]),
              ((4, 4, 4), [1, 2, 3], [1, 2])]
    for shape, predicted, target_occ in scenes:
        prediction, target = build_scene(shape, predicted, target_occ)
        feed(metric, prediction, target)
    metric.results()

    assert dump.exists()
    payload = json.loads(dump.read_text(encoding='utf-8'))
    assert payload['samples'] == 2
    # One entry per (scene, horizon).
    assert len(payload['per_sample']) == 2 * 4
    assert {entry['horizon'] for entry in payload['per_sample']} == {0, 2, 4, 6}

    # The per-scene matrices must reproduce the metric's own aggregate exactly.
    for horizon in (0, 2, 4, 6):
        total = np.zeros((18, 18), dtype=np.int64)
        for entry in payload['per_sample']:
            if entry['horizon'] == horizon:
                total += np.asarray(entry['semantic'], dtype=np.int64)
        assert np.array_equal(total, metric.semantic[horizon]), horizon


def test_per_sample_dump_records_the_scene_label(tmp_path):
    """The cluster bootstrap needs the scene, not the sample index.

    Two samples from the same scene must share a label, and the label must come
    from the camera filename when ``scene_token`` is absent -- ``img_filename``
    is in the collator's ``return_keys`` while ``scene_token`` is not, so the
    filename path is the one that actually fires during evaluation.
    """
    dump = tmp_path / 'labelled.json'
    metric = SparseWorldStrictMetric(boundary_stride=0, per_sample_path=dump)
    prediction, target = build_scene((4, 4, 4), [0], [0])
    names = {
        's1': 'samples/CAM_FRONT/scene_one__CAM_FRONT__1533151603512404.jpg',
        's2': 'samples/CAM_FRONT/scene_two__CAM_FRONT__1533151603512404.jpg',
    }
    for key in ('s1', 's1', 's2'):
        result = {'final_occ_grid': prediction.unsqueeze(0),
                  'future_occ_grids': [prediction.unsqueeze(0)] * 6}
        metas = {'occ_label': target.unsqueeze(0),
                 'future_occ_labels': target.unsqueeze(0).unsqueeze(0).expand(
                     1, 6, *prediction.shape).contiguous(),
                 'img_filename': [[names[key]]]}
        metric.add_batch(result, metas)
    metric.results()

    payload = json.loads(dump.read_text(encoding='utf-8'))
    by_sample = {}
    for entry in payload['per_sample']:
        by_sample.setdefault(entry['sample'], set()).add(entry['scene'])
    assert by_sample[0] == {'scene_one'}
    assert by_sample[1] == {'scene_one'}
    assert by_sample[2] == {'scene_two'}

    ids, scenes, miou, occupied, precision, recall = load_scores(dump)
    assert scenes.tolist() == ['scene_one', 'scene_one', 'scene_two']
    # Two samples of scene_one => 2 clusters, so the cluster bootstrap is the
    # one the CLI will pick.
    assert len(set(scenes.tolist())) == 2


def test_per_sample_miou_matches_a_hand_computed_scene():
    # Two occupied target voxels, one predicted correctly -> TP=1, FN=1, FP=0.
    # Class 0: IoU = 1/2 = 50.  Class 17 (free) is excluded by [:17], and the
    # remaining 15 classes have zero target and zero prediction so their IoU is
    # NaN and nanmean skips them.  mIoU over 17 classes = 50 / 1 = 50.
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[0, 0] = 1        # TP for class 0
    hist[0, 17] = 1       # FN for class 0
    hist[17, 17] = 100    # true negatives, irrelevant to [:17]
    assert per_sample_miou(hist) == pytest.approx(50.0, abs=1e-9)


def test_bootstrap_of_a_constant_is_degenerate():
    values = np.full(50, 14.0)
    mean, low, high, stderr = bootstrap_ci(values, repetitions=200, seed=0)
    assert mean == pytest.approx(14.0)
    assert low == pytest.approx(14.0) and high == pytest.approx(14.0)
    assert stderr == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_interval_brackets_the_mean_and_narrows_with_n():
    generator = np.random.default_rng(0)
    small = generator.normal(14.0, 1.0, size=30)
    large = generator.normal(14.0, 1.0, size=3000)
    _, low_s, high_s, se_s = bootstrap_ci(small, repetitions=2000, seed=1)
    _, low_l, high_l, se_l = bootstrap_ci(large, repetitions=2000, seed=1)
    assert low_s < small.mean() < high_s
    assert low_l < large.mean() < high_l
    assert (high_l - low_l) < (high_s - low_s)
    assert se_l < se_s


def test_paired_bootstrap_cancels_a_shared_component():
    generator = np.random.default_rng(2)
    shared = generator.normal(0.0, 1.0, size=200)
    a = 14.0 + shared
    b = 13.9 + shared
    observed, low, high, stderr = paired_bootstrap_ci(a, b, repetitions=2000, seed=3)
    assert observed == pytest.approx(0.1, abs=1e-9)
    # The shared component cancels exactly, so the difference has no spread.
    assert stderr == pytest.approx(0.0, abs=1e-9)
    assert low == pytest.approx(0.1) and high == pytest.approx(0.1)


def test_paired_bootstrap_requires_the_same_scenes():
    with pytest.raises(AssertionError):
        paired_bootstrap_ci(np.zeros(3), np.zeros(4), repetitions=10, seed=0)


def test_cli_reports_ci_and_paired_difference(tmp_path):
    def write_dump(path, offset, with_scenes=True):
        hist = np.zeros((18, 18), dtype=np.int64)
        hist[0, 0] = 10
        hist[0, 17] = offset
        entries = []
        for sample in range(6):
            for horizon in (0, 2, 4, 6):
                entry = {'sample': sample, 'horizon': horizon,
                         'semantic': hist.tolist(),
                         'binary': [[5, 1], [2, 20]]}
                if with_scenes:
                    entry['scene'] = f'scene{sample % 2}'
                entries.append(entry)
        path.write_text(json.dumps({'samples': 6, 'horizons': [0, 2, 4, 6],
                                    'class_names': ['others'],
                                    'per_sample': entries}), encoding='utf-8')
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    write_dump(first, 1)
    write_dump(second, 2)
    out = tmp_path / 'out.json'
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p1/bootstrap_ci.py', '--variants',
         f'a={first}', f'b={second}', '--repetitions', '500', '--json', str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-400:]
    assert '95% CI' in completed.stdout
    assert 'paired difference' in completed.stdout
    # The dumps carry scene labels, so the headline interval must be the
    # scene-level (cluster) one, with the sample-level one printed beside it.
    assert '(scene)' in completed.stdout and '(sample)' in completed.stdout
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert set(payload) == {'a', 'b'}
    assert payload['a']['miou'].__len__() == 6
    assert payload['a']['unit'] == 'scene'
    assert payload['a']['ci']['unit'] == 'scene'
    assert payload['a']['ci']['clusters'] == 2
    assert payload['a']['ci_sample_level']['unit'] == 'sample'
    assert 'paired' in payload['a']
    # ``a`` has fewer false negatives, so it must score higher.
    paired = payload['a']['paired']['b']
    assert paired['observed'] > 0
    assert paired['shared_samples'] == 6
    assert paired['unit'] == 'scene'


def test_cli_warns_when_scene_labels_are_missing(tmp_path):
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[0, 0] = 10
    hist[0, 17] = 1
    entries = [{'sample': s, 'horizon': 0, 'semantic': hist.tolist(),
                'binary': [[5, 1], [2, 20]]} for s in range(4)]
    path = tmp_path / 'unlabelled.json'
    path.write_text(json.dumps({'samples': 4, 'horizons': [0, 2, 4, 6],
                                'class_names': [], 'per_sample': entries}),
                    encoding='utf-8')
    out = tmp_path / 'out.json'
    completed = subprocess.run(
        [sys.executable, 'benchmarks/p1/bootstrap_ci.py', '--variants',
         f'x={path}', '--repetitions', '200', '--json', str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-400:]
    assert 'no scene label' in completed.stdout
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert payload['x']['unit'] == 'sample'
    assert 'ci_sample_level' not in payload['x']


def test_cluster_bootstrap_widens_the_interval_for_clustered_samples():
    """Pseudo-replication, in numbers.

    Fifty samples of scenes A/B are all 0 and fifty of C/D are all 1.  Samples
    are perfectly correlated *within* a scene, so:

    * a sample-level bootstrap sees 100 draws from a 50/50 pool and reports
      SE = sqrt(0.25/100) = 0.05;
    * a scene-level bootstrap sees 4 clusters from {0, 0, 1, 1} and reports
      SE = sqrt(0.25/4) = 0.25 -- five times wider.

    Ignoring the clustering is exactly how an evaluation-side interval ends up
    looking tighter than the data supports.
    """
    values = np.array([0.0] * 50 + [1.0] * 50)
    groups = np.array(['a'] * 25 + ['b'] * 25 + ['c'] * 25 + ['d'] * 25,
                      dtype=object)
    mean, low, high, stderr = cluster_bootstrap_ci(
        values, groups, repetitions=4000, seed=0)
    sample_mean, sample_low, sample_high, sample_stderr = bootstrap_ci(
        values, repetitions=4000, seed=0)
    assert mean == pytest.approx(0.5)
    assert sample_mean == pytest.approx(0.5)
    assert sample_stderr == pytest.approx(0.05, abs=0.01)
    assert stderr == pytest.approx(0.25, abs=0.03)
    assert stderr > 3 * sample_stderr
    assert (high - low) > 3 * (sample_high - sample_low)


def test_paired_cluster_bootstrap_cancels_the_shared_scene_component():
    scene_means = np.array([10.0, 12.0, 14.0, 16.0])
    groups = np.array([f's{i}' for i in range(4) for _ in range(3)],
                      dtype=object)
    base = np.repeat(scene_means, 3)
    other = base + 0.4
    observed, low, high, stderr = paired_cluster_bootstrap_ci(
        base, other, groups, repetitions=2000, seed=0)
    assert observed == pytest.approx(-0.4)
    # The same clusters are drawn for both arms, so the shared component
    # cancels exactly and only the (constant) shift remains.
    assert stderr == pytest.approx(0.0, abs=1e-9)
    assert low == pytest.approx(-0.4) and high == pytest.approx(-0.4)


def test_load_scores_averages_over_horizons(tmp_path):
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[0, 0] = 1
    hist[0, 17] = 1        # class-0 IoU = 50 for every horizon
    entries = [{'sample': 0, 'horizon': h, 'semantic': hist.tolist(),
                'binary': [[1, 0], [0, 1]]} for h in (0, 2, 4, 6)]
    path = tmp_path / 'one.json'
    path.write_text(json.dumps({'samples': 1, 'horizons': [0, 2, 4, 6],
                                'class_names': [], 'per_sample': entries}),
                    encoding='utf-8')
    ids, scenes, miou, occupied, precision, recall = load_scores(path)
    assert ids.tolist() == [0]
    assert miou[0] == pytest.approx(50.0)
    # A dump written before scene labels existed must still load: the label is
    # None and callers fall back to a sample-level bootstrap (with a warning).
    assert scenes.tolist() == [None]


def test_occupancy_pair_orientation_is_not_transposed():
    """``precision != recall`` on an asymmetric matrix, so a swap is detectable.

    The 2x2 is ``hist[target, prediction]``.  With 10 true positives, 30 false
    positives and 90 false negatives: precision = 10/40 = 25%, recall = 10/100 =
    10%.  Swapping the axes would return (10%, 25%) instead, which the assertion
    below rejects.
    """
    from benchmarks.p1.bootstrap_ci import per_sample_occupancy_pair
    hist = [[0, 30],      # target 0 (free): 30 predicted occupied
            [90, 10]]     # target 1 (occupied): 90 missed, 10 hit
    precision, recall = per_sample_occupancy_pair(hist)
    assert precision == pytest.approx(25.0)
    assert recall == pytest.approx(10.0)
    assert precision != recall
    # Degenerate denominators yield NaN, not 0 and not a division error.  The
    # two cases are different: no predicted occupancy makes precision undefined
    # (recall is a genuine 0), and an empty occupancy target makes recall
    # undefined (precision is a genuine 0).
    precision, recall = per_sample_occupancy_pair([[5, 0], [5, 0]])
    assert precision != precision and recall == pytest.approx(0.0)
    precision, recall = per_sample_occupancy_pair([[5, 3], [0, 0]])
    assert precision == pytest.approx(0.0) and recall != recall
