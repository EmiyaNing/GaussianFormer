"""Independent geometry and denominator checks for the decoder census."""
import numpy as np
import pytest
import torch
from tools.decoder_layer_stats.common import quintiles, summarize_frame


def test_fixed_load_groups_are_disjoint_and_match_historical_top20_ties():
    load = np.array([0, 4, 4, 8, 0, 7, 9, 0, 4, 10])
    groups = quintiles(load)
    np.testing.assert_array_equal(np.bincount(groups), [2] * 5)
    np.testing.assert_array_equal(np.flatnonzero(groups == 0), np.sort(np.argsort(load, kind='stable')[-2:]))
    assert load[groups == 0].min() >= load[groups == 1].max()
    np.testing.assert_array_equal(quintiles(np.zeros(10)), [4, 4, 3, 3, 2, 2, 1, 1, 0, 0])


def test_aggregate_keeps_voxel_unions_and_all_GS_denominators_separate():
    load = np.arange(5)
    groups = quintiles(load)
    distances = np.arange(20).reshape(5, 4).astype(float)
    counts = np.array([[0, 0, 0, 0], [1, 1, 2, 1], [2, 1, 4, 2], [2, 0, 5, 0], [1, 1, 3, 3]])
    unions = np.array([[8, 9, 10, 5], [3, 3, 4, 3], [4, 5, 5, 0],
                       [3, 4, 5, 2], [2, 3, 3, 1], [0, 0, 0, 0]])
    rows = summarize_frame(distances, counts, load, groups, unions, 10, 2)
    assert rows[0]['GS'] == 10 and rows[0]['same_class_core_hit_GS'] == 3
    assert rows[0]['core_1sigma_union'] == 8
    assert sum(r['core_1sigma_union'] for r in rows[1:]) == 12  # overlapping groups
    assert rows[1]['assigned_foreground'] == 4 and rows[1]['queries'] == 1
    assert rows[0]['query_reference_center_to_foreground_region_grid_sum'] == distances[:, 1].sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('empty_foreground', [False, True])
def test_support_matches_FP64_enumeration_across_query_word_boundary(empty_foreground):
    from tools.decoder_layer_stats import extension
    rng = np.random.default_rng(19)
    shape, h, q, k = (5, 6, 4), .4, 70, 2
    xyz = ((np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), -1).reshape(-1, 3) + .5) * h).astype(np.float32)
    means = (xyz[rng.integers(0, len(xyz), q*k)] + rng.uniform(-.1, .1, (q*k, 3))).astype(np.float32)
    scales = rng.uniform(.1, .4, (q*k, 3))
    inverse = []
    for scale in scales:
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        mat = rotation.T @ np.diag(scale ** -2) @ rotation
        inverse.append(mat.flat[[0, 4, 8, 1, 5, 2]])
    inverse = np.array(inverse, np.float32)
    alpha = rng.uniform(.01, 1, q*k).astype(np.float32)
    alpha[::13] = 0
    centers = (means / h).astype(np.int32)
    radii = np.ceil(scales.max(1) * 3 / h).astype(np.int32)
    labels = rng.integers(0, 18, len(xyz), dtype=np.int32)
    camera = rng.random(len(xyz)) > .2
    targets = np.flatnonzero(camera & (labels < 17))
    if empty_foreground:
        targets = targets[:0]
    lookup = np.full(len(xyz), -1, np.int32)
    lookup[targets] = np.arange(len(targets))
    classes = rng.integers(0, 17, q*k, dtype=np.int32)
    groups = quintiles(rng.integers(0, 100, q))
    inputs = [torch.from_numpy(x).cuda().contiguous() for x in
              (means, inverse, alpha, centers, radii, xyz, lookup, labels, classes, groups)]
    bits, coverage, unions, counts = extension().support(*inputs, len(targets), q, k, *shape)
    expected = np.zeros((4, len(targets), q), bool)
    gs_expected = np.zeros((q*k, 2), np.int32)
    grid = (xyz[targets] / h).astype(np.int32)
    for g in range(q*k):
        delta = means[g].astype(float) - xyz[targets].astype(float)
        power = -.5 * (inverse[g, :3] * delta ** 2).sum(1) - (inverse[g, 3:] * delta[:, [0, 1, 0]] * delta[:, [1, 2, 2]]).sum(1)
        positive = ((np.abs(grid - centers[g]) <= radii[g]).all(1) & (alpha[g] > 0)
                    & (np.exp(power.astype(np.float32)) * alpha[g] > 0))
        core = positive & (power >= -.5)
        same = core & (labels[targets] == classes[g])
        for s, hits in enumerate((core, positive & (power >= -4.5), positive, same)):
            expected[s, :, g//k] |= hits
        gs_expected[g] = [core.sum(), same.sum()]
    np.testing.assert_array_equal(counts.cpu(), gs_expected)
    np.testing.assert_array_equal(coverage.cpu(), expected.sum(1))
    np.testing.assert_array_equal(unions.cpu()[0], expected.any(2).sum(1))
    for b in range(5):
        np.testing.assert_array_equal(unions.cpu()[b+1], expected[:, :, groups == b].any(2).sum(1))
    packed = bits.cpu().numpy().view(np.uint64)
    for j in range(q):
        np.testing.assert_array_equal((packed[:, :, j//64] >> np.uint64(j%64)) & np.uint64(1), expected[:, :, j])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_exact_voxel_region_minimum_is_independent_of_nearest_voxel_center():
    from tools.query_foreground_distance import extension
    # Nearest center is (1.25,0,0), nearest unit cube is centered at (1,1,0).
    means = np.array([[0, 0, 0], [1, 1, 0]], np.float32)
    gt = np.array([[1.25, 0, 0], [1, 1, 0]], np.float32)
    actual = extension().distances(torch.from_numpy(means).cuda(), torch.from_numpy(gt).cuda(), 1).cpu().numpy()
    truth = []
    for mu in means.astype(float):
        delta = np.abs(gt.astype(float) - mu)
        truth.append([np.sqrt((delta**2).sum(1).min()), np.sqrt((np.maximum(delta-.5, 0)**2).sum(1).min())])
    np.testing.assert_allclose(actual, truth, atol=1e-6)
