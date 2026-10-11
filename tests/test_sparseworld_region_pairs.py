import numpy as np

from tools.sparseworld_region_utils import joint_counts, states, state_metrics, percentile_interval


def test_joint_states_void_mask_and_uint8_labels():
    gt = np.asarray([0, 4, 17, 4, 4, 255], dtype=np.uint8)
    f = np.asarray([0, 4, 4, 17, 5, 0], dtype=np.uint8)
    p = np.asarray([0, 17, 17, 4, 6, 0], dtype=np.uint8)
    joint = joint_counts(gt, f, p)
    assert joint.sum() == 5
    assert joint[17, 4, 17] == 1
    assert joint[4, 5, 6] == 1
    counts = np.bincount(states(gt, f, p)[gt < 18], minlength=4)
    np.testing.assert_array_equal(counts, [1, 2, 1, 1])
    metric = state_metrics(counts)
    assert metric['advantage'] == 20
    assert metric['oracle_accuracy'] == 80
    mask = np.asarray([False, True, False, False, True, True])
    masked = joint_counts(gt, f, p, mask)
    assert masked.sum() == 2
    assert masked.sum(2)[4, 4] == 1
    assert masked.sum(1)[4, 17] == 1


def test_binary_occupancy_is_distinct_from_semantic_correctness():
    gt = np.asarray([4, 17, 4], dtype=np.uint8)
    f = np.asarray([5, 4, 17], dtype=np.uint8)
    p = np.asarray([6, 17, 4], dtype=np.uint8)
    np.testing.assert_array_equal(states(gt, f, p), [0, 1, 1])
    np.testing.assert_array_equal(states(gt, f, p, binary=True), [3, 1, 1])


def test_bootstrap_intervals_preserve_horizon_and_mask_axes():
    # Equal dimension sizes can conceal a horizon/mask transpose in shape checks.
    distinct_metrics = np.asarray([[1, 10, 100], [2, 20, 200], [3, 30, 300]])
    samples = np.repeat(distinct_metrics[None], 8, axis=0)
    interval = percentile_interval(samples)
    assert interval.shape == (3, 3, 2)
    np.testing.assert_array_equal(interval[..., 0], distinct_metrics)
    np.testing.assert_array_equal(interval[..., 1], distinct_metrics)
