import numpy as np

from tools.summarize_future_image_joint import bootstrap_pair, score


def counts():
    hist = np.zeros((3, 3, 18, 18))
    hist[..., 0, 0] = 8
    hist[..., 0, 17] = 2
    hist[..., 17, 0] = 3
    return dict(hist=hist, intersection=np.ones((3, 3))*8,
                union=np.ones((3, 3))*13)


def test_scores_recalculate_iou_from_counts():
    result = score(**{key: value.sum(0) for key, value in counts().items()})
    np.testing.assert_allclose(result['mIoU'], 100*8/13)
    np.testing.assert_allclose(result['OccIoU'], 100*8/13)


def test_identical_arms_have_zero_paired_scene_interval():
    arrays = [counts(), counts()]
    result = bootstrap_pair(arrays, arrays, ['a', 'a', 'b'], [0, 1, 2], 20, 11)
    for metric in ('mIoU', 'OccIoU'):
        for interval in result[metric].values():
            np.testing.assert_equal(interval, [0., 0.])
