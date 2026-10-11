import numpy as np

from tools.analyze_future2current_regions import grouped_metrics, scene_hist


def test_group_metrics_distinguish_exact_class_and_group_membership():
    hist = np.zeros((18, 18), dtype=np.float64)
    hist[4, 4] = 6          # GT car, predicted car
    hist[4, 3] = 2          # GT car, predicted bus: dynamic but wrong class
    hist[4, 17] = 2         # GT car, predicted free
    hist[11, 11] = 9        # GT road, predicted road
    hist[11, 4] = 1         # GT road, predicted car
    hist[17, 4] = 3         # GT free, predicted car
    got = grouped_metrics(hist)
    assert np.isclose(got['dynamic']['gt_exact_accuracy'], 60.0)
    assert np.isclose(got['dynamic']['group_iou'], 100.0 * 8 / 14)
    assert np.isclose(got['dynamic']['macro_iou'], 100.0 * 6 / 14 / 2)
    assert np.isclose(got['static']['gt_exact_accuracy'], 90.0)
    assert np.isclose(got['static']['group_iou'], 90.0)


def test_scene_hist_keeps_samples_of_one_scene_together():
    hist = np.zeros((3, 3, 18, 18), dtype=np.float64)
    hist[0, :, 4, 4] = 1
    hist[1, :, 4, 4] = 2
    hist[2, :, 11, 11] = 3
    output = scene_hist(hist, ['a', 'a', 'b'], ['a', 'b'])
    assert output.shape == (2, 3, 18, 18)
    assert np.all(output[0, :, 4, 4] == 3)
    assert np.all(output[1, :, 11, 11] == 3)
