import pytest
import torch

from tools.diagnose_gaussianformer_empty import extended_metrics


def test_empty_error_rates_condition_on_the_correct_ground_truth_population():
    counts = torch.zeros(18, 18, dtype=torch.float64)
    counts[0, 0], counts[0, 1], counts[0, 17] = 30, 10, 10
    counts[17, 0], counts[17, 17] = 5, 45
    result = extended_metrics(counts)
    assert result['occupied_missed_as_empty_percent'] == pytest.approx(20.)
    assert result['empty_falsely_occupied_percent'] == pytest.approx(10.)
    assert result['recall'] == pytest.approx(80.)
    assert result['precision'] == pytest.approx(100 * 40 / 45)
    assert result['IoU'] == pytest.approx(100 * 40 / 55)
    assert result['predicted_occupied_fraction'] == pytest.approx(45.)
    assert result['gt_occupied_fraction'] == pytest.approx(50.)
    assert result['confusion'] == counts.long().tolist()
