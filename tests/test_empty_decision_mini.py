"""Scientific accounting checks for the frozen empty-decision experiment."""
import numpy as np
import torch

from tools.diagnose_empty_decision_mini import transition_masks, FEATURES
from tools.analyze_empty_decision_mini import (
    GROUPS, MODELS, precision_at_fraction, histogram_scores, scores)
from tools.diagnose_opus_v1_gaussian import calibrated_predictions


def test_transition_partition_counts_semantic_rescue_separately():
    gt = torch.tensor([2, 3, 17, 1, 17])
    fg = torch.tensor([2, 0, 4, 1, 5])
    native = torch.tensor([17, 17, 17, 1, 17])
    shifted = torch.tensor([2, 0, 4, 1, 17])
    masks = transition_masks(gt, fg, native, shifted)
    assert [int(masks[k].sum()) for k in masks] == [3, 1, 1, 1]
    assert torch.equal(masks['flips'].long(), sum(masks[k].long() for k in masks if k != 'flips'))


def test_native_float_subtraction_rule_is_used_at_threshold():
    # Computing maxfg-empty first changes some float32 boundary decisions.
    empty = torch.tensor([3.000000238418579, 2., 2.])
    maxfg = torch.tensor([2.000000238418579, 1., .999])
    fg = torch.tensor([3, 4, 5])
    shifted = calibrated_predictions(maxfg, fg, empty, 1.)
    native = calibrated_predictions(maxfg, fg, empty, 0.)
    m = transition_masks(torch.tensor([3, 17, 5]), fg, native, shifted)
    assert shifted.tolist() == [3, 4, 17]
    assert m['correct_rescue'].tolist() == [True, False, False]


def test_precision_budget_uses_population_weights_and_tie_fraction():
    y = np.array([1., 0., 1.])
    p = np.array([.9, .9, .1])
    w = np.array([1., 3., 6.])
    assert np.isclose(precision_at_fraction(y, p, w, .2), .25)
    for order in [np.array([1, 0, 2]), np.array([2, 1, 0])]:
        assert np.isclose(precision_at_fraction(y[order], p[order], w[order], .2), .25)


def test_histogram_metrics_match_exact_metrics_with_ties():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([.1, .1, .4, .8, .8, .8])
    w = np.array([2., 1., 3., 1., 4., 2.])
    pos = np.bincount((p*10).astype(int), weights=w*y, minlength=10)
    neg = np.bincount((p*10).astype(int), weights=w*(1-y), minlength=10)
    auc, ap = histogram_scores(pos, neg)
    exact = scores(y, p, w)
    assert np.isclose(auc, exact['auroc'])
    assert np.isclose(ap, exact['auprc'])


def test_probe_inputs_exclude_ground_truth_and_cover_features_once():
    assert sum(GROUPS.values(), []) == list(range(len(FEATURES)+17))
    forbidden = ['gt', 'mask', 'label', 'lidar', 'depth']
    assert all(not any(word in name for word in forbidden) for name in FEATURES)
    for groups in MODELS.values():
        columns = sum([GROUPS[g] for g in groups], [])
        assert len(columns) == len(set(columns))
