"""Checks for count-based decisions, floating boundaries and directional errors."""
import numpy as np
import torch

from tools.empty_decision_optimization import confusion
from tools.fit_empty_decision_optimization import Dense, decision_statistics
from tools.eval_empty_decision_optimization import (
    EDGES, hist_tensor, histogram_counts, histogram_counts_torch, histogram_curve_counts, metrics, chosen_index)


def test_histogram_preserves_zero_and_near_zero_native_decisions():
    score = torch.tensor([-.03125,-1e-8,0.,1e-8,.03125,-20.,20.])
    gt = torch.tensor([17,2,2,17,3,17,4])
    fg = torch.tensor([2,2,2,2,3,4,4])
    region = torch.tensor([0,0,0,1,1,0,1])
    native = score >= 0
    h = hist_tensor(score,gt,fg,region,native)
    cut = int(np.searchsorted(EDGES,0))+1
    actual,events = histogram_counts(h.numpy(),np.zeros((2,18,18),np.int64),cut)
    expected = confusion(gt,torch.where(native,fg,17),region)
    np.testing.assert_array_equal(actual,expected)
    assert events.sum() == 0


def test_class_threshold_histograms_match_dense_predictions_and_transitions():
    rng = np.random.default_rng(12)
    n = 1000
    score = torch.from_numpy(rng.normal(size=n).astype(np.float32))
    gt = torch.from_numpy(rng.integers(0,18,n))
    fg = torch.from_numpy(rng.integers(0,17,n))
    region = torch.from_numpy(rng.integers(0,2,n))
    native = torch.from_numpy(rng.integers(0,2,n).astype(bool))
    h = hist_tensor(score,gt,fg,region,native)
    cuts = rng.integers(480,545,17)
    threshold = torch.from_numpy(EDGES[cuts-1])[fg]
    occupied = score >= threshold
    actual,events = histogram_counts(h.numpy(),np.zeros((2,18,18),np.int64),cuts)
    expected = confusion(gt,torch.where(occupied,fg,17),region)
    np.testing.assert_array_equal(actual,expected)
    tc,te = histogram_counts_torch(h,torch.zeros((2,18,18),dtype=torch.int64),[cuts])
    np.testing.assert_array_equal(tc[0].numpy(),expected)
    np.testing.assert_array_equal(te[0].numpy(),events)
    curve_cuts = np.array([500,513,530])
    curve = histogram_curve_counts(h.numpy(),np.zeros((2,18,18),np.int64),curve_cuts)
    for i,cut in enumerate(curve_cuts):
        np.testing.assert_array_equal(curve[i],histogram_counts(h.numpy(),np.zeros((2,18,18),np.int64),cut)[0])
    for r in [0,1]:
        addition = (region == r)&(~native)&occupied
        removal = (region == r)&native&(~occupied)
        direct = []
        for mask in [addition,removal]:
            direct += [int((mask&(gt == fg)).sum()),int((mask&(gt != fg)&(gt != 17)).sum()),int((mask&(gt == 17)).sum())]
        np.testing.assert_array_equal(events[r],direct)


def test_iou_gradient_matches_local_fractional_utility():
    d = torch.tensor([.3,.7],requires_grad=True)
    q = torch.tensor([.8,.1])
    # Expected counts for one class, independent of the implementation's labels.
    tp = 20+(q*d).sum()
    union = 100+((1-q)*d).sum()
    j = tp/union
    gradient, = torch.autograd.grad(j,d)
    expected = (q-j.detach()*(1-q))/union.detach()
    torch.testing.assert_close(gradient,expected)


def test_decision_statistics_distinguishes_wrong_semantics_and_empty_risk():
    context = dict(fg=torch.tensor([0,0,0]),w=torch.ones(3),correct=torch.tensor([True,False,False]),
        camera=torch.tensor([True,True,False]),out_empty=torch.tensor([False,False,True]),
        gt=torch.ones(17),tp=torch.zeros(17),fp=torch.zeros(17),out_fixed=torch.tensor(0.),out_count=torch.tensor(1.))
    d = torch.tensor([1.,1.,0.],requires_grad=True)
    iou,risk = decision_statistics(d,context)
    torch.testing.assert_close(iou,torch.tensor(.5/17))
    assert risk == 0
    gradient, = torch.autograd.grad(iou,d,retain_graph=True)
    assert gradient[0] > 0 and gradient[1] < 0 and gradient[2] == 0
    risk_gradient, = torch.autograd.grad(risk,d)
    torch.testing.assert_close(risk_gradient,torch.tensor([0.,0.,1.]))


def test_zero_residual_initialization_preserves_baseline():
    gate = Dense(29,16,1,zero_output=True)
    assert torch.count_nonzero(gate(torch.randn(30,29))) == 0


def test_operating_point_selection_enforces_training_risk_budget():
    index,feasible = chosen_index(np.array([20.,25.,27.]),np.array([.08,.10,.14]),.005,.10)
    assert feasible and index == 1
    index,feasible = chosen_index(np.array([20.,25.,27.]),np.array([.08,.10,.14]),None,.10)
    assert feasible and index == 2
