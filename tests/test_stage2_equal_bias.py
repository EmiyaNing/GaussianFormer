import numpy as np
from tools.calibrate_empty_mechanism_stage2 import counts_from_hist,fitting_thresholds,NB,EDGES


def test_histogram_thresholds_match_exact_decisions_including_edge_ties():
    margin=np.array([0.,-.03125,.03125,-257.,257.,-.5,.5,0.])
    gt=np.array([1,17,1,17,3,3,17,2])
    fg=np.array([1,1,1,2,3,3,3,2])
    bins=np.searchsorted(EDGES,margin,side="right")
    code=(gt*17+fg)*NB+bins
    hist=np.bincount(code,minlength=18*17*NB).reshape(18,17,NB)
    for shift in [-.03125,0.,.03125]:
        thresholds=np.full(17,shift)
        pred=np.where(margin>=thresholds[fg],fg,17)
        actual=np.bincount(gt*18+pred,minlength=324).reshape(18,18)
        assert np.array_equal(counts_from_hist(hist,thresholds),actual)


def test_fitting_optimizes_each_class_with_correct_ground_truth_denominator():
    margin=np.array([-.25,-.5,-1.,.5,-.5])
    gt=np.array([1,17,17,2,17]);fg=np.array([1,1,1,2,2])
    bins=np.searchsorted(EDGES,margin,side="right")
    hist=np.bincount((gt*17+fg)*NB+bins,minlength=18*17*NB).reshape(18,17,NB)
    thresholds,boundary=fitting_thresholds(hist)
    assert thresholds[1]==-.25 and thresholds[2]==0.
    assert not any(boundary)
    pred=np.where(margin>=thresholds[fg],fg,17)
    assert np.array_equal(pred,gt)
