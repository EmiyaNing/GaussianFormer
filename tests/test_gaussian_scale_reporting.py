"""Margin histograms must reproduce native tie handling and FP budgets."""
import numpy as np
from tools.probe_gaussian_scale_supervision import MARGIN_EDGES
from tools.report_gaussian_scale_supervision import curve,metrics


def test_zero_margin_is_foreground_and_budget_selection_is_feasible():
    hist=np.zeros((18,17,len(MARGIN_EDGES)+1),np.int64)
    examples=[(0,0,0.),(0,1,-.1),(17,0,0.),(17,1,-.2)]
    confusion=np.zeros((18,18),np.int64)
    for y,c,m in examples:
        index=np.searchsorted(MARGIN_EDGES,m,side='right');hist[y,c,index]+=1
        confusion[y,c if m>=0 else 17]+=1
    rows,selected=curve(hist,1)
    zero=rows[int(np.argmin(np.abs(MARGIN_EDGES)))]
    expected=metrics(confusion)
    assert zero['false_positives']==expected['false_positives']==1
    assert zero['misses']==expected['misses']==1
    assert zero['mIoU']==expected['mIoU']
    assert selected['false_positives']<=1
    assert selected['threshold']<=0
