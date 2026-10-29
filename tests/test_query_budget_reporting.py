"""Denominators and composition controls used by the query-budget report."""
import numpy as np
import pytest
from tools.report_query_budget import standardized, summarize, matched_complexity, bootstrap_standardized
from tools.stat_query_budget import FIELDS


def test_standardization_removes_composition_reversal():
    a=np.zeros((11,17,4,4,5),dtype=np.int64)
    # Low-load queries occur mainly in the difficult class, high-load in easy.
    a[1,0,0,0,:]=[100,10,10,0,90]
    a[1,1,0,0,:]=[900,810,810,0,90]
    a[9,0,0,0,:]=[900,180,180,0,720]
    a[9,1,0,0,:]=[100,100,100,0,0]
    assert a[1,...,1].sum()/a[1,...,0].sum()>a[9,...,1].sum()/a[9,...,0].sum()
    out=standardized(a)
    assert out['retained_foreground_fraction']==1
    assert out['low']['hole_rate']==pytest.approx(.5)
    assert out['high']['hole_rate']==pytest.approx(.6)


def test_report_uses_pooled_foreground_denominator():
    a=np.zeros((2,len(FIELDS)),dtype=np.float32)
    values={'gt_foreground':[10,30],'gt_no_core':[2,12],
            'core_coverage_all_fg':[8,20],'core_shared_other_query':[6,12]}
    for key,value in values.items():a[:,FIELDS.index(key)]=value
    out=summarize(a)
    assert out['hole_rate']==pytest.approx(14/40)
    assert out['shared_coverage_fraction']==pytest.approx(18/28)
    assert out['core_fg_coverage_per_gaussian']==pytest.approx(28/256)


def test_complexity_matching_uses_common_query_strata():
    a=np.zeros((2,len(FIELDS)),dtype=np.float32)
    for key,value in {'gt_foreground':[100,100],'gt_no_core':[20,30],
        'gt_miss':[10,15],'gt_components':[1,2],
        'gt_horizontal_range_sum':[1500,1500],'gt_coarse_distance_sum':[20,20]}.items():
        a[:,FIELDS.index(key)]=value
    cls=np.zeros((2,17),dtype=np.int32);cls[:,0]=100
    out=matched_complexity(a,cls)
    assert out['retained_foreground_fraction']==1
    assert out['single']['hole_rate']==pytest.approx(.2)
    assert out['multiple']['hole_rate']==pytest.approx(.3)


def test_scene_bootstrap_preserves_identical_scene_effects():
    pairs=np.zeros((3,2,272,5))
    pairs[:,0,0,:]=[100,10,20,0,90]
    pairs[:,1,0,:]=[100,30,25,0,70]
    all_fg=np.zeros((3,272));all_fg[:,0]=200
    out=bootstrap_standardized(pairs,all_fg,repeats=20)
    np.testing.assert_allclose(out['hole_gap_interval'],[.2,.2])
    np.testing.assert_allclose(out['miss_gap_interval'],[.05,.05])
