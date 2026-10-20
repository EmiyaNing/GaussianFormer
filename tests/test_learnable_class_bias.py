"""Verify the class-bias rule, full CE sufficient statistics and exact counts."""
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.learnable_class_bias import sorted_statistics,exact_counts
from tools.fit_learnable_class_bias import ce_objective

spec=importlib.util.spec_from_file_location('bias_component',Path(__file__).resolve().parents[1]/'model/head/learnable_class_empty_bias.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
LearnableClassEmptyBias=module.LearnableClassEmptyBias


def test_bias_replaces_global_once_and_preserves_foreground_order_and_band():
    head=LearnableClassEmptyBias(3.5)
    scores=torch.full((1,18,5),-10.,dtype=torch.float32)
    scores[:,4]=torch.tensor([.7,1.,1.2,-2.,4.]);scores[:,17]=1.
    native=scores.argmax(1)
    torch.testing.assert_close(head.predict(scores),native,rtol=0,atol=0)
    with torch.no_grad():head.class_bias[4]=4.
    predicted=head.predict(scores)
    assert predicted.tolist()==[[4,4,4,17,4]]
    changed=head(scores)
    torch.testing.assert_close(changed[:,:17],scores[:,:17].double(),rtol=0,atol=0)
    assert changed[0,17,0]==.5
    assert changed[0,17,3]==1. # outside candidate band
    with torch.no_grad():head.class_bias[4]=3.
    assert head.predict(scores).tolist()==[[17,17,17,17,4]]


def test_full18_ce_sufficient_statistics_have_exact_loss_and_bias_gradient():
    generator=torch.Generator().manual_seed(4)
    scores=torch.randn(113,18,generator=generator,dtype=torch.float64)*2
    gt=torch.randint(18,(113,),generator=generator)
    fg=scores[:,:17].argmax(-1)
    lse=scores[:,:17].logsumexp(-1)-scores[:,17]
    target=scores.gather(1,gt[:,None])[:,0]-scores[:,17]
    weight=torch.rand(113,generator=generator,dtype=torch.float64);weight/=weight.sum()
    a=torch.randn(17,generator=generator,dtype=torch.float64,requires_grad=True)
    adjusted=torch.cat([scores[:,:17],(scores[:,17]-a[fg])[:,None]],-1)
    expected=(F.cross_entropy(adjusted,gt,reduction='none')*weight).sum()
    actual=ce_objective(a[fg],lse,target,(gt==17).double(),weight)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    ga,=torch.autograd.grad(actual,a,retain_graph=True);ge,=torch.autograd.grad(expected,a)
    torch.testing.assert_close(ga,ge,atol=1e-12,rtol=1e-12)


def test_exact_sorted_counts_include_ties_wrong_foreground_and_both_directions():
    rng=np.random.default_rng(928);n=531
    margin=rng.uniform(-2,2,n).astype(np.float32);margin[:3]=[0.,-.5,.5]
    gt=rng.integers(0,18,n);fg=rng.integers(0,17,n);region=rng.integers(0,2,n)
    fixed=np.zeros((2,18,18),np.int64);fixed[0,17,2]=5;fixed[1,3,3]=7
    stat=sorted_statistics(margin,region*306+gt*17+fg,fixed)
    thresholds=np.stack([np.zeros(17),np.full(17,.5),np.full(17,-.5),rng.uniform(-3,3,17)])
    counts,events=exact_counts(stat,thresholds,events=True)
    native=np.where(margin>=0,fg,17)
    for i,t in enumerate(thresholds):
        prediction=np.where(margin>=t[fg],fg,17)
        code=region*324+gt*18+prediction
        direct=fixed+np.bincount(code,minlength=648).reshape(2,18,18)
        np.testing.assert_array_equal(counts[i],direct)
        up=(native==17)&(prediction!=17);down=(native!=17)&(prediction==17)
        correct=gt==fg;empty=gt==17;wrong=~correct&~empty
        expected=np.array([[np.sum((region==r)&v) for v in
            [up&correct,up&wrong,up&empty,down&correct,down&wrong,down&empty]] for r in [0,1]])
        np.testing.assert_array_equal(events[i],expected)
