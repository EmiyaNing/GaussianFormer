import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from tools.empty_mechanism_stage3 import negative_focal_gradient,suppressive_part


def test_negative_focal_gradient_has_original_normalization_and_weights():
    torch.manual_seed(31928)
    raw=torch.randn(13,17,dtype=torch.double,requires_grad=True)
    labels=torch.arange(13)%17;weights=torch.linspace(.2,2,13,dtype=torch.double)
    m=SimpleNamespace(focal_gamma=2.,focal_alpha=.25)
    targets=F.one_hot(labels,17).to(raw);q=raw.sigmoid()
    loss=((1-targets)*.75*q.square()*F.softplus(raw)*weights[:,None]*2/13).sum()
    expected=torch.autograd.grad(loss,raw)[0]
    actual=negative_focal_gradient(raw,labels,weights,m,2.)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)


def test_only_suppressive_amplitude_component_is_removed():
    torch.manual_seed(31928)
    raw=torch.randn(11,17,dtype=torch.double);d=raw.sigmoid()
    g=torch.randn_like(raw)
    removed,_=suppressive_part(raw,g)
    after=g-removed;dot=(d*g).sum(-1)
    torch.testing.assert_close(after[dot<=0],g[dot<=0],atol=0,rtol=0)
    torch.testing.assert_close((d*after[...,:]).sum(-1)[dot>0],torch.zeros_like(dot[dot>0]),atol=1e-14,rtol=0)
    eps=1e-6
    finite=(F.softplus(raw-eps*after).sum(-1)-F.softplus(raw+eps*after).sum(-1))/(2*eps)
    torch.testing.assert_close(finite,-dot.clamp_max(0),atol=1e-8,rtol=1e-7)


def test_ce_correction_preserves_geometry_and_empty_direct_gradients():
    torch.manual_seed(31928)
    u=torch.randn(3,17,dtype=torch.double,requires_grad=True)
    weights=torch.rand(5,3,dtype=torch.double,requires_grad=True)
    empty=torch.randn(5,dtype=torch.double,requires_grad=True)
    sem=F.softplus(u);scores=torch.cat([weights@sem,empty[:,None]],-1)
    y=torch.tensor([17,17,2,4,17]);loss=F.cross_entropy(scores,y)
    g=torch.autograd.grad(loss,sem,retain_graph=True)[0]*u.detach().sigmoid()
    removed,_=suppressive_part(u,g)
    correction=-(u*removed).sum();modified=loss+(correction-correction.detach())
    a=torch.autograd.grad(loss,(u,weights,empty),retain_graph=True)
    b=torch.autograd.grad(modified,(u,weights,empty))
    torch.testing.assert_close(modified,loss,atol=0,rtol=0)
    torch.testing.assert_close(a[0]-removed,b[0])
    torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)
    torch.testing.assert_close(a[2],b[2],atol=0,rtol=0)


def test_matched_threshold_uses_inclusive_foreground_ties():
    from tools.eval_empty_mechanism_stage3 import matched_threshold,EDGES,NB
    hist=np.zeros((18,17,NB),dtype=np.int64)
    margin=np.array([-1.,-.25,0.,.5,2.])
    hist[17,0]=np.bincount(np.searchsorted(EDGES,margin,side='right'),minlength=NB)
    threshold,fp=matched_threshold(hist,2)
    assert threshold==1/32 and fp==2
    assert int((margin>=threshold).sum())==2


def test_matched_rescue_reselects_threshold_in_every_scene_resample():
    from tools.eval_empty_mechanism_stage3 import EDGES,NB
    from tools.report_empty_mechanism_stage3 import recomputed_matched_rescue
    empty=np.zeros((2,NB),dtype=np.int64);rescue=np.zeros_like(empty)
    for i,values in enumerate([[-1.,0.,1.],[-2.,.5,2.]]):
        empty[i]=np.bincount(np.searchsorted(EDGES,values,side='right'),minlength=NB)
    for i,values in enumerate([[.1,1.5],[.6,3.]]):
        rescue[i]=np.bincount(np.searchsorted(EDGES,values,side='right'),minlength=NB)
    curves=dict(empty_hist=empty,rescue_hist=rescue,baseline_fp=np.array([1,1]),cohort_size=np.array([2,2]))
    effect,threshold=recomputed_matched_rescue(curves,np.array([[1.,1.],[2.,0.],[0.,2.]]))
    np.testing.assert_array_equal(effect,[75.,100.,100.])
    np.testing.assert_array_equal(threshold,[.53125,.03125,.53125])
