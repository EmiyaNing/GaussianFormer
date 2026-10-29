"""Diagnostic semantics, overlap attribution and GPU/independent-reference parity."""
import numpy as np
import pytest
import torch

from tools.boundary_gaussian_stats import rates, gaussian_counts
from tools.boundary_gaussian_stats.reference import reference


def fixture():
    shape=(7,6,4); step=np.float32(.4); lower=np.array([-1.,-1.,-1.],np.float32)
    rng=np.random.default_rng(928)
    xyz=(np.stack(np.meshgrid(*(np.arange(n) for n in shape),indexing='ij'),axis=-1).reshape(-1,3)+.5)*step+lower
    xyz=xyz.astype(np.float32); ng=13
    means=xyz[rng.integers(len(xyz),size=ng)].copy()
    scales=rng.uniform(.1,.39,size=(ng,3)).astype(np.float32)
    inverse=np.zeros((ng,6),np.float32); inverse[:,:3]=1/scales**2
    alpha=rng.uniform(.01,1,size=ng).astype(np.float32)
    semantics=rng.uniform(.1,2,size=(ng,17)).astype(np.float32)
    labels=rng.choice([0,2,4,17,255],size=len(xyz)).astype(np.int32)
    camera=rng.random(len(xyz))>.4
    winner=rng.integers(0,17,size=len(xyz),dtype=np.int32)
    prediction=rng.choice([0,2,4,17],size=len(xyz)).astype(np.int32)
    # Exact ties exercise deterministic min-Gaussian-index ownership.
    means[1]=means[0]; scales[1]=scales[0]; inverse[1]=inverse[0]
    alpha[1]=alpha[0]; semantics[1]=semantics[0]
    return [means,scales,inverse,alpha,semantics,xyz,labels,camera,winner,prediction], shape, step, lower


def test_denominators_and_missing_groups():
    r=rates([10,5,3,4,20])
    assert r['voxels']==42 and r['gt_foreground']==18 and r['gt_empty']==24
    assert r['wrong_class_fraction']==5/42
    assert r['miss_rate_gt_foreground']==3/18
    assert r['false_positive_rate_gt_empty']==4/24
    assert all(v is None for k,v in rates([0]*5).items() if 'fraction' in k or 'rate_' in k)


def test_boundary_includes_empty_and_separates_foreground_mixing():
    bits=torch.tensor([[0,1<<3,(1<<3)|(1<<17),(1<<3)|(1<<4),(1<<3)|(1<<4)|(1<<17)]],dtype=torch.int32)
    c=gaussian_counts(bits)[0]
    assert c['predicted']==5 and c['supported']==4
    assert c['boundary_any']==3 and c['boundary_foreground']==2 and c['boundary_fg_empty']==2


def test_scene_pooling_uses_raw_counts():
    first=rates([1,1,0,0,0]); second=rates([98,0,0,0,0])
    combined=rates(np.array([1,1,0,0,0])+np.array([98,0,0,0,0]))
    assert combined['wrong_class_fraction']==.01
    assert combined['wrong_class_fraction'] != (first['wrong_class_fraction']+second['wrong_class_fraction'])/2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_matches_independent_numpy_including_overlap_and_attribution():
    from tools.boundary_gaussian_stats import extension
    values,shape,step,lower=fixture()
    expected=reference(*values,shape,step,3.,lower)
    actual=extension().diagnose(*[torch.as_tensor(a,device='cuda').contiguous() for a in values],
                                *shape,float(step),3.,*map(float,lower))
    for i in (0,1,2,4):
        np.testing.assert_array_equal(actual[i].cpu().numpy(),expected[i])
    packed=actual[3].cpu().numpy().view(np.uint64)
    owners=np.where(packed>0,0xffffffff-(packed&0xffffffff),-1).astype(np.int64)
    np.testing.assert_array_equal(owners,expected[3])
    # Baseline = disjoint boundary-dominant + nonboundary-dominant + unowned.
    count=actual[4].cpu().numpy()
    np.testing.assert_array_equal(count[:,0],count[:,4]+count[:,7]+count[:,8])
    assert (count[:,1]>=count[:,4]).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('condition',['pure_empty','zero_opacity','invalid_gt'])
def test_no_boundary_and_no_support_cases(condition):
    from tools.boundary_gaussian_stats import extension
    values,shape,step,lower=fixture()
    if condition=='pure_empty':
        values[6][:]=17
        values[7][:]=False
    elif condition=='zero_opacity':
        values[3][:]=0
    else:
        values[6][:]=255
    expected=reference(*values,shape,step,3.,lower)
    actual=extension().diagnose(*[torch.as_tensor(a,device='cuda').contiguous() for a in values],
                                *shape,float(step),3.,*map(float,lower))
    for i in (0,1,2,4):
        np.testing.assert_array_equal(actual[i].cpu().numpy(),expected[i])
    assert not actual[1].any()
    assert not actual[2].any()
    if condition in ('zero_opacity','invalid_gt'):
        assert not actual[0].any() and not actual[3].any()
