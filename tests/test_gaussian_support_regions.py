"""New support groups and independently evaluated geometry mass."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from tools.stat_gaussian_support_regions import gaussian_types,native_mass


def test_support_groups_partition_foreground_empty_and_unobserved():
    bits=torch.tensor([0,1<<2,1<<17,(1<<2)|(1<<17),(1<<2)|(1<<5),
                       (1<<2)|(1<<5)|(1<<17)],dtype=torch.int32)
    assert gaussian_types(bits).tolist()==[4,2,3,1,0,0]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('zero_alpha',[False,True])
def test_native_geometry_mass_matches_independent_enumeration(zero_alpha):
    shape=(6,5,4);step=.4;lower=np.array([-1.,-1.,-1.],np.float32)
    xyz=((np.stack(np.meshgrid(*(np.arange(n) for n in shape),indexing='ij'),-1).reshape(-1,3)+.5)*step+lower).astype(np.float32)
    means=xyz[[0,38,73]].copy();scales=np.array([[.1,.2,.3],[.4,.2,.1],[.25,.3,.2]],np.float32)
    inv=np.zeros((3,6),np.float32);inv[:,:3]=1/scales**2
    alpha=np.array([.8,0.,.2],np.float32)
    if zero_alpha:alpha[:]=0
    camera=np.arange(len(xyz))%3!=0
    reference=np.zeros((2,len(xyz)),np.float64)
    for g in range(3):
        if alpha[g]<=0:continue
        center=((means[g]-lower)/step).astype(int);radius=int(np.ceil(scales[g].max()*3/step))
        cells=((xyz-lower)/step).astype(int)
        inside=(np.abs(cells-center)<=radius).all(1)&camera
        kernel=np.exp(-.5*((xyz-means[g])**2*inv[g,:3]).sum(1))
        reference[0,inside]+=kernel[inside];reference[1,inside]+=alpha[g]*kernel[inside]
    from model.head.localagg.local_aggregate import LocalAggregator
    aggregator=LocalAggregator(scale_multiplier=3.,H=shape[0],W=shape[1],D=shape[2],
        pc_min=lower.tolist(),grid_size=step)
    # Append a dummy fixed empty Gaussian: native_mass must exclude it.
    full_means=np.concatenate((means,means[:1]));full_scales=np.concatenate((scales,scales[:1]))
    full_alpha=np.concatenate((alpha,np.ones(1,np.float32)))
    covariance_inverse=np.stack([np.diag(v**-2) for v in full_scales])
    args=[torch.tensor(v,device='cuda').unsqueeze(0).contiguous() for v in
        (full_means,full_alpha[:,None],np.ones((4,18),np.float32),full_scales,covariance_inverse)]
    head=SimpleNamespace(prepare_gaussian_args=lambda _:args,aggregator=aggregator.cuda())
    actual=native_mass(head,dict(gaussian=None,sampled_xyz=torch.tensor(xyz,device='cuda').unsqueeze(0)))
    np.testing.assert_allclose(actual.cpu().numpy()[:,camera],reference[:,camera],atol=2e-6,rtol=2e-6)
