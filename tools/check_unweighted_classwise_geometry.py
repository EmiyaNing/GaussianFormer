"""Check the corrected objective: equal geometry classes, original focal targets."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from tools.unweighted_classwise_geometry_experiment import RUN,config,write
from loss.classwise_chamfer_set_loss import ClasswiseChamferSetLoss
from loss.opus_set_loss import OPUSSetLoss

def main():
    torch.set_num_threads(4);torch.manual_seed(928)
    old,new=config('classwise'),config()
    a,b=old.to_dict(),new.to_dict()
    old_weights=a['loss']['loss_cfgs'][0].pop('geometry_class_weights')
    unit=b['loss']['loss_cfgs'][0].pop('geometry_class_weights')
    assert a==b,'The corrected ablation must differ only in geometry_class_weights'
    assert unit==[1.]*17 and old_weights==[10,5,10,5,5,10,10,5,10,5,5,1,5,1,1,2,1]
    kwargs=dict(new.loss.loss_cfgs[0]);kwargs.pop('type')
    module=ClasswiseChamferSetLoss(**kwargs)
    original=OPUSSetLoss(class_weights=list(new.loss.loss_cfgs[0].class_weights))
    predicted=torch.arange(17).repeat_interleave(2)
    labels=torch.arange(16).repeat_interleave(3)  # one GS class has no GT
    p=torch.randn(len(predicted),3,dtype=torch.double,requires_grad=True)
    g=torch.randn(len(labels),3,dtype=torch.double)
    z=(F.one_hot(predicted,17)*3.+torch.randn(len(predicted),17)*.1).double().requires_grad_()
    camera=torch.ones(len(labels),dtype=torch.bool)
    result=module._single(p,z,g,labels,camera)
    expected=original._single(p,z,g,labels,camera)
    torch.testing.assert_close(result[0],expected[0],rtol=0,atol=0)
    actual_grad=torch.autograd.grad(result[0],z,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected[0],z,retain_graph=True)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)
    assert torch.count_nonzero(actual_grad[predicted==16])>0,'Missing same-class GT must not skip original classifier'
    plosses,glosses=[],[]
    for c in range(17):
        pm,gm=predicted==c,labels==c
        if not pm.any() or not gm.any():continue
        pp,gg=p[pm],g[gm];dist=torch.cdist(pp.detach(),gg)
        p2g,g2p=dist.argmin(1),dist.argmin(0)
        plosses.append(F.smooth_l1_loss(pp,gg[p2g],beta=.2,reduction='none').sum(-1).mean())
        weight=torch.where(dist[g2p,torch.arange(len(gg))]>.2,5.,1.)
        glosses.append((F.smooth_l1_loss(gg,pp[g2p],beta=.2,reduction='none').sum(-1)*weight).mean())
    refs=(torch.stack(plosses).mean(),torch.stack(glosses).mean())
    for actual,ref in zip(result[1:3],refs):
        torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(actual,p,retain_graph=True)[0],
                                   torch.autograd.grad(ref,p,retain_graph=True)[0],rtol=1e-12,atol=1e-12)
    out=dict(status='passed',only_geometry_class_weights_changed=True,
        original_classification_weights_retained=True,original_focal_value_and_logits_gradient_exact=True,
        missing_same_class_GT_still_receives_original_focal=True,
        uniform_geometry_value_and_center_gradient_match_independent_reference=True)
    RUN.mkdir(parents=True,exist_ok=True);write(RUN/'numerical_checks.json',out)
    print(json.dumps(out))

if __name__=='__main__':main()
