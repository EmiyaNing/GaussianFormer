"""Training-only gradient calibration and real-frame audits before freezing protocol."""
import gc
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.semantic_field_experiment_common import *


def pilot():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from loss.semantic_gaussian_field_loss import SemanticGaussianFieldLoss
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    c,net,opt,objective,state=build('geometry',training=True)
    del opt,state
    dataset=OPENOCC_DATASET.build(c.train_dataset_config)
    assert len(dataset)==4096 and len({k[0] for k in dataset.keyframes})==698
    wrapped=TrainingFrames(dataset,c.seed)
    order=np.random.default_rng(c.seed).permutation(len(dataset)).tolist()
    chosen=order[:4]
    modes={a:SemanticGaussianFieldLoss(mode=a,weight=1.,seed=c.seed).cuda() for a in ('geometry','semantic')}
    raw=[]
    handle=net.encoder.attribute_branches['5'].net[-1].register_forward_hook(lambda m,i,o:raw.append(o))
    rows=[]
    for index in chosen:
        seed_all(c.seed*1000000+index)
        batch=to_cuda(custom_collate_fn_temporal([wrapped[index]]))
        out=forward(net,batch);inp=loss_input(c,out,batch);inp['global_iter']=512
        pieces=[part(inp) for part in list(objective.losses)[:2]]
        original=sum(v[0] if isinstance(v,tuple) else v for v in pieces)
        param=net.encoder.layers[-1].reg_branch[-1].weight
        base_grad=torch.autograd.grad(original,param,retain_graph=True)[0].norm()
        row=dict(index=index,original_loss=float(original.detach()),original_geometry_parameter_grad_norm=float(base_grad),arms={})
        for a,loss in modes.items():
            torch.cuda.synchronize();start=__import__('time').monotonic()
            value,metrics=loss(inp)
            torch.cuda.synchronize();seconds=__import__('time').monotonic()-start
            variables=[param,out['gaussian'].means,out['gaussian'].scales,out['gaussian'].rotations,raw[-1],out['gaussian'].semantics,out['gaussian'].opacities]
            grads=torch.autograd.grad(value,variables,retain_graph=True,allow_unused=True)
            assert grads[5] is None and grads[6] is None,'Field objective has a direct semantic/opacity gradient'
            norms={name:float(v.norm()) if v is not None else None for name,v in zip(
                ('geometry_parameter','physical_center','physical_scale','rotation','raw_attribute','semantic','opacity'),grads)}
            assert all(torch.isfinite(v).all() for v in grads if v is not None)
            assert norms['physical_center']>0 and norms['physical_scale']>0 and norms['rotation']>0
            raw_scale=grads[4][...].reshape(-1,8)[:,:3]
            # Amplitude changes must not directly lower either field objective.
            with torch.no_grad():
                g=out['gaussian']
                altered=GaussianPrediction(g.means,g.scales,g.rotations,g.opacities*.1,g.semantics*5.)
                again,_=loss(dict(inp,gaussian=altered))
                torch.testing.assert_close(again,value.detach(),atol=2e-6,rtol=2e-6)
            row['arms'][a]=dict(loss=float(value.detach()),seconds=seconds,gradient_norms=norms,
                raw_scale_gradient_norm=float(raw_scale.norm()),raw_scale_gradient_nonzero_fraction=float((raw_scale!=0).float().mean()),
                diagnostics={k:float(v) for k,v in metrics.items()},amplitude_opacity_invariant=True)
            loss.last_raw=None;loss.last_geometry=None
        rows.append(row)
        print(json.dumps(row),flush=True)
        del out,batch,inp,value,original,grads,raw_scale,g,altered,again
        raw.clear();gc.collect();torch.cuda.empty_cache()
    handle.remove()
    # Same scalar for G/S, selected on four TRAIN frames only. The stronger
    # median geometry-parameter gradient is capped at 10% of original objective.
    ratios=[]
    for row in rows:
        largest=max(row['arms'][a]['gradient_norms']['geometry_parameter'] for a in modes)
        ratios.append(.1*row['original_geometry_parameter_grad_norm']/max(largest,1e-12))
    weight=float(np.clip(np.median(ratios),.005,.5))
    result=dict(status='passed',training_indices=chosen,rows=rows,shared_field_weight=weight,
        selection='0.1 times original final reg-branch gradient / max(G,S) field gradient; median of 4 train frames, clipped [0.005,0.5]',
        same_camera_set_start=True,amplitude_opacity_invariance=True)
    write(RUN/'pilot.json',result)
    for a in ('geometry','semantic'):
        path=ROOT/f'config/mini_semantic_field/small_{a}_1ep.py'
        cfg=Config.fromfile(path);cfg.loss.loss_cfgs[-1].weight=weight;cfg.dump(path)
    print(json.dumps(dict(status='passed',shared_field_weight=weight)),flush=True)


if __name__=='__main__':pilot()
