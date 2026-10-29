"""Four prespecified train frames: field/original geometry-parameter gradient balance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,numpy as np,torch
from tools.semantic_field_experiment_common import *
from loss.semantic_gaussian_field_loss import SemanticGaussianFieldLoss

def main():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from loss import OPENOCC_LOSS
    indices=json.loads((RUN/'pilot.json').read_text())['training_indices'];records=[]
    modules={n:SemanticGaussianFieldLoss(mode=n,weight=.5).cuda() for n in ('geometry','semantic')}
    for arm in ('initial',*ARMS):
        checkpoint=INITIAL if arm=='initial' else RUN/'train'/arm/'epoch_1.pth'
        c,net=build('baseline',checkpoint);net.train()
        objective=OPENOCC_LOSS.build(c.loss).cuda();dataset=TrainingFrames(OPENOCC_DATASET.build(c.train_dataset_config),928)
        raw=[];handle=net.encoder.attribute_branches['5'].net[-1].register_forward_hook(lambda m,i,o:raw.append(o))
        for index in indices:
            seed_all(c.seed*1000000+index)
            batch=to_cuda(custom_collate_fn_temporal([dataset[index]]));out=forward(net,batch)
            inputs=loss_input(c,out,batch);inputs['global_iter']=512
            parts,_=objective.losses[0].components(inputs)
            geometry=sum(v for k,v in parts.items() if k.endswith(('/pred_to_gt','/gt_to_pred')))
            occupancy=objective.losses[1](inputs);occupancy=occupancy[0] if isinstance(occupancy,tuple) else occupancy
            original=sum(parts.values())+occupancy
            p=net.encoder.layers[-1].reg_branch[-1].weight
            def grad(v,variables):return torch.autograd.grad(v,variables,retain_graph=True,allow_unused=True)
            original_grad,raw_grad=grad(original,[p,raw[-1]])
            cg=grad(geometry,[p])[0];og=grad(occupancy,[p])[0]
            def cosine(a,b):return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-20))
            for mode,module in modules.items():
                value,diag=module(inputs);fg,fr=grad(value,[p,raw[-1]])
                a=fr.reshape(-1,8)[:,:3];b=raw_grad.reshape(-1,8)[:,:3]
                records.append(dict(arm=arm,index=index,mode=mode,original_grad_norm=float(original_grad.norm()),
                    center_CD_grad_norm=float(cg.norm()),occupancy_grad_norm=float(og.norm()),
                    weighted_field_grad_norm=float(fg.norm()),field_over_original=float(fg.norm()/original_grad.norm()),
                    cosine_with_original=cosine(fg,original_grad),cosine_with_CD=cosine(fg,cg),
                    cosine_with_occupancy=cosine(fg,og),field_raw_scale_norm=float(a.norm()),
                    original_raw_scale_norm=float(b.norm()),field_raw_scale_nonzero_fraction=float((a!=0).float().mean()),
                    known_sample_fraction=float(diag['known_sample_fraction'])))
                module.last_raw=None;module.last_geometry=None
            raw.clear();del out,batch,inputs,parts,geometry,occupancy,original,original_grad,raw_grad,cg,og,fg,fr,a,b,value
        handle.remove();net.flush_memory_phase();del net,objective,dataset;torch.cuda.empty_cache()
    write(RUN/'gradient_balance_audit.json',dict(status='complete',indices=indices,records=records,
        source_sha256=sha(Path(__file__)),interpretation='Local unpreconditioned gradients on four prespecified train frames; not a causal decomposition of AdamW trajectory or full validation.'))

if __name__=='__main__':main()
