"""Four prespecified training-frame checks, no validation selection or tuning."""
from pathlib import Path
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.classwise_chamfer_experiment import *


def main():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from loss import OPENOCC_LOSS
    c,net,opt,objective,state=build(ARMS[0],training=True);del opt,state
    b=config('baseline')
    assert c.model==b.model and c.optimizer==b.optimizer
    assert len(c.loss.loss_cfgs)==2 and c.loss.loss_cfgs[1]==b.loss.loss_cfgs[1]
    before=dict(b.loss.loss_cfgs[0]);after=dict(c.loss.loss_cfgs[0])
    after.pop('geometry_class_weights');after['type']=before['type'];assert before==after
    base_set=OPENOCC_LOSS.build(b.loss.loss_cfgs[0]).cuda()
    wrapped=TrainingFrames(OPENOCC_DATASET.build(c.train_dataset_config),c.seed)
    indices=np.random.default_rng(c.seed).permutation(4096)[:4].tolist();records=[]
    for index in indices:
        seed_all(c.seed*1000000+index)
        batch=to_cuda(custom_collate_fn_temporal([wrapped[index]]));out=forward(net,batch);inputs=loss_input(c,out,batch)
        parts,diag=objective.losses[0].components(inputs)
        original,original_diag=base_set.components(inputs)
        diffs={k:float((parts[k]-v).detach()) for k,v in original.items() if k.endswith('/classification')}
        assert max(abs(v) for v in diffs.values())==0
        p=net.encoder.layers[-1].reg_branch[-1].weight
        new_geom=sum(v for k,v in parts.items() if not k.endswith('/classification'))
        old_geom=sum(v for k,v in original.items() if not k.endswith('/classification'))
        ng=torch.autograd.grad(new_geom,p,retain_graph=True)[0]
        og=torch.autograd.grad(old_geom,p,retain_graph=True)[0]
        value,details=objective(inputs);value.backward()
        assert torch.isfinite(value) and all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None)
        row=dict(index=index,classification_maximum_difference=max(abs(v) for v in diffs.values()),
            new_geometry=float(new_geom.detach()),original_geometry=float(old_geom.detach()),total_loss=float(value.detach()),
            final_regression_gradient_ratio=float(ng.norm()/og.norm()),
            final_regression_gradient_cosine=float((ng*og).sum()/(ng.norm()*og.norm()).clamp_min(1e-20)),
            diagnostics={k:float(v) for k,v in diag.items()},full_backward_finite=True)
        records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='diagnostics'}),flush=True)
        net.zero_grad(set_to_none=True)
        del batch,out,inputs,parts,original,new_geom,old_geom,ng,og,value,details,diag,original_diag
        gc.collect();torch.cuda.empty_cache()
    write(RUN/'pilot.json',dict(status='passed',training_indices=indices,records=records,
        original_classification_exact=True,original_occupancy_and_model_and_optimizer_preserved=True,
        only_two_losses_no_previous_extras=True,all_six_geometry_stages_replaced=True,no_validation_tuning=True))
    net.flush_memory_phase()

if __name__=='__main__':main()
