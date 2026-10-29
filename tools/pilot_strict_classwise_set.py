"""Four fixed training frames: frozen geometry, changed focal, finite backward."""
from pathlib import Path
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.strict_classwise_set_experiment import *


def main():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from loss import OPENOCC_LOSS
    c,net,opt,objective,state=build(ARMS[0],training=True);del opt,state
    previous=config('classwise');baseline=config('baseline')
    for other in (previous,baseline):
        assert c.model==other.model and c.optimizer==other.optimizer
        assert c.loss.loss_cfgs[1]==other.loss.loss_cfgs[1]
        assert c.train_dataset_config==other.train_dataset_config
    new=dict(c.loss.loss_cfgs[0]);old=dict(previous.loss.loss_cfgs[0])
    new['type']=old['type'];assert new==old
    prior=OPENOCC_LOSS.build(previous.loss.loss_cfgs[0]).cuda()
    dataset=TrainingFrames(OPENOCC_DATASET.build(c.train_dataset_config),c.seed)
    indices=np.random.default_rng(c.seed).permutation(4096)[:4].tolist();records=[]
    for index in indices:
        seed_all(c.seed*1000000+index)
        batch=to_cuda(custom_collate_fn_temporal([dataset[index]]))
        out=forward(net,batch);inputs=loss_input(c,out,batch)
        parts,diag=objective.losses[0].components(inputs);original,_=prior.components(inputs)
        diffs={k:float((parts[k]-v).detach()) for k,v in original.items() if not k.endswith('/classification')}
        assert max(abs(v) for v in diffs.values())==0
        p=net.encoder.layers[-1].reg_branch[-1].weight
        ng=torch.autograd.grad(parts['stage_5/pred_to_gt']+parts['stage_5/gt_to_pred'],p,retain_graph=True)[0]
        og=torch.autograd.grad(original['stage_5/pred_to_gt']+original['stage_5/gt_to_pred'],p,retain_graph=True)[0]
        torch.testing.assert_close(ng,og,atol=0,rtol=0)
        value,details=objective(inputs);value.backward()
        assert torch.isfinite(value) and all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None)
        row=dict(index=index,geometry_maximum_difference=max(abs(v) for v in diffs.values()),
            geometry_gradient_exact=True,total_loss=float(value.detach()),
            old_classification=float(sum(v for k,v in original.items() if k.endswith('/classification')).detach()),
            new_classification=float(sum(v for k,v in parts.items() if k.endswith('/classification')).detach()),
            diagnostics={k:float(v) for k,v in diag.items()},full_backward_finite=True)
        records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='diagnostics'}),flush=True)
        net.zero_grad(set_to_none=True)
        del batch,out,inputs,parts,original,ng,og,value,details,diag
        gc.collect();torch.cuda.empty_cache()
    write(RUN/'pilot.json',dict(status='passed',training_indices=indices,records=records,
        geometry_all_six_stages_exact=True,original_occupancy_and_model_and_optimizer_preserved=True,
        classification_is_sameclass_macro_focal=True,no_validation_tuning=True))
    net.flush_memory_phase()

if __name__=='__main__':main()
