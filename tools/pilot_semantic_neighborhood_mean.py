"""Prespecified training-frame sanity checks; no validation-based tuning."""
from pathlib import Path
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.semantic_neighborhood_experiment import *

def main():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    arm=ARMS[0];c,net,opt,objective,state=build(arm,training=True);del opt,state
    baseline=config('baseline')
    assert c.model==baseline.model and c.optimizer==baseline.optimizer
    assert c.loss.loss_cfgs[:2]==baseline.loss.loss_cfgs and len(c.loss.loss_cfgs)==3
    assert c.loss.loss_cfgs[0].final_geometry_weight==1 and c.loss.loss_cfgs[0].final_classification_weight==1
    wrapped=TrainingFrames(OPENOCC_DATASET.build(c.train_dataset_config),c.seed)
    indices=np.random.default_rng(c.seed).permutation(4096)[:4].tolist();records=[]
    module=objective.losses[-1]
    for index in indices:
        seed_all(c.seed*1000000+index)
        batch=to_cuda(custom_collate_fn_temporal([wrapped[index]]));out=forward(net,batch);inputs=loss_input(c,out,batch)
        original=sum((v[0] if isinstance(v,tuple) else v) for v in [p(inputs) for p in objective.losses[:2]])
        value,diag=module(inputs)
        p=net.encoder.layers[-1].reg_branch[-1].weight;g=out['gaussian']
        variables=[p,g.means,g.scales,g.rotations,g.semantics,g.opacities]
        grads=torch.autograd.grad(value,variables,retain_graph=True,allow_unused=True)
        assert grads[0] is not None and grads[1] is not None
        assert all(x is None for x in grads[2:]),'Only center has direct added-loss gradients'
        original_grad=torch.autograd.grad(original,p,retain_graph=True)[0]
        assert all(torch.isfinite(x).all() for x in grads if x is not None)
        assert float(grads[0].norm())>0
        (original+value).backward()
        assert all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None)
        with torch.no_grad():
            altered=GaussianPrediction(g.means,g.scales,g.rotations,g.opacities*.1,g.semantics*5.)
            again,_=module(dict(inputs,gaussian=altered))
            torch.testing.assert_close(again,value.detach(),atol=2e-6,rtol=2e-6)
        row=dict(index=index,original_loss=float(original.detach()),added_loss=float(value.detach()),
            direct_gradient_norms={key:float(v.norm()) if v is not None else None for key,v in zip(
                ('final_regression','center','scale','rotation','semantics','opacity'),grads)},
            gradient_ratio_to_original=float(grads[0].norm()/original_grad.norm()),
            cosine_with_original=float((grads[0]*original_grad).sum()/(grads[0].norm()*original_grad.norm()).clamp_min(1e-20)),
            diagnostics={k:float(v) for k,v in diag.items()},amplitude_opacity_invariant=True)
        records.append(row);print(json.dumps(row),flush=True)
        net.zero_grad(set_to_none=True)
        del batch,out,inputs,original,value,grads,original_grad,g,altered,again,variables
        gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed',training_indices=indices,records=records,
        original_CD_and_classification_preserved=True,previous_added_losses_removed=True,
        original_model_and_optimizer_preserved=True,weights=dict(mean3=.5,uncovered1=.5),
        no_validation_tuning=True,full_combined_backward_finite=True)
    write(RUN/'pilot.json',result);net.flush_memory_phase()

if __name__=='__main__':main()
