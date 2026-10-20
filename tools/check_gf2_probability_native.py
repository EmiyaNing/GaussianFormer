"""One native accumulated optimizer update before formal probability training."""
from pathlib import Path
import runpy
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.gf2_probability_experiment import RUN,write,read


class PilotComplete(Exception):pass


def main():
    from mmengine import Config
    import model.head.opus_gf2_probability_head as head
    folder=RUN/'native_pilot';folder.mkdir(exist_ok=True)
    if (folder/'microbatch_trace.jsonl').exists():
        raise RuntimeError('Pilot trace exists; inspect before explicit fresh retry')
    cfg=Config.fromfile('config/gf2_probability_decomposition/probability.py')
    cfg.loss.trace_dir=str(folder.relative_to(ROOT));cfg.dump(folder/'pilot.py')
    original_forward=head.OPUSGF2ProbabilityHead.forward;checks={}
    def checked_forward(self,*args,**kwargs):
        result=original_forward(self,*args,**kwargs)
        if not checks:
            p=result['pred_occ'][-1];q=result['conditional_semantics'][-1]
            assert torch.isfinite(p).all() and (p>=0).all() and (p<=1).all()
            normalization=float((p.sum(1)-1).abs().max().detach())
            assert normalization<2e-5
            # With geometry held fixed, empty likelihood has no output-logit
            # path into semantic values. Shared backbone/geometry gradients
            # are a separate question and are not claimed to vanish.
            empty_nll=-p[:,17].clamp_min(1e-6).log().mean()
            grad=torch.autograd.grad(empty_nll,result['opus_pred_logits'][-1],retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.abs().max()==0
            xyz=result['sampled_xyz'][0]
            cells=((xyz-xyz.new_tensor([-40,-40,-1]))/.4).long()
            codes=(cells[:,0]*200+cells[:,1])*16+cells[:,2]
            assert len(torch.unique(codes))==len(codes)==640000
            checks.update(probability_sum_max_error=normalization,empty_nll_semantic_gradient_max=float(grad.abs().max()),
                unique_query_cells=len(codes),conditional_sum_error=float((q.sum(1)-1).abs().max().detach()))
        return result
    head.OPUSGF2ProbabilityHead.forward=checked_forward
    original_step=torch.optim.AdamW.step
    def step(self,*args,**kwargs):
        assert all(torch.isfinite(p.grad).all() for group in self.param_groups for p in group['params'] if p.grad is not None)
        original_step(self,*args,**kwargs)
        raise PilotComplete()
    torch.optim.AdamW.step=step
    sys.argv=['tools/train_center_support.py','--py-config',str(folder/'pilot.py'),
        '--work-dir',str(folder),'--seed','928','--gradient-accumulation','4']
    try:runpy.run_path(str(ROOT/'tools/train_center_support.py'),run_name='__main__')
    except PilotComplete:pass
    else:raise RuntimeError('Native pilot did not reach optimizer update')
    rows=[__import__('json').loads(x) for x in (folder/'microbatch_trace.jsonl').read_text().splitlines()]
    assert [x['global_iter'] for x in rows]==[0,1,2,3]
    reference=ROOT/'work_dirs/empty_mechanism_history/train/B0'
    baseline=[__import__('json').loads(x) for x in (reference/'microbatch_trace.jsonl').read_text().splitlines()[:4]]
    assert [x['data_fingerprint'] for x in rows]==[x['data_fingerprint'] for x in baseline]
    a=np.load(folder/'initial_forward.npz');b=np.load(reference/'initial_forward.npz')
    np.testing.assert_array_equal(a['labels'],b['labels'])
    np.testing.assert_allclose(a['raw'],b['raw'],rtol=1e-5,atol=1e-6)
    write(folder/'pilot_success.json',dict(status='passed',microbatches=4,optimizer_updates=1,
        finite_gradients=True,paired_inputs=True,initial_raw_max_error=float(abs(a['raw']-b['raw']).max()),**checks))


if __name__=='__main__':main()
