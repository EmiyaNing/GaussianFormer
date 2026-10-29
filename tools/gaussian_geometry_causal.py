"""Paired, forward-identical Occupancy-gradient interventions (additive renderer).

This is an independent experimental runner. Production model/loss sources and
previous frozen statistical experiments are deliberately left unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config

RUN = ROOT / 'work_dirs/gaussian_geometry_causal'
CONFIG = ROOT / 'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py'
ARMS = ('standard', 'no_amplitude', 'no_geometry', 'no_both', 'no_semantic_amplitude')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DirectionOnly(torch.autograd.Function):
    """Identity values; backward of stopgrad(A) * (a/A), A=sum(a).

    This removes the explicit renderer amplitude route, not all possible
    changes to A under Set loss, shared features or Adam's coordinate metric.
    """
    @staticmethod
    def forward(ctx, a):
        p = a / a.sum(-1, keepdim=True).clamp_min(torch.finfo(a.dtype).tiny)
        ctx.save_for_backward(p)
        return a.view_as(a)

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        return g - (g * p).sum(-1, keepdim=True)


class Routes:
    def __init__(self, network, arm='standard'):
        self.arm = arm
        self.enabled = True
        self.original = network.head.prepare_gaussian_args
        self.head = network.head
        def prepare(head, gaussian):
            means, alpha, sem, scales, inverse = self.original(gaussian)
            if not self.enabled:
                return means, alpha, sem, scales, inverse
            if self.arm in ('no_amplitude', 'no_both', 'no_semantic_amplitude'):
                # Do not project the appended empty Gaussian or its scalar.
                sem = torch.cat((torch.cat((DirectionOnly.apply(sem[:, :-1, :17]),
                    sem[:, :-1, 17:]), -1), sem[:, -1:]), 1)
            if self.arm in ('no_amplitude', 'no_both'):
                alpha = torch.cat((alpha[:, :-1].detach(), alpha[:, -1:]), 1)
            if self.arm in ('no_geometry', 'no_both'):
                means = torch.cat((means[:, :-1].detach(), means[:, -1:]), 1)
                scales = torch.cat((scales[:, :-1].detach(), scales[:, -1:]), 1)
                inverse = torch.cat((inverse[:, :-1].detach(), inverse[:, -1:]), 1)
            return means, alpha, sem, scales, inverse
        network.head.prepare_gaussian_args = types.MethodType(prepare, network.head)

    def close(self):
        self.head.prepare_gaussian_args = self.original


def numerical_audit():
    seed_all(928)
    a = torch.exp(torch.randn(2, 23, 17, dtype=torch.float64)).requires_grad_()
    g = torch.randn_like(a)
    value = DirectionOnly.apply(a)
    actual, = torch.autograd.grad((value*g).sum(), a)
    reference = a.sum(-1, keepdim=True).detach() * a / a.sum(-1, keepdim=True)
    expected, = torch.autograd.grad((reference*g).sum(), a)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    assert torch.equal(value, a)
    # Uniform radial scaling of a changes amplitude only; its VJP must vanish.
    orth = (actual*a).sum(-1).abs().max().item()
    assert orth < 1e-12
    result = dict(status='passed', identity_forward_exact=True,
        reference_vjp_max_error=(actual-expected).abs().max().item(), radial_dot_max=orth)
    write(RUN/'numerical_audit.json', result)
    return result


def cfg():
    c = Config.fromfile(CONFIG)
    # Retain checkpointing initially; the pilot records real training throughput.
    return c


def build(initial=None, freeze_empty=False):
    from mmseg.models import build_segmentor
    from mmengine.optim import build_optim_wrapper
    from loss import OPENOCC_LOSS
    import model
    c = cfg()
    net = build_segmentor(c.model)
    net.init_weights()
    if initial:
        state = torch.load(initial, map_location='cpu', weights_only=False)
        net.load_state_dict(state['state_dict'], strict=True)
    else:
        state = None
        pre = torch.load(ROOT/c.load_from, map_location='cpu', weights_only=False)
        weights = pre.get('state_dict', pre)
        revised = {('img_backbone.'+k[len('backbone.'):] if k.startswith('backbone.') else k):v
                   for k,v in weights.items()}
        loaded = net.load_state_dict(revised, strict=False)
        assert not any(k.startswith('img_backbone.') for k in loaded.missing_keys)
    net = net.cuda()
    opt = build_optim_wrapper(net, c.optimizer)
    if state is not None and 'optimizer' in state:
        opt.load_state_dict(state['optimizer'])
    # Preserve all original optimizer parameter groups when restoring moments.
    # MMEngine omits frozen parameters if they are frozen before construction.
    if freeze_empty:
        for name, p in net.named_parameters():
            if name in ('head.empty_scalar', 'head.empty_bias'):
                p.requires_grad_(False)
    loss = OPENOCC_LOSS.build(c.loss).cuda()
    return c, net, opt, loss, state


def to_cuda(batch):
    return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def forward(net, batch):
    return net(imgs=batch['img'], metas={k:v for k,v in batch.items() if k!='img'})


def loss_input(c, out, batch):
    value = {'metas':batch, 'global_iter':0}
    for key, source in c.loss_input_convertion.items():
        value[key] = out[source]
    return value


def loader(split, seed, steps=None, augmentation=True):
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Sampler
    c = cfg()
    settings = copy.deepcopy(c[('train' if augmentation else 'val')+'_dataset_config'])
    settings.imageset = c[split+'_dataset_config'].imageset
    if not augmentation:
        settings.phase = 'val'
    dataset = OPENOCC_DATASET.build(settings)
    order = np.random.default_rng(seed).permutation(len(dataset)).tolist() if augmentation else list(range(len(dataset)))
    if steps is not None:
        order = (order*((steps+len(order)-1)//len(order)))[:steps]
    class FixedSampler(Sampler):
        def __iter__(self): return iter(order)
        def __len__(self): return len(order)
    def init_worker(worker):
        seed_all(seed+100000+worker)
    data = DataLoader(dataset,batch_size=1,sampler=FixedSampler(),num_workers=4,
        collate_fn=custom_collate_fn_temporal,pin_memory=True,worker_init_fn=init_worker,
        persistent_workers=True,prefetch_factor=2)
    return dataset, data, order


def digest_batch(batch):
    h=hashlib.sha256()
    for key in ('img','projection_mat','occ_label','occ_cam_mask'):
        h.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gaussian_snapshot(out):
    g=out['gaussian']
    return {name:getattr(g,name).detach().cpu().clone() for name in
            ('means','scales','rotations','opacities','semantics')}


def training(arm, steps, seed, initial=None, accumulation=8, pilot=False):
    assert steps % accumulation == 0
    start=time.perf_counter()
    source_at_start=sha(__file__)
    seed_all(seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    folder=RUN/('pilot' if pilot else ('train' if initial else 'common'))/f'{arm}_s{seed}'
    if (folder/'complete.json').exists():
        existing=json.loads((folder/'complete.json').read_text())
        assert existing['initial']==(str(initial) if initial else None)
        assert existing['microbatches']==steps and existing['seed']==seed
        return existing
    folder.mkdir(parents=True,exist_ok=True)
    c,net,opt,loss,state=build(initial,freeze_empty=initial is not None)
    control=Routes(net,arm)
    dataset,data,order=loader('train',seed+(300000 if initial else 0),steps)
    seed_all(seed+200000)  # Network dropout independent of loader construction.
    opt.zero_grad();net.train()
    trace=[];reference=None
    # Warmup only for a newly initialized common trunk. Branches share a fixed,
    # original-group-ratio LR; optimizer moments and update counts are restored.
    # The maximum group is empty_bias at 25x. Use configured LR per group instead.
    group_lr=[]
    for group in opt.param_groups:
        configured=group.get('initial_lr',group['lr'])
        group_lr.append(configured)
    for i,batch in enumerate(data):
        fingerprint=digest_batch(batch)
        batch=to_cuda(batch)
        out=forward(net,batch)
        total,details=loss(loss_input(c,out,batch))
        if reference is None:
            np.savez_compressed(folder/'initial_forward.npz',
                logits=out['pred_occ'][-1].detach().cpu().numpy(),
                **{k:v.numpy() for k,v in gaussian_snapshot(out).items()})
            reference=True
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss')
        (total/accumulation).backward()
        row=dict(microbatch=i,index=order[i],fingerprint=fingerprint,total=float(total.detach()),
                 losses={k:float(v) for k,v in details.items()})
        if (i+1)%32==0 or i==0:
            g=out['gaussian']
            row['gaussian']=dict(amplitude=float(g.semantics.detach().sum(-1).mean()),
                opacity=float(g.opacities.detach().mean()),
                scale_mean=float(g.scales.detach().mean()),
                scale_max_saturated_fraction=float((g.scales.detach().max(-1).values>=.3999).float().mean()))
        if (i+1)%accumulation==0:
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),c.grad_max_norm,error_if_nonfinite=True)
            # Ratio-preserving LR. This independent diagnostic schedule is not
            # a continuation of the production 30-epoch cosine training.
            factor=(1/3+(2/3)*min((i+1)/min(steps,512),1)) if initial is None else .5
            for group,rate in zip(opt.param_groups,group_lr):group['lr']=rate*factor
            opt.step();opt.zero_grad()
            row.update(optimizer_step=(i+1)//accumulation,grad_norm=float(norm))
        trace.append(row)
        if (i+1)%32==0 or i==0:
            elapsed=time.perf_counter()-start
            progress=dict(status='running',arm=arm,seed=seed,microbatches=i+1,total=steps,
                seconds=elapsed,seconds_per_microbatch=elapsed/(i+1),loss=row['total'])
            write(folder/'progress.json',progress)
            print(json.dumps(progress),flush=True)
        del out,total,batch
    checkpoint=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),
        microbatches=steps,optimizer_steps=steps//accumulation,seed=seed,arm=arm,
        initial=str(initial) if initial else None)
    torch.save(checkpoint,folder/'final.pth')
    with (folder/'trace.jsonl').open('w') as f:
        for row in trace:f.write(json.dumps(row)+'\n')
    result=dict(status='complete',arm=arm,seed=seed,microbatches=steps,
        optimizer_steps=steps//accumulation,seconds=time.perf_counter()-start,
        initial=str(initial) if initial else None,initial_sha256=sha(initial) if initial else None,
        source_sha256=source_at_start,data_indices_sha256=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),
        checkpoint_sha256=sha(folder/'final.pth'),peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        empty_bias=float(net.head.empty_bias.detach()),empty_scalar=float(net.head.empty_scalar.detach()))
    write(folder/'complete.json',result)
    control.close()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','train'])
    p.add_argument('--arm',choices=ARMS,default='standard')
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--seed',type=int,default=928)
    p.add_argument('--initial',type=Path)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--accumulation',type=int,default=8)
    args=p.parse_args()
    if args.action=='audit':print(numerical_audit())
    elif args.action=='train':print(training(args.arm,args.steps,args.seed,args.initial,args.accumulation,args.pilot))


if __name__=='__main__':main()
