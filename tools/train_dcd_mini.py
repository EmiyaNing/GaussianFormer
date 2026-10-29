"""One-epoch 4096-frame DCD geometry fine-tuning; see run_dcd_mini_experiment.py."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import Dataset, DataLoader, Sampler
import tools.gaussian_geometry_causal as core

DEFAULT_CONFIG='config/mini_dcd_finetune/opusv1_gaussian_tiny_8f_dcd_1ep.py'


def numpy_rng_checkpoint():
    name,keys,position,has_gauss,cached=np.random.get_state()
    # Native eval.py uses torch.load's weights_only default. Keep recovery
    # metadata limited to tensors and basic Python types, including RNG keys.
    return (name,torch.from_numpy(keys.astype(np.int64)),position,has_gauss,cached)


def restore_numpy_rng(value):
    name,keys,position,has_gauss,cached=value
    np.random.set_state((name,keys.cpu().numpy().astype(np.uint32),position,has_gauss,cached))


class TrainingFrames(Dataset):
    def __init__(self, dataset, seed):
        self.dataset=dataset;self.seed=seed
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        np.random.seed(self.seed*1000000+index)
        random.seed(self.seed*1000000+index)
        return self.dataset[index]


def check_config(path):
    c=Config.fromfile(path)
    original=Config.fromfile(ROOT/'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py')
    assert c.model==original.model
    assert len(c.loss.loss_cfgs)==2
    assert c.loss.loss_cfgs[1]==original.loss.loss_cfgs[1]
    first=dict(c.loss.loss_cfgs[0]);first['type']='OPUSSetLoss'
    first={k:v for k,v in first.items() if not k.startswith('dcd_')}
    assert first==original.loss.loss_cfgs[0]
    assert c.max_epochs==1 and c.gradient_accumulation==4
    for p in [c.train_load_from,c.train_dataset_config.imageset,c.val_dataset_config.imageset]:
        if not (ROOT/p).is_file():raise FileNotFoundError(ROOT/p)
    return c


def configure(args):
    core.CONFIG=args.py_config
    return check_config(args.py_config)


def training(args):
    c=configure(args)
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    folder=args.work_dir/'train';folder.mkdir(parents=True,exist_ok=True)
    initial=(ROOT/c.train_load_from).resolve()
    files=[args.py_config,Path(__file__),ROOT/'loss/opus_set_dcd_loss.py',
           ROOT/'loss/opus_set_loss.py',ROOT/'tools/gaussian_geometry_causal.py']
    protocol=dict(config_sha256=hashlib.sha256(c.pretty_text.encode()).hexdigest(),
                  initial_sha256=core.sha(initial),epochs=1,seed=c.seed,
                  train_frames=c.expected_train_frames,val_frames=c.expected_val_frames,
                  source_sha256={str(p.relative_to(ROOT)):core.sha(p) for p in files})
    if (args.work_dir/'protocol.json').exists():
        assert json.loads((args.work_dir/'protocol.json').read_text())==protocol,'Configuration/source changed; use a new work directory'
    else:
        core.write(args.work_dir/'protocol.json',protocol)
        c.dump(args.work_dir/'resolved_config.py')
    if (folder/'complete.json').exists():return
    core.seed_all(c.seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    latest=folder/'latest.pth'
    source=latest if latest.exists() else initial
    c,net,opt,objective,state=core.build(source,freeze_empty=False)
    if 'optimizer' not in state:raise ValueError('Checkpoint must contain AdamW state')
    completed=int(state.get('finetune_microbatches',0)) if source==latest else 0
    updates=int(state.get('finetune_optimizer_updates',0)) if source==latest else 0
    group_bases=state.get('finetune_group_base_lrs') or [g.get('initial_lr',g['lr']) for g in opt.param_groups]
    dataset=OPENOCC_DATASET.build(c.train_dataset_config)
    assert len(dataset)==c.expected_train_frames
    assert len({x[0] for x in dataset.keyframes})==c.expected_train_scenes
    order=np.random.default_rng(c.seed).permutation(len(dataset)).tolist()
    order_sha=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest()
    if source==latest:assert state['data_order_sha256']==order_sha
    trace_path=folder/'trace.jsonl'
    if trace_path.exists():
        # A crash may leave executed frames after the last saved optimizer
        # state. Preserve that trace, then resume a single checkpoint-aligned
        # history rather than silently duplicating those frame numbers.
        lines=trace_path.read_text().splitlines()
        kept=[]
        for line in lines:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row['microbatch']<=completed:kept.append(line)
        if len(kept)!=len(lines):
            trace_path.rename(folder/f'trace_before_resume_{time.time_ns()}.jsonl')
            trace_path.write_text('\n'.join(kept)+ ('\n' if kept else ''))
    class FixedSampler(Sampler):
        def __iter__(self):return iter(order[completed:])
        def __len__(self):return len(order)-completed
    loader=DataLoader(TrainingFrames(dataset,c.seed),batch_size=1,sampler=FixedSampler(),
        num_workers=c.train_loader.num_workers,collate_fn=custom_collate_fn_temporal,
        pin_memory=True,**(dict(persistent_workers=True,prefetch_factor=2) if c.train_loader.num_workers else {}))
    if source==latest:
        torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
        restore_numpy_rng(state['rng_numpy']);random.setstate(state['rng_random'])
    else:core.seed_all(c.seed*1000000+200000)
    source_epoch=state.get('source_epoch',state.get('epoch'))
    del state
    net.train();opt.zero_grad()
    total=len(order);accumulation=c.gradient_accumulation
    total_updates=math.ceil(total/accumulation)
    started=time.monotonic();beginning=completed
    def checkpoint(count):
        ckpt=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),epoch=1,source_epoch=source_epoch,
            finetune_microbatches=count,finetune_optimizer_updates=updates,
            finetune_group_base_lrs=group_bases,data_order_sha256=order_sha,
            rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            rng_numpy=numpy_rng_checkpoint(),rng_random=random.getstate())
        temporary=latest.with_suffix('.tmp');torch.save(ckpt,temporary);temporary.replace(latest)
    try:
        with (folder/'trace.jsonl').open('a',buffering=1) as trace:
            for local,batch in enumerate(loader,completed):
                if local%accumulation==0:
                    warmup=min(1.,(updates+1)/c.finetune_warmup_updates)
                    cosine=.5*(1+math.cos(math.pi*updates/max(total_updates-1,1)))
                    factor=c.finetune_base_lr/c.source_optimizer_base_lr*(.1+.9*warmup)*(c.finetune_min_ratio+(1-c.finetune_min_ratio)*cosine)
                    for g,base in zip(opt.param_groups,group_bases):g['lr']=base*factor
                batch=core.to_cuda(batch);out=core.forward(net,batch)
                value,details=objective(core.loss_input(c,out,batch))
                if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite loss at frame {local+1}')
                group_length=min(accumulation,total-(local//accumulation)*accumulation)
                (value/group_length).backward()
                row=dict(microbatch=local+1,index=order[local],loss=float(value.detach()),
                         losses={k:float(v) for k,v in details.items()})
                if (local+1)%accumulation==0 or local+1==total:
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),c.grad_max_norm,error_if_nonfinite=True)
                    opt.step();opt.zero_grad();updates+=1
                    row.update(optimizer_update=updates,grad_norm=float(norm))
                trace.write(json.dumps(row)+'\n')
                count=local+1
                if count%32==0 or count in (beginning+1,total):
                    rate=(time.monotonic()-started)/(count-beginning)
                    progress=dict(status='running',microbatches=count,total=total,optimizer_updates=updates,
                        seconds_per_frame=rate,remaining_hours=(total-count)*rate/3600,
                        loss=row['loss'],dcd_geometry_loss=row['losses']['OPUSSetDCDLoss/loss_pts'],
                        pred_exponential_mean=row['losses']['OPUSSetDCDLoss/pred_exponential_mean'],
                        gt_exponential_mean=row['losses']['OPUSSetDCDLoss/gt_exponential_mean'],
                        lr=opt.param_groups[0]['lr'])
                    core.write(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
                if count%(c.save_every_updates*accumulation)==0 or count==total:checkpoint(count)
                del batch,out,value
        final=folder/'epoch_1.pth'
        if final.exists():final.unlink()
        os.link(latest,final)
        summary=dict(status='complete',frames=total,unique_frames=len(set(order)),epochs=1,
                     optimizer_updates=updates,checkpoint=str(final),checkpoint_sha256=core.sha(final))
        core.write(folder/'complete.json',summary);core.write(folder/'progress.json',summary)
    finally:net.flush_memory_phase()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=['train','check'],default='train')
    parser.add_argument('--py-config',type=Path,default=Path(DEFAULT_CONFIG))
    parser.add_argument('--work-dir',type=Path,default=Path('work_dirs/dcd_mini_1ep'))
    args=parser.parse_args();args.py_config=args.py_config.resolve();args.work_dir=args.work_dir.resolve()
    os.chdir(ROOT)
    if args.action=='check':
        c=configure(args)
        print(json.dumps(dict(status='passed',train_frames=c.expected_train_frames,
              val_frames=c.expected_val_frames,model_and_classification_preserved=True)))
        return
    if torch.cuda.device_count()!=1:raise ValueError('Use one visible GPU')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    with (args.work_dir/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        training(args)


if __name__=='__main__':main()
