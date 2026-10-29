"""Matched mini-set fine-tuning for bidirectional local semantic field supervision."""
from pathlib import Path
import argparse,fcntl,hashlib,json,math,os,random,sys,time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from tools.semantic_field_experiment_common import *
from tools.train_dcd_mini import numpy_rng_checkpoint,restore_numpy_rng


def train(arm):
    protocol=assert_protocol()
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    folder=RUN/'train'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    latest=folder/'latest.pth'
    c,net,opt,objective,state=build(arm,latest if latest.exists() else INITIAL,True)
    completed=int(state.get('field_microbatches',0));updates=int(state.get('field_updates',0))
    bases=state.get('field_base_lrs',[g['lr'] for g in opt.param_groups])
    dataset=OPENOCC_DATASET.build(c.train_dataset_config)
    assert len(dataset)==4096 and len({x[0] for x in dataset.keyframes})==698
    order=np.random.default_rng(c.seed).permutation(len(dataset)).tolist()
    order_sha=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest()
    if completed:assert state['data_order_sha256']==order_sha
    tracefile=folder/'trace.jsonl'
    if tracefile.exists():
        lines=tracefile.read_text().splitlines();keep=[]
        for line in lines:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row['microbatch']<=completed:keep.append(line)
        if keep!=lines:
            tracefile.rename(folder/f'trace_before_resume_{time.time_ns()}.jsonl')
            tracefile.write_text('\n'.join(keep)+('\n' if keep else ''))
    class FixedSampler(Sampler):
        def __iter__(self):return iter(order[completed:])
        def __len__(self):return len(order)-completed
    loader=DataLoader(TrainingFrames(dataset,c.seed,arm!='baseline'),batch_size=1,sampler=FixedSampler(),
        num_workers=c.train_loader.num_workers,collate_fn=custom_collate_fn_temporal,
        pin_memory=True,persistent_workers=True,prefetch_factor=2)
    if completed:
        torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
        restore_numpy_rng(state['rng_numpy']);random.setstate(state['rng_random'])
    else:seed_all(c.seed*1000000+200000)
    del state
    net.train();opt.zero_grad();total=len(order);acc=c.gradient_accumulation
    total_updates=math.ceil(total/acc);start=time.monotonic();begin=completed
    def checkpoint(count):
        ckpt=dict(state_dict=net.state_dict(),optimizer=opt.state_dict(),epoch=1,
            field_microbatches=count,field_updates=updates,field_base_lrs=bases,
            data_order_sha256=order_sha,protocol_sha256=sha(RUN/'protocol.json'),
            rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            rng_numpy=numpy_rng_checkpoint(),rng_random=random.getstate())
        temporary=latest.with_suffix('.tmp');torch.save(ckpt,temporary);temporary.replace(latest)
    try:
        with tracefile.open('a',buffering=1) as trace:
            for i,batch in enumerate(loader,completed):
                if i%acc==0:
                    warmup=min(1.,(updates+1)/c.finetune_warmup_updates)
                    cosine=.5*(1+math.cos(math.pi*updates/max(total_updates-1,1)))
                    factor=(.1+.9*warmup)*(c.finetune_min_ratio+(1-c.finetune_min_ratio)*cosine)
                    for group,base in zip(opt.param_groups,bases):group['lr']=base*factor
                batch=to_cuda(batch);out=forward(net,batch)
                inputs=loss_input(c,out,batch);inputs['global_iter']=i
                value,details=objective(inputs)
                if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite loss {arm} frame{i+1}')
                (value/min(acc,total-(i//acc)*acc)).backward()
                row=dict(microbatch=i+1,index=order[i],loss=float(value.detach()),
                    losses={k:float(v) for k,v in details.items()})
                for module in objective.modules():
                    if hasattr(module,'last_raw'):module.last_raw=None;module.last_geometry=None
                if (i+1)%acc==0 or i+1==total:
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),c.grad_max_norm,error_if_nonfinite=True)
                    opt.step();opt.zero_grad();updates+=1;row.update(optimizer_update=updates,grad_norm=float(norm))
                trace.write(json.dumps(row)+'\n');count=i+1
                if count%32==0 or count in (begin+1,total):
                    torch.cuda.synchronize();rate=(time.monotonic()-start)/(count-begin)
                    progress=dict(status='running',arm=arm,microbatches=count,total=total,optimizer_updates=updates,
                        seconds_per_frame=rate,remaining_hours=(total-count)*rate/3600,loss=row['loss'],
                        lr=opt.param_groups[0]['lr'],peak_cuda_GB=torch.cuda.max_memory_allocated()/1e9,
                        field={k:v for k,v in row['losses'].items() if k.startswith('SemanticGaussianFieldLoss')})
                    write(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
                if count%(c.save_every_updates*acc)==0 or count==total:checkpoint(count)
                del batch,out,inputs,value
        final=folder/'epoch_1.pth'
        if final.exists():final.unlink()
        os.link(latest,final)
        summary=dict(status='complete',frames=total,unique_frames=len(set(order)),scenes=698,epochs=1,
            optimizer_updates=updates,checkpoint=str(final),checkpoint_sha256=sha(final),
            data_order_sha256=order_sha,seconds=time.monotonic()-start,
            peak_cuda_GB=torch.cuda.max_memory_allocated()/1e9)
        write(folder/'complete.json',summary);write(folder/'progress.json',summary)
    finally:net.flush_memory_phase()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('arm',choices=ARMS);args=parser.parse_args()
    os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);train(args.arm)
