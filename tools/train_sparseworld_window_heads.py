#!/usr/bin/env python3
"""A1 matched final cls/reg adaptation; frozen image/interaction encoders."""
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from loss.opus_set_loss import OPUSSetLoss
from tools.diagnose_teacher_alignment import seed_all
from tools.experiment_future_response_distillation import build_student
from tools.sparseworld_region_utils import ROOT,HORIZONS,write_json,sha256
from tools.sparseworld_region_ablation_common import OUT,GROUPS,CENTERS,records,masks,evaluate
from tools.run_sparseworld_region_ablations import render

STEPS=768
SEEDS=[20261008,20261009,20261010]

def decode(model,cls,reg,state):
    f=state['features'].cuda();c=state['centers'].cuda()
    points=c[:,:,None]+model.encoder.layers[-1].scale*reg(f).reshape(1,-1,48,3)
    # Keep the exact native normalization / metric decoding roundtrip.
    points=model._world(model._norm(points)).flatten(1,2)
    logits=cls(f).reshape(1,-1,17)
    return points,logits

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py');strict=Config.fromfile(cfg.student_model_config)
    model=build_student(cfg,strict);dest=OUT/'adaptation';dest.mkdir(exist_ok=True)
    selections=json.loads((OUT/'train_selection.json').read_text());val=records()
    assert len(selections)==256 and len(set(r['scene'] for r in selections))==256
    assert not {r['scene'] for r in selections}&{r['scene'] for r in val}
    print('[adaptation] loading frozen caches',flush=True)
    train=[torch.load(OUT/'train_cache'/(r['token']+'.pth'),map_location='cpu') for r in selections]
    objective=OPUSSetLoss(**{k:v for k,v in strict.loss.current_loss.items() if k!='type'}).cuda()
    centers=torch.as_tensor(CENTERS,device='cuda')
    initial_cls=copy.deepcopy(model.encoder.layers[-1].cls.state_dict());initial_reg=copy.deepcopy(model.encoder.layers[-1].reg.state_dict())
    with torch.no_grad():
        r=val[0];cache=torch.load(OUT/'val_cache'/(r['token']+'.pth'),map_location='cpu')
        pair=dict(np.load(ROOT/'pairs'/(r['token']+'.npz')))
        for j,h in enumerate(HORIZONS):
            p,l=decode(model,model.encoder.layers[-1].cls,model.encoder.layers[-1].reg,cache[f'P_{h}'])
            assert np.array_equal(render(model,p,l),pair['P'][j]),'Cached native head parity failed'
    print('[adaptation] cached native head exact replay passed',flush=True)
    write_json(dest/'protocol.json',dict(steps=STEPS,seeds=SEEDS,arms=['H','P','M'],train_samples=len(train),train_scenes=256,val_samples=len(val),
          trainable='Only original final encoder layer cls and reg branches; no encoder adaptation',lr=2e-5,weight_decay=.01,
          optimizer='AdamW',grad_clip=5.,batch_size=1,horizons=[2,4,6],bn='fixed',loss='original OPUS full-set focal + bidirectional SmoothL1, all GT occupied, camera weights all true',
          schedule='same per seed across arms, permutation of 256 x 3 tasks',checkpoint_sha256=sha256(cfg.student_checkpoint)))
    for seed in SEEDS:
        schedule=np.random.default_rng(seed).permutation(len(train)*3)
        for arm in ['H','P','M']:
            output=dest/f'{arm}_{seed}.npz'
            if output.exists():continue
            seed_all(seed)
            cls=model.encoder.layers[-1].cls;reg=model.encoder.layers[-1].reg
            cls.load_state_dict(initial_cls);reg.load_state_dict(initial_reg)
            cls.requires_grad_(True);reg.requires_grad_(True)
            parameters=list(cls.parameters())+list(reg.parameters())
            opt=torch.optim.AdamW(parameters,lr=2e-5,weight_decay=.01)
            log=[];start=time.monotonic()
            for step,index in enumerate(schedule[:STEPS]):
                row=train[index//3];j=index%3;h=HORIZONS[j]
                p,l=decode(model,cls,reg,row[f'{arm}_{h}'])
                gt=row['GT'][j].reshape(-1).cuda().long();valid=gt<17
                if valid.any():
                    c,g1,g2,_,_=objective._single(p[0],l[0],centers[valid],gt[valid],torch.ones_like(gt[valid],dtype=torch.bool))
                    loss=2*c+.5*(g1+g2)
                else: loss=p.sum()*0+l.sum()*0
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.);opt.step()
                if step%64==0 or step+1==STEPS:
                    log.append(dict(step=step+1,loss=float(loss),gradient_norm=float(norm),seconds=time.monotonic()-start))
                    print(f'[adaptation {arm} {seed}] {step+1}/{STEPS} loss={float(loss):.3f} {time.monotonic()-start:.1f}s',flush=True)
            train_seconds=time.monotonic()-start
            torch.save(dict(cls=cls.state_dict(),reg=reg.state_dict(),seed=seed,arm=arm,updates=STEPS),dest/f'{arm}_{seed}.pth')
            cls.requires_grad_(False);reg.requires_grad_(False)
            hist=[];counts=[]
            with torch.no_grad():
                for r in val:
                    cache=torch.load(OUT/'val_cache'/(r['token']+'.pth'),map_location='cpu')
                    pair=dict(np.load(ROOT/'pairs'/(r['token']+'.npz')));attrs=dict(np.load(ROOT/'attributes'/(r['token']+'.npz')))
                    hs=[];cs=[]
                    for j,h in enumerate(HORIZONS):
                        p,l=decode(model,cls,reg,cache[f'{arm}_{h}']);pred=render(model,p,l)
                        hh,cc=evaluate(pair['GT'][j],pred,masks(r,j,pair,attrs));hs.append(hh);cs.append(cc)
                    hist.append(hs);counts.append(cs)
            np.savez_compressed(output,hist=np.asarray(hist),counts=np.asarray(counts),tokens=np.array([r['token'] for r in val]))
            write_json(dest/f'{arm}_{seed}.json',dict(train_seconds=train_seconds,total_seconds=time.monotonic()-start,updates=STEPS,log=log,
                       parameters=sum(p.numel() for p in parameters),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            print(f'[adaptation {arm} {seed}] saved evaluation',flush=True)
if __name__=='__main__':main()
