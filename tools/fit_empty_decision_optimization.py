"""Fit small calibration/decision modules on training scenes, never validation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tools.empty_decision_optimization import (RUN, SEEDS, BAND, PRIMARY_BUDGET,
    initialize, read, seed_all, write_json, file_sha256)


class Dense(nn.Module):
    def __init__(self, inputs, hidden, outputs, zero_output=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


def prepare_data():
    protocol = initialize()
    outputs = [RUN/'samples_fit.npz', RUN/'samples_tune.npz']
    if all(p.exists() for p in outputs) and (RUN/'sample_manifest.json').exists():
        return read(RUN/'sample_manifest.json')
    summary = read(RUN/'train/summary.json')
    assert len(summary['frames']) == protocol['train_frames'] == 4096
    accum = {cohort: {k:[] for k in ['X','gt','fg','region','weight','frame']} for cohort in ['fit','tune']}
    fixed = {k:np.zeros((2,18,18), np.int64) for k in ['fit','tune']}
    native = {k:np.zeros((2,18,18), np.int64) for k in ['fit','tune']}
    counts = {k:0 for k in ['fit','tune']}
    for frame in summary['frames']:
        cohort = 'fit' if frame['scene'] in protocol['fit_scenes'] else 'tune'
        stem = ROOT/frame['path']
        meta = read(stem.with_suffix('.json'))
        fixed[cohort] += np.asarray(meta['fixed_confusion'])
        native[cohort] += np.asarray(meta['native_confusion'])
        with np.load(stem.with_suffix('.npz')) as z:
            for key in ['X','gt','fg','region','weight']:
                accum[cohort][key].append(z[key])
            accum[cohort]['frame'].append(np.full(len(z['gt']),frame['index'],np.int16))
        counts[cohort] += 1
    manifest = dict(status='complete', cohorts={}, supervision='both regions; no validation rows')
    for ci, cohort in enumerate(['fit','tune']):
        data = {k:np.concatenate(v) for k,v in accum[cohort].items()}
        n = len(data['gt'])
        cap = protocol[cohort+'_row_cap']
        if n > cap:
            ids = np.random.default_rng(928+ci).choice(n,cap,replace=False)
            data = {k:v[ids] for k,v in data.items()}
            data['weight'] *= n/cap
        np.savez_compressed(RUN/f'samples_{cohort}.npz', **data,
            fixed_confusion=fixed[cohort], native_confusion=native[cohort])
        manifest['cohorts'][cohort] = dict(frames=counts[cohort], original_rows=n,
            fitted_rows=len(data['gt']), population_estimate=float(data['weight'].sum()),
            strata='region and native margin sign, label independent; uniform row cap inverse weighted')
    write_json(RUN/'sample_manifest.json',manifest)
    return manifest


def load_samples(cohort):
    with np.load(RUN/f'samples_{cohort}.npz') as z:
        return {k:z[k] for k in z.files}


def scaler(data, support):
    cols = list(range(9 if support else 8))
    x, w = data['X'][:,cols].astype(np.float64), data['weight']
    mean = np.average(x,axis=0,weights=w)
    std = np.sqrt(np.average((x-mean)**2,axis=0,weights=w)).clip(.05)
    return dict(cols=cols, mean=mean.astype(np.float32), std=std.astype(np.float32))


def encoded(data, scale, device='cuda'):
    x = torch.as_tensor(data['X'],device=device)
    fg = torch.as_tensor(data['fg'].astype(np.int64),device=device)
    mean = torch.as_tensor(scale['mean'],device=device)
    std = torch.as_tensor(scale['std'],device=device)
    values = (x[:,scale['cols']]-mean)/std
    return torch.cat([values,F.one_hot(fg,17).to(values)],-1)


def outcome(gt, fg):
    return np.where(gt == 17,1,np.where(gt == fg,0,2)).astype(np.int64)


def probability_fit(fit, tune, seed, support=True, binary=False):
    name = f'B3_seed{seed}' if binary else f'prob_{"support" if support else "no_support"}_seed{seed}'
    path = RUN/'models'/f'{name}.pt'
    if path.exists():
        return path, read(path.with_suffix('.json'))
    seed_all(seed)
    scale = scaler(fit,support)
    a,b = encoded(fit,scale), encoded(tune,scale)
    y = outcome(fit['gt'],fit['fg'])
    yt = outcome(tune['gt'],tune['fg'])
    y = torch.as_tensor((y == 0).astype(np.float32) if binary else y,device='cuda')
    yt = torch.as_tensor((yt == 0).astype(np.float32) if binary else yt,device='cuda')
    w = torch.as_tensor(fit['weight']/fit['weight'].sum(),device='cuda',dtype=torch.float32)
    wt = torch.as_tensor(tune['weight']/tune['weight'].sum(),device='cuda',dtype=torch.float32)
    hidden, outputs = (40,1) if binary else (32,3)
    network = Dense(a.shape[1],hidden,outputs).cuda()
    optimizer = torch.optim.AdamW(network.parameters(),lr=.003,weight_decay=1e-4)
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1,801):
        optimizer.zero_grad(set_to_none=True)
        logits = network(a)
        loss = ((F.binary_cross_entropy_with_logits(logits[:,0],y,reduction='none') if binary else
                 F.cross_entropy(logits,y,reduction='none'))*w).sum()
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0:
            with torch.no_grad():
                p = network(b)
                val_loss = ((F.binary_cross_entropy_with_logits(p[:,0],yt,reduction='none') if binary else
                            F.cross_entropy(p,yt,reduction='none'))*wt).sum().item()
            history.append(dict(epoch=epoch,fit_nll=loss.item(),tuning_nll=val_loss))
            if best is None or val_loss < best[0]:
                best = (val_loss,epoch,copy.deepcopy(network.state_dict()))
    network.load_state_dict(best[2])
    with torch.no_grad():
        raw = network(b)
        temperatures = [.5,.75,1.,1.25,1.5,2.]
        choices = []
        for temp in temperatures:
            loss = ((F.binary_cross_entropy_with_logits(raw[:,0]/temp,yt,reduction='none') if binary else
                     F.cross_entropy(raw/temp,yt,reduction='none'))*wt).sum().item()
            choices.append((loss,abs(temp-1),temp))
        calibrated_nll,_,temperature = min(choices)
    artifact = dict(kind='binary' if binary else 'probability',support=support,seed=seed,
        scale=scale,hidden=hidden,outputs=outputs,temperature=temperature,
        state_dict={k:v.cpu() for k,v in network.state_dict().items()})
    torch.save(artifact,path)
    result = dict(name=name,kind=artifact['kind'],support=support,seed=seed,parameters=sum(p.numel() for p in network.parameters()),
        selected_epoch=best[1],tuning_nll_before_temperature=best[0],temperature=temperature,
        tuning_nll=calibrated_nll,history=history,elapsed_seconds=time.monotonic()-started,
        artifact_sha256=file_sha256(path))
    write_json(path.with_suffix('.json'),result)
    print(name,'NLL',calibrated_nll,'elapsed',result['elapsed_seconds'],flush=True)
    return path,result


def load_artifact(path,device='cuda'):
    record = torch.load(path,map_location='cpu',weights_only=False)
    inputs = len(record['scale']['cols'])+17
    network = Dense(inputs,record['hidden'],record['outputs']).to(device).eval()
    network.load_state_dict(record['state_dict'])
    return record,network


def loss_context(data):
    gt = torch.as_tensor(data['gt'].astype(np.int64),device='cuda')
    fg = torch.as_tensor(data['fg'].astype(np.int64),device='cuda')
    region = torch.as_tensor(data['region'].astype(np.int64),device='cuda')
    w = torch.as_tensor(data['weight'],device='cuda',dtype=torch.float32)
    fixed = torch.as_tensor(data['fixed_confusion'],device='cuda',dtype=torch.float32)
    camera = region == 0
    gt_count = fixed[0].sum(-1)+torch.zeros(18,device='cuda').scatter_add_(0,gt,w*camera)
    empty_out = fixed[1,17].sum()+(w*(region == 1)*(gt == 17)).sum()
    context = dict(fg=fg, w=w, correct=gt == fg, camera=camera,
        out_empty=(region == 1)&(gt == 17), gt=gt_count[:17],
        tp=fixed[0].diag()[:17],fp=fixed[0].sum(0)[:17]-fixed[0].diag()[:17],
        out_fixed=fixed[1,17,:17].sum(),out_count=empty_out)
    return context


def decision_statistics(d,c):
    mass = c['w']*d
    tp = c['tp']+torch.zeros(17,device=d.device).scatter_add_(0,c['fg'],mass*c['camera']*c['correct'])
    fp = c['fp']+torch.zeros(17,device=d.device).scatter_add_(0,c['fg'],mass*c['camera']*(~c['correct']))
    iou = (tp/(c['gt']+fp).clamp_min(1)).mean()
    risk = (c['out_fixed']+(mass*c['out_empty']).sum())/c['out_count'].clamp_min(1)
    return iou,risk


def policy_fit(fit,tune,seed,kind,probability_path):
    name = f'{kind}_seed{seed}'
    path = RUN/'models'/f'{name}.pt'
    if path.exists():
        return path,read(path.with_suffix('.json'))
    seed_all(seed)
    record,probability = load_artifact(probability_path)
    a,b = encoded(fit,record['scale']),encoded(tune,record['scale'])
    with torch.no_grad():
        qa = (probability(a)/record['temperature']).softmax(-1)
        qb = (probability(b)/record['temperature']).softmax(-1)
    a,b = torch.cat([a,qa],-1),torch.cat([b,qb],-1)
    ma = torch.as_tensor(fit['X'][:,0],device='cuda')
    mb = torch.as_tensor(tune['X'][:,0],device='cuda')
    ca,cb = loss_context(fit),loss_context(tune)
    with torch.no_grad():
        refa = decision_statistics((ma >= 0).float(),ca)[1].item()
        refb = decision_statistics((mb >= 0).float(),cb)[1].item()
    gate = Dense(a.shape[1],16,1,zero_output=True).cuda()
    optimizer = torch.optim.AdamW(gate.parameters(),lr=.008,weight_decay=1e-4)
    constrained = kind != 'P2'
    one_way = kind == 'P3_oneway'
    multiplier = 0.
    history,best = [],None
    started = time.monotonic()
    for step in range(601):
        if step:
            optimizer.zero_grad(set_to_none=True)
            score = ma+BAND*gate(a)[:,0].tanh()
            d = torch.sigmoid(score/.4)
            if one_way:
                d = torch.where(ma >= 0,torch.ones_like(d),d)
            iou,risk = decision_statistics(d,ca)
            loss = -iou+(multiplier*(risk-refa-PRIMARY_BUDGET) if constrained else 0.)
            loss.backward()
            optimizer.step()
            if constrained:
                multiplier = max(0.,multiplier+10.*(risk.item()-refa-PRIMARY_BUDGET))
        if step % 20 == 0:
            with torch.no_grad():
                score = mb+BAND*gate(b)[:,0].tanh()
                d = (score >= 0).float()
                if one_way:
                    d = torch.where(mb >= 0,torch.ones_like(d),d)
                iou,risk = decision_statistics(d,cb)
                violation = max(0.,risk.item()-refb-PRIMARY_BUDGET)
                rank = (int(violation <= 1e-8),iou.item()) if constrained and violation <= 1e-8 else (
                    (0,-violation) if constrained else (1,iou.item()))
            history.append(dict(step=step,tuning_camera_mIoU=iou.item()*100,tuning_out_fpr=risk.item(),
                reference_out_fpr=refb,lambda_value=multiplier))
            if best is None or rank > best[0]:
                best = (rank,step,copy.deepcopy(gate.state_dict()))
    gate.load_state_dict(best[2])
    artifact = dict(kind='policy',policy_kind=kind,support=record['support'],seed=seed,
        probability_path=str(probability_path.relative_to(ROOT)),scale=record['scale'],
        hidden=16,outputs=1,one_way=one_way,band=BAND,
        state_dict={k:v.cpu() for k,v in gate.state_dict().items()})
    torch.save(artifact,path)
    result = dict(name=name,kind=kind,support=record['support'],seed=seed,selected_step=best[1],
        gate_parameters=sum(p.numel() for p in gate.parameters()),
        probability_parameters=sum(p.numel() for p in probability.parameters()),
        training_objective='camera soft mIoU'+(' with complement FPR primal-dual constraint' if constrained else ''),
        risk_budget=PRIMARY_BUDGET if constrained else None,one_way=one_way,
        history=history,elapsed_seconds=time.monotonic()-started,artifact_sha256=file_sha256(path))
    write_json(path.with_suffix('.json'),result)
    print(name,'selected step',best[1],'elapsed',result['elapsed_seconds'],flush=True)
    return path,result


class Predictor:
    """Inputs contain no GT or camera mask. Reuses frozen probabilities per frame."""
    def __init__(self,path):
        self.path = Path(path)
        self.record = torch.load(path,map_location='cpu',weights_only=False)
        r = self.record
        n = len(r['scale']['cols'])+17+(3 if r['kind'] == 'policy' else 0)
        self.network = Dense(n,r['hidden'],r['outputs']).cuda().eval()
        self.network.load_state_dict(r['state_dict'])
        self.probability = Predictor(ROOT/r['probability_path']) if r['kind'] == 'policy' else None

    def __call__(self,features,fg,cache=None):
        cache = {} if cache is None else cache
        key = str(self.path)
        if key in cache:
            return cache[key]
        r = self.record
        scale = r['scale']
        x = (features[:,scale['cols']]-torch.as_tensor(scale['mean'],device=features.device))/torch.as_tensor(scale['std'],device=features.device)
        x = torch.cat([x,F.one_hot(fg.long(),17).to(x)],-1)
        if r['kind'] == 'policy':
            probability = self.probability(features,fg,cache)
            score = features[:,0]+BAND*self.network(torch.cat([x,probability],-1))[:,0].tanh()
            if r['one_way']:
                score = torch.where(features[:,0] >= 0,torch.full_like(score,1e6),score)
            result = score
        elif r['kind'] == 'binary':
            result = self.network(x)[:,0]/r['temperature']
        else:
            result = (self.network(x)/r['temperature']).softmax(-1)
        cache[key] = result
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args = parser.parse_args()
    initialize()
    write_json(RUN/'training_schedule.json',dict(probability_max_epochs=800,
        probability_checkpoint_interval=40,decision_steps=600,decision_checkpoint_interval=20,
        supersedes_initial_protocol_schedule=True,
        reason='Pilot tuning NLL was still decreasing at 160; extended before validation access.',
        selection='all network selection uses training tuning scenes only'))
    (RUN/'models').mkdir(exist_ok=True)
    prepare_data()
    fit,tune = load_samples('fit'),load_samples('tune')
    result = read(RUN/'fit_results.json') if (RUN/'fit_results.json').exists() else dict(models={})
    for seed in ([args.seed] if args.seed else SEEDS):
        paths = {}
        for support in [True,False]:
            path,meta = probability_fit(fit,tune,seed,support=support)
            result['models'][meta['name']] = dict(path=str(path.relative_to(ROOT)),**meta)
            paths[support] = path
            if support:
                result['models'][f'P1_seed{seed}'] = dict(path=str(path.relative_to(ROOT)),kind='probability',seed=seed,support=True)
            write_json(RUN/'fit_results.json',result)
        path,meta = probability_fit(fit,tune,seed,support=True,binary=True)
        result['models'][meta['name']] = dict(path=str(path.relative_to(ROOT)),**meta)
        for kind in ['P2','P3','P4','P3_oneway']:
            path,meta = policy_fit(fit,tune,seed,kind,paths[kind != 'P4'])
            result['models'][meta['name']] = dict(path=str(path.relative_to(ROOT)),**meta)
            write_json(RUN/'fit_results.json',result)
    result.update(status='complete',seeds=sorted({v['seed'] for v in result['models'].values()}),
        source_sha256=file_sha256(__file__),finished_at=datetime.now().astimezone().isoformat())
    write_json(RUN/'fit_results.json',result)


if __name__ == '__main__':
    main()
