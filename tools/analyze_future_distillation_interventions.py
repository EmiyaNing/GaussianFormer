#!/usr/bin/env python3
"""Additional geometry measurements and paired-scene uncertainty for T1."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from tools.future_distillation_causal_common import *


def histogram_miou(hist):
    diagonal=np.diagonal(hist,axis1=-2,axis2=-1)
    union=hist.sum(-1)+hist.sum(-2)-diagonal
    with np.errstate(divide='ignore',invalid='ignore'):return np.nanmean(100*diagonal[...,:17]/union[...,:17],axis=-1)


def paired_bootstrap(root,mix,repetitions=1000):
    scenes=mix['validation']['scenes'];unique=sorted(set(scenes));locations={s:i for i,s in enumerate(unique)}
    rng=np.random.default_rng(20260930)
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repetitions)
    def grouped(path):
        with np.load(path) as values:hist=values['hist'].copy()
        rows=np.zeros((len(unique),3,18,18),dtype=np.float64)
        for row,scene in zip(hist,scenes):rows[locations[scene]]+=row
        return np.einsum('bs,shij->bhij',weights,rows,optimize=True)
    base=histogram_miou(grouped(root/'s1_baseline.npz'))
    result=dict(repetitions=repetitions,scenes=len(unique),protocol='paired scene bootstrap, frozen checkpoint; no retraining variance; exploratory unadjusted intervals',cases={})
    for name,row in mix['cases'].items():
        values=histogram_miou(grouped(Path(row['rows'])))-base
        result['cases'][name]={h:np.quantile(values[:,i],[.025,.975]).tolist() for i,h in enumerate(('1s','2s','3s'))}
        result['cases'][name]['mean_1_2_3s']=np.quantile(values.mean(-1),[.025,.975]).tolist()
    save_json(root/'s1_intervention_bootstrap.json',result)


def screen_bootstrap(root,repetitions=1000):
    """Pair scenes and completed training seeds; intervals remain exploratory."""
    screen=json.loads((root/'s2_screens.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())['val']
    scenes=manifest['scenes'];unique=sorted(set(scenes));locations={s:i for i,s in enumerate(unique)}
    def grouped(path):
        with np.load(path) as values:hist=values['hist'].copy()
        if len(hist)!=len(scenes):raise RuntimeError('Screen bootstrap sample count differs')
        rows=np.zeros((len(unique),3,18,18),dtype=np.float64)
        for row,scene in zip(hist,scenes):rows[locations[scene]]+=row
        return rows
    rng=np.random.default_rng(20261001)
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repetitions)
    untouched=histogram_miou(np.einsum('bs,shij->bhij',weights,grouped(root/'s2_baseline.npz'),optimize=True))
    arms={}
    for name,row in screen['arms'].items():
        if row.get('status')=='complete':arms.setdefault(name.split('_seed')[0],{})[row['seed']]=row
    result=dict(repetitions=repetitions,scenes=len(unique),cases={},
                protocol='Paired scene bootstrap and paired seed resampling; average per-seed native mIoU. Three development seeds only, exploratory unadjusted intervals, not independent confirmation.')
    def samples(runs,seeds,selection):
        values=np.stack([histogram_miou(np.einsum('bs,shij->bhij',weights,grouped(runs[seed]['rows']),optimize=True)) for seed in seeds])
        return values[selection,np.arange(repetitions)[None]].mean(0)
    for arm,runs in arms.items():
        if len(runs)!=3:continue
        control='task_heads_frozen' if arm.endswith('heads_frozen') else 'task_heads_train'
        seeds=sorted(set(runs)&set(arms.get(control,{})))
        if len(seeds)!=3:continue
        selection=rng.integers(0,3,size=(3,repetitions))
        score=samples(runs,seeds,selection);task=samples(arms[control],seeds,selection)
        record=dict(control=control,seeds=seeds,baseline_change_ci={},paired_task_change_ci={})
        for label,delta in [('baseline_change_ci',score-untouched),('paired_task_change_ci',score-task)]:
            record[label]={h:np.quantile(delta[:,i],[.025,.975]).tolist() for i,h in enumerate(('1s','2s','3s'))}
            record[label]['mean_1_2_3s']=np.quantile(delta.mean(-1),[.025,.975]).tolist()
        result['cases'][arm]=record
    result['status']='complete' if screen['status']=='complete' and len(result['cases'])==7 else 'partial'
    save_json(root/'s2_bootstrap.json',result)


@torch.no_grad()
def geometry(cfg,root,mix):
    manifest=json.loads((root/'manifest.json').read_text())['val']
    instance,_=model_for(cfg);instance.requires_grad_(False)
    sums={}
    for source in source_records(root,manifest,'val',len(mix['validation']['indices'])):
        output=cached_rollout(instance,source)
        targets=matched_targets(instance,output,teacher_record(cfg,'val',source['index'],source['scene']))
        for h,(target,valid) in targets.items():
            state=output['future_query_states'][h-1]
            original=instance._world(state['positions'])
            for alpha in cfg.causal_mix:
                features=state['features'].clone()
                features[:,valid]=(1-alpha)*features[:,valid]+alpha*target[:,valid].to(features)
                positions,_=instance.decode(features,state['interaction_positions'])
                points=instance._world(positions)
                distance=(points-original).norm(dim=-1)[0,valid]
                inside=((points>=points.new_tensor(instance.pc_range[:3]))&(points<points.new_tensor(instance.pc_range[3:]))).all(-1)[0,valid]
                row=sums.setdefault(f'mix{alpha:g}/{h/2:g}s',dict(points=0,shift_m=0.,inside=0,outside=0))
                row['points']+=distance.numel();row['shift_m']+=float(distance.sum());row['inside']+=int(inside.sum());row['outside']+=int((~inside).sum())
    for row in sums.values():
        row['mean_shift_m']=row['shift_m']/max(row['points'],1)
        row['inside_ratio']=row['inside']/max(row['points'],1)
    save_json(root/'s1_geometry.json',dict(status='complete',cases=sums,
                protocol='native joint head, valid matched queries only; point displacement from untouched student; fixed interaction positions and no recurrent feedback'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_causal.py')
    parser.add_argument('--output-dir')
    parser.add_argument('--screens-only',action='store_true')
    args=parser.parse_args();cfg=Config.fromfile(args.config);root=Path(args.output_dir or cfg.causal_root)
    if args.screens_only:
        screen_bootstrap(root);sys.exit(0)
    deterministic();torch.cuda.set_per_process_memory_fraction(.15)
    mix=json.loads((root/'s1_feature_intervention.json').read_text())
    paired_bootstrap(root,mix);geometry(cfg,root,mix)
