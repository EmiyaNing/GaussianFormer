#!/usr/bin/env python3
"""CPU paired-scene bootstrap; does not train or choose checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np


def miou(hist):
    diagonal=np.diagonal(hist,axis1=-2,axis2=-1)
    denominator=hist.sum(-1)+hist.sum(-2)-diagonal
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.nanmean(100*diagonal[...,:17]/denominator[...,:17],axis=-1)


def main(root,repetitions=1000):
    root=Path(root);report=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    scenes=manifest['val']['scenes'];unique=sorted(set(scenes));lookup={s:i for i,s in enumerate(unique)}
    generator=np.random.default_rng(20261002)
    weights=generator.multinomial(len(unique),np.ones(len(unique))/len(unique),size=repetitions)
    def sampled(path):
        with np.load(path) as saved:hist=saved['hist'].copy()
        if len(hist)!=len(scenes):raise RuntimeError('Scene count mismatch')
        summed=np.zeros((len(unique),3,18,18),dtype=np.float64)
        for value,scene in zip(hist,scenes):summed[lookup[scene]]+=value
        return miou(np.einsum('bs,shij->bhij',weights,summed,optimize=True))
    def intervals(difference):
        result={h:np.quantile(difference[:,i],[.025,.975]).tolist() for i,h in enumerate(('1s','2s','3s'))}
        result['mean_1_2_3s']=np.quantile(difference.mean(-1),[.025,.975]).tolist()
        return result
    base=sampled(root/'baseline.npz');final=str(manifest['evaluation_steps'][-1]);values={}
    result=dict(status='partial',repetitions=repetitions,scenes=len(unique),baseline_changes={},paired_changes={},
                protocol='Paired scene bootstrap; one fixed training seed; exploratory unadjusted intervals, not independent confirmation or training-seed uncertainty.')
    for label,row in report['arms'].items():
        if final not in row.get('steps',{}):continue
        values[label]=sampled(row['steps'][final]['rows'])
        result['baseline_changes'][label]=intervals(values[label]-base)
    for first,second in [('full_bn_train','full_bn_fixed'),('future_bn_train','future_bn_fixed'),
                         ('full_bn_fixed','future_bn_fixed'),('full_bn_fixed','full_bn_fixed_detached_geom'),
                         ('future_bn_fixed','future_bn_fixed_detached_geom')]:
        for seed in manifest['seeds']:
            a,b=f'{first}_task_seed{seed}',f'{second}_task_seed{seed}'
            if a in values and b in values:result['paired_changes'][b+' vs '+a]=intervals(values[b]-values[a])
    for label,row in report['arms'].items():
        task=f'{row["arm"]}_task_seed{row["seed"]}'
        if row['supervision']!='task' and label in values and task in values:
            result['paired_changes'][label+' vs '+task]=intervals(values[label]-values[task])
    for seed in manifest['seeds']:
        for arm in {row['arm'] for row in report['arms'].values()}:
            correct=f'{arm}_teacher_seed{seed}';shuffled=f'{arm}_shuffled_seed{seed}'
            if correct in values and shuffled in values:
                result['paired_changes'][correct+' vs '+shuffled]=intervals(values[correct]-values[shuffled])
    if report['status']=='complete':result['status']='complete'
    destination=root/'bootstrap.json';temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(destination)
    print('[protocol bootstrap]',result['status'],len(result['baseline_changes']),len(result['paired_changes']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('out/future_distillation_causal/20261001_training_protocol'))
    parser.add_argument('--repetitions',type=int,default=1000)
    args=parser.parse_args();main(args.root,args.repetitions)
