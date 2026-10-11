#!/usr/bin/env python3
"""Compare screen features against one fixed original spatial target cohort."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mmengine import Config
from tools.future_distillation_causal_common import *


@torch.no_grad()
def audit(cfg,root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())['val']
    screen=json.loads((root/'s2_screens.json').read_text())
    sources=list(source_records(root,manifest,'val'))
    original,_=model_for(cfg);original.requires_grad_(False)
    targets=[];baseline_losses=[]
    for source in sources:
        output=cached_rollout(original,source)
        teacher=teacher_record(cfg,'val',source['index'],source['scene'])
        selected=matched_targets(original,output,teacher)
        baseline_losses.append(float(feature_loss(output,selected)))
        targets.append({h:(feature.cpu(),valid.cpu()) for h,(feature,valid) in selected.items()})
    del original;torch.cuda.empty_cache()
    result=dict(status='running',baseline_fixed_feature_loss=sum(baseline_losses)/len(baseline_losses),arms={},
                protocol='Teacher targets and masks fixed at untouched baseline geometry; same frozen-history query slots. Trained points may move; not claimed to be identical physical sampling after training.')
    previous=root/'s2_alignment_audit.json'
    if previous.exists():result['arms']=json.loads(previous.read_text()).get('arms',{})
    for label,row in screen['arms'].items():
        if row.get('status')!='complete' or label in result['arms']:continue
        instance,_=model_for(cfg,row['checkpoint']);instance.requires_grad_(False)
        losses=[]
        for source,target in zip(sources,targets):
            output=cached_rollout(instance,source)
            selected={h:(feature.cuda(),valid.cuda()) for h,(feature,valid) in target.items()}
            losses.append(float(feature_loss(output,selected)))
        result['arms'][label]=dict(fixed_feature_loss=sum(losses)/len(losses),moving_feature_loss=row['feature_validation_loss'])
        save_json(previous,result)
        print('[fixed alignment audit]',label,result['arms'][label],flush=True)
        del instance;torch.cuda.empty_cache()
    result['status']='complete';save_json(previous,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/opus/future_distillation_causal.py')
    parser.add_argument('--output-dir')
    args=parser.parse_args();cfg=Config.fromfile(args.config);deterministic();torch.cuda.set_per_process_memory_fraction(.15)
    audit(cfg,Path(args.output_dir or cfg.causal_root))
