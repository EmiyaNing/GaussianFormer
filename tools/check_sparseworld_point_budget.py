#!/usr/bin/env python3
"""A3 follow-up fixed after initial screen: repeated thinning x padding at 3 s."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from tools.experiment_future_response_distillation import build_student,paths_for
from tools.experiment_future2current import baseline
from tools.sparseworld_region_ablation_common import OUT,ROOT,GROUPS,records,masks,evaluate
from tools.run_sparseworld_region_ablations import render,thin,spatial_equal
from tools.train_sparseworld_window_heads import decode
from tools.sparseworld_region_utils import write_json

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py');model=build_student(cfg,Config.fromfile(cfg.student_model_config))
    manifest=json.load(open('out/future2current/feature_identity_2048x512/manifest.json'))
    sources=paths_for(Path(manifest['source_cache']),'val',manifest['val'])
    dest=OUT/'budget_followup';dest.mkdir(exist_ok=True)
    seeds=[20261008,20261009,20261010]
    write_json(dest/'protocol.json',dict(status='posthoc_followup',reason='Initial single-seed equal-query intervention reversed F advantage',
               seeds=seeds,query_budgets=[360,540,720],padding=[True,False],horizon=3,
               limitation='Same 150 analytical scenes; a controlled budget intervention, not a new independent confirmation'))
    for i,r in enumerate(records()):
        path=dest/(r['token']+'.npz')
        if path.exists():continue
        source=torch.load(sources[r['position']],map_location='cpu');roll=baseline(model,source)
        f=(roll['points'][5],roll['logits'][5]);cache=torch.load(OUT/'val_cache'/(r['token']+'.pth'),map_location='cpu')
        p=decode(model,model.encoder.layers[-1].cls,model.encoder.layers[-1].reg,cache['P_6'])
        pair=dict(np.load(ROOT/'pairs'/(r['token']+'.npz')));attrs=dict(np.load(ROOT/'attributes'/(r['token']+'.npz')))
        assert np.array_equal(render(model,*p),pair['P'][2])
        groups=masks(r,2,pair,attrs);hist=[];counts=[];names=[]
        for seed in seeds:
            for budget in (360,540,720):
                for a,pr in [('F',f),('P',p)]:
                    selected=thin(*pr,budget,seed)
                    for mode in ('native','no_padding'):
                        pred=render(model,*selected,mode);h,c=evaluate(pair['GT'][2],pred,groups)
                        names.append(f'{a}_{budget}_{mode}_{seed}');hist.append(h);counts.append(c)
            fe,pe,q=spatial_equal(f,p,seed)
            for a,pr in [('F',fe),('P',pe)]:
                h,c=evaluate(pair['GT'][2],render(model,*pr),groups)
                names.append(f'{a}_spatial_{seed}');hist.append(h);counts.append(c)
        np.savez_compressed(path,names=np.array(names),hist=np.array(hist),counts=np.array(counts))
        if i%10==0:print(f'[budget follow-up] {i+1}/150',flush=True)
if __name__=='__main__':main()
