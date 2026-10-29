"""Scene bootstrap for pure-foreground misses and pure-empty false positives.

Reuses the original complete census; no inference or training.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(928);boot={}
    for split in ('train','val'):
        rows=[]
        for path in sorted((ROOT/'work_dirs/boundary_gaussian_full/scenes'/split).glob('*.json')):
            e=np.array(json.loads(path.read_text())['error_counts']);row=[]
            for scope in (1,3):
                n=e[scope,7];row.extend([n[2],n[:3].sum(),n[3],n[3:].sum()])
            rows.append(row)
        a=np.array(rows,dtype=np.float64)
        assert len(a)==(700 if split=='train' else 150)
        weights=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=5000).astype(np.float64)
        counts=weights@a;boot[split]=counts[:,::2]/counts[:,1::2]
    result={}
    for j,name in enumerate(('native_pure_fg_miss','native_pure_empty_fp',
                             'core3sigma_pure_fg_miss','core3sigma_pure_empty_fp')):
        result[name]={split+'_95pct_interval':(100*np.quantile(boot[split][:,j],[.025,.975])).tolist()
                      for split in ('train','val')}
        result[name]['val_minus_train_pp_95pct_interval']=(100*np.quantile(
            boot['val'][:,j]-boot['train'][:,j],[.025,.975])).tolist()
    result['protocol']=dict(replicates=5000,seed=928,
        unit='scene, independently resampled within train and val',
        estimator='pooled error counts / pooled conditional GT counts',
        scope='fixed checkpoint; not training seed variability')
    target=ROOT/'work_dirs/gaussian_support_regions_full/error_scene_bootstrap.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
