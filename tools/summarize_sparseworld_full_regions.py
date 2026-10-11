#!/usr/bin/env python3
"""Full train/val paired scene statistics; refuse incomplete censuses."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_region_utils import state_metrics,metric_from_hist,percentile_interval,write_json,sha256,DYNAMIC
from tools.analyze_sparseworld_region_pairs import attribute_counts,standardized_contrast


def boot_sum(values,weights):
    # All integers are far below 2**53; float64 BLAS sums remain exact counts.
    return (weights.astype(np.float64)@values.reshape(len(values),-1).astype(np.float64)).reshape(len(weights),*values.shape[1:])


def summarize_states(c,weights):
    point=state_metrics(c.sum(0));boot=state_metrics(boot_sum(c,weights))
    result={k:dict(value=v,ci95=percentile_interval(boot[k])) for k,v in point.items() if k!='total'}
    result.update(total=point['total'],scenes=(c.sum(-1)>0).sum(0))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--pilot',action='store_true')
    args=p.parse_args();root=args.root/('smoke' if args.pilot else '')
    for name in ('status.json','statistics_status.json'):assert json.loads((root/name).read_text())['status']=='complete'
    protocol=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    d=dict(np.load(root/'scene_counts.npz'));scene_names=d['scene_names'].tolist();mapping={s:i for i,s in enumerate(scene_names)}
    common={k:d[k].copy() for k in ('basic','joint','covariates')};common_samples=d['samples'][:,2].copy()
    for r in manifest['samples']:
        if not all(t['available'] for t in r['targets']):
            with np.load(root/'sample_stats'/(r['token']+'.npz')) as row:
                for k in common:common[k][mapping[r['scene']]]-=row[k]
    result=dict(status='complete',sources=len(manifest['samples']),paired_targets=manifest['paired_targets'],protocol=protocol,
        units=dict(accuracy='percent',advantage='P minus F percentage points',interval='95% paired scene resampling, 2000 repeats; census counts exact'),splits={})
    for split in ('train','val'):
        ids=np.flatnonzero(d['split']==split);n=len(ids);weights=np.random.default_rng(20261010).multinomial(n,np.ones(n)/n,size=2000)
        cov=d['covariates'][ids];attrs=attribute_counts(cov);attrs['lidar']=d['lidar'][ids];attrs['boundary08']=d['boundary08'][ids]
        item=dict(sources=int(d['sources'][ids].sum()),samples=d['samples'][ids].sum(0),scenes=n,
            states=summarize_states(d['basic'][ids],weights),attributes={k:summarize_states(v,weights) for k,v in attrs.items()},
            common_six_future_sources=int(common_samples[ids].sum()),common_states=summarize_states(common['basic'][ids],weights),
            common_attributes={k:summarize_states(v,weights) for k,v in attribute_counts(common['covariates'][ids]).items()})
        item['models']={};item['model_difference']={};models_boot={}
        for name,axis in [('F',-1),('P',-2)]:
            hist=d['joint'][ids].sum(axis);point=metric_from_hist(hist.sum(0));boot=metric_from_hist(boot_sum(hist,weights));models_boot[name]=boot
            item['models'][name]={k:dict(value=v,ci95=percentile_interval(boot[k])) for k,v in point.items()}
        for k in models_boot['F']:
            item['model_difference'][k]=dict(value=np.asarray(item['models']['P'][k]['value'])-np.asarray(item['models']['F'][k]['value']),ci95=percentile_interval(models_boot['P'][k]-models_boot['F'][k]))
        item['standardized']={name:standardized_contrast(cov,factor,levels,weights,dynamic=dynamic) for name,factor,levels,dynamic in [
            ('fast_vs_slow','speed',[0,2],True),('boundary_vs_interior','boundary',[0,1],False),
            ('camera_visible_vs_invisible','camera',[0,1],False),('future_only_vs_both','frustum',[3,1],False)]}
        item['joint_error_counts']=d['errors'][ids].sum(0);item['spatial_sensitivity_counts']=d['sensitivity'][ids].sum(0)
        x,y=np.meshgrid(np.arange(-39,40,2),np.arange(-39,40,2),indexing='ij')
        near=x*x+y*y<100
        groups={'前左':(x>=0)&(y>=0),'前右':(x>=0)&(y<0),'后左':(x<0)&(y>=0),'后右':(x<0)&(y<0),
                '近车 10m':near,'近车前左':near&(x>=0)&(y>=0),'近车前右':near&(x>=0)&(y<0),
                '近车后左':near&(x<0)&(y>=0),'近车后右':near&(x<0)&(y<0)}
        xy=d['xy'][ids].reshape(n,3,40,40,4)
        spatial=np.stack([xy[:,:,mask,:].sum(2) for mask in groups.values()],axis=2)
        item['spatial_groups']=dict(names=list(groups),statistics=summarize_states(spatial,weights),
            definition='Occupied GT; x forward/y left; 2m block-center radial cutoff for near-car groups')
        result['splits'][split]=item
        print('[full summary]',split,'fast P-F',np.asarray(item['attributes']['speed']['advantage']['value'])[:,2],flush=True)
    train=np.asarray(result['splits']['train']['attributes']['speed']['advantage']['value'])[:,2]
    val=result['splits']['val']['attributes']['speed']['advantage'];vv=np.asarray(val['value'])[:,2];ci=np.asarray(val['ci95'])[:,2]
    result['fast_gate']=dict(passed=bool((train>0).all() and (vv>0).all() and (ci[:,0]>0).all()),
        train_advantage=train,val_advantage=vv,val_ci95=ci,rule='All three train/val fast P-F point differences >0; all three val paired 95% intervals >0')
    write_json(root/'summary.json',result)
    print('[fast gate]',result['fast_gate'],flush=True)

if __name__=='__main__':main()
