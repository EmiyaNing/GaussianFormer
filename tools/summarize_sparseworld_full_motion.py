#!/usr/bin/env python3
"""Paired scene summaries of occupancy oracles and historical-query controls."""
import argparse,json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_region_utils import write_json,percentile_interval,CLASS_NAMES

# Empty per-class/horizon groups are retained as NaN support, rendered as "—".
warnings.filterwarnings('ignore',message='All-NaN slice encountered',category=RuntimeWarning)


def estimate(frame,names,weights,scene_names,weight_column=None):
    out={};lookup={s:i for i,s in enumerate(scene_names)}
    for name in names:
        num=np.zeros((len(scene_names),3));den=np.zeros_like(num)
        good=frame[name].notna()
        for (scene,h),group in frame[good].groupby(['scene','horizon_seconds']):
            w=group[weight_column].to_numpy() if weight_column else np.ones(len(group))
            num[lookup[scene],h-1]=np.sum(group[name].to_numpy()*w);den[lookup[scene],h-1]=w.sum()
        total_den=den.sum(0);point=np.divide(num.sum(0),total_den,out=np.full(3,np.nan),where=total_den>0)
        bn=weights@num;bd=weights@den;boot=np.divide(bn,bd,out=np.full_like(bn,np.nan),where=bd>0)
        out[name]=dict(value=point,ci95=percentile_interval(boot),support=total_den,scenes=(den>0).sum(0))
    return out


def all_fast_counts(frame,weights,scene_names):
    lookup={s:i for i,s in enumerate(scene_names)}
    columns=['gt_voxels']+[m+'_'+k for m in ('F','P') for k in ['native_correct','native_free','native_wrong_nonfree','native_wrong_static']+
        [x+'_correct' for x in ('translation','yaw','rigid','size','rigid_size')]]
    data=np.zeros((len(scene_names),3,len(columns)),np.float64)
    for (s,h),g in frame.groupby(['scene','horizon_seconds']):data[lookup[s],h-1]=g[columns].sum().to_numpy()
    total=data.sum(0);boot=(weights@data.reshape(len(scene_names),-1)).reshape(len(weights),3,len(columns))
    out={}
    for i,c in enumerate(columns[1:],1):out[c]=dict(value=100*total[:,i]/total[:,0],ci95=percentile_interval(100*boot[:,:,i]/boot[:,:,0]))
    delta=out['P_native_correct']['value']-out['F_native_correct']['value']
    gaps={}
    fi=columns.index('F_native_correct');pi=columns.index('P_native_correct');original_b=100*(boot[:,:,pi]-boot[:,:,fi])/boot[:,:,0]
    for mode in ('translation','yaw','rigid','size','rigid_size'):
        f=columns.index('F_'+mode+'_correct');p=columns.index('P_'+mode+'_correct')
        remaining=100*(total[:,p]-total[:,f])/total[:,0];remaining_b=100*(boot[:,:,p]-boot[:,:,f])/boot[:,:,0]
        shrink=delta-remaining;shrink_b=original_b-remaining_b
        gaps[mode]=dict(remaining_gap=dict(value=remaining,ci95=percentile_interval(remaining_b)),
            gap_reduction=dict(value=shrink,ci95=percentile_interval(shrink_b)),
            reduction_fraction_of_original_gap=dict(value=100*shrink/delta,ci95=percentile_interval(100*shrink_b/original_b)),
            forecast_gain=dict(value=100*(total[:,f]-total[:,fi])/total[:,0],ci95=percentile_interval(100*(boot[:,:,f]-boot[:,:,fi])/boot[:,:,0])),
            perception_gain=dict(value=100*(total[:,p]-total[:,pi])/total[:,0],ci95=percentile_interval(100*(boot[:,:,p]-boot[:,:,pi])/boot[:,:,0])))
    return dict(voxels=total[:,0],metrics=out,gap_changes=gaps)


def weighted_component_share(frame,numerator,denominator,weights,scene_names):
    lookup={s:i for i,s in enumerate(scene_names)};a=np.zeros((len(scene_names),3));b=np.zeros_like(a)
    for (s,h),g in frame.groupby(['scene','horizon_seconds']):
        w=g.gt_voxels.to_numpy();a[lookup[s],h-1]=np.sum(w*g[numerator]);b[lookup[s],h-1]=np.sum(w*g[denominator])
    ba=weights@a;bb=weights@b
    point=np.divide(a.sum(0),b.sum(0),out=np.full(3,np.nan),where=b.sum(0)>0)*100
    boot=np.divide(ba,bb,out=np.full_like(ba,np.nan),where=bb>0)*100
    return dict(value=point,ci95=percentile_interval(boot))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();root=args.root/('smoke' if args.pilot else '')
    region=json.loads((root/'summary.json').read_text())
    assert args.pilot or region['fast_gate']['passed'],'Full fast advantage not confirmed'
    for p in ('motion_status.json','query_status.json'):assert json.loads((root/p).read_text())['status']=='complete'
    d=dict(np.load(root/'scene_counts.npz'));oracles=pd.read_parquet(root/'motion_instance_oracles.parquet');lineage=pd.read_parquet(root/'query_instance_diagnostics.parquet')
    result=dict(status='smoke_complete' if args.pilot else 'complete',gate=region['fast_gate'],splits={},
                scope='Pilot pipeline check only, no full-data mechanism claim' if args.pilot else 'Full fast-instance denominators; reliable occupancy-match and historical-query subsets explicitly separate')
    for split in ('train','val'):
        scenes=d['scene_names'][d['split']==split].tolist();n=len(scenes);weights=np.random.default_rng(20261010).multinomial(n,np.ones(n)/n,size=2000).astype(float)
        item={}
        for radius in (2.,4.):
            f=oracles[(oracles.split==split)&(oracles.radius_m==radius)].copy();e=f[f.eligible]
            fields=[m+'_'+mode+'_'+metric for m in ('F','P') for mode in ('base','translation','yaw','rigid','size','rigid_size') for metric in ('iou','recall','precision')]
            fields+=[m+'_'+k for m in ('F','P') for k in ('position','orientation','pose','extent','residual','raw_error','centroid_displacement','yaw_abs_degrees')]
            geometric=dict(instance_equal=estimate(e,fields,weights,scenes),voxel_weighted=estimate(e,fields,weights,scenes,'gt_voxels'))
            f['match_fraction']=f.eligible.astype(float)
            item[str(radius)]=dict(instances=int(len(f)),eligible=int(f.eligible.sum()),
                coverage=estimate(f,['match_fraction'],weights,scenes,'gt_voxels'),all_fast=all_fast_counts(f,weights,scenes),geometry=geometric,
                exclusions=f.groupby('exclusion').agg(instances=('gt_voxels','size'),voxels=('gt_voxels','sum')).reset_index().to_dict('records'))
        body=oracles[(oracles.split==split)&(oracles.radius_m==2.)][['source_token','horizon_seconds','instance_token','current_gt_voxels','current_border','future_border']]
        f=lineage[lineage.split==split].merge(body,on=['source_token','horizon_seconds','instance_token'],validate='one_to_one')
        f['bound_fraction']=f.eligible.astype(float);e=f[f.eligible].copy()
        e['template_advantage_over_aligned_actual']=e.template_gt_iou-e.aligned_actual_gt_iou
        comparable=e[(e.current_gt_voxels>=20)&~e.current_border.fillna(True)&~e.future_border&(e.gt_voxels>=20)].copy()
        fields=(['raw_squared','pose_removable_squared','nonrigid_residual_squared','centroid_error_xy','centroid_error_z','yaw_correction_degrees','pair_distance_change_m',
            'template_gt_iou','actual_gt_iou','aligned_actual_gt_iou','template_advantage_over_aligned_actual','box_dimension_change']+
            [k+'_fraction' for k in ('drop_geometry','drop_range','drop_confidence','kept_wrong_class','kept_object_class')])
        item['query_lineage']=dict(instances=len(f),eligible=int(f.eligible.sum()),coverage=estimate(f,['bound_fraction'],weights,scenes,'gt_voxels'),
            instance_equal=estimate(e,fields,weights,scenes),voxel_weighted=estimate(e,fields,weights,scenes,'gt_voxels'),
            pose_share=weighted_component_share(e,'pose_removable_squared','raw_squared',weights,scenes),
            shape_comparable=dict(instances=len(comparable),metrics=estimate(comparable,['template_gt_iou','aligned_actual_gt_iou','template_advantage_over_aligned_actual'],weights,scenes,'gt_voxels'),
                rule='Historical binding; current and future GT >=20 voxels; neither GT box truncated; shape GT comparison only, not the main raw rigidity denominator'),
            per_horizon_instances=[int((f.horizon_seconds==h).sum()) for h in (1,2,3)],
            per_horizon_eligible=[int((e.horizon_seconds==h).sum()) for h in (1,2,3)])
        # Body GT change control excludes unavailable current GT and truncation.
        g=oracles[(oracles.split==split)&(oracles.radius_m==2)&(oracles.current_gt_voxels>=20)&~oracles.future_border&~oracles.current_border.fillna(True)]
        item['gt_body']=dict(instances=len(g),instance_equal=estimate(g,['gt_body_iou','box_dimension_change'],weights,scenes),
                            voxel_weighted=estimate(g,['gt_body_iou','box_dimension_change'],weights,scenes,'gt_voxels'))
        item['by_class']={}
        for label in sorted(f.label.unique()):
            cf=f[f.label==label];ce=e[e.label==label];cg=g[g.label==label];cs=comparable[comparable.label==label]
            item['by_class'][CLASS_NAMES[int(label)]]=dict(instances=len(cf),eligible=int(cf.eligible.sum()),
                query_coverage=estimate(cf,['bound_fraction'],weights,scenes,'gt_voxels'),
                query_geometry=estimate(ce,['raw_squared','pose_removable_squared','nonrigid_residual_squared','pair_distance_change_m','template_advantage_over_aligned_actual'],weights,scenes,'gt_voxels'),
                pose_share=weighted_component_share(ce,'pose_removable_squared','raw_squared',weights,scenes),
                shape_comparable=estimate(cs,['template_advantage_over_aligned_actual'],weights,scenes,'gt_voxels'),
                gt_body=estimate(cg,['gt_body_iou','box_dimension_change'],weights,scenes,'gt_voxels'))
        result['splits'][split]=item
        # The oracle denominator and original correct count must replay exactly.
        cov=region['splits'][split]['attributes']['speed'];native=item['2.0']['all_fast']
        np.testing.assert_array_equal(native['voxels'],np.asarray(cov['total'])[:,2])
        for m,key in [('F','forecast_accuracy'),('P','perception_accuracy')]:
            np.testing.assert_allclose(native['metrics'][m+'_native_correct']['value'],np.asarray(cov[key]['value'])[:,2],rtol=0,atol=1e-10)
    write_json(root/'motion_summary.json',result)
    print('Full native denominators replayed; motion/query summaries complete.',flush=True)

if __name__=='__main__':main()
