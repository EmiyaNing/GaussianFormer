"""Summarize registered mechanism probes with scene-resampled uncertainty."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage1 import RUN,EPOCHS,read,write_json,file_sha256
from tools.eval_empty_decision_optimization import metrics
REPORT=ROOT/'analysis/empty_mechanism/stage1'


def arrays(outcomes):
    return dict(counts=np.asarray([o['counts'] for o in outcomes],np.float64),
        loss=np.asarray([o['ce']+o['lovasz'] for o in outcomes]),
        cls=np.asarray([o['set_cls'] for o in outcomes]),
        rescued=np.asarray([o['rescued'] for o in outcomes]),
        correct_to_empty=np.asarray([o['correct_to_empty'] for o in outcomes]),
        winner_changed=np.asarray([o['foreground_winner_changed'] for o in outcomes]),
        foreground_correct=np.asarray([o['foreground_correct'] for o in outcomes]),
        recoverable=np.asarray([o['cohorts']['recoverable']['n'] for o in outcomes]),
        correct=np.asarray([o['cohorts']['correct']['n'] for o in outcomes]),
        amplitude=np.asarray([o['cohorts']['recoverable']['amplitude_sum'] for o in outcomes]),
        supported=np.asarray([o['cohorts']['recoverable']['supported'] for o in outcomes]),
        share=np.asarray([o['cohorts']['recoverable']['share_sum'] for o in outcomes]),
        margin=np.asarray([o['cohorts']['recoverable']['margin_sum'] for o in outcomes]))


def statistics(a,w):
    c=(w@a['counts'].reshape(len(a['counts']),-1)).reshape(-1,18,18)
    m=metrics(c);occupied=c[:,:17,:].sum((1,2));total=c.sum((1,2))
    def ratio(x,y):return np.divide(w@a[x],y,out=np.zeros(len(w)),where=y>0)
    return dict(miou=m['mIoU'],fnr=m['occupied_FNR'],fpr=m['empty_FPR'],occupancy_iou=m['IoU'],
        field=(w@a['loss'])/w.sum(1),set_cls=(w@a['cls'])/w.sum(1),
        rescue_percent=100*ratio('rescued',w@a['recoverable']),
        correct_to_empty_percent=100*ratio('correct_to_empty',w@a['correct']),
        winner_changed_percent=100*ratio('winner_changed',total),
        foreground_correct_percent=100*ratio('foreground_correct',occupied),
        recoverable_amplitude=ratio('amplitude',w@a['supported']),
        recoverable_share=ratio('share',w@a['recoverable']),
        recoverable_margin=ratio('margin',w@a['recoverable']))


def summarize_group(records,role=None,seed=928):
    bases=[r['baseline'] if role is None else r['baselines'][role] for r in records]
    n=len(records);rng=np.random.default_rng(seed)
    w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=500)])
    baseline=statistics(arrays(bases),w)
    keys=sorted(set((r['branch'],r['radius'],r['pathway']) for r in records[0]['interventions'] if role is None or r['role']==role))
    result=[]
    for branch,radius,pathway in keys:
        selected=[next(x for x in r['interventions'] if (x['branch'],x['radius'],x['pathway'])==(branch,radius,pathway) and (role is None or x['role']==role)) for r in records]
        outcomes=[x['outcome'] for x in selected];value=statistics(arrays(outcomes),w);effect={}
        for name,v in value.items():
            delta=v-baseline[name]
            effect[name]=dict(delta=float(delta[0]),ci95=np.percentile(delta[1:],[2.5,97.5]).tolist(),after=float(v[0]))
        harmed=sum(o['empty_fn']>b['empty_fn'] for o,b in zip(outcomes,bases))
        improved=sum(o['empty_fn']<b['empty_fn'] for o,b in zip(outcomes,bases))
        def own(o):
            if branch=='set_cls':return o['set_cls']
            if branch=='voxel_ce':return o['ce']
            if branch=='voxel_lovasz':return o['lovasz']
            return o['ce']+o['lovasz']+(o['set_cls'] if branch=='joint' else 0.)
        result.append(dict(branch=branch,radius=radius,pathway=pathway,scenes=n,effect=effect,
            empty_fn_delta=sum(o['empty_fn']-b['empty_fn'] for o,b in zip(outcomes,bases)),
            empty_fp_delta=sum(o['empty_fp']-b['empty_fp'] for o,b in zip(outcomes,bases)),
            rescued=sum(o['rescued'] for o in outcomes),correct_to_empty=sum(o['correct_to_empty'] for o in outcomes),
            fn_harmed_scenes=harmed,fn_improved_scenes=improved,
            own_loss_improved_scenes=sum(own(o)<own(b) for o,b in zip(outcomes,bases)),
            gradient_norm_mean=float(np.mean([x['gradient_norm'] for x in selected]))))
    return dict(baseline={name:float(v[0]) for name,v in baseline.items()},conditions=result)


def matched_summary(records,directory,census=False):
    scene={};total=dict(valid=0,occupied=0,recoverable=0,correct=0,recoverable_zero_support=0,matched_population=0,pairs=0)
    pair_sums=np.zeros(4);pair_weight=0.;maxerror=0.;frames_with=0;supporting_frames=0
    for r in records:
        m=r['profile'];meta=r['frame'];key=meta['scene'];index=meta['index']
        fname=f'frame_{index:04d}.npz' if census else f'{r["split"]}_{index:04d}_profile.npz'
        a=np.load(directory/fname)
        for name in total:total[name]+=m[name]
        maxerror=max(maxerror,m['renderer_max_error'])
        if m['recoverable']:frames_with+=1
        w=a['paired_weight'];d=a['paired_delta'];part=(w[:,None]*d).sum(0);weight=w.sum()
        scene.setdefault(key,np.zeros(5));scene[key]+=np.r_[part,weight]
        pair_sums+=part;pair_weight+=weight
        if len(d) and part[1]<0:supporting_frames+=1
    group=np.array(list(scene.values()));n=len(group);rng=np.random.default_rng(928)
    weights=rng.multinomial(n,np.full(n,1/n),size=500);sample=weights@group
    ci=np.percentile(sample[:,:4]/np.maximum(sample[:,4:],1e-30),[2.5,97.5],axis=0)
    mean=pair_sums/max(pair_weight,1e-30)
    differences={}
    for j,name in enumerate(['log_support','log_amplitude','log_share','empty_score']):
        differences[name]=dict(mean=float(mean[j]),ci95=ci[:,j].tolist())
        if j<3:differences[name]['geometric_ratio']=float(np.exp(mean[j]));differences[name]['ratio_ci95']=np.exp(ci[:,j]).tolist()
    return dict(totals=total,frames=len(records),scenes=n,frames_with_recoverable=frames_with,
        frames_lower_matched_amplitude=supporting_frames,matched_differences=differences,renderer_max_error=maxerror)


def gradient_cosines(records):
    values=[]
    for r in records:
        norms={x['branch']:x['gradient_norm'] for x in r['interventions'] if x['pathway']=='full'}
        a=norms['set_cls'];b=norms['voxel_total'];c=norms['joint']
        values.append((c*c-a*a-b*b)/max(2*a*b,1e-30))
    return dict(mean=float(np.mean(values)),minimum=float(min(values)),maximum=float(max(values)),negative_scenes=int((np.array(values)<0).sum()),values=values)


def channel_trace():
    directory=RUN/'amplitude_trace';manifest=read(directory/'manifest.json')
    assert manifest['status']=='complete' and len(manifest['files'])==144
    result={};max_error=0.;max_loss_error=0.
    for epoch in EPOCHS:
        result[str(epoch)]={}
        for split in ['train','val']:
            records=[read(f) for f in sorted(directory.glob(f'epoch{epoch}_{split}_*.json'))]
            assert len(records)==(16 if split=='train' else 32)
            n=len(records);rng=np.random.default_rng(928)
            weights=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=500)])
            groups=[]
            for radius in [1e-4,1e-3,1e-2]:
                rows=[next(t for t in r['rows'] if t['radius']==radius) for r in records]
                entry=dict(radius=radius,cohorts={})
                for cohort in ['recoverable','correct','true_empty','gaussian']:
                    values=[r['gaussian'] if cohort=='gaussian' else r['cohorts'][cohort] for r in rows]
                    count='count' if cohort=='gaussian' else 'supported'
                    denominator=weights@np.asarray([v[count] for v in values])
                    report={}
                    for key in ['positive_sum','negative_sum','total_sum','amplitude_declined']:
                        mean=(weights@np.asarray([v[key] for v in values]))/np.maximum(denominator,1e-30)
                        if key=='amplitude_declined':mean*=100
                        report[key]=dict(mean=float(mean[0]),ci95=np.percentile(mean[1:],[2.5,97.5]).tolist())
                    entry['cohorts'][cohort]=report
                groups.append(entry)
            result[str(epoch)][split]=groups
            for r in records:
                assert r['source_sha256']==file_sha256(ROOT/'tools/trace_empty_mechanism_stage1.py')
                max_loss_error=max(max_loss_error,r['original_training_field_loss_error'])
                for row in r['rows']:max_error=max(max_error,row['rendered_additivity_max_error'])
    return dict(exploratory=True,frames=144,interventions=432,conditions=result,
        max_rendered_additivity_error=max_error,max_original_field_loss_error=max_loss_error)


def audit(summary):
    protocol=read(RUN/'protocol.json');checks=[]
    for epoch in EPOCHS:
        checkpoint=ROOT/protocol['checkpoint'][str(epoch)]['path']
        checks.append(dict(name=f'checkpoint_epoch{epoch}_immutable',passed=file_sha256(checkpoint)==protocol['checkpoint'][str(epoch)]['sha256']))
        for split,number in [('train',16),('val',32)]:
            records=[read(x) for x in sorted((RUN/f'epoch{epoch}').glob(split+'_*.json'))]
            checks.append(dict(name=f'epoch{epoch}_{split}_complete',passed=len(records)==number and all(len(x['interventions'])==45 and x['status']=='complete' for x in records)))
            checks.append(dict(name=f'epoch{epoch}_{split}_source',passed=all(x['source_sha256']==file_sha256(ROOT/'tools/empty_mechanism_stage1.py') and x['protocol_sha256']==file_sha256(RUN/'protocol.json') for x in records)))
            for r in records:
                # Counts, fixed empty counts and transition totals must agree for every path.
                b=np.asarray(r['baseline']['counts']);valid=b.sum();truth=b.sum(1)
                for row in r['interventions']:
                    a=np.asarray(row['outcome']['counts']);o=row['outcome']
                    assert np.array_equal(a.sum(1),truth) and a.sum()==valid
                    assert a[:17,17].sum()==o['empty_fn'] and a[17,:17].sum()==o['empty_fp']
                    assert abs(row['raw_step_relative']/row['radius']-1)<.002
                    for name,cohort in o['cohorts'].items():
                        assert cohort['empty_sum']==r['baseline']['cohorts'][name]['empty_sum']
                assert r['profile']['renderer_max_error']<2e-4
        p=[read(x) for x in sorted((RUN/f'epoch{epoch}'/'parameters').glob('pair_*.json'))]
        checks.append(dict(name=f'epoch{epoch}_parameters_complete',passed=len(p)==8 and all(len(x['interventions'])==36 and x['frozen_geometry_max_error']<1e-6 for x in p)))
        checks.append(dict(name=f'epoch{epoch}_parameters_source',passed=all(x['source_sha256']==file_sha256(ROOT/'tools/empty_mechanism_stage1.py') and x['protocol_sha256']==file_sha256(RUN/'protocol.json') for x in p)))
        for r in p:
            for row in r['interventions']:
                b=r['baselines'][row['role']];o=row['outcome']
                assert np.array_equal(np.sum(o['counts'],axis=1),np.sum(b['counts'],axis=1))
                for name,cohort in o['cohorts'].items():assert cohort['empty_sum']==b['cohorts'][name]['empty_sum']
    c=[read(x) for x in sorted((RUN/'census').glob('frame_*.json'))]
    checks.append(dict(name='census_complete',passed=len(c)==1024 and len({x['frame']['index'] for x in c})==1024))
    checks.append(dict(name='census_source',passed=all(x['source_sha256']==file_sha256(ROOT/'tools/empty_mechanism_stage1.py') for x in c)))
    controls=[r['controls'] for e in EPOCHS for f in sorted((RUN/f'epoch{e}').glob('val_*.json')) if (r:=read(f))['controls']]
    checks.append(dict(name='six_numerical_and_loss_controls',passed=len(controls)==6 and all(x['loss_parity']['geometry_gradient_to_raw_logits_norm']==0 for x in controls)))
    checks.append(dict(name='production_query_grid_unique',passed=read(RUN/'query_grid_audit.json')['duplicates']==0))
    checks.append(dict(name='exploratory_channel_trace_complete',passed=summary['channel_trace']['frames']==144))
    result=dict(status='passed' if all(x['passed'] for x in checks) else 'failed',checks=checks,
        controls=controls,source_sha256=file_sha256(ROOT/'tools/empty_mechanism_stage1.py'),
        local_frames=144,output_interventions=6480,parameter_pairs=24,parameter_interventions=864,census_frames=len(c),channel_trace_interventions=432)
    write_json(RUN/'completion_audit.json',result)
    assert result['status']=='passed',result
    return result


def csv_export(summary):
    rows=[]
    for mode in ['output','parameters']:
        for epoch,groups in summary[mode].items():
            for cohort,group in groups.items():
                for r in group['conditions']:
                    row=dict(mode=mode,epoch=epoch,cohort=cohort,branch=r['branch'],radius=r['radius'],pathway=r['pathway'],scenes=r['scenes'],empty_fn_delta=r['empty_fn_delta'],empty_fp_delta=r['empty_fp_delta'],fn_harmed_scenes=r['fn_harmed_scenes'],fn_improved_scenes=r['fn_improved_scenes'])
                    for metric,d in r['effect'].items():
                        row[metric+'_delta']=d['delta'];row[metric+'_ci_low']=d['ci95'][0];row[metric+'_ci_high']=d['ci95'][1]
                    rows.append(row)
    with (REPORT/'all_interventions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    colors={'amplitude':'#d55e00','composition':'#0072b2','full':'#333333'}
    names={'amplitude':'仅幅度','composition':'仅组成（恢复幅度）','full':'完整更新'}
    labels=['Set 分类','体素 CE','Lovász','体素合计','联合目标']
    branches=['set_cls','voxel_ce','voxel_lovasz','voxel_total','joint']
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.5),sharey=True)
    for ax,epoch in zip(axes,EPOCHS):
        group=summary['output'][str(epoch)]['val'];x=np.arange(5)
        for j,path in enumerate(['amplitude','composition','full']):
            r=[next(r for r in group['conditions'] if r['branch']==b and r['radius']==.001 and r['pathway']==path) for b in branches]
            y=np.array([t['effect']['fnr']['delta'] for t in r]);ci=np.array([t['effect']['fnr']['ci95'] for t in r]).T
            ax.errorbar(x+(j-1)*.18,y,yerr=np.maximum(np.vstack([y-ci[0],ci[1]-y]),0),fmt='o',capsize=3,color=colors[path],label=names[path])
        ax.axhline(0,color='#bbbbbb',lw=1);ax.set_xticks(x,labels,rotation=25);ax.set_title(f'Epoch {epoch} · 32 个验证场景');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('occupied→empty 漏检率变化（百分点）');axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3,frameon=False)
    fig.suptitle('输出路径干预：相同 raw-logit 相对更新量 0.001',y=1.06);fig.tight_layout();fig.savefig(REPORT/'output_paths.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.3),sharey=True)
    for ax,epoch in zip(axes,EPOCHS):
        for j,path in enumerate(['amplitude','composition','full']):
            group=summary['parameters'][str(epoch)]['heldout'];r=[next(r for r in group['conditions'] if r['branch']==b and r['radius']==.0001 and r['pathway']==path) for b in ['set_cls','voxel_total','joint']]
            y=np.array([t['effect']['fnr']['delta'] for t in r]);ci=np.array([t['effect']['fnr']['ci95'] for t in r]).T
            ax.errorbar(np.arange(3)+(j-1)*.18,y,yerr=np.maximum(np.vstack([y-ci[0],ci[1]-y]),0),fmt='o',capsize=3,color=colors[path],label=names[path])
        ax.axhline(0,color='#bbbbbb',lw=1);ax.set_xticks(range(3),['Set 分类','体素合计','联合目标']);ax.set_title(f'Epoch {epoch} · 8 个独立场景');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('occupied→empty 漏检率变化（百分点）');axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3,frameon=False)
    fig.suptitle('真实分类参数更新后：独立场景的路径效应，相对步长 0.0001',y=1.06);fig.tight_layout();fig.savefig(REPORT/'parameter_heldout_paths.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,key,title in zip(axes,['fnr','field'],['漏检率变化（百分点）','CE＋Lovász 变化']):
        for path in ['amplitude','composition','full']:
            group=summary['output']['5']['val'];r=sorted([r for r in group['conditions'] if r['branch']=='set_cls' and r['pathway']==path],key=lambda r:r['radius'])
            ax.plot([t['radius'] for t in r],[t['effect'][key]['delta'] for t in r],'-o',color=colors[path],label=names[path])
        ax.axhline(0,color='#bbbbbb',lw=1);ax.set_xscale('log');ax.set_xlabel('raw-logit 相对 L2 更新量');ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(frameon=False);fig.suptitle('Epoch 5：Set 分类方向的步长敏感性（32 个验证场景）');fig.tight_layout();fig.savefig(REPORT/'dose_response_epoch5.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    REPORT.mkdir(parents=True, exist_ok=True);summary=dict(output={},parameters={},matched={},gradient_alignment={})
    for epoch in EPOCHS:
        summary['output'][str(epoch)]={};summary['matched'][str(epoch)]={};summary['gradient_alignment'][str(epoch)]={}
        for split in ['train','val']:
            records=[read(f) for f in sorted((RUN/f'epoch{epoch}').glob(split+'_*.json'))]
            assert len(records)==(16 if split=='train' else 32)
            summary['output'][str(epoch)][split]=summarize_group(records)
            summary['matched'][str(epoch)][split]=matched_summary(records,RUN/f'epoch{epoch}')
            summary['gradient_alignment'][str(epoch)][split]=gradient_cosines(records)
        records=[read(f) for f in sorted((RUN/f'epoch{epoch}'/'parameters').glob('pair_*.json'))]
        assert len(records)==8
        summary['parameters'][str(epoch)]={role:summarize_group(records,role) for role in ['source','heldout']}
    records=[read(f) for f in sorted((RUN/'census').glob('frame_*.json'))];assert len(records)==1024
    counts=np.sum([r['baseline']['counts'] for r in records],axis=0)
    summary['census']=matched_summary(records,RUN/'census',True);summary['census']['metrics']={k:v.tolist() if np.ndim(v) else float(v) for k,v in metrics(counts).items()}
    summary['census']['counts']=counts.tolist()
    summary['channel_trace']=channel_trace()
    summary['audit']=audit(summary)
    write_json(RUN/'summary.json',summary);csv_export(summary);figures(summary)
    print('Completed summary, audit, full CSV and figures.',flush=True)

if __name__=='__main__':main()
