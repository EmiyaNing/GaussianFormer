"""Unified read-only report: both new losses, eight matched historical controls, all queries."""
from pathlib import Path
import csv,itertools,json,math,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.bidirectional_coverage_experiment import *
from tools.report_classwise_chamfer_mini import metrics,csv_write
from tools.report_semantic_field_mini import paired_interval
from tools.report_gaussian_geometry_causal import CLASS_NAMES

REPORT=ROOT/'analysis/bidirectional_coverage_mini_results.md'
NAMES=tuple(CONTROL_FOLDERS)+ARMS
SHORT=dict(baseline='B',classwise='C',strict_classwise='D',unweighted_classwise='E',
    unweighted_geometry='F',neighborhood='N',field_geometry='OG',field_semantic='OS',union='U',decision='R')
LABELS=dict(baseline='B：原目标',classwise='C：逐类加权几何＋原分类',
    strict_classwise='D：逐类几何＋逐类分类',unweighted_classwise='E：D的类间等权分支',
    unweighted_geometry='F：逐类等权几何＋原分类',neighborhood='N：原CD＋局部均距附加项',
    field_geometry='OG：旧几何场附加项',field_semantic='OS：旧局部同类场附加项',
    union='U：核心覆盖＋max错误支持',decision='R：核心覆盖＋真实渲染错误')
FOLDERS=dict(CONTROL_FOLDERS,**{a:RUN/'eval'/a for a in ARMS})

def aggregate(name):
    folder=FOLDERS[name];meta=json.loads((folder/'complete.json').read_text())
    paths=sorted((folder/'scenes').glob('*.npz'));assert len(paths)==150
    total,scenes,indices,tokens,loads,cores={},[],[],[],[],[]
    for path in paths:
        info=json.loads(path.with_suffix('.json').read_text())
        assert info['checkpoint_sha256']==meta['checkpoint_sha256']
        indices+=info['indices'];tokens+=info['tokens']
        with np.load(path) as z:
            row={k:z[k].sum(0) for k in z.files if k not in ('load','query_core','aux_geometry')}
            loads.append(z['load']);cores.append(z['query_core'])
        for k,v in row.items():total[k]=total.get(k,np.zeros_like(v))+v
        scenes.append((path.stem,row))
    assert sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    total['load']=np.concatenate(loads);total['query_core']=np.concatenate(cores)
    total['frame_indices']=np.asarray(indices,dtype=np.int64)
    assert total['load'].shape==total['query_core'].shape==(6019,1200)
    assert int(total['load'].sum())==int(total['class_coverage'][:,0].sum())==121325057
    out=RUN/'aggregate';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'{name}.npz',**total)
    return total,scenes

def global_metrics(a):
    value=metrics(a);load=a['load'];core=a['query_core'];denom=load.sum(1)
    assert (denom>0).all()
    ascending=np.sort(load,axis=1)
    gini=2*(ascending*np.arange(1,1201)).sum(1)/(1200*denom)-(1201/1200)
    value.update(query_gini_frame_mean=float(gini.mean()),
        query_gini_GT_weighted=float(np.average(gini,weights=denom)),
        query_load_cv_frame_mean=float((load.std(1)/load.mean(1)).mean()),
        mean_assigned_GT_per_query=float(load.mean()),mean_core_GT_per_query=float(core.mean()))
    for field,data in [('assigned_GT',load),('core_GT',core)]:
        for q in (.25,.5,.75,.9,.99):value[f'{field}_p{int(q*100)}']=float(np.quantile(data,q))
    descending=ascending[:,::-1]
    for group in range(5):value[f'nearest_share_{20*group}_{20*(group+1)}']=100*float(descending[:,240*group:240*(group+1)].sum()/denom.sum())
    return {k:float(v) for k,v in value.items()}

def class_statistics(name,a):
    rows=[];cm=a['confusion'];cc=a['class_coverage']
    for c,label in enumerate(CLASS_NAMES):
        gt=int(cc[c,0]);tp=int(cm[c,c]);fp=int(cm[:,c].sum()-tp)
        row=dict(arm=name,class_id=c,class_name=label,GT=gt,TP=tp,
            FN_empty=int(cm[c,17]),wrong_foreground=int(cm[c,:17].sum()-tp),
            FP=fp,FP_empty=int(cm[17,c]),FP_other_foreground=fp-int(cm[17,c]),
            IoU=100*tp/(gt+fp),recall=100*tp/gt,precision=100*tp/max(tp+fp,1),
            FN_empty_rate=100*cm[c,17]/gt,FP_over_GT=100*fp/gt)
        for i,key in enumerate(('core1','core2','core3','same_core1','same_core2','same_core3'),1):row[key]=100*cc[c,i]/gt
        rows.append(row)
    return rows

def query_statistics(name,a):
    load,core=a['load'],a['query_core'];total=load.sum()
    median_load=np.median(load,axis=0);median_core=np.median(core,axis=0)
    p90_load=np.quantile(load,.9,axis=0);p90_core=np.quantile(core,.9,axis=0)
    means_load=load.mean(0);means_core=core.mean(0)
    shares=100*load.sum(0)/total;zero_load=100*(load==0).mean(0);zero_core=100*(core==0).mean(0)
    low=100*(core<=20).mean(0)
    return [dict(arm=name,query_id=q,GT_share=float(shares[q]),mean_assigned_GT=float(means_load[q]),
        median_assigned_GT=float(median_load[q]),p90_assigned_GT=float(p90_load[q]),zero_assigned_percent=float(zero_load[q]),
        mean_core_GT=float(means_core[q]),median_core_GT=float(median_core[q]),p90_core_GT=float(p90_core[q]),
        zero_core_percent=float(zero_core[q]),core_le20_percent=float(low[q])) for q in range(1200)]

def training_audit(arm):
    order,norms,sums=[],[],{}
    with (RUN/'train'/arm/'trace.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);assert math.isfinite(row['loss']);order.append(row['index'])
            for key,value in row['losses'].items():
                assert math.isfinite(value);sums[key]=sums.get(key,0.)+value
            if 'grad_norm' in row:assert math.isfinite(row['grad_norm']);norms.append(row['grad_norm'])
    with (REFERENCE_RUN/'train/baseline/trace.jsonl').open() as stream:
        original=[json.loads(line)['index'] for line in stream]
    assert order==original and len(order)==len(set(order))==4096 and len(norms)==2048
    first=json.loads((RUN/'train'/arm/'first_frame_check.json').read_text());assert first['status']=='passed'
    return dict(data_order_exact=True,first_frame=first,
        gradient_norm_median=float(np.median(norms)),gradient_norm_p95=float(np.quantile(norms,.95)),
        gradient_clip_percent=100*float((np.asarray(norms)>35).mean()),
        mean_diagnostics={k:v/4096 for k,v in sums.items()})

def comparison_data(all_metrics,class_rows,query_rows,arrays):
    pairs=[];class_pairs=[];query_pairs=[];query_distributions=[]
    cb={(r['arm'],r['class_id']):r for r in class_rows}
    qb={(r['arm'],r['query_id']):r for r in query_rows}
    for subject,reference in itertools.permutations(NAMES,2):
        pairs.append(dict(subject=subject,reference=reference,
            **{k+'_delta':all_metrics[subject][k]-all_metrics[reference][k] for k in all_metrics[subject]}))
        for c in range(17):
            a,b=cb[subject,c],cb[reference,c]
            class_pairs.append(dict(subject=subject,reference=reference,class_id=c,class_name=CLASS_NAMES[c],
                **{k+'_delta':a[k]-b[k] for k in a if k not in ('arm','class_id','class_name','GT')}))
        if subject not in ARMS:continue
        for q in range(1200):
            a,b=qb[subject,q],qb[reference,q]
            query_pairs.append(dict(subject=subject,reference=reference,query_id=q,
                **{k+'_delta':a[k]-b[k] for k in a if k not in ('arm','query_id')}))
        a,b=arrays[subject]['load'],arrays[reference]['load']
        ac,bc=arrays[subject]['query_core'],arrays[reference]['query_core']
        query_distributions.append(dict(subject=subject,reference=reference,
            pooled_assignment_distribution_TV_percent=50*float(np.abs(a-b).sum()/a.sum()),
            mean_absolute_assigned_GT_change=float(np.abs(a-b).mean()),
            load_increase_entries_percent=100*float((a>b).mean()),load_decrease_entries_percent=100*float((a<b).mean()),
            load_unchanged_entries_percent=100*float((a==b).mean()),
            mean_absolute_core_GT_change=float(np.abs(ac-bc).mean()),
            core_increase_entries_percent=100*float((ac>bc).mean()),core_decrease_entries_percent=100*float((ac<bc).mean())))
    return pairs,class_pairs,query_pairs,query_distributions

def plots(m,classes,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'analysis/figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(folder/f'bidirectional_coverage_{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,[('mIoU','mIoU'),('IoU','Binary occupancy IoU'),
        ('geo1','1-sigma geometric coverage'),('same1','Same-class 1-sigma coverage'),
        ('FN_rate','Misses / GT foreground'),('top20_nearest_share','Top20 nearest-query share')]):
        values=[m[n][key] for n in NAMES]
        ax.bar(range(len(NAMES)),values,color=['#8c969d' if n not in ARMS else ('#397aa4' if n=='union' else '#4c9b78') for n in NAMES])
        ax.set_xticks(range(len(NAMES)),[SHORT[n] for n in NAMES]);ax.set_title(title);ax.set_ylabel('%')
        ax.set_ylim(0,max(values)*1.2)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=7)
    fig.suptitle('Matched 4096-frame fine-tuning; full 6019-frame camera validation')
    save(fig,'all_models')
    cb={(r['arm'],r['class_id']):r for r in classes}
    fig,axes=plt.subplots(3,1,figsize=(14,7.5),constrained_layout=True)
    for ax,key,title in zip(axes,('IoU','core1','same_core1'),('Class IoU change','1-sigma geometric coverage change','Same-class 1-sigma coverage change')):
        data=np.array([[cb[n,c][key]-cb['baseline',c][key] for c in range(17)] for n in ARMS])
        bound=max(.05,float(np.abs(data).max()));im=ax.imshow(data,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_yticks([0,1],['U','R']);ax.set_xticks(range(17),CLASS_NAMES,rotation=55,ha='right',fontsize=8)
        ax.set_title(title+' vs B (percentage points)');fig.colorbar(im,ax=ax,shrink=.8)
        for i in range(2):
            for j in range(17):ax.text(j,i,f'{data[i,j]:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(data[i,j])>.6*bound else 'black')
    save(fig,'class_changes')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for name in ('baseline','classwise','unweighted_geometry',*ARMS):
        load=np.sort(arrays[name]['load'],axis=1)[:,::-1].sum(0)
        axes[0].plot(np.arange(1201)/1200,np.r_[0,np.cumsum(load)/load.sum()],label=SHORT[name])
        thresholds=np.array([0,1,5,10,20,50,100,200,500,1000,5000])
        core=arrays[name]['query_core'];axes[1].plot(thresholds,[100*(core<=x).mean() for x in thresholds],label=SHORT[name])
    axes[0].plot([0,1],[0,1],'--',color='.6');axes[0].set(xlabel='Fraction of queries, ranked per frame',ylabel='Cumulative assigned GT share',title='All-query nearest assignment')
    axes[1].set_xscale('symlog',linthresh=1);axes[1].set(xlabel='Actual 1-sigma core GT contacts per query',ylabel='Query entries at or below threshold (%)',title='All-query core contact distribution')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'query_distributions')

def report():
    if not all((RUN/'eval'/a/'complete.json').exists() for a in ARMS):
        print('Both full evaluations must complete before final reporting',flush=True);return
    protocol=assert_protocol();arrays,scenes,m={}, {}, {}
    classes,queries=[],[]
    for name in NAMES:
        print('Aggregating '+name,flush=True)
        arrays[name],scenes[name]=aggregate(name);m[name]=global_metrics(arrays[name])
        classes.extend(class_statistics(name,arrays[name]));queries.extend(query_statistics(name,arrays[name]))
    for name in NAMES[1:]:
        np.testing.assert_array_equal(arrays[name]['frame_indices'],arrays['baseline']['frame_indices'])
        np.testing.assert_array_equal(arrays[name]['class_coverage'][:,0],arrays['baseline']['class_coverage'][:,0])
    pairs,class_pairs,query_pairs,query_distributions=comparison_data(m,classes,queries,arrays)
    quints=[]
    for name in NAMES:
        load=np.sort(arrays[name]['load'],axis=1)[:,::-1]
        for q in range(5):
            block=load[:,240*q:240*(q+1)]
            quints.append(dict(arm=name,group=f'{20*q}-{20*(q+1)}%',GT_count=int(block.sum()),
                GT_share=100*float(block.sum()/load.sum()),mean_GT_per_query=float(block.mean())))
    for name,rows in [('metrics',[dict(arm=n,**m[n]) for n in NAMES]),('per_class',classes),
        ('query_quintiles',quints),('per_query',queries),('metric_differences',pairs),
        ('per_class_differences',class_pairs),('per_query_differences',query_pairs),
        ('query_distribution_differences',query_distributions)]:
        csv_write(RUN/f'aggregate/{name}.csv',rows)
    intervals={}
    for arm in ARMS:
        for reference in NAMES:
            if reference==arm:continue
            reverse_key=reference+'__minus__'+arm;key=arm+'__minus__'+reference
            if reverse_key in intervals:
                reverse=intervals[reverse_key]
                intervals[key]={k:([-v[1],-v[0]] if k.endswith('_95CI') else v) for k,v in reverse.items()}
            else:intervals[key]=paired_interval(scenes[reference],scenes[arm])
    audits={a:training_audit(a) for a in ARMS}
    write(RUN/'optimization_audit.json',audits)
    training={a:json.loads((RUN/'train'/a/'complete.json').read_text()) for a in ARMS}
    pilot=json.loads((RUN/'pilot.json').read_text())
    summary=dict(status='complete',metrics=m,paired_comparisons=intervals,training=training,
        optimization_audit=audits,pilot=pilot,query_quintiles=quints,
        query_distribution_differences=query_distributions,
        historical_controls=list(CONTROL_FOLDERS),report_source_sha256=sha(Path(__file__)))
    write(RUN/'summary.json',summary);plots(m,classes,arrays)
    b,u,r=[m[n] for n in ('baseline','union','decision')]
    cb={(row['arm'],row['class_id']):row for row in classes}
    delta=lambda subject,reference,key:m[subject][key]-m[reference][key]
    pairfmt=lambda row:f"{row['IoU']:.3f} / {row['core1']:.3f}"
    def classdelta(subject,reference,c):
        a,z=cb[subject,c],cb[reference,c]
        return f"{a['IoU']-z['IoU']:+.3f} / {a['core1']-z['core1']:+.3f}"
    lines=['# 两种双向覆盖损失：完整实验结果','',
        '**两组均已完成。** U和R分别从共同起点在4096帧、698场景上微调1 epoch，各自完成6019帧、150场景的完整camera有效区域验证。','',
        f"**U相对B：mIoU {u['mIoU']-b['mIoU']:+.3f}、1σ覆盖 {u['geo1']-b['geo1']:+.3f} 个百分点；R相对B分别为 {r['mIoU']-b['mIoU']:+.3f}、{r['geo1']-b['geo1']:+.3f}。R−U的mIoU差为 {r['mIoU']-u['mIoU']:+.3f} 个百分点。**",'',
        '两者只替换最终第六级几何，全部原分类与前五级中心CD保留；CE＋Lovasz也保留。U使用max联合错误支持，R使用固定语义/opacity/empty参数后的真实渲染间隔。公式、训练定标和抽样口径见[实验设计](bidirectional_coverage_mini_design.md)。','',
        '## 1. 所有方案的整体指标','',
        '| 方案 | mIoU | ΔmIoU对B | 二值IoU | 1σ覆盖 | 同类1σ | Top20最近归属 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['mIoU']:.3f}% | {v['mIoU']-b['mIoU']:+.3f} | {v['IoU']:.3f}% | {v['geo1']:.3f}% | {v['same1']:.3f}% | {v['top20_nearest_share']:.3f}% |")
    lines+=['','B/C/D/E/F/N/OG/OS的历史权重与逐场景结果只读复用，均核验相同训练帧顺序、2048次更新和6019帧验证。OG/OS为更早的附加场损失，N为局部均距附加项；E是曾误执行的严格分类等权分支，和保留原分类的F不同。','',
        '![全部方案整体比较](figures/bidirectional_coverage_all_models.png)','',
        '## 2. 新两组相对每个方案的变化','',
        '除计数CSV外，本节差值单位均为百分点。漏检、FP下降代表对应错误减少；归属更均衡不单独代表容量问题已解决。','',
        '| 比较 | ΔmIoU | Δ二值IoU | Δ1σ | Δ同类1σ | Δ漏检率 | ΔFP/GT前景 | ΔTop20归属 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    comparisons=[(a,ref) for a in ARMS for ref in CONTROL_FOLDERS]+[('decision','union')]
    for a,ref in comparisons:
        lines.append('| '+SHORT[a]+'−'+SHORT[ref]+' | '+' | '.join(f'{delta(a,ref,k):+.3f}' for k in ('mIoU','IoU','geo1','same1','FN_rate','FP_over_GTfg','top20_nearest_share'))+' |')
    lines+=['','| 核心比较 | mIoU差值95%区间 | 1σ差值95%区间 |','| --- | --- | --- |']
    for a,ref in [('union','baseline'),('decision','baseline'),('decision','union')]:
        ci=intervals[a+'__minus__'+ref];lo,hi=ci['mIoU_delta_95CI'];x,y=ci['geo1_delta_95CI']
        lines.append(f'| {SHORT[a]}−{SHORT[ref]} | [{lo:+.3f}, {hi:+.3f}] | [{x:+.3f}, {y:+.3f}] |')
    lines+=['','场景配对bootstrap按150场景重采样1500次，种子928；区间不包含训练种子方差。所有新组对历史组的区间保存在[汇总JSON](../work_dirs/bidirectional_coverage_mini_v1/summary.json)。','',
        '## 3. 逐类IoU和几何覆盖放在同一张表','',
        '**每格为“IoU / 1σ几何覆盖率”，数值单位为%；最后两列为相对B的变化，单位为百分点。** 原分类始终保留于U/R。','',
        '| 类别 | B：原目标 | F：几何等权＋原分类 | U：max支持 | R：渲染错误 | U−B | R−B |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c,label in enumerate(CLASS_NAMES):
        lines.append('| '+label+' | '+' | '.join(pairfmt(cb[n,c]) for n in ('baseline','unweighted_geometry','union','decision'))+
                     ' | '+classdelta('union','baseline',c)+' | '+classdelta('decision','baseline',c)+' |')
    lines+=['','主要前几版及两新组彼此的逐类变化如下。每格仍为“ΔIoU / Δ1σ覆盖”，单位为百分点。','',
        '| 类别 | U−C | R−C | U−F | R−F | R−U |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for c,label in enumerate(CLASS_NAMES):lines.append('| '+label+' | '+' | '.join(classdelta(a,ref,c) for a,ref in [('union','classwise'),('decision','classwise'),('union','unweighted_geometry'),('decision','unweighted_geometry'),('decision','union')])+' |')
    for reference in ('strict_classwise','unweighted_classwise','neighborhood','field_geometry','field_semantic'):
        lines+=['','<details>',f'<summary>展开：U/R相对{LABELS[reference]}的17类变化</summary>','',
            '| 类别 | U−'+SHORT[reference]+'：ΔIoU / Δ1σ | R−'+SHORT[reference]+'：ΔIoU / Δ1σ |',
            '| --- | ---: | ---: |']
        for c,label in enumerate(CLASS_NAMES):lines.append(f'| {label} | {classdelta("union",reference,c)} | {classdelta("decision",reference,c)} |')
        lines+=['','</details>']
    lines+=['','![新两组逐类变化](figures/bidirectional_coverage_class_changes.png)','',
        '1σ覆盖不要求正确分类；同类覆盖还要求至少一个GS的预测类别与GT相同。两者都不保证该GS能主导渲染结果，也不计其周围新增FP。','',
        '## 4. 1σ、2σ、3σ与同类覆盖','',
        '| 方案 | 1σ | 2σ | 3σ | 同类1σ | 同类2σ | 同类3σ | 非道路1σ | 类别等权1σ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:lines.append('| '+SHORT[n]+' | '+' | '.join(f'{m[n][k]:.3f}%' for k in ('geo1','geo2','geo3','same1','same2','same3','nonroad_core1','macro_core1'))+' |')
    lines+=['','全部17类的1σ/2σ/3σ与同类覆盖、绝对TP、两类FP和漏检见[逐类绝对指标](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_class.csv)；任意两组的逐类变化见[逐类差值](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_class_differences.csv)。','',
        '## 5. 全部query的负载变化','',
        '每帧1200个query、每个query最终64个GS。每个有效前景GT体素先分给最近的最终GS中心，再归给其父query，形成独占的最近归属计数。实际核心接触量则是该query的64个GS在1σ内覆盖的GT并集，不同query之间允许重叠。两种指标分别报告。','',
        '| 方案 | Top20% | 第二20% | 第三20% | 第四20% | 最后20% |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:lines.append('| '+SHORT[n]+' | '+' | '.join(f"{m[n][f'nearest_share_{20*q}_{20*(q+1)}']:.3f}%" for q in range(5))+' |')
    lines+=['','每帧各自排序后划分240个query一组，表中按全验证GT数量汇总。不同模型的同一排名组不代表同一query编号。','',
        '| 方案 | 零最近归属query | 零核心接触query | 核心接触≤20 | 最近归属Gini | 最近归属中位数 | 核心接触中位数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {SHORT[n]} | {v['zero_nearest']:.3f}% | {v['zero_core']:.3f}% | {v['core_le20']:.3f}% | {v['query_gini_frame_mean']:.4f} | {v['assigned_GT_p50']:.1f} | {v['core_GT_p50']:.1f} |")
    lines+=['','Gini先在每帧1200个query上计算再对帧平均；中位数覆盖全部6019×1200个query条目。分布均衡本身不是方法目标，也不等于已验证的容量上限。','',
        '![全部query分布](figures/bidirectional_coverage_query_distributions.png)','',
        '| 方案 | 固定高负载区1σ | 固定高负载区同类1σ | 固定高负载区二值召回 |',
        '| --- | ---: | ---: | ---: |']
    for n in NAMES:lines.append('| '+SHORT[n]+' | '+' | '.join(f'{m[n][k]:.3f}%' for k in ('high_core','high_same_core','high_recall'))+' |')
    lines+=['','固定高负载区域由共同初始权重确定，所有方案评价同一批GT，避免每个模型换一片区域后直接比较。','',
        '- [全部1200个query编号的统计](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_query.csv)：平均/中位/90%分位归属、实际核心接触、零/低接触频率。',
        '- [每个query的新组−历史组变化](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_query_differences.csv)：按相同query编号配对；编号表示模型槽位，不宣称它在不同模型中对应固定地理实体。',
        '- [逐帧分布差异](../work_dirs/bidirectional_coverage_mini_v1/aggregate/query_distribution_differences.csv)：归属分布总变差、负载增加/减少的query条目比例。总变差比较计数分布，不是逐GT匹配变更的准确比例。','',
        '完整逐帧数组保存在下列NPZ的 `load`、`query_core` 两个6019×1200数组中；`frame_indices`记录对应原验证帧索引。由它们可以还原任意两组、任意一帧和query的差值。','',
        '| 方案 | 全部query逐帧数组 |','| --- | --- |']
    for n in NAMES:lines.append(f'| {SHORT[n]} | [NPZ](../work_dirs/bidirectional_coverage_mini_v1/aggregate/{n}.npz) |')
    lines+=['','## 6. 预测错误与场输出','',
        '| 方案 | 漏检/GT前景 | 二值FP/GT前景 | 二值FP/预测前景 | 检出前景语义准确率 | GS中心位于camera前景 | 平均opacity | 平均语义总幅度 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {SHORT[n]} | {v['FN_rate']:.3f}% | {v['FP_over_GTfg']:.3f}% | {v['FP_over_predfg']:.3f}% | {v['detected_semantic_accuracy']:.3f}% | {v['center_camera_fg']:.3f}% | {v['opacity']:.4f} | {v['semantic_amplitude']:.4f} |")
    lines+=['','二值FP只表示empty被预测为前景；逐类FP还包括其他前景类别被预测成本类。逐类CSV将二者分别保存。','',
        '## 7. 训练强度与实现核验','',
        '| 新组 | 外层系数 | 反向系数 | 梯度范数中位数 | 梯度范数95%分位 | 裁剪更新比例 | 训练分钟 | 峰值已分配显存/GB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in ARMS:
        z=audits[a];t=training[a];beta=1. if a=='union' else pilot['decision_reverse_weight']
        lines.append(f"| {SHORT[a]} | {pilot['weights'][a]:.6f} | {beta:.6f} | {z['gradient_norm_median']:.3f} | {z['gradient_norm_p95']:.3f} | {z['gradient_clip_percent']:.3f}% | {t['seconds']/60:.2f} | {t['peak_cuda_GB']:.3f} |")
    lines+=['','两组正向核心距离相同；反向形式及训练定标系数不同，因此U−R比较的是按预定定标规则实现的完整方案。定标只使用预定4个训练帧，不使用验证集。','',
        '三维独立参考、重复GS、稳定类别份额、核心内停止吸引、远缺口梯度、真实渲染前向一致性和直接梯度通路检查均通过。实际训练首帧核对原分类、前五级几何与Occupancy和B一致；两组4096帧顺序逐项相同，全部损失与裁剪前梯度有限。','',
        '反向每帧均匀抽样最多4096个有效体素，并用采样概率加权；它是完整反向目标的随机估计，存在抽样方差。U的每个样本仍精确搜索所有可能最优的GS，没有固定K或指数尾部截断。R保留原渲染候选范围与empty标定。推理结构不变，本轮未重新测FPS。','',
        '## 8. 结论的边界','']
    for a in ARMS:
        change=m[a]['mIoU']-b['mIoU'];lo,hi=intervals[a+'__minus__baseline']['mIoU_delta_95CI']
        if lo>0:verdict='本次固定权重比较的场景配对区间位于零以上。'
        elif hi<0:verdict='本次固定权重比较的场景配对区间位于零以下，尚未带来任务收益。'
        else:verdict='本次场景配对区间跨零，尚无清晰的任务增益。'
        lines += [f"{SHORT[a]}相对B的mIoU变化 {change:+.3f} 个百分点，覆盖变化 {m[a]['geo1']-b['geo1']:+.3f}、Top20归属变化 {m[a]['top20_nearest_share']-b['top20_nearest_share']:+.3f} 个百分点。{verdict}",'']
    lines+=['U还有一个由定义直接决定的限制：实体核心内部的支持为常数1，所以不只正确覆盖会饱和，错误位置若位于获胜GS核心内部，该处spill对几何的直接梯度也为零；其几何调整需要来自其他位置或其他损失。R则按实际渲染错误间隔求导，不使用这一实体核心平台。这个数学差异不等于已证明它是两组性能差异的主要原因。','',
        '新目标改变几何的直接梯度，但原分类、Occupancy和共享网络仍可改变语义幅度、opacity与empty参数。覆盖上升、FP下降或负载更均衡不能单独当作最终机制已被解决的证明；需和IoU、固定区域召回一起判断。','',
        '与C/F等方案比较的是整体接入：它们修改六级几何，U/R只替换最终级。所有比较只有一个训练种子、1 epoch微调；不估计从头训练收益或种子方差。原设计文档中的“未实现/未训练”描述形成设计时的状态，本报告给出后续三维实现与实验。','',
        '## 9. 完整数据与复现','',
        '- [所有绝对指标](../work_dirs/bidirectional_coverage_mini_v1/aggregate/metrics.csv)与[任意两方案整体差值](../work_dirs/bidirectional_coverage_mini_v1/aggregate/metric_differences.csv)。',
        '- [所有逐类指标](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_class.csv)与[任意两方案逐类差值](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_class_differences.csv)。',
        '- [五等分query负载](../work_dirs/bidirectional_coverage_mini_v1/aggregate/query_quintiles.csv)、[全部query统计](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_query.csv)及[新组对全部对照的query变化](../work_dirs/bidirectional_coverage_mini_v1/aggregate/per_query_differences.csv)。',
        '- [U权重](../work_dirs/bidirectional_coverage_mini_v1/train/union/epoch_1.pth)、[R权重](../work_dirs/bidirectional_coverage_mini_v1/train/decision/epoch_1.pth)。',
        '- [训练定标](../work_dirs/bidirectional_coverage_mini_v1/pilot.json)、[优化记录](../work_dirs/bidirectional_coverage_mini_v1/optimization_audit.json)、[源代码与对照协议](../work_dirs/bidirectional_coverage_mini_v1/protocol.json)。','',
        '复现汇总：','', '```bash','python3.10 tools/report_bidirectional_coverage_mini.py','```','']
    REPORT.write_text('\n'.join(lines))
    validate_and_audit(protocol,summary,arrays)

def validate_and_audit(protocol,summary,arrays):
    for name,digest in protocol['sources'].items():assert sha(RUN/'sources'/name)==digest,name
    for name,digest in protocol['checks'].items():assert sha(RUN/name)==digest,name
    def finite(value):
        if isinstance(value,float):assert math.isfinite(value)
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    finite(summary)
    for arm in ARMS:
        t=summary['training'][arm];e=json.loads((RUN/'eval'/arm/'complete.json').read_text())
        assert t['frames']==t['unique_frames']==4096 and t['optimizer_updates']==2048
        assert e['frames']==6019 and e['scenes']==150
        assert sha(Path(t['checkpoint']))==t['checkpoint_sha256']==e['checkpoint_sha256']
        z=arrays[arm];cm=z['confusion'];tp=np.diag(cm)[:17]
        union=cm[:17].sum(1)+cm[:,:17].sum(0)-tp
        assert abs(100*(tp/union).mean()-summary['metrics'][arm]['mIoU'])<1e-10
        assert z['GS_class_count'].sum()==6019*76800
        for ref in ('initial','baseline'):
            np.testing.assert_array_equal(z[ref+'_to_current_classes'].sum(0),z['GS_class_count'])
            np.testing.assert_array_equal(z[ref+'_to_current_classes'].sum(1),z[ref+'_GS_class_count'])
        assert abs(sum(summary['metrics'][arm][f'nearest_share_{20*q}_{20*(q+1)}'] for q in range(5))-100)<1e-9
    check=subprocess.run(['node','.chk.mjs',str(ROOT/'analysis/bidirectional_coverage_mini_design.md'),str(REPORT)],cwd=ROOT,capture_output=True,text=True,check=True)
    artifacts=[REPORT,ROOT/'analysis/bidirectional_coverage_mini_design.md',RUN/'summary.json',RUN/'optimization_audit.json',
        *sorted((RUN/'aggregate').glob('*.csv')),*sorted((ROOT/'analysis/figures').glob('bidirectional_coverage_*.png')),
        *sorted((ROOT/'analysis/figures').glob('bidirectional_coverage_*.pdf')),Path(__file__).resolve()]
    audit=dict(status='passed',checked=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        frozen_sources_verified=len(protocol['sources']),controls_verified=len(protocol['controls']),
        new_train_frames_per_arm=4096,updates_per_arm=2048,val_frames_per_arm=6019,val_scenes=150,
        historical_controls=len(CONTROL_FOLDERS),all_query_entries_per_model=6019*1200,
        all_losses_and_gradients_finite=True,training_order_exact=True,first_frame_preserved_losses_exact=True,
        independent_mIoU_exact=True,all_model_frame_indices_aligned=True,
        document_check=check.stdout.strip(),report_source_sha256=sha(Path(__file__)),
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts})
    write(RUN/'final_integrity_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='artifacts'}),flush=True)

if __name__=='__main__':report()
