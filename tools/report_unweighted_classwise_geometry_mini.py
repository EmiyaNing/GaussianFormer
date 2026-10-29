"""Three-arm report: original Set, weighted classwise geometry and uniform geometry."""
from pathlib import Path
import json,math,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.unweighted_classwise_geometry_experiment import *
from tools.report_classwise_chamfer_mini import metrics,csv_write
from tools.report_semantic_field_mini import paired_interval
from tools.report_gaussian_geometry_causal import CLASS_NAMES

REPORT=ROOT/'analysis/unweighted_classwise_geometry_mini_results.md'
NAMES=('baseline','classwise','unweighted_geometry')
LABELS={'baseline':'B：原始几何＋原分类','classwise':'C：逐类加权几何＋原分类',
        'unweighted_geometry':'F：逐类等权几何＋原分类'}
FOLDERS={'baseline':BASELINE_EVAL,'classwise':CLASSWISE_RUN/'eval/classwise',
         'unweighted_geometry':RUN/'eval/unweighted_geometry'}

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
    assert total['load'].shape==total['query_core'].shape==(6019,1200)
    out=RUN/'aggregate';out.mkdir(exist_ok=True);np.savez_compressed(out/f'{name}.npz',**total)
    return total,scenes

def training_audit():
    sums,norms,order={},[],[]
    with (RUN/'train/unweighted_geometry/trace.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);assert math.isfinite(row['loss']);order.append(row['index'])
            for k,v in row['losses'].items():assert math.isfinite(v);sums[k]=sums.get(k,0.)+v
            if 'grad_norm' in row:assert math.isfinite(row['grad_norm']);norms.append(row['grad_norm'])
    with (CLASSWISE_RUN/'train/classwise/trace.jsonl').open() as stream:
        oldorder=[json.loads(line)['index'] for line in stream]
    assert order==oldorder and len(set(order))==len(order)==4096 and len(norms)==2048
    first=json.loads((RUN/'first_frame_check.json').read_text());assert first['status']=='passed'
    means={k:v/4096 for k,v in sums.items()}
    return dict(data_order_exact=True,first_frame=first,means=means,
        gradient_norm_median=float(np.median(norms)),gradient_norm_p95=float(np.quantile(norms,.95)),
        gradient_clip_percent=100*float((np.asarray(norms)>35).mean()),
        geometry_mean=sum(v for k,v in means.items() if k.endswith(('/pred_to_gt','/gt_to_pred'))),
        classification_mean=sum(v for k,v in means.items() if k.endswith('/classification')))

def plot(m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,[('mIoU','mIoU'),('IoU','Binary occupancy IoU'),
        ('geo1','1-sigma coverage'),('same1','Same-class 1-sigma coverage'),
        ('FN_rate','Misses / GT foreground'),('top20_nearest_share','Top20 nearest-query share')]):
        vals=[m[n][key] for n in NAMES];ax.bar(range(3),vals,color=['#8a939b','#cc8c48','#4c9b78'])
        ax.set_xticks(range(3),['B','C','F']);ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,max(vals)*1.2)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Original Set classifier retained; matched 4096-frame fine-tuning, full validation')
    for ext in ('png','pdf'):fig.savefig(ROOT/f'analysis/figures/unweighted_classwise_geometry_mini_comparison.{ext}',dpi=180)
    plt.close(fig)

def report():
    protocol=assert_protocol();arrays,scenes,m={},{},{}
    for n in NAMES:arrays[n],scenes[n]=aggregate(n);m[n]=metrics(arrays[n])
    for n in NAMES[1:]:np.testing.assert_array_equal(arrays[n]['class_coverage'][:,0],arrays['baseline']['class_coverage'][:,0])
    intervals={n:paired_interval(scenes[n],scenes['unweighted_geometry']) for n in NAMES[:2]}
    rows,groups=[],[]
    for n in NAMES:
        cm,cc=arrays[n]['confusion'],arrays[n]['class_coverage']
        for c,label in enumerate(CLASS_NAMES):
            tp=cm[c,c];union=cm[c].sum()+cm[:,c].sum()-tp;gt=cc[c,0]
            rows.append(dict(arm=n,class_id=c,class_name=label,GT=int(gt),IoU=100*tp/max(union,1),
                core1=100*cc[c,1]/max(gt,1),same_core1=100*cc[c,4]/max(gt,1),
                FN=100*cm[c,17]/max(gt,1),class_FP=int(cm[:,c].sum()-tp),FP_from_empty=int(cm[17,c])))
        sorted_load=np.sort(arrays[n]['load'],axis=1)[:,::-1]
        for q in range(5):
            block=sorted_load[:,q*240:(q+1)*240]
            groups.append(dict(arm=n,group=f'{q*20}-{(q+1)*20}%',GT_count=int(block.sum()),
                GT_share=100*block.sum()/cc[:,0].sum(),mean_GT_per_query=float(block.mean())))
    for name,data in [('metrics',[dict(arm=n,**m[n]) for n in NAMES]),('per_class',rows),('query_quintiles',groups)]:csv_write(RUN/f'aggregate/{name}.csv',data)
    audit=training_audit();write(RUN/'optimization_audit.json',audit)
    train=json.loads((RUN/'train/unweighted_geometry/complete.json').read_text())
    evaluation=json.loads((RUN/'eval/unweighted_geometry/complete.json').read_text())
    assert sha(Path(train['checkpoint']))==train['checkpoint_sha256']==evaluation['checkpoint_sha256']
    summary=dict(status='complete',metrics=m,paired_comparisons=intervals,training=train,
        optimization_audit=audit,query_quintiles=groups,classification='Original all-class NN focal; original class and spatial weights',
        geometry='Uniform mean of active class directional means')
    write(RUN/'summary.json',summary);plot(m)
    f,c,b=[m[n] for n in ('unweighted_geometry','classwise','baseline')]
    lo,hi=intervals['classwise']['mIoU_delta_95CI']
    verdict='场景配对区间位于零以上。' if lo>0 else ('场景配对区间位于零以下。' if hi<0 else '场景配对区间跨零。')
    lines=['# 逐类几何等权＋原分类：实验结果','',
        '**实验已完成。** F 使用共同起点在4096帧、698场景上训练1 epoch，完成6019帧、150场景的完整camera区域验证。','',
        f"**相对逐类几何加权的 C，mIoU {f['mIoU']-c['mIoU']:+.3f} 个百分点，二值IoU {f['IoU']-c['IoU']:+.3f}，1σ覆盖 {f['geo1']-c['geo1']:+.3f}。** {verdict}",'',
        '本轮只改变几何类间权重。**分类保留全类最近GT匹配、原类别权重与空间权重**；它不是此前误执行的“逐类几何＋逐类分类”等权分支。详见[实验设计](unweighted_classwise_geometry_mini_design.md)。','',
        '## 1. 任务指标','',
        '| 模型 | mIoU | 二值IoU | 漏检/GT前景 | FP/GT前景 | 检出前景语义准确率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:lines.append('| '+LABELS[n]+' | '+' | '.join(f'{m[n][k]:.3f}%' for k in ('mIoU','IoU','FN_rate','FP_over_GTfg','detected_semantic_accuracy'))+' |')
    lines+=['','| 比较 | mIoU差值95%区间/百分点 | 1σ覆盖差值95%区间/百分点 |','| --- | --- | --- |']
    for n,label in [('baseline','B'),('classwise','C')]:
        a,z=intervals[n]['mIoU_delta_95CI'];u,v=intervals[n]['geo1_delta_95CI']
        lines.append(f'| F−{label} | [{a:+.3f}, {z:+.3f}] | [{u:+.3f}, {v:+.3f}] |')
    lines+=['','区间按150场景配对重采样1500次，种子928；不包含训练种子方差。','',
        '![三组完整验证对比](figures/unweighted_classwise_geometry_mini_comparison.png)','',
        '## 2. 覆盖与query归属','',
        '| 模型 | 1σ | 2σ | 3σ | 同类1σ | 非道路1σ | Top20最近归属 | 零核心接触query |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:lines.append('| '+LABELS[n]+' | '+' | '.join(f'{m[n][k]:.3f}%' for k in ('geo1','geo2','geo3','same1','nonroad_core1','top20_nearest_share','zero_core'))+' |')
    lines+=['','| Query负载组 | B | C | F |','| --- | ---: | ---: | ---: |']
    for q in range(5):lines.append('| '+groups[q]['group']+' | '+' | '.join(f"{groups[i*5+q]['GT_share']:.3f}%" for i in range(3))+' |')
    lines+=['','每帧1200个query独立按最近GS中心的GT归属量排序，每组240个，比例按全验证前景GT汇总。最近归属不要求实际覆盖；它不等于已证明的容量上限。','',
        '## 3. 逐类IoU与覆盖','',
        '| 类别 | B IoU | C IoU | F IoU | F−C IoU/百分点 | F−C 1σ/百分点 | F−C 同类1σ/百分点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    by={(r['arm'],r['class_id']):r for r in rows}
    for i,label in enumerate(CLASS_NAMES):
        bv,cv,fv=[by[n,i] for n in NAMES]
        lines.append(f"| {label} | {bv['IoU']:.3f}% | {cv['IoU']:.3f}% | {fv['IoU']:.3f}% | {fv['IoU']-cv['IoU']:+.3f} | {fv['core1']-cv['core1']:+.3f} | {fv['same_core1']-cv['same_core1']:+.3f} |")
    lines+=['','完整绝对覆盖、漏检率与FP计数见[逐类CSV](../work_dirs/unweighted_classwise_geometry_mini_v1/aggregate/per_class.csv)。','',
        '## 4. 核验与解释范围','',
        '完整配置比较确认只改变 `geometry_class_weights`，原分类 `class_weights` 保持不变。独立小集合参考确认几何值与中心梯度正确，分类值与全部logits梯度和原实现一致；没有同类GT的GS仍受到原分类监督。首个真实训练帧的六级分类和Occupancy loss与C完全相同，4096帧顺序也逐项相同。','',
        f"训练耗时 {train['seconds']/60:.2f} 分钟，峰值已分配CUDA显存 {train['peak_cuda_GB']:.3f} GB。裁剪前总梯度范数中位数 {audit['gradient_norm_median']:.3f}、95%分位 {audit['gradient_norm_p95']:.3f}，触发范数35裁剪的更新占 {audit['gradient_clip_percent']:.3f}%。",'',
        '本轮直接检验几何项的类别权重。有效类别仍先各自求均值，因此等权不等于原始的全局样本归一化。没有修改优化系数或按验证集选择权重；结论仅适用于本次单种子1 epoch微调。','',
        '- [完整指标与区间](../work_dirs/unweighted_classwise_geometry_mini_v1/summary.json)；[优化记录](../work_dirs/unweighted_classwise_geometry_mini_v1/optimization_audit.json)。','']
    REPORT.write_text('\n'.join(lines))
    subprocess.run(['node','.chk.mjs',str(ROOT/'analysis/unweighted_classwise_geometry_mini_design.md'),str(REPORT)],cwd=ROOT,check=True,capture_output=True,text=True)
    for name,digest in protocol['sources'].items():assert sha(RUN/'sources'/name)==digest
    z=arrays['unweighted_geometry'];assert int(z['class_coverage'][:,0].sum())==121325057
    assert z['GS_class_count'].sum()==6019*76800
    cm=z['confusion'];tp=np.diag(cm)[:17];union=cm[:17].sum(1)+cm[:,:17].sum(0)-tp
    assert abs(100*(tp/union).mean()-f['mIoU'])<1e-10
    artifacts=[REPORT,ROOT/'analysis/unweighted_classwise_geometry_mini_design.md',RUN/'summary.json',RUN/'optimization_audit.json',
        *sorted((RUN/'aggregate').glob('*.csv')),ROOT/'analysis/figures/unweighted_classwise_geometry_mini_comparison.png']
    write(RUN/'final_integrity_audit.json',dict(status='passed',checked=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        frozen_sources_verified=len(protocol['sources']),controls_verified=len(protocol['controls']),
        train_frames=4096,optimizer_updates=2048,val_frames=6019,val_scenes=150,
        original_classification_exact=True,data_order_exact=True,checkpoint_sha256=train['checkpoint_sha256'],
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts}))

if __name__=='__main__':report()
