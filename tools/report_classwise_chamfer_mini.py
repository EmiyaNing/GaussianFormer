"""Readable matched full-validation report for semantic classwise CD."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.classwise_chamfer_experiment import *
from tools.report_semantic_field_mini import metrics as base_metrics,paired_interval
from tools.report_gaussian_geometry_causal import CLASS_NAMES
REPORT=ROOT/'analysis/classwise_chamfer_mini_results.md'
LABELS={'baseline':'B：原双向CD','classwise':'C：逐类加权双向CD'}
NAMES=('baseline','classwise')


def progress():
    t=RUN/'train/classwise/progress.json';e=RUN/'eval/classwise/progress.json'
    td=json.loads(t.read_text()) if t.exists() else {};ed=json.loads(e.read_text()) if e.exists() else {}
    train='4096/4096帧' if td.get('status')=='complete' else f"{td.get('microbatches',0)}/4096帧"
    val='6019/6019帧' if ed.get('status')=='complete' else f"{ed.get('processed',0)}/6019帧，{ed.get('scenes',0)}/150场景"
    lines=['# 按语义类别计算双向 CD：实验结果','',
        '**状态：实验尚未全部完成，暂不判断是否有效。**','',
        '本轮移除前几轮的附加目标，直接替换六个阶段的两个几何CD方向；分类、Occupancy和camera筛选保留。详见[设计报告](classwise_chamfer_mini_design.md)。','',
        '| 对象 | 训练 | 完整验证 |','| --- | --- | --- |',
        '| B：原目标 | 复用已核验同预算权重 | 复用已核验6019帧逐场景结果 |',
        f'| C：逐类加权CD | {train} | {val} |','',
        '数值检查及四帧联合反向已通过。四帧的最终几何回归梯度约为原几何项的2.3–15.4倍；没有据此调整原几何系数。有效类别和无同类匹配的比例将随训练记录。','']
    q=RUN/'queue_progress.json'
    if q.exists():
        state=json.loads(q.read_text());lines += [f"当前阶段：`{state.get('phase','准备')}`；状态：`{state.get('status','准备')}`。",'']
        if state.get('status')=='failed':lines += [f"错误：`{state.get('error')}`。",'']
    REPORT.write_text('\n'.join(lines))


def aggregate(arm):
    folder=BASELINE_EVAL if arm=='baseline' else RUN/'eval'/arm
    meta=json.loads((folder/'complete.json').read_text());paths=sorted((folder/'scenes').glob('*.npz'))
    assert len(paths)==150
    total={};scenes=[];indices=[];tokens=[];load=[];core=[]
    for path in paths:
        info=json.loads(path.with_suffix('.json').read_text());indices+=info['indices'];tokens+=info['tokens']
        assert info['checkpoint_sha256']==meta['checkpoint_sha256']
        with np.load(path) as z:
            # Baseline's previous loss-specific aux_geometry is unrelated here.
            row={k:z[k].sum(0) for k in z.files if k not in ('load','query_core','aux_geometry')}
            load.append(z['load']);core.append(z['query_core'])
        for k,v in row.items():total[k]=total.get(k,np.zeros_like(v))+v
        scenes.append((path.stem,row))
    assert sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    total['load']=np.concatenate(load);total['query_core']=np.concatenate(core)
    folder=RUN/'aggregate';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{arm}.npz',**total)
    return total,scenes


def metrics(a):
    v=base_metrics(a);cc=a['class_coverage'];other=np.arange(17)!=11
    v['macro_core1']=100*float((cc[:,1]/cc[:,0]).mean())
    v['nonroad_core1']=100*float(cc[other,1].sum()/cc[other,0].sum())
    return {k:float(x) for k,x in v.items()}


def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot(m,interval):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(10.5,5.7),constrained_layout=True)
    panels=[('mIoU','mIoU','mIoU_delta_95CI'),('IoU','Binary occupancy IoU',None),
        ('geo1','1-sigma foreground coverage','geo1_delta_95CI'),('same1','Same-class core coverage',None),
        ('FN_rate','Misses / GT foreground',None),('top20_nearest_share','Top 20% nearest-query share',None)]
    for ax,(key,title,ci) in zip(axes.flat,panels):
        delta=m['classwise'][key]-m['baseline'][key];ax.axhline(0,color='.5',lw=.8)
        ax.bar([0],[delta],color='#397aa4',width=.4);anchor=delta
        if ci:
            lo,hi=interval[ci];ax.vlines(0,lo,hi,color='black');ax.scatter([0,0],[lo,hi],marker='_',color='black')
            anchor=max(delta,hi) if delta>=0 else min(delta,lo)
        ax.annotate(f'{delta:+.3f}',(0,anchor),xytext=(0,5 if delta>=0 else -5),textcoords='offset points',ha='center',va='bottom' if delta>=0 else 'top')
        ax.set_xticks([0],['Classwise weighted CD']);ax.set_title(title,fontsize=10);ax.set_ylabel('Change vs baseline (pp)')
        ax.grid(axis='y',alpha=.2);lo,hi=ax.get_ylim();margin=max(.03,(hi-lo)*.25);ax.set_ylim(lo-margin,hi+margin)
    fig.suptitle('Full Occ3D validation: 4096-frame matched fine-tuning')
    folder=ROOT/'analysis/figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(folder/f'classwise_chamfer_mini_comparison.{ext}',dpi=180)
    plt.close(fig)


def report():
    if not (RUN/'eval/classwise/complete.json').exists():progress();return
    assert_protocol();arrays={};scenes={};m={}
    for arm in NAMES:arrays[arm],scenes[arm]=aggregate(arm);m[arm]=metrics(arrays[arm])
    old=np.load(REFERENCE_RUN/'aggregate/baseline.npz')
    for key in ('confusion','class_coverage'):np.testing.assert_array_equal(old[key],arrays['baseline'][key])
    np.testing.assert_array_equal(arrays['baseline']['class_coverage'][:,0],arrays['classwise']['class_coverage'][:,0])
    interval=paired_interval(scenes['baseline'],scenes['classwise'])
    csv_write(RUN/'aggregate/metrics.csv',[dict(arm=n,**m[n]) for n in NAMES])
    classrows=[]
    for n in NAMES:
        cm=arrays[n]['confusion'];cc=arrays[n]['class_coverage']
        for c,label in enumerate(CLASS_NAMES):
            tp=cm[c,c];union=cm[c].sum()+cm[:,c].sum()-tp;gt=cc[c,0]
            classrows.append(dict(arm=n,class_id=c,class_name=label,GT=int(gt),IoU=100*tp/max(union,1),
                core1=100*cc[c,1]/max(gt,1),same_core1=100*cc[c,4]/max(gt,1),FN=100*cm[c,17]/max(gt,1),FP_from_empty=int(cm[17,c])))
    csv_write(RUN/'aggregate/per_class.csv',classrows)
    trace=[json.loads(s) for s in (RUN/'train/classwise/trace.jsonl').read_text().splitlines()]
    oldtrace=[json.loads(s) for s in (REFERENCE_RUN/'train/baseline/trace.jsonl').read_text().splitlines()]
    assert len(trace)==4096 and [x['index'] for x in trace]==[x['index'] for x in oldtrace]
    # First identical input: all six original classification terms and Occupancy exact.
    first=trace[0]['losses'];other=oldtrace[0]['losses'];diffs={}
    for k,v in first.items():
        if k.endswith('/classification'):
            oldkey=k.replace('ClasswiseChamferSetLoss/','CenterSupportDiagnosticSetLoss/')
            diffs[k]=abs(v-other[oldkey])
        elif k=='OccupancyLoss' or k.startswith('OccupancyLoss/'):
            diffs[k]=abs(v-other[k])
    assert diffs and max(diffs.values())==0
    diagnostic_keys=sorted({k for x in trace for k in x['losses'] if '/classwise/' in k})
    trainmeans={k:float(np.mean([x['losses'][k] for x in trace])) for k in diagnostic_keys}
    class_train=[]
    stage_train=[]
    prefix='ClasswiseChamferSetLoss/'
    for s in range(6):
        tag=f'{prefix}stage_{s}/classwise'
        stage_train.append(dict(stage=s,active_classes=trainmeans[tag+'/active_classes_mean'],
            unmatched_pred_percent=100*trainmeans[tag+'/unmatched_pred_fraction'],unmatched_GT_percent=100*trainmeans[tag+'/unmatched_GT_fraction'],
            p2g_same_m=trainmeans[tag+'/p2g_same_m'],g2p_same_m=trainmeans[tag+'/g2p_same_m']))
        for c,name in enumerate(CLASS_NAMES):
            ct=f'{tag}/class_{c}';npred=np.array([x['losses'][ct+'/pred_count'] for x in trace]);ngt=np.array([x['losses'][ct+'/GT_count'] for x in trace])
            active=(npred>0)&(ngt>0)
            class_train.append(dict(stage=s,class_id=c,class_name=name,mean_pred_count=float(npred.mean()),mean_GT_count=float(ngt.mean()),
                GT_present_frames=int((ngt>0).sum()),GS_present_frames=int((npred>0).sum()),active_frames=int(active.sum()),
                GT_without_GS_frames=int(((ngt>0)&(npred==0)).sum()),GS_without_GT_frames=int(((npred>0)&(ngt==0)).sum()),
                mean_pred_term_all_frames=trainmeans[ct+'/pred_term'],mean_GT_term_all_frames=trainmeans[ct+'/GT_term']))
    csv_write(RUN/'aggregate/training_class_counts.csv',class_train)
    training=json.loads((RUN/'train/classwise/complete.json').read_text());pilot=json.loads((RUN/'pilot.json').read_text())
    summary=dict(status='complete',metrics=m,paired_vs_baseline=interval,training=training,
        first_frame_unchanged_loss_maximum_difference=max(diffs.values()),data_order_exact=True,
        training_frame_mean_diagnostics=trainmeans,training_stage_summary=stage_train,pilot=pilot,
        baseline_eval_source=str(BASELINE_EVAL.relative_to(ROOT)))
    write(RUN/'summary.json',summary);plot(m,interval)
    b,c=m['baseline'],m['classwise'];d=c['mIoU']-b['mIoU'];a,z=interval['mIoU_delta_95CI']
    if a>0:verdict='本次固定模型比较中，mIoU提高，场景配对区间也位于零以上。'
    elif z<0:verdict='本轮mIoU下降，场景配对区间也位于零以下，当前版本没有得到任务收益。'
    else:verdict='本轮mIoU差值的场景配对区间包含零，尚无清晰的任务收益。'
    lines=['# 按语义类别计算双向 CD：实验结果','',
        '**实验已完成。** C组使用原small初始化，在4096帧、698个场景上微调1epoch；以下均为完整6019帧、150个验证场景的camera有效区域统计。','',
        f"**相对同预算B组，mIoU变化{d:+.3f}个百分点，1σ核心覆盖变化{c['geo1']-b['geo1']:+.3f}个百分点，漏检率变化{c['FN_rate']-b['FN_rate']:+.3f}个百分点。** {verdict}",'',
        '设计与公式见[设计报告](classwise_chamfer_mini_design.md)。本轮直接替换六级双向几何CD，不叠加上一轮附加项目。B为相同起点、顺序、优化器和预算的原目标继续训练；B的权重及逐场景验证结果只读复用。','',
        '## 1. 预测性能','',
        '| 模型 | mIoU ↑ | 二值IoU ↑ | 漏检/GT前景 ↓ | FP/GT前景 ↓ | FP/预测前景 ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['mIoU']:.3f}% | {v['IoU']:.3f}% | {v['FN_rate']:.3f}% | {v['FP_over_GTfg']:.3f}% | {v['FP_over_predfg']:.3f}% |")
    lo,hi=interval['geo1_delta_95CI']
    lines += ['',f'mIoU差值的场景配对95%区间为[{a:+.3f}, {z:+.3f}]个百分点；1σ覆盖差值区间为[{lo:+.3f}, {hi:+.3f}]个百分点。按150个场景配对重采样1500次，固定种子928。它不包含训练种子变化。','',
        'mIoU是17个前景类别IoU的等权平均。漏检是真实前景预测为empty；FP是真实empty预测为前景。先合并所有帧的混淆矩阵，再计算上表比例。','',
        '![逐类CD相对原目标的变化](figures/classwise_chamfer_mini_comparison.png)','',
        '## 2. 几何与同类覆盖','',
        '| 模型 | 1σ几何覆盖 | 2σ几何覆盖 | 3σ几何覆盖 | 同预测类别1σ覆盖 | 非道路1σ覆盖 | 17类等权1σ覆盖 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append('| '+LABELS[n]+' | '+' | '.join(f'{v[k]:.3f}%' for k in ('geo1','geo2','geo3','same1','nonroad_core1','macro_core1'))+' |')
    p=arrays['classwise']['paired_baseline'];r=100*p[:4]/max(p[4],1)
    lines += ['',f'相对B，新增/丢失1σ几何覆盖分别占GT前景{r[0]:.3f}%/{r[1]:.3f}%；新增/丢失同预测类别核心覆盖为{r[2]:.3f}%/{r[3]:.3f}%。一个GT被多个GS覆盖仅计一次。','',
        '覆盖统计使用最终第六级实际GS和旋转椭球，而非第五级中心或最近距离的替代指标。同类覆盖要求GS当前预测类别与GT相同；非道路排除driveable_surface。','',
        '## 3. Query负载与固定高负载区域','',
        '| 模型 | 最近归属top20%份额 | 零核心接触query | 核心接触≤20体素query | 核心覆盖top20%的GT并集 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append('| '+LABELS[n]+' | '+' | '.join(f'{v[k]:.3f}%' for k in ('top20_nearest_share','zero_core','core_le20','top20_core_union'))+' |')
    lines += ['', '最近归属把每个GT分给最近最终GS的父query，不要求该GS实际覆盖它；核心接触统计一个query的64个GS覆盖并集，不同query可以重叠。因此归属份额下降不能单独证明负载问题已解决。','',
        '| 固定初始高负载GT区域 | 1σ几何覆盖 | 同预测类别核心覆盖 | 二值召回 | 正确语义召回 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append('| '+LABELS[n]+' | '+' | '.join(f'{v[k]:.3f}%' for k in ('high_core','high_same_core','high_recall','high_semantic_recall'))+' |')
    lines += ['', '高负载区域由初始权重逐帧确定，之后两模型在同一GT集合上比较，避免用变化后的集合混淆区域收益。','',
        '## 4. 逐类别结果','',
        '| 类别 | B IoU | C IoU | IoU变化/百分点 | 核心覆盖变化/百分点 | 同类核心变化/百分点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    by={(x['arm'],x['class_id']):x for x in classrows}
    for i,label in enumerate(CLASS_NAMES):
        u,v=by['baseline',i],by['classwise',i]
        lines.append(f"| {label} | {u['IoU']:.3f}% | {v['IoU']:.3f}% | {v['IoU']-u['IoU']:+.3f} | {v['core1']-u['core1']:+.3f} | {v['same_core1']-u['same_core1']:+.3f} |")
    lines += ['', '## 5. 训练中有多少类别真正参与匹配','',
        '| 阶段（从0开始） | 平均有效类数 | 无同类GT的GS占比 | 无同类GS的GT占比 | 同类GS→GT距离/m | 同类GT→GS距离/m |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s in stage_train:
        lines.append(f"| {s['stage']} | {s['active_classes']:.3f}/17 | {s['unmatched_pred_percent']:.3f}% | {s['unmatched_GT_percent']:.3f}% | {s['p2g_same_m']:.3f} | {s['g2p_same_m']:.3f} |")
    ratios=[x['final_regression_gradient_ratio'] for x in pilot['records']]
    lines += ['', '上表为训练帧等权平均，不是完整验证指标。距离只统计有同类目标的点，不含空类回退；原全类最近距离另存训练日志。第5阶段是最终第六级。逐阶段逐类别的数量和缺失帧数见[类别匹配CSV](../work_dirs/classwise_chamfer_mini_v1/aggregate/training_class_counts.csv)。','',
        f"四个预定训练帧的新几何回归梯度范数是原几何项的{min(ratios):.2f}–{max(ratios):.2f}倍。由于同类匹配距离可更远且每类先平均，沿用原系数不等于沿用相同的有效优化强度。本轮没有调整系数，也没有按验证结果选权重。",'',
        '## 6. 中心、形状与证据','',
        '| 模型 | 中心在有效前景 | 中心在有效empty | 中心在未知区 | 最大轴饱和 | 平均最大轴/m | 平均opacity | 平均前景幅度和 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        v=m[n];lines.append(f"| {LABELS[n]} | {v['center_camera_fg']:.3f}% | {v['center_camera_empty']:.3f}% | {v['center_unknown']:.3f}% | {v['major_scale_saturated']:.3f}% | {v['major_scale']:.5f} | {v['opacity']:.5f} | {v['semantic_amplitude']:.5f} |")
    lines += ['', '未知包括camera无效、标签无效或范围外。几何CD的直接梯度只到中心；尺度、opacity和语义仍可由Occupancy及共享网络间接改变。这些表格记录相关变化，不能单凭它们认定某个属性是性能变化的主因。','',
        '## 7. 对照核验与结论范围','',
        '首个真实训练输入上的六级Set分类和Occupancy损失与B完全一致，最大差值为0。4096帧训练顺序与B逐项相同；原分类匹配及权重不变，GT继续使用camera筛选。配置中只有新Set loss与原Occupancy两项，没有上一轮附加损失。','',
        '数值检查覆盖17类穷举参考、中心梯度与有限差分、CUDA最近邻、单类退化为原CD、缺少GS/GT、所有类不相交、类别加权归一化以及分类损失完全一致。验证统计使用旧覆盖位图和16个GT位置的全部GS穷举复核。','',
        '本轮同时改变同类匹配与类别加权汇总，并使用一次固定系数、单种子、1epoch微调。无论结果正负，都不能单独证明某项改变的贡献，也不代表已经建立训练因果机制或穷尽所有类别几何监督形式。','',
        f"训练耗时{training['seconds']/60:.2f}分钟，峰值已分配CUDA显存{training['peak_cuda_GB']:.3f}GB。推理结构不变，本轮未重新测FPS。",'',
        '- [完整指标与区间](../work_dirs/classwise_chamfer_mini_v1/summary.json)；[逐类CSV](../work_dirs/classwise_chamfer_mini_v1/aggregate/per_class.csv)。',
        '- [冻结协议](../work_dirs/classwise_chamfer_mini_v1/protocol.json)；[数值检查](../work_dirs/classwise_chamfer_mini_v1/numerical_checks.json)；[四帧梯度预检](../work_dirs/classwise_chamfer_mini_v1/pilot.json)。',
        '- 权重：`work_dirs/classwise_chamfer_mini_v1/train/classwise/epoch_1.pth`；逐帧训练日志及150个场景的完整统计保存在同一实验目录。','']
    REPORT.write_text('\n'.join(lines));print(json.dumps(dict(status='complete',metrics=m,interval=interval),ensure_ascii=False))

if __name__=='__main__':report()
