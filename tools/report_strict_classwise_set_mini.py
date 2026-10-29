"""Three-arm full-validation report; new strict loss vs both frozen controls."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.strict_classwise_set_experiment import *
from tools.report_classwise_chamfer_mini import metrics,csv_write
from tools.report_semantic_field_mini import paired_interval
from tools.report_gaussian_geometry_causal import CLASS_NAMES

REPORT=ROOT/'analysis/strict_classwise_set_mini_results.md'
NAMES=('baseline','classwise','strict_classwise')
LABELS={'baseline':'B：原几何＋原分类','classwise':'C：逐类几何＋原分类',
        'strict_classwise':'D：逐类几何＋逐类分类'}


def progress():
    train=RUN/'train/strict_classwise/progress.json';evaluation=RUN/'eval/strict_classwise/progress.json'
    t=json.loads(train.read_text()) if train.exists() else {}
    e=json.loads(evaluation.read_text()) if evaluation.exists() else {}
    ts='4096/4096' if t.get('status')=='complete' else f"{t.get('microbatches',0)}/4096"
    es='6019/6019' if e.get('status')=='complete' else f"{e.get('scenes',0)}/150 场景"
    REPORT.write_text('\n'.join(['# 几何与分类统一逐类匹配：实验结果','',
        '**实验尚未全部完成，当前不判断是否有效。**','',
        '执行细节与自强化含义见[实验设计](strict_classwise_set_mini_design.md)。','',
        '| 模型 | 训练 | 完整验证 |','| --- | --- | --- |',
        '| B：原目标 | 复用已核验同预算权重 | 复用完整结果 |',
        '| C：逐类几何＋原分类 | 复用已核验同预算权重 | 复用完整结果 |',
        f'| D：逐类几何＋逐类分类 | {ts} | {es} |','',
        'D 从同一预训练起点重新训练；分类按预测类别同类匹配，每类平均后加权，权重不重复使用。','']))


def aggregate(arm):
    folders={'baseline':BASELINE_EVAL,'classwise':PREVIOUS_RUN/'eval/classwise',
             'strict_classwise':RUN/'eval/strict_classwise'}
    folder=folders[arm];meta=json.loads((folder/'complete.json').read_text())
    paths=sorted((folder/'scenes').glob('*.npz'));assert len(paths)==150
    total={};scenes=[];indices=[];tokens=[];loads=[];cores=[]
    for p in paths:
        info=json.loads(p.with_suffix('.json').read_text())
        assert info['checkpoint_sha256']==meta['checkpoint_sha256']
        indices+=info['indices'];tokens+=info['tokens']
        with np.load(p) as z:
            row={k:z[k].sum(0) for k in z.files if k not in ('load','query_core','aux_geometry')}
            loads.append(z['load']);cores.append(z['query_core'])
        for k,v in row.items():total[k]=total.get(k,np.zeros_like(v))+v
        scenes.append((p.stem,row))
    assert sorted(indices)==list(range(6019)) and len(set(tokens))==6019
    total['load']=np.concatenate(loads);total['query_core']=np.concatenate(cores)
    assert total['load'].shape==total['query_core'].shape==(6019,1200)
    folder=RUN/'aggregate';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{arm}.npz',**total)
    return total,scenes


def trace_audit():
    sums={};norms=[];order=[];first=None
    with (RUN/'train/strict_classwise/trace.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);first=first or row;order.append(row['index'])
            for k,v in row['losses'].items():sums[k]=sums.get(k,0.)+v
            if 'grad_norm' in row:norms.append(row['grad_norm'])
    with (PREVIOUS_RUN/'train/classwise/trace.jsonl').open() as stream:
        oldfirst=json.loads(next(stream));oldorder=[oldfirst['index']]
        oldorder.extend(json.loads(line)['index'] for line in stream)
    assert order==oldorder and len(order)==len(set(order))==4096 and len(norms)==2048
    diffs={}
    for k,v in first['losses'].items():
        if k.endswith(('/pred_to_gt','/gt_to_pred')):
            key=k.replace('StrictClasswiseSetLoss/','ClasswiseChamferSetLoss/')
            diffs[k]=abs(v-oldfirst['losses'][key])
        elif k=='OccupancyLoss' or k.startswith('OccupancyLoss/'):
            diffs[k]=abs(v-oldfirst['losses'][k])
    assert diffs and max(diffs.values())==0
    means={k:v/4096 for k,v in sums.items()}
    geom=sum(v for k,v in means.items() if k.endswith(('/pred_to_gt','/gt_to_pred')))
    cls=sum(v for k,v in means.items() if k.endswith('/classification'))
    return dict(data_order_exact=True,first_geometry_and_occupancy_max_difference=max(diffs.values()),
        means=means,gradient_norm_median=float(np.median(norms)),gradient_norm_p95=float(np.quantile(norms,.95)),
        gradient_clip_percent=100*float((np.asarray(norms)>35).mean()),
        mean_weighted_geometry=geom,mean_weighted_classification=cls)


def plot(m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    panels=[('mIoU','mIoU'),('IoU','Binary occupancy IoU'),('geo1','1-sigma coverage'),
            ('same1','Same-class 1-sigma coverage'),('FN_rate','Misses / GT foreground'),
            ('top20_nearest_share','Top20 nearest-query share')]
    for ax,(key,title) in zip(axes.flat,panels):
        values=[m[n][key] for n in NAMES]
        ax.bar(range(3),values,color=['#8a939b','#cc8c48','#397aa4'])
        ax.set_xticks(range(3),['B','C','D']);ax.set_title(title);ax.set_ylabel('%')
        ax.set_ylim(0,max(values)*1.19)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Matched 4096-frame fine-tuning; full 6019-frame camera validation')
    folder=ROOT/'analysis/figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(folder/f'strict_classwise_set_mini_comparison.{ext}',dpi=180)
    plt.close(fig)


def report():
    if not (RUN/'eval/strict_classwise/complete.json').exists():progress();return
    assert_protocol();arrays={};scenes={};m={}
    for name in NAMES:arrays[name],scenes[name]=aggregate(name);m[name]=metrics(arrays[name])
    for n in NAMES[1:]:np.testing.assert_array_equal(arrays[n]['class_coverage'][:,0],arrays['baseline']['class_coverage'][:,0])
    intervals={n:paired_interval(scenes[n],scenes['strict_classwise']) for n in NAMES[:2]}
    csv_write(RUN/'aggregate/metrics.csv',[dict(arm=n,**m[n]) for n in NAMES])
    classes=[];groups=[]
    for name in NAMES:
        cm=arrays[name]['confusion'];cc=arrays[name]['class_coverage']
        for c,label in enumerate(CLASS_NAMES):
            tp=cm[c,c];union=cm[c].sum()+cm[:,c].sum()-tp;gt=cc[c,0]
            classes.append(dict(arm=name,class_id=c,class_name=label,GT=int(gt),IoU=100*tp/max(union,1),
                core1=100*cc[c,1]/max(gt,1),same_core1=100*cc[c,4]/max(gt,1),
                FN=100*cm[c,17]/max(gt,1),class_FP=int(cm[:,c].sum()-tp),FP_from_empty=int(cm[17,c])))
        load=arrays[name]['load'];sorted_load=np.sort(load,axis=1)[:,::-1]
        for q in range(5):
            counts=sorted_load[:,q*240:(q+1)*240]
            groups.append(dict(arm=name,group=f'{q*20}-{(q+1)*20}%',
                GT_count=int(counts.sum()),GT_share=100*counts.sum()/cc[:,0].sum(),mean_GT_per_query=float(counts.mean())))
    csv_write(RUN/'aggregate/per_class.csv',classes);csv_write(RUN/'aggregate/query_quintiles.csv',groups)
    audit=trace_audit();write(RUN/'optimization_audit.json',audit)
    final=arrays['strict_classwise'];hist=[]
    for c,label in enumerate(CLASS_NAMES):
        hist.append(dict(class_id=c,class_name=label,initial_GS=int(final['initial_GS_class_count'][c]),
            baseline_GS=int(final['baseline_GS_class_count'][c]),strict_GS=int(final['GS_class_count'][c])))
    csv_write(RUN/'aggregate/GS_class_counts.csv',hist)
    switch={n:100*(1-np.trace(final[n+'_to_current_classes'])/final[n+'_to_current_classes'].sum())
            for n in ('initial','baseline')}
    training=json.loads((RUN/'train/strict_classwise/complete.json').read_text())
    summary=dict(status='complete',metrics=m,paired_comparisons=intervals,training=training,
        optimization_audit=audit,query_quintiles=groups,GS_class_switch_percent=switch,
        no_validation_tuning=True,classification_target='detached current argmax via nearest same-class GT')
    write(RUN/'summary.json',summary);plot(m)
    d=m['strict_classwise'];c=m['classwise'];delta=d['mIoU']-c['mIoU'];lo,hi=intervals['classwise']['mIoU_delta_95CI']
    if lo>0:verdict='相对上一组，mIoU 提高，场景配对区间位于零以上。'
    elif hi<0:verdict='相对上一组，mIoU 下降，场景配对区间位于零以下。'
    else:verdict='相对上一组，mIoU 差值的场景配对区间跨零，当前没有清晰增益。'
    lines=['# 几何与分类统一逐类匹配：实验结果','',
        '**实验已完成。** D 在相同 4096 帧、698 场景上微调 1 epoch，并完成全部 6019 帧、150 场景的 camera 有效区域验证。','',
        f"**D 相对 C：mIoU {delta:+.3f} 个百分点，二值 IoU {d['IoU']-c['IoU']:+.3f} 个百分点，1σ 覆盖 {d['geo1']-c['geo1']:+.3f} 个百分点。** {verdict}",'',
        '执行规则、公式和自强化含义见[设计文档](strict_classwise_set_mini_design.md)。B、C 均复用已核验的同预算结果；D 从相同预训练起点重新训练。','',
        '## 1. 最终预测性能','',
        '| 模型 | mIoU | 二值 IoU | 漏检/GT前景 | FP/GT前景 | 检出前景语义准确率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    # The detected foreground semantic accuracy is computed directly from confusion.
    for n in NAMES:
        cm=arrays[n]['confusion'];accuracy=100*np.trace(cm[:17,:17])/max(cm[:17,:17].sum(),1)
        v=m[n];lines.append(f"| {LABELS[n]} | {v['mIoU']:.3f}% | {v['IoU']:.3f}% | {v['FN_rate']:.3f}% | {v['FP_over_GTfg']:.3f}% | {accuracy:.3f}% |")
    lines+=['','区间按 150 个场景配对重采样 1500 次，种子 928；单位为百分点，不包含训练种子方差。','',
        '| 比较 | mIoU 差值的 95% 区间 | 核心覆盖差值的 95% 区间 |','| --- | --- | --- |']
    for n in NAMES[:2]:
        a,b=intervals[n]['mIoU_delta_95CI'];u,v=intervals[n]['geo1_delta_95CI']
        lines.append(f'| D − {n} | [{a:+.3f}, {b:+.3f}] | [{u:+.3f}, {v:+.3f}] |')
    lines+=['','![三组完整验证对比](figures/strict_classwise_set_mini_comparison.png)','',
        '## 2. 最终第六级 GS 覆盖','',
        '| 模型 | 1σ | 2σ | 3σ | 同类1σ | 非道路1σ | 17类等权1σ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:
        lines.append('| '+LABELS[n]+' | '+' | '.join(f'{m[n][k]:.3f}%' for k in ('geo1','geo2','geo3','same1','nonroad_core1','macro_core1'))+' |')
    lines+=['','覆盖为 GT 前景落入旋转椭球支持的并集比例；同类要求 GS 当前预测类别与 GT 相同。存在覆盖不等于最终渲染预测正确。','',
        '## 3. Query 负载和固定困难区域','',
        '| 模型 | Top20最近归属 | 零核心接触query | 核心接触≤20体素query | 固定高负载区1σ | 固定高负载区二值召回 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for n in NAMES:lines.append('| '+LABELS[n]+' | '+' | '.join(f'{m[n][k]:.3f}%' for k in ('top20_nearest_share','zero_core','core_le20','high_core','high_recall'))+' |')
    lines+=['','高负载区域由共同初始权重确定，两模型比较同一 GT 集合。最近归属来自最近最终 GS 的父 query，不要求实际覆盖，不代表已验证的容量上限。','',
        '| Query负载排序组 | B GT份额 | C GT份额 | D GT份额 |','| --- | ---: | ---: | ---: |']
    for q in range(5):lines.append('| '+groups[q]['group']+' | '+' | '.join(f"{groups[i*5+q]['GT_share']:.3f}%" for i in range(3))+' |')
    lines+=['','每帧独立排序，每组 240/1200 个 query；比例按全验证 GT 计数汇总。不同模型同一组不代表相同 query 身份。','',
        '## 4. 逐类别性能与覆盖','',
        '| 类别 | B IoU | C IoU | D IoU | D−C IoU/百分点 | D−C 1σ/百分点 | D−C 同类1σ/百分点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    by={(r['arm'],r['class_id']):r for r in classes}
    for i,label in enumerate(CLASS_NAMES):
        b,c,d=[by[n,i] for n in NAMES]
        lines.append(f"| {label} | {b['IoU']:.3f}% | {c['IoU']:.3f}% | {d['IoU']:.3f}% | {d['IoU']-c['IoU']:+.3f} | {d['core1']-c['core1']:+.3f} | {d['same_core1']-c['same_core1']:+.3f} |")
    lines+=['','完整绝对覆盖、漏检和 FP 计数见[逐类CSV](../work_dirs/strict_classwise_set_mini_v1/aggregate/per_class.csv)。','',
        '## 5. 分类自强化与类别分布','',
        'D 的有效分类目标必然等于当前 argmax 类别，这由匹配规则决定，不作为学习正确的证据。以下分歧统计改用全类最近 GT 作为空间参照；它描述 Set 标签分歧，不是场景语义准确率。','',
        '| 阶段 | 有效类别均值 | 跳过GS占比 | 跳过GT占比 | 训练分类目标与最近GT类别一致率 |',
        '| --- | ---: | ---: | ---: | ---: |']
    means=audit['means']
    for s in range(6):
        p=f'StrictClasswiseSetLoss/stage_{s}'
        lines.append(f"| {s} | {means[p+'/classwise/active_classes_mean']:.3f} | {100*means[p+'/classwise/unmatched_pred_fraction']:.3f}% | {100*means[p+'/classwise/unmatched_GT_fraction']:.3f}% | {100*means[p+'/strict_classification/agreement_with_nearest_GT']:.3f}% |")
    lines+=['','阶段 0–5 对应六级，阶段 5 为最终 GS。训练表按帧等权平均，匹配 GT 继续先经过 camera 筛选。','',
        f"完整验证中，同索引 GS 相对初始化的类别变化比例为 {switch['initial']:.3f}%，相对 B 为 {switch['baseline']:.3f}%。这不是跟踪物理对象身份。C 历史验证未保存逐 GS 类别，因此不提供 C→D 类别切换比例。",'',
        '| 类别 | 初始化平均GS/帧 | B平均GS/帧 | D平均GS/帧 |','| --- | ---: | ---: | ---: |']
    for r in hist:lines.append(f"| {r['class_name']} | {r['initial_GS']/6019:.1f} | {r['baseline_GS']/6019:.1f} | {r['strict_GS']/6019:.1f} |")
    lines+=['','## 6. 优化强度与对照核验','',
        f"D 裁剪前总梯度范数中位数 {audit['gradient_norm_median']:.3f}，95%分位 {audit['gradient_norm_p95']:.3f}，触发范数35裁剪的更新比例 {audit['gradient_clip_percent']:.3f}%。六级加权几何均值 {audit['mean_weighted_geometry']:.3f}，加权分类均值 {audit['mean_weighted_classification']:.3f}。这些描述训练轨迹，不等于逐项梯度贡献。",'',
        '数值检查通过17类独立全对参考、分类logits有限差分、全部17维正负类梯度、缺失类别跳过、空交集零值及类别权重只使用一次。四个固定真实帧上，六级几何损失与 C 完全相同，最终几何回归梯度也完全相同，联合反向有限。','',
        '首个实际训练输入上的几何与 Occupancy 损失和 C 完全一致，4096帧顺序逐项相同。B、C权重与完整验证产物只读复用并校验哈希。完整验证覆盖全部150场景，无验证集调参。','',
        f"训练耗时 {training['seconds']/60:.2f} 分钟，峰值已分配 CUDA 显存 {training['peak_cuda_GB']:.3f} GB。推理结构不变，本轮未重测 FPS。",'',
        '## 7. 结论的适用范围','',
        'D 相对 C 同时改变了分类目标来源、分类的类内平均与类间汇总，以及缺少同类GT时的分类跳过规则。该比较检验用户确认的严格逐类版本，不能将收益或退化单独归因于消除分类—几何冲突。只有一轮固定系数、单训练种子的微调；自强化不能作为正确语义监督的替代证明。','',
        '- [完整指标与区间](../work_dirs/strict_classwise_set_mini_v1/summary.json)；[优化审计](../work_dirs/strict_classwise_set_mini_v1/optimization_audit.json)。',
        '- [冻结协议](../work_dirs/strict_classwise_set_mini_v1/protocol.json)；[数值检查](../work_dirs/strict_classwise_set_mini_v1/numerical_checks.json)；[真实帧预检](../work_dirs/strict_classwise_set_mini_v1/pilot.json)。',
        '- [五组query CSV](../work_dirs/strict_classwise_set_mini_v1/aggregate/query_quintiles.csv)；[GS类别数量CSV](../work_dirs/strict_classwise_set_mini_v1/aggregate/GS_class_counts.csv)。',
        '- 最终权重：`work_dirs/strict_classwise_set_mini_v1/train/strict_classwise/epoch_1.pth`。','']
    REPORT.write_text('\n'.join(lines))
    print(json.dumps({'status':'complete','metrics':m,'intervals':intervals},ensure_ascii=False))

if __name__=='__main__':report()
