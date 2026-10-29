"""Posthoc CPU audit of optimization strength; never changes the experiment."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.classwise_chamfer_experiment import RUN,REFERENCE_RUN,config,sha,write,assert_protocol
from tools.report_gaussian_geometry_causal import CLASS_NAMES
REPORT=ROOT/'analysis/classwise_chamfer_mini_results.md'
MARKER='<!-- classwise-optimization-audit-v1 -->'


def main():
    assert_protocol()
    assert (RUN/'summary.json').exists()
    traces={}
    for arm,path in [('baseline',REFERENCE_RUN/'train/baseline/trace.jsonl'),('classwise',RUN/'train/classwise/trace.jsonl')]:
        traces[arm]=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(traces[arm])==4096
    assert [v['index'] for v in traces['baseline']]==[v['index'] for v in traces['classwise']]
    stats={}
    for arm,rows in traces.items():
        norms=np.array([r['grad_norm'] for r in rows if 'grad_norm' in r]);assert len(norms)==2048 and np.isfinite(norms).all()
        prefix='CenterSupportDiagnosticSetLoss/' if arm=='baseline' else 'ClasswiseChamferSetLoss/'
        geometry=np.array([sum(v for k,v in r['losses'].items() if k.startswith(prefix) and (k.endswith('/pred_to_gt') or k.endswith('/gt_to_pred'))) for r in rows])
        stats[arm]=dict(updates=len(norms),gradient_norm_median=float(np.median(norms)),gradient_norm_p95=float(np.quantile(norms,.95)),
            gradient_norm_max=float(norms.max()),clipped_updates=int((norms>35).sum()),clipped_update_percent=100*float((norms>35).mean()),
            mean_total_six_stage_geometry=float(geometry.mean()),median_total_six_stage_geometry=float(np.median(geometry)))
    weights=np.asarray(config('classwise').loss.loss_cfgs[0].geometry_class_weights);contributions=np.zeros(17)
    maximum_error=0.
    for r in traces['classwise']:
        expected=0.;computed=0.
        for s in range(6):
            tag=f'ClasswiseChamferSetLoss/stage_{s}';d=r['losses'];den=d[tag+'/classwise/weight_sum_mean']
            expected+=d[tag+'/pred_to_gt']+d[tag+'/gt_to_pred']
            if den<=0:continue
            for c in range(17):
                ct=tag+f'/classwise/class_{c}'
                term=.5*weights[c]/den*(d[ct+'/pred_term']+d[ct+'/GT_term'])
                contributions[c]+=term;computed+=term
        maximum_error=max(maximum_error,abs(computed-expected))
    assert maximum_error<1e-3,maximum_error
    shares=100*contributions/contributions.sum()
    classrows=[dict(class_id=c,class_name=CLASS_NAMES[c],weight=float(weights[c]),
        mean_six_stage_geometry_contribution=float(contributions[c]/4096),share_percent=float(shares[c])) for c in range(17)]
    result=dict(status='passed',scope='Posthoc CPU analysis only; frozen training/evaluation sources unmodified',
        per_arm=stats,geometry_class_contributions=classrows,maximum_loss_reconstruction_error=maximum_error,
        note='Norms are recorded before clipping; all updates use max norm35. Clip frequency alone does not identify the cause of performance changes.')
    write(RUN/'optimization_strength_audit.json',result)
    lines=['',MARKER,'## 8. 相同几何系数是否代表相同优化强度','',
        '不代表。逐类平均取消了点数多类别的数量优势，同类约束也可能将最近邻换成更远的目标。这里用训练日志直接核对，避免仅用预检的四帧梯度推断整个过程。','',
        '| 同预算2048次更新 | B：原CD | C：逐类CD |','| --- | ---: | ---: |']
    for key,label,unit in [('gradient_norm_median','裁剪前总梯度范数中位数',''),('gradient_norm_p95','裁剪前总梯度范数95%分位',''),
        ('clipped_update_percent','触发范数35裁剪的更新比例','%'),('mean_total_six_stage_geometry','六级几何损失的训练帧均值','')]:
        lines.append(f"| {label} | {stats['baseline'][key]:.3f}{unit} | {stats['classwise'][key]:.3f}{unit} |")
    lines += ['', '范数为所有模型参数的总梯度，读取于裁剪之前；几何损失已包含原阶段权重与0.5系数。帧顺序完全相同，但模型更新后的状态不同，因此这些数字反映两条实际训练轨迹，不能看成固定参数下的逐帧因果分解。','',
        '本轮几何项按预测类别硬分组，分组切换处的目标还可能发生跳变。频繁裁剪说明这版直接替换改变了优化尺度；它本身既不能证明类别几何监督无效，也不能单独认定为性能变化的主导原因。AdamW的适应性更新也不能用裁剪比例直接换算成等效学习率。','',
        '各类别对六级几何损失的平均贡献如下。贡献已经过有效类别权重归一化，不等同于原始类别点数：','',
        '| 类别 | 外层权重 | 六级几何平均贡献 | 占新几何项比例 |','| --- | ---: | ---: | ---: |']
    for r in sorted(classrows,key=lambda x:x['share_percent'],reverse=True):
        lines.append(f"| {r['class_name']} | {r['weight']:g} | {r['mean_six_stage_geometry_contribution']:.4f} | {r['share_percent']:.3f}% |")
    lines += ['', '这些份额用于发现目标是否由少数类别主导，不能直接解释逐类别IoU变化。缺失同类目标、同类长距离匹配和整体优化强度需要分开判断。','',
        '[完整优化强度审计](../work_dirs/classwise_chamfer_mini_v1/optimization_strength_audit.json)。重新生成基础报告后，运行`python3.10 tools/finalize_classwise_chamfer_report.py`恢复本节。','']
    # Explain the macro/micro coverage split from the actual full-val counts.
    baseline=np.load(RUN/'aggregate/baseline.npz');current=np.load(RUN/'aggregate/classwise.npz')
    bc,tc=baseline['class_coverage'],current['class_coverage'];bcm,tcm=baseline['confusion'],current['confusion']
    foreground=int(bc[:,0].sum());groups=[]
    for name,ids in [('类别0–10',np.arange(11)),('类别11–16',np.arange(11,17))]:
        gt=int(bc[ids,0].sum());old=int(bc[ids,1].sum());new=int(tc[ids,1].sum())
        groups.append(dict(name=name,GT=gt,GT_share_percent=100*gt/foreground,
            baseline_core_percent=100*old/gt,current_core_percent=100*new/gt,net_core=new-old))
    examples=[]
    for c in (2,3,4,7):
        oldfp=int(bcm[:,c].sum()-bcm[c,c]);newfp=int(tcm[:,c].sum()-tcm[c,c])
        examples.append(dict(class_id=c,class_name=CLASS_NAMES[c],baseline_TP=int(bcm[c,c]),current_TP=int(tcm[c,c]),
            baseline_class_FP=oldfp,current_class_FP=newfp,added_TP=int(tcm[c,c]-bcm[c,c]),added_class_FP=newfp-oldfp,
            baseline_IoU=float(bcm[c,c]/(bc[c,0]+oldfp)),
            added_TP_over_added_FP=float((tcm[c,c]-bcm[c,c])/(newfp-oldfp))))
    mean_gt=np.array([np.mean([v['losses'][f'ClasswiseChamferSetLoss/stage_5/classwise/class_{c}/GT_count'] for v in traces['classwise']]) for c in range(17)])
    scalar_share=float(shares[[0,7]].sum());gt_share=100*float(mean_gt[[0,7]].sum()/mean_gt.sum())
    old_iou=np.diag(bcm)[:17]/(bcm[:17].sum(1)+bcm[:,:17].sum(0)-np.diag(bcm)[:17])
    new_iou=np.diag(tcm)[:17]/(tcm[:17].sum(1)+tcm[:,:17].sum(0)-np.diag(tcm)[:17])
    outcome=dict(status='passed',groups=groups,classes_with_core_gain=int((tc[:,1]>bc[:,1]).sum()),
        classes_with_IoU_gain=int((new_iou>old_iou).sum()),examples=examples,
        net_core_gain=int(tc[:,1].sum()-bc[:,1].sum()),vegetation_net_core_gain=int(tc[16,1]-bc[16,1]),
        added_FN=int(tcm[:17,17].sum()-bcm[:17,17].sum()),added_FP=int(tcm[17,:17].sum()-bcm[17,:17].sum()),
        others_plus_pedestrian_training_GT_share_percent=gt_share,
        others_plus_pedestrian_scalar_geometry_share_percent=scalar_share)
    write(RUN/'interpretation_audit.json',outcome)
    lines += ['## 9. 为什么类别等权覆盖上升，性能仍然下降','',
        f"本轮{outcome['classes_with_core_gain']}/17类的1σ覆盖提高，但17类的IoU全部下降。几何覆盖收益没有转化为对应的预测收益。类别分化尤其明显：",'',
        '| 类别组 | 占验证GT前景 | B核心覆盖 | C核心覆盖 | 净新增核心体素 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        lines.append(f"| {g['name']} | {g['GT_share_percent']:.3f}% | {g['baseline_core_percent']:.3f}% | {g['current_core_percent']:.3f}% | {g['net_core']:+,} |")
    lines += ['', '0–10为others、barrier、各类交通参与者等；11–16为道路、其他平面、人行道、地形、建筑和植被。前一组的11类覆盖都增加，后一组的6类都下降。', '',
        f"前11类合计获得276,665个核心覆盖，后6类丢失4,019,291个，整体净减少3,742,626个。仅植被就净减少2,470,436个。17类等权平均每类权重相同，因此会被较多类别的小幅改善抬高；按GT体素计数的整体覆盖则受到大体积类别损失影响。这里尚未统计验证GS类别数量变化，不能把覆盖分化直接解释为GS数量的重新分配。",'',
        '### 9.1 为什么覆盖改善的类别，其IoU也下降','',
        '以以下类别为例，正确分类的TP确实增加，但预测为该类的FP增加更多。这里FP包括真实empty与其他前景类别被错判为该类，和前表的二值误占用定义不同：','',
        '| 类别 | 新增正确分类TP | 新增该类FP | 新增TP/新增FP | 原IoU |', '| --- | ---: | ---: | ---: | ---: |']
    for e in examples:
        lines.append(f"| {e['class_name']} | {e['added_TP']:+,} | {e['added_class_FP']:+,} | {100*e['added_TP_over_added_FP']:.3f}% | {100*e['baseline_IoU']:.3f}% |")
    lines += ['', '固定类别GT数量时，类别IoU等于TP除以“GT＋该类FP”。设原GT数量为G、原TP为T、原该类FP为F，新增TP为t、新增FP为f。当f大于零时，IoU下降的准确条件为：', '', '$$', r'\frac{T+t}{G+F+f}<\frac{T}{G+F}\quad\Longleftrightarrow\quad\frac{t}{f}<\frac{T}{G+F},\qquad f>0.', '$$', '', '四个例子中，新增TP/新增FP都低于各自原IoU，因此IoU下降。单独比较新增TP和新增FP的数量大小不足以判断IoU；这里同时核对了原IoU和完整混淆矩阵。', '',
        f"从整体二值预测看，FP净减少{abs(outcome['added_FP']):,}个，FN却增加{outcome['added_FN']:,}个，二值IoU因此下降。核心覆盖与实际渲染决策需分别评价。",'',
        '### 9.2 本轮最需要注意的优化因素','',
        f"Others和pedestrian合计仅占这批训练GT的{gt_share:.3f}%，却贡献了新六级几何损失数值的{scalar_share:.3f}%。这来自类内先平均、外层类别权重及实际匹配距离的共同作用。它说明目标数值重心明显偏向部分小类；损失份额不等于梯度份额，本轮没有逐类别梯度分解。",'',
        '同时，新几何目标均值约为原版的4.56倍，98.78%的更新触发裁剪，而原版为3.96%。这些实测结果支持优先检查监督强度与类别平衡，不能单独证明频繁裁剪就是性能下降的原因。按预测类别硬分组也可能引入不合适的远距离吸引，但本轮没有固定分组的反事实实验，尚不能确定其贡献。', '',
        '**本轮判断：按当前权重、硬分类分组和原几何系数直接替换双向CD，未达到预期。** 它改善了部分类别的几何覆盖，却损害了大体积区域的覆盖，且所有类别IoU下降。结果不支持保留这版配置，也不足以否定所有同类几何监督；若继续，应把类别汇总、匹配规则和优化强度分开检验。','',
        '[类别分化与混淆计数审计](../work_dirs/classwise_chamfer_mini_v1/interpretation_audit.json)。','']
    existing=REPORT.read_text()
    if MARKER not in existing:REPORT.write_text(existing+'\n'.join(lines))
    else:REPORT.write_text(existing.split(MARKER)[0].rstrip()+'\n'+'\n'.join(lines))
    write(RUN/'report_provenance.json',dict(status='complete',sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in [REPORT,RUN/'optimization_strength_audit.json',RUN/'interpretation_audit.json',RUN/'summary.json']}))
    print(json.dumps(stats))

if __name__=='__main__':main()
