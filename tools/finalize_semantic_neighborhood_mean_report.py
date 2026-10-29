"""Interpretation of completed raw counts; does not alter training or frozen sources."""
from pathlib import Path
import json,sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.semantic_neighborhood_experiment import RUN,ARMS,write,sha
ARM=ARMS[0]

def main():
    result=json.loads((RUN/'summary.json').read_text());assert result['status']=='complete'
    m=result['metrics'];b=m['baseline'];v=m[ARM]
    aggregate={n:dict(np.load(RUN/'aggregate'/f'{n}.npz')) for n in ('baseline',ARM)}
    paths=sorted((RUN/'eval/baseline/scenes').glob('*.npz'));raw={}
    for n in ('baseline',ARM):
        rows=[]
        for path in paths:
            with np.load(RUN/'eval'/n/'scenes'/path.name) as z:
                high=z['high'].sum(0);q=z['queries'].sum(0);fg=z['class_coverage'].sum((0,1))[0]
                rows.append(np.r_[high,q,fg])
        raw[n]=np.stack(rows)
    # Input vector: high(5), queries(9), GT(1). Every ratio uses scene sums.
    rng=np.random.default_rng(928);selections=rng.integers(0,150,(1500,150));samples=[]
    for ids in selections:
        values=[]
        for n in ('baseline',ARM):
            a=raw[n][ids].sum(0)
            values.append(np.array([100*a[1]/a[0],100*a[3]/a[0],100*a[10]/a[14],100*a[9]/a[5]]))
        samples.append(values[1]-values[0])
    names=('high_core','high_recall','top20_nearest_share','core_le20')
    intervals={key:np.quantile(np.asarray(samples)[:,i],[.025,.975]).tolist() for i,key in enumerate(names)}
    total_gain=int(aggregate[ARM]['class_coverage'][:,1].sum()-aggregate['baseline']['class_coverage'][:,1].sum())
    high_gain=int(aggregate[ARM]['high'][1]-aggregate['baseline']['high'][1])
    rows=list(csv.DictReader((RUN/'aggregate/per_class.csv').open()));old={r['name']:r for r in rows if r['arm']=='baseline'}
    nbetter=sum(float(r['core1'])>float(old[r['name']]['core1']) for r in rows if r['arm']==ARM)
    write(RUN/'pressure_interpretation_audit.json',dict(status='complete',paired_95CI=intervals,
        net_core_GT_gain=total_gain,net_high_load_core_GT_gain=high_gain,net_other_core_GT_gain=total_gain-high_gain,
        classes_core_better=nbetter,classes_core_worse=17-nbetter,scene_resamples=1500,seed=928))
    target=ROOT/'analysis/semantic_neighborhood_mean_mini_results.md';text=target.read_text()
    marker='<!-- neighborhood-interpretation-v1 -->'
    if marker not in text:
        lines=[marker,'## 8. 怎样理解“几何略好，但预测没有变好”','',
            '**当前更准确的判断是：同类局部位置和负载代理指标有小幅正向变化，但本轮没有得到任务性能收益。** 这比上一轮核心覆盖下降的方向好，仍不足以作为已有效解决问题的方法。','',
            '### 8.1 几何改善存在，但幅度很小','',
            f"GS中心落在有效前景的比例增加{v['center_camera_fg']-b['center_camera_fg']:.3f}个百分点；有同类3σ目标的GS比例增加{v['active_same3_percent']-b['active_same3_percent']:.3f}个百分点，有效GS均距从{b['mean3_active_m']:.5f}m降至{v['mean3_active_m']:.5f}m。这组结果没有出现通过大量失去同类邻域目标而降低均距的迹象。",'',
            f"完整验证的1σ核心净新增{total_gain:,}个GT体素。固定初始高负载区域净新增{high_gain:,}个，其他区域净减少{high_gain-total_gain:,}个；净收益主要集中在所关心的高负载区域。但相对原有核心缺口，整体只减少了{100*(v['geo1']-b['geo1'])/(100-b['geo1']):.3f}%，实际幅度很小。",'',
            f"逐类看，{nbetter}/17类的核心覆盖提高，{17-nbetter}/17类下降。非道路覆盖提高，17类等权覆盖仅增加{v['macro_core1']-b['macro_core1']:.3f}个百分点；不能写成各类别普遍改善。",'',
            '| 固定模型间的变化 | 点估计/百分点 | 场景配对95%区间 |',
            '| --- | ---: | ---: |']
        for key,label in [('top20_nearest_share','最近归属top20%份额'),('high_core','固定高负载GT核心覆盖'),('high_recall','固定高负载GT二值召回')]:
            lo,hi=intervals[key];lines.append(f"| {label} | {v[key]-b[key]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        lines+=['','归属集中度和核心缺口向预期方向变化，但同一高负载区域的召回下降，说明几何代理指标的改善尚未转化为有效检测。区间仅反映场景重采样，不排除训练种子差异。','',
            '### 8.2 剩余缺口仍然难，渲染证据也可能改变','',
            f"未覆盖GT比例从{b['loss_uncovered_GT_percent']:.3f}%降到{v['loss_uncovered_GT_percent']:.3f}%，但剩余集合的最近中心距从{b['uncovered_center_distance_m']:.5f}m升至{v['uncovered_center_distance_m']:.5f}m。按全部GT归一的新增未覆盖距离项从{b['uncovered1_m']:.6f}m变为{v['uncovered1_m']:.6f}m，并未降低。剩余集合不同，条件均距增加含有筛选效应，不能据此声称每个剩余体素都离GS更远了。",'',
            f"任务上，FN总数增加{int(v['FN']-b['FN']):,}，FP总数仅减少{int(b['FP']-v['FP']):,}，二值IoU下降{b['IoU']-v['IoU']:.3f}个百分点。与此同时，2σ、3σ覆盖略降，平均opacity从{b['opacity']:.5f}降至{v['opacity']:.5f}，语义幅度也改变。因此新增核心覆盖不能保证聚合后的正确类别分数足够。",'',
            '这些关联提示下一步需要区分“新增核心是否真正救回FN”与“形状、opacity、语义和empty决策变化是否抵消几何收益”。本轮没有固定几何／属性的交叉替换实验，不能认定opacity下降就是漏检增加的主导原因。','',
            '### 8.3 本轮结论的范围','',
            '结果只针对本次0.5权重、4096帧、1epoch微调。基础CD保留，两个新增项在预定训练帧上的几何回归参数梯度约为原总目标梯度的4%–5%，因此这不是对更强监督或长期训练的穷尽检验。本轮也未改变每query固定64个GS的结构。','',
            '目前可以说目标带来了可测但很小的几何调整；不能说它已提高模型性能、解决query容量瓶颈或建立完整训练因果机制。后续方法应继续同时检查覆盖、召回和原生mIoU，而非只追求均距或负载比例改善。','',
            '训练耗时约52.39分钟，峰值已分配CUDA显存约4.455GB。推理结构不变，本轮未重新测FPS。','',
            '[补充的压力区间与净覆盖计数](../work_dirs/semantic_neighborhood_mean_mini_v1/pressure_interpretation_audit.json)。重建基础报告后，可执行`python3.10 tools/finalize_semantic_neighborhood_mean_report.py`恢复本节。','']
        text+='\n'+'\n'.join(lines);target.write_text(text)
    write(RUN/'report_provenance.json',dict(status='complete',sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in (target,RUN/'pressure_interpretation_audit.json',RUN/'summary.json')}))
if __name__=='__main__':main()
