"""Append count-derived interpretation without changing frozen experiment code."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.strict_classwise_set_experiment import RUN,PREVIOUS_RUN,write,assert_protocol


def main():
    assert_protocol()
    summary=json.loads((RUN/'summary.json').read_text())
    prior=json.loads((PREVIOUS_RUN/'optimization_strength_audit.json').read_text())['per_arm']
    m=summary['metrics'];c=m['classwise'];d=m['strict_classwise']
    a=np.load(RUN/'aggregate/strict_classwise.npz');b=np.load(RUN/'aggregate/classwise.npz')
    initial=a['initial_GS_class_count'];current=a['GS_class_count']
    first_initial=100*initial[:11].sum()/initial.sum()
    first_current=100*current[:11].sum()/current.sum()
    added_FN=int(a['confusion'][:17,17].sum()-b['confusion'][:17,17].sum())
    reduced_FP=int(b['confusion'][17,:17].sum()-a['confusion'][17,:17].sum())
    result=dict(status='complete',comparison='D minus C unless explicitly stated',
        delta={k:d[k]-c[k] for k in ('mIoU','IoU','FN_rate','FP_over_GTfg','geo1','same1',
                                   'top20_nearest_share','zero_core','high_core','high_recall')},
        added_FN=added_FN,reduced_binary_FP=reduced_FP,
        initial_first11_GS_share=float(first_initial),D_first11_GS_share=float(first_current),
        GS_histogram_comparison='D vs initialization; C per-GS validation classes were not saved',
        causal_limit='Targets, class reduction and skip rules changed together; no attribution to a single factor')
    write(RUN/'interpretation_audit.json',result)
    audit=summary['optimization_audit']
    lines=['<!-- strict-classwise-interpretation-v1 -->','## 8. 这组结果如何理解','',
        '**当前严格逐类版本不适合保留。** 它没有恢复上一组的性能，反而同时损害类别预测、核心覆盖与固定困难区域的召回。17 个类别的 IoU 均低于 C。','',
        f"相对 C，漏检增加 {added_FN:,} 个体素，而二值 FP 减少 {reduced_FP:,} 个。漏检率增加 {d['FN_rate']-c['FN_rate']:.3f} 个百分点，检出前景的语义准确率也从 88.852% 降至 83.033%，因此退化同时包含检出和类别判断两部分。",'',
        '### 8.1 本轮确实观察到了 GS 类别数量的明显变化','',
        f"类别 0–10 包括 others、barrier 和各类交通参与者等。它们占全部 GS 的比例从初始化的 {first_initial:.3f}% 增至 D 的 {first_current:.3f}%；总 GS 数仍固定为每帧 76,800。建筑的 GS 从每帧约 16,886 减至 10,008，植被从 11,333 减至 6,702，car 则从 3,592 增至 11,750。",'',
        '这是实际预测类别计数的变化，不是只根据覆盖下降反推数量。比较对象是共同初始化及 B；C 的完整验证未保存逐 GS 类别，不能据此声称已测出 C→D 的数量转移。类别数量变化也不意味着每个变类 GS 都预测错误。','',
        'D 与初始化同索引 GS 的类别变化比例约为 60.480%。自强化没有保证类别固定：标签在每次前向中重新生成，共享参数、Occupancy 和几何更新仍可改变 argmax，随后新的类别又成为 Set 分类目标。这与“冻结初始化伪标签”不同。','',
        '### 8.2 几何公式相同，但实际优化轨迹明显变化','',
        '| 指标 | B | C | D |','| --- | ---: | ---: | ---: |',
        f"| 六级加权几何损失均值 | {prior['baseline']['mean_total_six_stage_geometry']:.3f} | {prior['classwise']['mean_total_six_stage_geometry']:.3f} | {audit['mean_weighted_geometry']:.3f} |",
        f"| 裁剪前总梯度范数中位数 | {prior['baseline']['gradient_norm_median']:.3f} | {prior['classwise']['gradient_norm_median']:.3f} | {audit['gradient_norm_median']:.3f} |",
        f"| 触发范数35裁剪的更新比例 | {prior['baseline']['clipped_update_percent']:.3f}% | {prior['classwise']['clipped_update_percent']:.3f}% | {audit['gradient_clip_percent']:.3f}% |",'',
        'D 和 C 的两项 CD 公式、系数及共同初始输入上的梯度完全相同，但分类更新会改变预测类别、同类集合、空间位置与共享特征，进而改变后续几何匹配。因此，几何损失与总梯度的增加是两条实际轨迹的差异，不能归为几何实现被改动。','',
        '当前更值得考虑的解释是：失去最近 GT 类别的直接纠偏后，错误的预测类别可能同时成为分类强化目标和几何吸引依据；类内平均及原类别权重又共同改变了监督重心。这是与结果相符的机制候选，尚未分离各因素的贡献。','',
        '浅层还出现了较多跳过：阶段0平均有 9.374% 的 GS 无同类 GT，本轮连分类也跳过；C 对这部分 GS 仍保留原最近 GT 分类监督。这同样是本轮处理的一部分，不能忽略。','',
        '### 8.3 更多 query 接触前景，没有转化为有效分担','',
        f"零核心接触 query 从 C 的 {c['zero_core']:.3f}% 减至 {d['zero_core']:.3f}%，但 Top20 最近归属份额从 {c['top20_nearest_share']:.3f}% 增至 {d['top20_nearest_share']:.3f}%。固定初始高负载区域的核心覆盖从 {c['high_core']:.3f}% 降至 {d['high_core']:.3f}%，二值召回从 {c['high_recall']:.3f}% 降至 {d['high_recall']:.3f}%。",'',
        '这说明，仅让更多 query 接触少量前景不足以改善有效覆盖；最近归属更集中与困难区域的覆盖不足在本轮同时加剧。它仍不能单独证明每个 query 的输出容量已到上限。','',
        '**本轮支持的判断是：统一匹配规则本身不足以改善模型，当前严格自强化版本明显退化。** 若继续设计同类监督，应保留可靠的外部 GT 类别纠偏信号，并重新控制优化强度；不能把本轮结果单独归因为自强化、类别加权或跳过规则中的某一项。','',
        '[本节计数审计](../work_dirs/strict_classwise_set_mini_v1/interpretation_audit.json)；[最终完整性核验](../work_dirs/strict_classwise_set_mini_v1/final_integrity_audit.json)。基础报告重新生成后，运行 `python3.10 tools/finalize_strict_classwise_set_report.py` 恢复本节。','']
    report=ROOT/'analysis/strict_classwise_set_mini_results.md'
    prefix=report.read_text().split('<!-- strict-classwise-interpretation-v1 -->')[0].rstrip()
    report.write_text(prefix+'\n\n'+'\n'.join(lines))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
