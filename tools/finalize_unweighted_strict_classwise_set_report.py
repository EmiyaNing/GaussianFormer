"""Add readable interpretation without changing the frozen experiment protocol."""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.unweighted_strict_classwise_set_experiment import RUN, STRICT_RUN, write

MARKER = '<!-- unweighted-classwise-interpretation-v1 -->'


def main():
    summary = json.loads((RUN/'summary.json').read_text())
    m = summary['metrics']; e, d, c, b = [m[n] for n in
        ('unweighted_classwise', 'strict_classwise', 'classwise', 'baseline')]
    old = json.loads((STRICT_RUN/'optimization_audit.json').read_text())
    new = summary['optimization_audit']
    path = ROOT/'analysis/unweighted_strict_classwise_set_mini_results.md'
    text = path.read_text().split(MARKER)[0].rstrip()
    for name, label in [('baseline', 'B'), ('classwise', 'C'), ('strict_classwise', 'D')]:
        text = text.replace(f'| E − {name} |', f'| E − {label} |')
    text = text.replace(
        '各类的绝对漏检和 FP 计数见',
        '各类的漏检率和 FP 计数见')
    text = text.replace(
        '类别内空间距离权重、几何阈值与稀有类 GT 几何权重保留；',
        '类别内空间位置与距离阈值权重、稀有类相关配置保留；')
    verdict = ('**去掉类别加权明显缓解了上一组的退化，但尚未恢复到原始目标或仅逐类几何版本。** '
               f'E 比 D 高 {e["mIoU"]-d["mIoU"]:.3f} 个百分点，仍比 C 低 '
               f'{c["mIoU"]-e["mIoU"]:.3f}、比 B 低 {b["mIoU"]-e["mIoU"]:.3f} 个百分点。')
    if verdict not in text:
        at = text.index('\n\n', text.index('**实验已完成。**'))
        text = text[:at] + '\n\n' + verdict + text[at:]
    lines = [MARKER, '', '## 6. 这次消融说明了什么', '',
        '**类别加权确实放大了 D 的退化。** E 保留同类匹配、分类目标、缺类跳过规则与每类内部均值，只将 Set 类间汇总改为有效类等权。完整验证中 17 类 IoU 全部高于 D；增幅最大的五类是 vegetation、manmade、sidewalk、terrain 和 other_flat。', '',
        f'改善主要体现为前景召回恢复：漏检减少 {int(d["FN"]-e["FN"]):,} 个体素，漏检率从 {d["FN_rate"]:.3f}% 降到 {e["FN_rate"]:.3f}%；同时增加 {int(e["FP"]-d["FP"]):,} 个二值 FP，FP/GT前景从 {d["FP_over_GTfg"]:.3f}% 升到 {e["FP_over_GTfg"]:.3f}%。召回、语义准确率与覆盖共同恢复，最终使 mIoU 上升。', '',
        f'同类 1σ覆盖增加 {e["same1"]-d["same1"]:.3f} 个百分点，Top20 最近归属份额降低 {d["top20_nearest_share"]-e["top20_nearest_share"]:.3f} 个百分点。但 E 的整体覆盖和任务性能仍低于 B、C，query 最近归属也仍更集中。零核心接触 query 的比例反而从 {d["zero_core"]:.3f}% 升到 {e["zero_core"]:.3f}%，再次说明“更少 query 完全不接触前景”不能替代整体覆盖和预测质量。', '',
        '| 优化记录 | D：类别加权 | E：类间等权 |', '| --- | ---: | ---: |',
        f'| 梯度范数中位数 | {old["gradient_norm_median"]:.3f} | {new["gradient_norm_median"]:.3f} |',
        f'| 梯度范数95%分位 | {old["gradient_norm_p95"]:.3f} | {new["gradient_norm_p95"]:.3f} |',
        f'| 触发范数35裁剪的更新 | {old["gradient_clip_percent"]:.3f}% | {new["gradient_clip_percent"]:.3f}% |', '',
        '权重变化也改变了 Set 相对 Occupancy 的优化强度；这些梯度记录描述差异，不能单独作为性能变化的因果解释。', '',
        '**这里的“类间等权”仍然保留了按类平衡。** 每个有效类别先求自己的均值，再由各类均分总权重；一个只含少量 GS/GT 的类别，与道路等大类别获得相同的类间系数。这与把所有 GS 或 GT 放在一起求全局均值不同。因此，本轮没有恢复原 Set loss 的样本归一化，也没有恢复其基于全类最近 GT 的分类纠偏。', '',
        '当前证据支持“原类别权重是退化的重要放大因素”，也表明它不是唯一因素。E 与 C 仍同时存在分类匹配、分类归一化、缺类跳过和几何类间权重的差异，不能把剩余差距全部归因于其中某一项。', '',
        '实现说明：设计中“保留稀有类几何权重”指沿用 `rare_classes/rare_weight` 配置与原实现。原布尔索引后的 `clamp_` 不会写回权重张量，本轮未额外修正；实际改变的类间权重由已冻结的 17 维全1向量确定。', '']
    path.write_text(text + '\n\n' + '\n'.join(lines))
    write(RUN/'interpretation_audit.json', dict(status='complete',
        E_minus_D={key:e[key]-d[key] for key in ('mIoU', 'IoU', 'FN_rate', 'FP_over_GTfg', 'geo1', 'same1', 'top20_nearest_share')},
        E_minus_C_mIoU=e['mIoU']-c['mIoU'], E_minus_B_mIoU=e['mIoU']-b['mIoU'],
        removed_FN=int(d['FN']-e['FN']), added_FP=int(e['FP']-d['FP']),
        unweighted_means_uniform_active_classes=True, original_global_mean_not_restored=True))
    print(json.dumps({'status':'complete', 'report':str(path)}))


if __name__ == '__main__':
    main()
