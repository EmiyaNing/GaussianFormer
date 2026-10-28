"""Compare the single M-repair run with cached T-A0 and M-current counts."""

import csv
import json
from pathlib import Path

import numpy as np
from tools.pmg_world_stage3.summarize_experiments import regional_sufficient, paired_intervals
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


ROOT = Path("analysis/pmg_world_repair_20261024")
WORK = Path("out/pmg_world_repair_20261024/M_repair")
SOURCES = [
    ("T-A0", Path("analysis/pmg_world_stage2/T_A0_dense"), "epoch_10"),
    ("M-current", Path("analysis/pmg_world_stage3_20261023/frozen_M_current"), "epoch_01"),
    ("M-repair", ROOT, "epoch_01"),
]


def read():
    metrics, globals_, regions = [], [], []
    scenes = samples = None
    for name, folder, prefix in SOURCES:
        metrics.append(json.loads((folder / f"{prefix}_metrics.json").read_text()))
        with np.load(folder / f"{prefix}_counts.npz") as data:
            if scenes is None:
                scenes, samples = data["scenes"].copy(), data["sample_counts"].copy()
            else:
                assert np.array_equal(scenes, data["scenes"]), name
                assert np.array_equal(samples, data["sample_counts"]), name
            globals_.append(data["scene_confusion"].copy())
            regions.append(regional_sufficient(data["scene_region_confusion"]))
    assert int(samples.sum()) == 1024
    return metrics, scenes, np.stack(globals_, 1), np.stack(regions, 1)


def main():
    if not (WORK / "completed.json").exists():
        raise RuntimeError("The one declared training and validation must complete first")
    metrics, scenes, global_counts, region_parts = read()
    weights = np.random.default_rng(20261023).multinomial(
        len(scenes), np.full(len(scenes), 1 / len(scenes)), size=2000)
    boot = (weights.astype(float) @ global_counts.reshape(len(scenes), -1)).reshape(
        2000, 3, 7, 18, 18)
    diagonal = np.diagonal(boot, axis1=-2, axis2=-1)
    union = boot.sum(-1) + boot.sum(-2) - diagonal
    miou = (100 * np.divide(diagonal, union, out=np.zeros_like(diagonal),
                           where=union > 0))[..., :17].mean(-1)
    comparisons = {}
    for right in (0, 1):
        values = paired_intervals(region_parts, 2, right, weights)
        delta = miou[:, 2] - miou[:, right]
        values["global_miou"] = dict(
            per_horizon_delta_95ci_pp=np.quantile(delta, [.025, .975], axis=0).T.tolist(),
            primary_delta_95ci_pp=np.quantile(delta[:, [0, 2, 4, 6]].mean(-1), [.025, .975]).tolist())
        comparisons["M-repair_minus_" + SOURCES[right][0]] = values
    baseline, current, repair = metrics
    near_delta = float(np.mean([
        repair["regions"][h]["metrics"]["fast_neighborhood"]["occupied_iou"] -
        baseline["regions"][h]["metrics"]["fast_neighborhood"]["occupied_iou"] for h in range(3)]))
    ci = comparisons["M-repair_minus_T-A0"]
    checks = dict(fast_neighborhood_mean_gain_at_least_1pp=near_delta >= 1.,
        fast_neighborhood_mean_ci_lower_positive=ci["fast_neighborhood_iou"]["mean_delta_95ci_pp"][0] > 0,
        F_only_each_horizon_noninferior=all(x[0] >= -.5 for x in ci["F_only"]["per_horizon_delta_95ci_pp"]),
        global_primary_noninferior=ci["global_miou"]["primary_delta_95ci_pp"][0] >= -.2,
        current_miou_noninferior=ci["global_miou"]["per_horizon_delta_95ci_pp"][0][0] >= -.2)
    passed = all(checks.values())
    protocol = json.loads((WORK / "protocol.json").read_text())
    completion = json.loads((WORK / "completed.json").read_text())
    invariant = json.loads((WORK / "frozen_state_invariants.json").read_text())
    engineering = json.loads((WORK / "engineering_check.json").read_text())
    assert invariant["native_state_exact"] and invariant["query_assignment_exact"]
    assert np.array_equal(global_counts[:, 0, 0], global_counts[:, 2, 0])
    runtime.atomic_json(ROOT / "comparison_statistics.json", dict(iterations=2000,
        scene_count=len(scenes), seed=20261023, comparisons=comparisons,
        mean_fast_neighborhood_delta_pp=near_delta, acceptance_checks=checks,
        joint_target_passed=passed, current_confusion_exact=True,
        note="Paired scene sampling conditional on fixed weights; reused development split and one training seed."))

    rows = []
    for h in range(7):
        for c, name in enumerate(CLASS_NAMES):
            row = dict(seconds=h * .5, class_index=c, class_name=name)
            for label, value in zip(("T_A0", "M_current", "M_repair"), metrics):
                row[label + "_iou_percent"] = value["horizons"][h]["class_iou"][c]
            row["repair_minus_T_A0_pp"] = row["M_repair_iou_percent"] - row["T_A0_iou_percent"]
            row["repair_minus_M_current_pp"] = row["M_repair_iou_percent"] - row["M_current_iou_percent"]
            rows.append(row)
    with (ROOT / "per_horizon_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    class_lines = ["# 七时刻、18类别IoU对比", "",
        "单位为百分数，差值为百分点。mIoU平均17个非空类，free单列。完整三组原始值和差值见[CSV](per_horizon_class_iou.csv)。", ""]
    for name, index, difference in (("M-repair原始IoU", 2, False),
                                   ("相对T-A0的变化", 0, True),
                                   ("相对M-current的变化", 1, True)):
        class_lines += [f"## {name}", "", "| 类别 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c, label in enumerate(CLASS_NAMES):
            vals = [repair["horizons"][h]["class_iou"][c] -
                    (metrics[index]["horizons"][h]["class_iou"][c] if difference else 0.) for h in range(7)]
            formatted = [f"{v:+.4f}" if difference else f"{v:.4f}" for v in vals]
            class_lines.append("| " + label + " | " + " | ".join(formatted) + " |")
        class_lines.append("")
    (ROOT / "per_class_comparison.md").write_text("\n".join(class_lines) + "\n")

    lines = ["# M-repair单组实验结果", "",
        "本轮完成一组修正版：逐Gaussian状态条件头，加未来focal距离权重的位置梯度停止。两项合并，不能分别归因。", "",
        "**结论：" + ("达到预设联合筛选目标。" if passed else "未达到预设联合筛选目标，本轮不自动延长训练。") + "**", "",
        "## 实际预算与检查", "",
        f"同一T-A0第10轮初始化，冻结原生状态；完成512次更新、2048个不同训练样本、1024个验证样本（{len(scenes)}场景）。不是从头15轮，也不是完整4096样本epoch。", "",
        "工程检查通过：零头复现旧T-A0七时刻计数，非零位置修正保留Gaussian语义/形状和出生状态，新头梯度有限且非零；距离权重控制保持前向损失、分类logit和点回归梯度。训练后原生参数和query分配逐元素一致。", "",
        f"新增头参数{protocol['trainable_parameters']:,}，旧M-current为{protocol['previous_M_current_parameters']:,}；不是等容量消融。正式训练{completion['training_seconds']:.1f}s，验证{completion['evaluation_seconds']:.1f}s，训练PyTorch峰值已分配显存{completion['peak_training_cuda_mib'] / 1024:.3f}GiB。该显存值不等于整个GPU进程占用。", "",
        "参数减少不等于运算减少：新头逐child执行，并额外读取原生Gaussian属性。本轮只报告实际训练/验证耗时，不据参数数量宣称推理加速。", "",
        "## 全域mIoU", "", "mIoU固定平均17个非空类。所有指标使用最终第512更新权重；旧组复用原记录。", "",
        "| 条件 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 0/1/2/3s均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, _, _), m in zip(SOURCES, metrics):
        lines.append("| " + name + " | " + " | ".join(f"{h['miou']:.4f}" for h in m["horizons"]) +
                     f" | {m['selection_mean_miou']:.4f} |")
    lines += ["", "## 快速目标与原优势区域", "",
        "快区正确率是GT快速占据体素上的语义准确率；邻域IoU是二值占据IoU；F-only是固定旧参考区域的语义保持率。单位均为百分数。", "",
        "| 条件 | 秒 | 快区正确率 | 邻域IoU | 邻域precision | 邻域recall | 邻域FP | 邻域FN | F-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, _, _), m in zip(SOURCES, metrics):
        for h, record in enumerate(m["regions"]):
            r = record["metrics"]
            near = r["fast_neighborhood"]
            fp = near["false_positive"]
            fn = near["false_negative"]
            lines.append(f"| {name} | {h + 1} | {r['fast']['occupied_semantic_accuracy']:.4f} | "
                f"{near['occupied_iou']:.4f} | {near['occupied_precision']:.4f} | {near['occupied_recall']:.4f} | "
                f"{fp} | {fn} | {r['F_only']['occupied_semantic_accuracy']:.4f} |")
    lines += ["", f"相对T-A0，平均快速邻域IoU变化 **{near_delta:+.4f}pp**。", "",
        "| 对照 | 平均快速邻域IoU差值95%区间pp | 1s F-only差值区间 | 2s | 3s |",
        "|---|---|---|---|---|"]
    interval = lambda x: f"[{x[0]:+.4f}, {x[1]:+.4f}]"
    for name, value in comparisons.items():
        lines.append("| " + name + " | " + interval(value["fast_neighborhood_iou"]["mean_delta_95ci_pp"]) +
                     " | " + " | ".join(interval(x) for x in value["F_only"]["per_horizon_delta_95ci_pp"]) + " |")
    lines += ["", "## 预设目标核对", "", "| 条件 | 结果 |", "|---|---|"]
    labels = dict(fast_neighborhood_mean_gain_at_least_1pp="平均快邻域提升≥1pp",
        fast_neighborhood_mean_ci_lower_positive="快邻域差值区间下界>0",
        F_only_each_horizon_noninferior="F-only各时距区间下界≥−0.5pp",
        global_primary_noninferior="全域主指标区间下界≥−0.2pp",
        current_miou_noninferior="当前mIoU区间下界≥−0.2pp")
    lines += [f"| {labels[k]} | {'通过' if v else '未通过'} |" for k, v in checks.items()]
    lines += ["", "## 这组修改具体带来了什么", "",
        "### 能力保持", "",
        "| 秒 | M-repair相对T-A0的F-only变化pp | 相对旧M-current的恢复pp |",
        "|---|---:|---:|"]
    for h in range(3):
        repaired = repair["regions"][h]["metrics"]["F_only"]["occupied_semantic_accuracy"]
        original = baseline["regions"][h]["metrics"]["F_only"]["occupied_semantic_accuracy"]
        old = current["regions"][h]["metrics"]["F_only"]["occupied_semantic_accuracy"]
        lines.append(f"| {h+1} | {repaired-original:+.4f} | {repaired-old:+.4f} |")
    lines += ["", "本组合对原优势区域的损伤较旧M-current小，各时距配对区间均满足−0.5pp容限；仍存在相对T-A0的损失，不能称作完全无损。两项修改合并，不能单独归功于停止距离梯度或child条件输入。", "",
        "### 快速目标恢复", "",
        "| 秒 | 快速GT语义正确率变化pp | 邻域IoU变化pp | 邻域FP变化 | 邻域FN变化 |",
        "|---|---:|---:|---:|---:|"]
    for h in range(3):
        r, b = repair["regions"][h]["metrics"], baseline["regions"][h]["metrics"]
        delta_accuracy = r["fast"]["occupied_semantic_accuracy"] - b["fast"]["occupied_semantic_accuracy"]
        rn, bn = r["fast_neighborhood"], b["fast_neighborhood"]
        lines.append(f"| {h+1} | {delta_accuracy:+.4f} | {rn['occupied_iou']-bn['occupied_iou']:+.4f} | "
                     f"{rn['false_positive']-bn['false_positive']:+d} | {rn['false_negative']-bn['false_negative']:+d} |")
    lines += ["", "这次邻域IoU的小幅收益伴随FP减少、FN略增，邻域召回没有恢复。平均快速GT语义正确率差值的场景区间包含0。因此，结果主要表现为误报与能力保持取舍的调整，尚未充分利用GT位置诊断显示的恢复空间。", "",
        "全空间逐类结果有少量改善，例如3s motorcycle、bicycle和bus；但这些类别不等于GT快速目标分组，不能据此宣称快速实例问题已解决。完整七时刻全部类别见逐类报告。", "",
        "## 解释边界与后续", "",
        "本轮优先检验组合是否有效，不通过单项收益解释两项修改各自的作用。原生参数保持不等于输出区域性能保持，必须以上表验收。", "",
        "场景配对区间只描述这三个固定检查点在开发场景上的抽样差异，不包含训练种子、重复开发选择或多重比较。一次短程冻结实验不能替代端到端长程确认。", "",
        "根据精简计划，只有组合通过联合目标才进入长程验证；未通过时保留当前结果，不自动启动其他训练。", "",
        "## 文件", "",
        "- [实验设计](experiment_design.md)", "- [逐时刻逐类别可读对比](per_class_comparison.md)",
        "- [逐时刻逐类别CSV](per_horizon_class_iou.csv)", "- [配对区间与验收JSON](comparison_statistics.json)",
        "- [最终指标](final_512updates_metrics.json)",
        "- [运行记录](../../out/pmg_world_repair_20261024/M_repair)"]
    (ROOT / "results_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "README.md").write_text("# M-repair实验\n\n[结果汇总](results_summary.md)；[实验设计](experiment_design.md)；[逐类对比](per_class_comparison.md)。\n")
    print(json.dumps(dict(joint_target_passed=passed, near_delta_pp=near_delta,
                         primary=repair["selection_mean_miou"], acceptance_checks=checks), indent=2))


if __name__ == "__main__":
    main()
