"""Report the minimal recurrent query-gate architecture with cached controls.

All comparisons use cached counts. This module does not run model inference,
select checkpoints, tune the gate amplitude, or modify earlier reports.
"""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world_native_adapt.report import interval, regional
from tools.pmg_world.regions import REGIONS
from tools.pmg_world_stage3.summarize_experiments import (
    paired_intervals,
    regional_sufficient,
)
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


ROOT = Path("analysis/pmg_world_recurrent_gate_20261025/G1")
WORK = Path("out/pmg_world_recurrent_gate_20261025/G1")
OLD = Path("analysis/pmg_world_native_adapt_20261024")
SOURCES = (
    ("T-A0", Path("analysis/pmg_world_stage2/T_A0_dense"), "epoch_10"),
    ("N-native", OLD, "epoch_01"),
    ("N-half", OLD / "fixed_half_step", "epoch_01"),
    ("N-group", OLD / "grouped_update", "epoch_01"),
    ("G1", ROOT, "epoch_01"),
)
BOOTSTRAP_SEED = 20261023
BOOTSTRAP_REPLICATES = 2000


def load_sources(sources):
    """Check that every cached result covers exactly the same fixed regions."""
    metrics, global_counts, regional_counts, raw_regions, hashes = [], [], [], [], {}
    reference = None
    for name, folder, prefix in sources:
        metric_path = folder / f"{prefix}_metrics.json"
        counts_path = folder / f"{prefix}_counts.npz"
        metric = json.loads(metric_path.read_text())
        if metric["num_samples"] != 1024 or metric.get("partial", False):
            raise ValueError(f"{name}: complete 1024-sample evaluation required")
        metrics.append(metric)
        hashes[name] = {str(p): runtime.digest(p) for p in (metric_path, counts_path)}
        with np.load(counts_path) as archive:
            checks = {
                "scenes": archive["scenes"].copy(),
                "sample_counts": archive["sample_counts"].copy(),
                "scene_current_confusion": archive["scene_confusion"][:, 0].copy(),
                "scene_global_gt": archive["scene_confusion"].sum(-1),
                "scene_region_gt": archive["scene_region_confusion"].sum(-1),
            }
            if reference is None:
                reference = checks
            else:
                for key, value in checks.items():
                    if not np.array_equal(value, reference[key]):
                        raise ValueError(f"{name}: mismatched {key}")
            if not np.array_equal(archive["scene_confusion"].sum(0), archive["confusion"]):
                raise ValueError(f"{name}: global counts disagree with scene totals")
            if not np.array_equal(archive["scene_region_confusion"].sum(0), archive["region_confusion"]):
                raise ValueError(f"{name}: regional counts disagree with scene totals")
            confusion = archive["confusion"]
            diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
            union = confusion.sum(-1) + confusion.sum(-2) - diagonal
            iou = 100 * np.divide(diagonal, union, out=np.zeros_like(diagonal, dtype=float), where=union > 0)
            np.testing.assert_allclose(iou, [h["class_iou"] for h in metric["horizons"]], atol=1e-10)
            global_counts.append(archive["scene_confusion"].copy())
            regional_counts.append(regional_sufficient(archive["scene_region_confusion"]))
            raw_regions.append(archive["scene_region_confusion"].copy())
    if len(reference["scenes"]) != 150 or int(reference["sample_counts"].sum()) != 1024:
        raise ValueError("Expected 150 scenes and 1024 samples")
    return metrics, np.stack(global_counts, 1), np.stack(regional_counts, 1), np.stack(raw_regions, 1), hashes


def paired_statistics(global_counts, regional_counts, raw_regions, sources):
    """Bootstrap scenes jointly, aggregating counts before computing ratios."""
    scene_count, model_count = global_counts.shape[:2]
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        scene_count, np.full(scene_count, 1 / scene_count), size=BOOTSTRAP_REPLICATES
    )
    sampled = (weights.astype(float) @ global_counts.reshape(scene_count, -1)).reshape(
        BOOTSTRAP_REPLICATES, model_count, 7, 18, 18
    )
    diagonal = np.diagonal(sampled, axis1=-2, axis2=-1)
    union = sampled.sum(-1) + sampled.sum(-2) - diagonal
    miou = (100 * np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union > 0))[..., :17].mean(-1)
    comparisons = {}
    candidate = model_count - 1
    extra_boot = {}
    for region_name in ("static_classes", "slow", "medium"):
        counts = raw_regions[..., REGIONS.index(region_name), :, :]
        parts = np.stack((np.trace(counts[..., :17, :17], axis1=-2, axis2=-1), counts[..., :17, :].sum((-1, -2))), -1)
        sampled_parts = (weights @ parts.reshape(scene_count, -1)).reshape(BOOTSTRAP_REPLICATES, model_count, 3, 2)
        extra_boot[region_name] = 100 * sampled_parts[..., 0] / np.maximum(sampled_parts[..., 1], 1)
    for reference in range(candidate):
        value = paired_intervals(regional_counts, candidate, reference, weights)
        delta = miou[:, candidate] - miou[:, reference]
        value["global_miou"] = {
            "per_horizon_delta_95ci_pp": np.quantile(delta, [.025, .975], axis=0).T.tolist(),
            "primary_delta_95ci_pp": np.quantile(delta[:, [0, 2, 4, 6]].mean(-1), [.025, .975]).tolist(),
        }
        for region_name, region_values in extra_boot.items():
            region_delta = region_values[:, candidate] - region_values[:, reference]
            value[region_name + "_accuracy"] = {
                "per_horizon_delta_95ci_pp": np.quantile(region_delta, [.025, .975], axis=0).T.tolist(),
                "mean_delta_95ci_pp": np.quantile(region_delta.mean(-1), [.025, .975]).tolist(),
                "scope": "Exploratory regional protection diagnostic; not an additional predeclared acceptance threshold.",
            }
        comparisons[f"G1_minus_{sources[reference][0]}"] = value
    return comparisons


def acceptance(metrics, comparisons):
    """Apply the written thresholds without rounding the observed gains."""
    baseline, candidate = metrics[0], metrics[-1]
    fast_delta = regional(candidate, "fast", "occupied_semantic_accuracy") - regional(baseline, "fast", "occupied_semantic_accuracy")
    baseline_ci = comparisons["G1_minus_T-A0"]
    return {
        "fast_semantic_mean_gain_at_least_1pp": float(fast_delta.mean()) >= 1.0,
        "fast_semantic_mean_ci_lower_positive": baseline_ci["fast_accuracy"]["mean_delta_95ci_pp"][0] > 0,
        "F_only_each_horizon_noninferior": all(x[0] >= -.5 for x in baseline_ci["F_only"]["per_horizon_delta_95ci_pp"]),
        "global_primary_noninferior": baseline_ci["global_miou"]["primary_delta_95ci_pp"][0] >= -.2,
        "current_confusion_exact": True,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_class_results(metrics):
    rows = []
    lines = ["# 七时刻逐类别 IoU", "", "单位为百分数；差值为百分点。mIoU 固定平均 17 个占据类别，free 单列展示。全空间类别 IoU 不等于快速 GT 分组表现。", ""]
    names = [name for name, _, _ in SOURCES]
    for horizon in range(7):
        lines += [f"## {horizon * .5:g} 秒", "", "| 类别 | " + " | ".join(names) + " | G1−T-A0 | G1−N-half |", "|---|" + "---:|" * 7]
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = [m["horizons"][horizon]["class_iou"][class_index] for m in metrics]
            row = dict(seconds=horizon * .5, class_index=class_index, class_name=class_name)
            row.update({name + "_iou_percent": value for name, value in zip(names, values)})
            row.update(G1_minus_T_A0_pp=values[-1] - values[0], G1_minus_N_half_pp=values[-1] - values[2])
            rows.append(row)
            lines.append("| " + class_name + " | " + " | ".join(f"{v:.4f}" for v in values) + f" | {values[-1] - values[0]:+.4f} | {values[-1] - values[2]:+.4f} |")
        lines.append("")
    write_csv(ROOT / "per_horizon_class_iou.csv", rows)
    (ROOT / "per_class_comparison.md").write_text("\n".join(lines) + "\n")


def region_rows(metrics):
    rows = []
    for (name, _, _), metric in zip(SOURCES, metrics):
        for horizon, region in enumerate(metric["regions"]):
            values = region["metrics"]
            near, fast = values["fast_neighborhood"], values["fast"]
            rows.append(dict(
                model=name, seconds=horizon + 1,
                fast_semantic_accuracy_percent=fast["occupied_semantic_accuracy"],
                F_only_accuracy_percent=values["F_only"]["occupied_semantic_accuracy"],
                fast_GT_recall_percent=fast["occupied_recall"],
                recalled_fast_GT_class_accuracy_percent=fast["conditional_semantic_accuracy"],
                fast_neighborhood_iou_percent=near["occupied_iou"],
                fast_neighborhood_precision_percent=near["occupied_precision"],
                fast_neighborhood_recall_percent=near["occupied_recall"],
                fast_neighborhood_FP=near["false_positive"],
                fast_neighborhood_FN=near["false_negative"],
            ))
    return rows


def write_regional_class_diagnosis(raw_regions):
    """Separate empty misses and occupied wrong labels within fixed GT regions."""
    rows = []
    total_counts = raw_regions.sum(0)
    for region_name in ("fast", "F_only"):
        region_index = REGIONS.index(region_name)
        for model_index, (model_name, _, _) in enumerate(SOURCES):
            for horizon in range(3):
                confusion = total_counts[model_index, horizon, region_index]
                region_gt = int(confusion[:17].sum())
                for class_index, class_name in enumerate(CLASS_NAMES[:17]):
                    gt = int(confusion[class_index].sum())
                    correct = int(confusion[class_index, class_index])
                    empty = int(confusion[class_index, 17])
                    wrong = gt - correct - empty
                    reference_correct = int(total_counts[0, horizon, region_index, class_index, class_index])
                    half_correct = int(total_counts[2, horizon, region_index, class_index, class_index])
                    scene_gt = raw_regions[:, model_index, horizon, region_index, class_index].sum(-1)
                    rows.append(dict(
                        region=region_name, model=model_name, seconds=horizon + 1,
                        class_index=class_index, class_name=class_name,
                        GT_voxels=gt, supporting_scenes=int((scene_gt > 0).sum()),
                        correct_voxels=correct, predicted_empty_voxels=empty,
                        occupied_wrong_class_voxels=wrong,
                        class_accuracy_percent=100 * correct / gt if gt else None,
                        predicted_empty_percent=100 * empty / gt if gt else None,
                        occupied_wrong_class_percent=100 * wrong / gt if gt else None,
                        contribution_to_region_accuracy_delta_vs_T_A0_pp=100 * (correct - reference_correct) / max(region_gt, 1),
                        contribution_to_region_accuracy_delta_vs_N_half_pp=100 * (correct - half_correct) / max(region_gt, 1),
                    ))
    write_csv(ROOT / "regional_class_errors.csv", rows)
    lines = ["# 快速目标与 F-only 的按类误差", "", "这些是固定 GT 区域内的类别准确率，不是全空间类别 IoU。错误分成预测为空与预测为其他占据类别，两者和正确率合计 100%。没有 GT 支持的类别显示 —。", "", "对区域总准确率的贡献差使用该区域全部 GT 体素为分母，各类贡献可相加还原整体变化；它不同于单个类别内部准确率差。以下分析为探索性诊断，没有新增验收门槛。", ""]
    for region_name in ("fast", "F_only"):
        for horizon in range(3):
            lines += [f"## {region_name}，{horizon + 1} 秒", "", "| 类别 | GT体素 | 支持场景 | T-A0准确率 | N-native | N-half | N-group | G1 | G1空预测率 | G1非空错类率 | 对区域准确率贡献差 vs N-half pp |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for class_index, class_name in enumerate(CLASS_NAMES[:17]):
                selected = [r for r in rows if r["region"] == region_name and r["seconds"] == horizon + 1 and r["class_index"] == class_index]
                candidate = selected[-1]
                def number(value):
                    return "—" if value is None else f"{value:.4f}"
                lines.append(f"| {class_name} | {candidate['GT_voxels']} | {candidate['supporting_scenes']} | " + " | ".join(number(r["class_accuracy_percent"]) for r in selected) + f" | {number(candidate['predicted_empty_percent'])} | {number(candidate['occupied_wrong_class_percent'])} | {candidate['contribution_to_region_accuracy_delta_vs_N_half_pp']:+.4f} |")
            lines.append("")
    lines += ["## 解读限制", "", "fast trailer、bicycle、motorcycle 等类别的支持场景较少，零准确率或较大变化都需结合 GT 数量与场景数量解释。这里没有对每个小类别做独立显著性宣称。", "", "F-only 由旧参考模型的正确/错误关系定义，不能当作静态区域。应特别检查 manmade、vegetation、sidewalk 等类别是否退化，以及是否被 terrain、driveable_surface 或 truck 的改善抵消。完整 CSV 保留每组的空预测与非空错类计数。", ""]
    (ROOT / "regional_class_errors.md").write_text("\n".join(lines) + "\n")


def static_and_speed_lines(metrics, comparisons):
    lines = ["## 静态语义与速度分组保护", "", "static_classes 指占据语义类别中排除标签 2–10 的集合，并非由真实零速度确定；slow 是已知速度小于 0.5 m/s，medium 是 0.5–2 m/s，fast 是至少 2 m/s。低速不等于严格静止，F-only 也不等于静态。速度未知目标不能自动归入静态组。", "", "| 条件 | 秒 | 静态语义类准确率 | slow准确率 | medium准确率 |", "|---|---:|---:|---:|---:|"]
    for (name, _, _), metric in zip(SOURCES, metrics):
        for horizon in range(3):
            values = [regional(metric, region, "occupied_semantic_accuracy")[horizon] for region in ("static_classes", "slow", "medium")]
            lines.append(f"| {name} | {horizon + 1} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    for reference_index in (0, 2):
        reference_name = SOURCES[reference_index][0]
        lines += ["", f"相对 {reference_name} 的变化与 95% 场景配对区间（探索性诊断）：", "", "| 区域 | 秒 | 准确率变化 pp | 95% 区间 |", "|---|---:|---:|---|"]
        for region_name in ("static_classes", "slow", "medium"):
            delta = regional(metrics[-1], region_name, "occupied_semantic_accuracy") - regional(metrics[reference_index], region_name, "occupied_semantic_accuracy")
            ci = comparisons[f"G1_minus_{reference_name}"][region_name + "_accuracy"]["per_horizon_delta_95ci_pp"]
            for horizon in range(3):
                lines.append(f"| {region_name} | {horizon + 1} | {delta[horizon]:+.4f} | {interval(ci[horizon])} |")
    return lines + [""]


def comparison_lines(metrics, comparisons, reference_index):
    name = SOURCES[reference_index][0]
    candidate, reference = metrics[-1], metrics[reference_index]
    fast = regional(candidate, "fast", "occupied_semantic_accuracy") - regional(reference, "fast", "occupied_semantic_accuracy")
    fonly = regional(candidate, "F_only", "occupied_semantic_accuracy") - regional(reference, "F_only", "occupied_semantic_accuracy")
    ci = comparisons[f"G1_minus_{name}"]
    lines = [f"## 相对 {name}", "", f"平均快速语义变化 **{fast.mean():+.4f} pp**，场景配对 95% 区间 **{interval(ci['fast_accuracy']['mean_delta_95ci_pp'])}**。", "", "| 时距 | 快速语义变化 pp | 95% 区间 | F-only 变化 pp | 95% 区间 |", "|---|---:|---|---:|---|"]
    for horizon in range(3):
        lines.append(f"| {horizon + 1}s | {fast[horizon]:+.4f} | {interval(ci['fast_accuracy']['per_horizon_delta_95ci_pp'][horizon])} | {fonly[horizon]:+.4f} | {interval(ci['F_only']['per_horizon_delta_95ci_pp'][horizon])} |")
    lines += ["", f"全域主 mIoU 变化 {candidate['selection_mean_miou'] - reference['selection_mean_miou']:+.4f} pp，区间 {interval(ci['global_miou']['primary_delta_95ci_pp'])}。", ""]
    return lines


def main():
    completed = json.loads((WORK / "completed.json").read_text())
    if completed.get("status") != "completed" or completed.get("updates") != 512:
        raise ValueError("The complete final 512-update evaluation is required")
    metrics, global_counts, regional_counts, raw_regions, hashes = load_sources(SOURCES)
    comparisons = paired_statistics(global_counts, regional_counts, raw_regions, SOURCES)
    checks = acceptance(metrics, comparisons)
    passed = all(checks.values())
    deltas = {}
    for reference_index in range(len(SOURCES) - 1):
        name = SOURCES[reference_index][0]
        deltas[name] = dict(
            fast_semantic_per_horizon_delta_pp=(regional(metrics[-1], "fast", "occupied_semantic_accuracy")
                - regional(metrics[reference_index], "fast", "occupied_semantic_accuracy")).tolist(),
            F_only_per_horizon_delta_pp=(regional(metrics[-1], "F_only", "occupied_semantic_accuracy")
                - regional(metrics[reference_index], "F_only", "occupied_semantic_accuracy")).tolist(),
            primary_miou_delta_pp=metrics[-1]["selection_mean_miou"] - metrics[reference_index]["selection_mean_miou"])
    positive_half_increment = comparisons["G1_minus_N-half"]["fast_accuracy"]["mean_delta_95ci_pp"][0] > 0
    gate = json.loads((ROOT / "gate_statistics.json").read_text())
    statistics = dict(comparisons=comparisons, point_estimate_differences=deltas,
        acceptance_checks=checks, joint_target_passed=passed,
        positive_fast_increment_vs_N_half_CI=positive_half_increment,
        unit_gate_intervention_recommended=passed and positive_half_increment,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        samples=1024, scenes=150, actual_training_updates=512,
        training_samples_consumed=2048, final_checkpoint_only=True, source_sha256=hashes,
        paired_count_checks=dict(scene_order_exact=True, sample_counts_exact=True,
            scene_current_confusion_exact=True, scene_GT_marginals_exact=True),
        scope="One minimal forward architecture candidate on reused development data; scene intervals do not include training-seed or model-selection uncertainty")
    runtime.atomic_json(ROOT / "comparison_statistics.json", statistics)
    write_class_results(metrics)
    write_regional_class_diagnosis(raw_regions)
    rows = region_rows(metrics)
    write_csv(ROOT / "regional_comparison.csv", rows)
    lines = ["# G1：逐 query 递推更新门控的结果", "",
        f"**相对 T-A0 的预设联合目标：{'通过短程筛选' if passed else '未通过'}；相对 N-half 的平均快速语义增量区间：{'下界大于零' if positive_half_increment else '未证实稳定正增量'}。**", "",
        "在原生特征更新处增加一个共享线性门控（514 参数），调节每个 query、每个时刻的原生更新。零初始化系数为 1，位置、类别、Gaussian 属性从共同特征解码，并进入后续递推。", "",
        "同一 T-A0 第 10 轮、同一 2048 个不同训练样本及顺序、512 更新；原生六个模块与门控共同训练。损失、优化器、冻结范围与 N-native 匹配，没有参数投影或事后半步插值。最终完整评估 1024 样本、150 场景。", "",
        f"训练 {completed['training_seconds']:.1f}s；验证 {completed['evaluation_seconds']:.1f}s。当前帧各场景混淆矩阵与 T-A0 完全一致。", "",
        "## 快速语义与 F-only", "", "单位为准确率百分数，不是 IoU。", "",
        "| 条件 | fast 1s | 2s | 3s | F-only 1s | 2s | 3s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for (name, _, _), metric in zip(SOURCES, metrics):
        values = [*regional(metric, "fast", "occupied_semantic_accuracy"),
                  *regional(metric, "F_only", "occupied_semantic_accuracy")]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [""] + comparison_lines(metrics, comparisons, 2)
    lines += comparison_lines(metrics, comparisons, 1)
    lines += comparison_lines(metrics, comparisons, 0)
    labels = ("平均快速语义至少 +1 pp", "平均快速语义区间下界大于零",
              "F-only 每时距区间下界至少 −0.5 pp", "主 mIoU 区间下界至少 −0.2 pp",
              "当前完整混淆计数一致")
    lines += ["| 预设条件（相对 T-A0） | 结果 |", "|---|---|"]
    lines += [f"| {label} | {'通过' if value else '未通过'} |" for label, value in zip(labels, checks.values())]
    lines += ["", "## 七时刻全域 mIoU", "", "主指标平均 0/1/2/3 秒。", "",
        "| 条件 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 主指标 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, _, _), metric in zip(SOURCES, metrics):
        lines.append("| " + name + " | " + " | ".join(f"{h['miou']:.4f}" for h in metric["horizons"])
                     + f" | {metric['selection_mean_miou']:.4f} |")
    lines += ["", "## 覆盖与误报", "",
        "快速 GT 召回仅针对 GT 快速占据体素；邻域包含自由空间及其他占据。召回后类别正确率的条件集合随模型变化，不是固定体素分类消融。", "",
        "| 条件 | 秒 | 邻域 IoU | precision | recall | FP | FN | fast GT召回 | 召回后类别正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['seconds']} | {row['fast_neighborhood_iou_percent']:.4f} | {row['fast_neighborhood_precision_percent']:.4f} | {row['fast_neighborhood_recall_percent']:.4f} | {row['fast_neighborhood_FP']} | {row['fast_neighborhood_FN']} | {row['fast_GT_recall_percent']:.4f} | {row['recalled_fast_GT_class_accuracy_percent']:.4f} |")
    lines += [""] + static_and_speed_lines(metrics, comparisons)
    lines += ["## 门控实际学到了什么", "",
        "下表在全部 query slot 上聚合，并非物理实例或 GT 动静标签。范围只约束相对更新系数，不保证状态或输出变化有界。", "",
        "| 秒 | query次数 | 均值 | 标准差 | 最小 | 最大 | <0.95比例 | >1.05比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in gate["rows"]:
        lines.append(f"| {record['seconds']:g} | {record['query_evaluations']} | {record['mean']:.6f} | {record['std']:.6f} | {record['minimum']:.6f} | {record['maximum']:.6f} | {record['fraction_below_095']:.4%} | {record['fraction_above_105']:.4%} |")
    lines += ["", "## 判断边界", "",
        "只有相对 T-A0 联合条件通过，且相对 N-half 平均快速语义增量区间下界大于零时，建议追加一次同检查点的强制单位门评估；不重新训练、不选择新检查点。该规则在候选结果出现前固定。", "",
        "若门控近乎常数或联合目标未改善，应降低此结构优先级，不自动加深 MLP 或扫描范围。普通门控自身不构成完整论文创新。", "",
        "static_classes、slow、F-only 是不同定义，分别报告。七时刻逐类 IoU 与区域逐类空预测/错类分解用于检查总量掩盖的类别损伤。场景配对区间不包含训练随机性及开发集反复选择的不确定性。", "",
        "## 文件", "",
        "- [实验设计](experiment_design.md)",
        "- [配对统计](comparison_statistics.json)",
        "- [七时刻逐类 IoU](per_class_comparison.md)",
        "- [逐类 IoU CSV](per_horizon_class_iou.csv)",
        "- [区域逐类误差](regional_class_errors.md)",
        "- [区域指标 CSV](regional_comparison.csv)",
        "- [门控统计](gate_statistics.json)",
        "- [训练记录](../../../out/pmg_world_recurrent_gate_20261025/G1)"]
    (ROOT / "results_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "README.md").write_text("# G1 门控架构探针\n\n[结果汇总](results_summary.md)、[配对统计](comparison_statistics.json)、[逐类结果](per_class_comparison.md)。\n\nepoch_01 是固定最终 512 更新评估序号，不表示训练了一个完整子集 epoch。\n")
    print(json.dumps(dict(joint_target_passed=passed,
        fast_mean_delta_vs_T_A0_pp=float(np.mean(deltas['T-A0']['fast_semantic_per_horizon_delta_pp'])),
        fast_mean_delta_vs_N_half_pp=float(np.mean(deltas['N-half']['fast_semantic_per_horizon_delta_pp'])),
        F_only_delta_vs_T_A0_pp=deltas['T-A0']['F_only_per_horizon_delta_pp'],
        unit_gate_intervention_recommended=passed and positive_half_increment), indent=2))


if __name__ == "__main__":
    main()

