"""Compare a gate diagnostic/candidate with fixed, cached development controls.

The default is the post-hoc G1 unit-gate intervention: no new training and no
checkpoint selection. This CPU-only report never changes source evaluations.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.pmg_world.regions import REGIONS, region_result
from tools.pmg_world_native_adapt import gate_report
from tools.pmg_world_native_adapt.report import interval, regional
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


CONTROLS = tuple(source for source in gate_report.SOURCES if source[0] != "N-group")
DEFAULT_DIRECTORY = Path("analysis/pmg_world_gate_optimization_20261026/G1_unit")
STATIC_CLASSES = (0, 1, 11, 12, 13, 14, 15, 16)
REPORT_FILES = (
    "comparison_statistics.json", "results_summary.md", "README.md",
    "per_horizon_class_iou.csv", "per_class_comparison.md",
    "regional_comparison.csv", "regional_class_errors.csv", "regional_class_errors.md",
)


def validate_metric_fields(metrics, global_counts, raw_regions, sources):
    """Independently verify displayed scalar metrics against saved counts."""
    for model, ((name, _, _), metric) in enumerate(zip(sources, metrics)):
        if metric.get("miou_divisor") != 17 or metric.get("evaluation_mask") != "none":
            raise ValueError(f"{name}: incompatible class or visibility protocol")
        if len(metric["horizons"]) != 7 or len(metric["regions"]) != 3:
            raise ValueError(f"{name}: wrong number of horizons")
        total_global = global_counts[:, model].sum(0)
        total_regions = raw_regions[:, model].sum(0)
        for horizon, value in enumerate(metric["horizons"]):
            if value["seconds"] != horizon * .5:
                raise ValueError(f"{name}: global horizon order mismatch")
            hist = total_global[horizon]
            np.testing.assert_array_equal(hist.sum(-1), value["class_gt_count"])
            np.testing.assert_allclose(value["miou"], np.mean(value["class_iou"][:17]), atol=1e-10, rtol=0)
            np.testing.assert_allclose(value["static_class_miou_fullspace"],
                np.asarray(value["class_iou"])[list(STATIC_CLASSES)].mean(), atol=1e-10, rtol=0)
            rebuilt = region_result(hist)
            for field in ("occupied_iou", "occupied_precision", "occupied_recall"):
                np.testing.assert_allclose(value[field], rebuilt[field], atol=1e-10, rtol=0)
        expected_primary = np.mean([metric["horizons"][h]["miou"] for h in (0, 2, 4, 6)])
        np.testing.assert_allclose(metric["selection_mean_miou"], expected_primary, atol=1e-10, rtol=0)
        for horizon, value in enumerate(metric["regions"]):
            if value["seconds"] != horizon + 1:
                raise ValueError(f"{name}: regional horizon order mismatch")
            for region_index, region_name in enumerate(REGIONS):
                rebuilt = region_result(total_regions[horizon, region_index])
                for field, expected in rebuilt.items():
                    actual = value["metrics"][region_name][field]
                    if expected is None:
                        if actual is not None:
                            raise ValueError(f"{name}/{region_name}/{field}: expected undefined value")
                    else:
                        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0,
                            err_msg=f"{name}/{horizon}/{region_name}/{field}")


def build_statistics(metrics, global_counts, regional_counts, raw_regions, sources, hashes, kind):
    candidate_name = sources[-1][0]
    # The legacy helper hardcodes G1 as the candidate label. Rename by known
    # reference order, without replacing occurrences inside reference names.
    legacy = gate_report.paired_statistics(global_counts, regional_counts, raw_regions, sources)
    comparisons = {
        f"{candidate_name}_minus_{reference[0]}": legacy[f"G1_minus_{reference[0]}"]
        for reference in sources[:-1]
    }
    checks = gate_report.acceptance(metrics, legacy)
    differences = {}
    candidate = metrics[-1]
    for (name, _, _), reference in zip(sources[:-1], metrics[:-1]):
        values = {}
        for region in ("fast", "F_only", "static_classes", "slow", "medium"):
            delta = regional(candidate, region, "occupied_semantic_accuracy") - regional(reference, region, "occupied_semantic_accuracy")
            values[f"{region}_semantic_per_horizon_delta_pp"] = delta.tolist()
            values[f"{region}_semantic_mean_delta_pp"] = float(delta.mean())
        values["primary_miou_delta_pp"] = candidate["selection_mean_miou"] - reference["selection_mean_miou"]
        values["global_miou_per_horizon_delta_pp"] = [
            left["miou"] - right["miou"]
            for left, right in zip(candidate["horizons"], reference["horizons"])
        ]
        differences[name] = values
    return dict(
        candidate=candidate_name, candidate_kind=kind,
        additional_training_updates=0 if kind == "posthoc-intervention" else None,
        samples=1024, scenes=150, comparisons=comparisons,
        point_estimate_differences=differences,
        acceptance_checks=checks, joint_target_passed=all(checks.values()),
        bootstrap_replicates=gate_report.BOOTSTRAP_REPLICATES,
        bootstrap_seed=gate_report.BOOTSTRAP_SEED,
        source_sha256=hashes,
        paired_count_checks=dict(scene_order_exact=True, sample_counts_exact=True,
            scene_current_confusion_exact=True, scene_global_GT_marginals_exact=True,
            scene_region_GT_marginals_exact=True, aggregate_counts_equal_scene_totals=True,
            all_displayed_metrics_recomputed_from_counts=True),
        scope="Post-hoc development analysis at fixed checkpoints; scene intervals exclude training-seed, repeated-selection and multiplicity uncertainty. Original thresholds retained, not new confirmation criteria.",
    )


def write_class_results(output, sources, metrics):
    names = [source[0] for source in sources]
    candidate = names[-1]
    rows = []
    lines = ["# 七时刻、18 类的全空间 IoU", "",
        "IoU 单位为百分数，差值为百分点；mIoU 固定平均 17 个非空类。free 单独展示，不计入 mIoU。全空间类别 IoU 不等于快速 GT 语义准确率。", ""]
    for horizon in range(7):
        headers = ["类别", *names, *[f"{candidate}−{name}" for name in names[:-1]]]
        lines += [f"## {horizon * .5:g} 秒", "", "| " + " | ".join(headers) + " |",
                  "|---|" + "---:|" * (len(headers) - 1)]
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = [metric["horizons"][horizon]["class_iou"][class_index] for metric in metrics]
            deltas = [values[-1] - reference for reference in values[:-1]]
            row = dict(seconds=horizon * .5, class_index=class_index, class_name=class_name)
            row.update({f"{name}_iou_percent": value for name, value in zip(names, values)})
            row.update({f"{candidate}_minus_{name}_pp": value for name, value in zip(names[:-1], deltas)})
            rows.append(row)
            lines.append("| " + class_name + " | " + " | ".join(f"{v:.4f}" for v in values)
                         + " | " + " | ".join(f"{v:+.4f}" for v in deltas) + " |")
        lines.append("")
    gate_report.write_csv(output / "per_horizon_class_iou.csv", rows)
    (output / "per_class_comparison.md").write_text("\n".join(lines) + "\n")


def write_regional_results(output, sources, metrics, raw_regions):
    rows, class_rows = [], []
    totals = raw_regions.sum(0)
    names = [source[0] for source in sources]
    for model, ((name, _, _), metric) in enumerate(zip(sources, metrics)):
        for horizon, region in enumerate(metric["regions"]):
            values = region["metrics"]
            fast, near = values["fast"], values["fast_neighborhood"]
            row = dict(model=name, seconds=horizon + 1)
            for region_name in ("fast", "F_only", "static_classes", "slow", "medium"):
                row[f"{region_name}_semantic_accuracy_percent"] = values[region_name]["occupied_semantic_accuracy"]
            row.update(
                fast_GT_recall_percent=fast["occupied_recall"],
                recalled_fast_GT_class_accuracy_percent=fast["conditional_semantic_accuracy"],
                fast_neighborhood_iou_percent=near["occupied_iou"],
                fast_neighborhood_precision_percent=near["occupied_precision"],
                fast_neighborhood_recall_percent=near["occupied_recall"],
                fast_neighborhood_FP=near["false_positive"],
                fast_neighborhood_FN=near["false_negative"],
                static_class_miou_fullspace_percent=metric["horizons"][2 * (horizon + 1)]["static_class_miou_fullspace"],
            )
            rows.append(row)
            for region_name in ("fast", "F_only", "static_classes", "slow"):
                region_index = REGIONS.index(region_name)
                confusion = totals[model, horizon, region_index]
                region_gt = int(confusion[:17].sum())
                for class_index, class_name in enumerate(CLASS_NAMES[:17]):
                    gt = int(confusion[class_index].sum())
                    correct = int(confusion[class_index, class_index])
                    empty = int(confusion[class_index, 17])
                    wrong = gt - correct - empty
                    scene_gt = raw_regions[:, model, horizon, region_index, class_index].sum(-1)
                    record = dict(region=region_name, model=name, seconds=horizon + 1,
                        class_index=class_index, class_name=class_name,
                        GT_voxels=gt, supporting_scenes=int((scene_gt > 0).sum()),
                        correct_voxels=correct, predicted_empty_voxels=empty,
                        occupied_wrong_class_voxels=wrong,
                        class_accuracy_percent=100 * correct / gt if gt else None,
                        predicted_empty_percent=100 * empty / gt if gt else None,
                        occupied_wrong_class_percent=100 * wrong / gt if gt else None)
                    for reference, reference_name in enumerate(names[:-1]):
                        reference_correct = int(totals[reference, horizon, region_index, class_index, class_index])
                        record[f"contribution_to_region_accuracy_delta_vs_{reference_name}_pp"] = 100 * (correct - reference_correct) / max(region_gt, 1)
                    class_rows.append(record)
    gate_report.write_csv(output / "regional_comparison.csv", rows)
    gate_report.write_csv(output / "regional_class_errors.csv", class_rows)
    lines = ["# 固定 GT 区域中的逐类空预测与错类", "",
        "这里报告区域内类准确率，不是全空间 IoU。正确、预测为空、预测成其他非空类别三项相加为 100%；无 GT 支持显示 —。", "",
        "F-only 是旧参考 F 正确、P 错误的固定区域，不等同静态。static_classes 是语义标签不在 2–10 的非空类，slow 是可靠速度低于 0.5m/s 的已关联 GT。类别结果为探索性诊断，未做多重比较校正。", ""]
    lookup = {(row["region"], row["seconds"], row["class_index"], row["model"]): row for row in class_rows}
    for region_name in ("fast", "F_only", "static_classes", "slow"):
        for horizon in range(3):
            headers = ["类别", "GT体素", "支持场景", *names, f"{names[-1]}空预测率", f"{names[-1]}非空错类率"]
            lines += [f"## {region_name}，{horizon + 1} 秒", "", "| " + " | ".join(headers) + " |",
                      "|---|" + "---:|" * (len(headers) - 1)]
            for class_index, class_name in enumerate(CLASS_NAMES[:17]):
                selected = [lookup[(region_name, horizon + 1, class_index, name)] for name in names]
                candidate = selected[-1]
                values = [row["class_accuracy_percent"] for row in selected]
                values += [candidate["predicted_empty_percent"], candidate["occupied_wrong_class_percent"]]
                displayed = ["—" if value is None else f"{value:.4f}" for value in values]
                lines.append(f"| {class_name} | {candidate['GT_voxels']} | {candidate['supporting_scenes']} | " + " | ".join(displayed) + " |")
            lines.append("")
    lines += ["CSV 同时保留所有模型的空预测、错类计数与对区域总准确率的贡献差。贡献差以区域总 GT 为分母，可以按类别求和还原整体差值；它与类内准确率差不同。", ""]
    (output / "regional_class_errors.md").write_text("\n".join(lines))
    return rows


def write_summary(output, sources, metrics, statistics, rows, description):
    candidate = sources[-1][0]
    passed = statistics["joint_target_passed"]
    lines = [f"# {candidate}：门控优化的固定检查点对照", "",
        f"**沿用相对 T-A0 的原联合门槛：{'通过' if passed else '未通过'}。**", "", description, "",
        "这是事后开发诊断，所有指标来自完整 1024 样本、150 场景的缓存。没有按结果挑选新检查点或改变阈值。单位门干预只判断共同训练后模型对门的依赖，关闭组件可能造成分布变化，不能独立识别完整失败机制。", "",
        "## 七时刻全空间指标", "", "mIoU 固定平均 17 个非空类，主指标平均 0/1/2/3 秒；下表单位为百分数。", "",
        "| 模型 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 主 mIoU |", "|---|" + "---:|" * 8]
    for (name, _, _), metric in zip(sources, metrics):
        values = [value["miou"] for value in metric["horizons"]] + [metric["selection_mean_miou"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "| 模型 | 二值占据 IoU 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s |", "|---|" + "---:|" * 7]
    for (name, _, _), metric in zip(sources, metrics):
        lines.append("| " + name + " | " + " | ".join(f"{h['occupied_iou']:.4f}" for h in metric["horizons"]) + " |")
    lines += ["", "## 快速语义、F-only 与覆盖", "",
        "GT 快速准确率与 F-only 准确率不是 IoU。快速 GT 为可靠速度至少 2m/s 的已关联占据体素；邻域还包含空背景和其他占据。", "",
        "| 模型 | 秒 | fast准确率 | F-only准确率 | fast GT召回 | 召回后类别正确率 | 邻域IoU | 邻域precision | 邻域recall | FP | FN |", "|---|" + "---:|" * 10]
    for row in rows:
        fields = ("fast_semantic_accuracy_percent", "F_only_semantic_accuracy_percent", "fast_GT_recall_percent",
                  "recalled_fast_GT_class_accuracy_percent", "fast_neighborhood_iou_percent",
                  "fast_neighborhood_precision_percent", "fast_neighborhood_recall_percent")
        lines.append(f"| {row['model']} | {row['seconds']} | " + " | ".join(f"{row[field]:.4f}" for field in fields)
                     + f" | {row['fast_neighborhood_FP']} | {row['fast_neighborhood_FN']} |")
    for reference_name, _, _ in sources[:-1]:
        differences = statistics["point_estimate_differences"][reference_name]
        ci = statistics["comparisons"][f"{candidate}_minus_{reference_name}"]
        lines += ["", f"## {candidate} 相对 {reference_name}", "",
            f"三时距平均快速语义变化 **{differences['fast_semantic_mean_delta_pp']:+.4f} pp**，95% 场景配对区间 **{interval(ci['fast_accuracy']['mean_delta_95ci_pp'])}**。", "",
            "| 秒 | fast变化 pp | 95%区间 | F-only变化 pp | 95%区间 |", "|---|---:|---|---:|---|"]
        for horizon in range(3):
            lines.append(f"| {horizon + 1} | {differences['fast_semantic_per_horizon_delta_pp'][horizon]:+.4f} | {interval(ci['fast_accuracy']['per_horizon_delta_95ci_pp'][horizon])} | {differences['F_only_semantic_per_horizon_delta_pp'][horizon]:+.4f} | {interval(ci['F_only']['per_horizon_delta_95ci_pp'][horizon])} |")
        lines += ["", f"主 mIoU 变化 {differences['primary_miou_delta_pp']:+.4f} pp，95% 区间 {interval(ci['global_miou']['primary_delta_95ci_pp'])}。",
            f"平均 F-only 变化 {differences['F_only_semantic_mean_delta_pp']:+.4f} pp，95% 区间 {interval(ci['F_only']['mean_delta_95ci_pp'])}。"]
    lines += ["", "## 原联合目标检查", "", "与历史报告使用同一判定，不把事后诊断包装成新的独立确认。", "", "| 条件 | 结果 |", "|---|---|"]
    labels = ("平均快速语义至少 +1pp", "平均快速语义区间下界大于 0", "F-only 各时距区间下界至少 −0.5pp", "主 mIoU 区间下界至少 −0.2pp", "当前各场景完整混淆计数一致")
    for label, value in zip(labels, statistics["acceptance_checks"].values()):
        lines.append(f"| {label} | {'通过' if value else '未通过'} |")
    lines += ["", "## 静态语义与低速区域", "",
        "static_classes、slow 与 F-only 定义不同；静态语义类准确率与全空间静态类 mIoU 的分母也不同。未知速度不能归为静态。", "",
        "| 模型 | 秒 | static_classes准确率 | slow准确率 | medium准确率 | 静态类全空间mIoU |", "|---|" + "---:|" * 5]
    for row in rows:
        fields = ("static_classes_semantic_accuracy_percent", "slow_semantic_accuracy_percent", "medium_semantic_accuracy_percent", "static_class_miou_fullspace_percent")
        lines.append(f"| {row['model']} | {row['seconds']} | " + " | ".join(f"{row[field]:.4f}" for field in fields) + " |")
    for reference_name, _, _ in sources[:-1]:
        ci = statistics["comparisons"][f"{candidate}_minus_{reference_name}"]
        delta = statistics["point_estimate_differences"][reference_name]
        lines += ["", f"相对 {reference_name} 的探索性区域差异：", "", "| 区域 | 秒 | 变化 pp | 95%区间 |", "|---|---:|---:|---|"]
        for region in ("static_classes", "slow", "medium"):
            for horizon in range(3):
                lines.append(f"| {region} | {horizon + 1} | {delta[f'{region}_semantic_per_horizon_delta_pp'][horizon]:+.4f} | {interval(ci[f'{region}_accuracy']['per_horizon_delta_95ci_pp'][horizon])} |")
    lines += ["", "## 统计与复核范围", "",
        "以场景为单位做 2000 次配对 bootstrap，种子 20261023。每次先聚合混淆计数再计算指标，不平均单场景比率。区间仅反映这些固定检查点的场景抽样差异，不包含训练种子、反复使用开发集、模型选择及多重比较的不确定性。", "",
        "逐场景顺序、样本数量、当前帧混淆矩阵、全域及区域 GT 边际均严格相同；区域和全域计数均等于场景求和。所有展示的 JSON 标量已从混淆计数复核。", "",
        "- [配对统计与来源哈希](comparison_statistics.json)",
        "- [七时刻逐类 IoU](per_class_comparison.md) / [CSV](per_horizon_class_iou.csv)",
        "- [区域逐类空预测与错类](regional_class_errors.md) / [CSV](regional_class_errors.csv)",
        "- [完整区域指标 CSV](regional_comparison.csv)", ""]
    (output / "results_summary.md").write_text("\n".join(lines))
    (output / "README.md").write_text(f"# {candidate} 固定检查点诊断\n\n[结果汇总](results_summary.md)、[配对统计](comparison_statistics.json)、[逐类 IoU](per_class_comparison.md)、[区域错误分解](regional_class_errors.md)。\n\n本工具只读取已有评估缓存，不训练或推理。默认 G1-unit 为同一 G1 权重强制单位门的事后诊断，没有新增训练。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", default="G1-unit")
    parser.add_argument("--candidate-directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--candidate-prefix", default="epoch_01")
    parser.add_argument("--candidate-kind", choices=("posthoc-intervention", "trained-candidate"), default="posthoc-intervention")
    parser.add_argument("--description")
    args = parser.parse_args()
    if args.candidate_name in {source[0] for source in CONTROLS}:
        raise ValueError("Candidate name must differ from the fixed controls")
    if any(character in args.candidate_name for character in "|\n\r"):
        raise ValueError("Candidate name must be a single table-safe label")
    output = args.output_directory or args.candidate_directory
    if output.resolve() in {folder.resolve() for _, folder, _ in CONTROLS}:
        raise ValueError("Never write into a historical control report directory")
    if any((output / filename).exists() for filename in REPORT_FILES):
        raise FileExistsError("Report output already exists; choose a new output directory")
    sources = (*CONTROLS, (args.candidate_name, args.candidate_directory, args.candidate_prefix))
    metrics, global_counts, regional_counts, raw_regions, hashes = gate_report.load_sources(sources)
    validate_metric_fields(metrics, global_counts, raw_regions, sources)
    statistics = build_statistics(metrics, global_counts, regional_counts, raw_regions, sources, hashes, args.candidate_kind)
    description = args.description
    if description is None:
        description = ("同一 G1 最终第 512 更新权重，仅将所有 query 门强制为 1；其余训练参数不变，没有新增训练。"
                       if args.candidate_name == "G1-unit" else
                       "对指定候选的固定评估缓存进行对照；候选训练和结构协议需参照该候选独立实验设计。")
    output.mkdir(parents=True, exist_ok=True)
    statistics["description"] = description
    statistics["report_tool_sha256"] = runtime.digest(Path(__file__))
    runtime.atomic_json(output / "comparison_statistics.json", statistics)
    write_class_results(output, sources, metrics)
    rows = write_regional_results(output, sources, metrics, raw_regions)
    write_summary(output, sources, metrics, statistics, rows, description)
    print(json.dumps(dict(candidate=args.candidate_name, joint_target_passed=statistics["joint_target_passed"],
        differences=statistics["point_estimate_differences"], output_directory=str(output)), indent=2))


if __name__ == "__main__":
    main()
