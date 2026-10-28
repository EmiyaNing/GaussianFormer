"""Compare cached controls and native adaptation without new model inference."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world_stage3.summarize_experiments import regional_sufficient, paired_intervals
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


ROOT = Path("analysis/pmg_world_native_adapt_20261024")
WORK = Path("out/pmg_world_native_adapt_20261024/N_native")
SOURCES = [
    ("T-A0", Path("analysis/pmg_world_stage2/T_A0_dense"), "epoch_10"),
    ("M-current", Path("analysis/pmg_world_stage3_20261023/frozen_M_current"), "epoch_01"),
    ("M-repair", Path("analysis/pmg_world_repair_20261024"), "epoch_01"),
    ("N-native", ROOT, "epoch_01"),
]


def regional(metrics, region, field):
    return np.asarray([r["metrics"][region][field] for r in metrics["regions"]], dtype=float)


def interval(values):
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def main():
    if not (WORK / "completed.json").exists():
        raise RuntimeError("Formal training and evaluation must finish before reporting")
    metrics, global_counts, region_parts, source_hashes = [], [], [], {}
    scenes = samples = baseline_current = None
    for name, folder, prefix in SOURCES:
        path = folder / f"{prefix}_metrics.json"
        metrics.append(json.loads(path.read_text()))
        counts_path = folder / f"{prefix}_counts.npz"
        source_hashes[name] = {str(p): runtime.digest(p) for p in (path, counts_path)}
        with np.load(counts_path) as data:
            if scenes is None:
                scenes, samples = data["scenes"].copy(), data["sample_counts"].copy()
                baseline_current = data["confusion"][0].copy()
            else:
                assert np.array_equal(scenes, data["scenes"]), name
                assert np.array_equal(samples, data["sample_counts"]), name
                assert np.array_equal(baseline_current, data["confusion"][0]), name
            global_counts.append(data["scene_confusion"].copy())
            region_parts.append(regional_sufficient(data["scene_region_confusion"]))
    assert int(samples.sum()) == 1024 and len(scenes) == 150
    global_counts = np.stack(global_counts, 1)
    region_parts = np.stack(region_parts, 1)
    weights = np.random.default_rng(20261023).multinomial(
        len(scenes), np.full(len(scenes), 1 / len(scenes)), size=2000)
    boot = (weights.astype(float) @ global_counts.reshape(len(scenes), -1)).reshape(
        2000, len(SOURCES), 7, 18, 18)
    diagonal = np.diagonal(boot, axis1=-2, axis2=-1)
    union = boot.sum(-1) + boot.sum(-2) - diagonal
    miou = (100 * np.divide(diagonal, union, out=np.zeros_like(diagonal),
                           where=union > 0))[..., :17].mean(-1)
    comparisons = {}
    for right in range(3):
        values = paired_intervals(region_parts, 3, right, weights)
        delta = miou[:, 3] - miou[:, right]
        values["global_miou"] = dict(
            per_horizon_delta_95ci_pp=np.quantile(delta, [.025, .975], axis=0).T.tolist(),
            primary_delta_95ci_pp=np.quantile(delta[:, [0, 2, 4, 6]].mean(-1), [.025, .975]).tolist())
        comparisons["N-native_minus_" + SOURCES[right][0]] = values
    base, _, _, native = metrics
    fast_delta = regional(native, "fast", "occupied_semantic_accuracy") - regional(base, "fast", "occupied_semantic_accuracy")
    fonly_delta = regional(native, "F_only", "occupied_semantic_accuracy") - regional(base, "F_only", "occupied_semantic_accuracy")
    near_delta = regional(native, "fast_neighborhood", "occupied_iou") - regional(base, "fast_neighborhood", "occupied_iou")
    ci = comparisons["N-native_minus_T-A0"]
    checks = dict(fast_semantic_mean_gain_at_least_1pp=float(fast_delta.mean()) >= 1.,
        fast_semantic_mean_ci_lower_positive=ci["fast_accuracy"]["mean_delta_95ci_pp"][0] > 0,
        F_only_each_horizon_noninferior=all(x[0] >= -.5 for x in ci["F_only"]["per_horizon_delta_95ci_pp"]),
        global_primary_noninferior=ci["global_miou"]["primary_delta_95ci_pp"][0] >= -.2,
        current_confusion_exact=True)
    passed = all(checks.values())
    runtime.atomic_json(ROOT / "comparison_statistics.json", dict(
        unit="percentage point", bootstrap_seed=20261023, bootstrap_replicates=2000,
        scenes=len(scenes), samples=int(samples.sum()), comparisons=comparisons,
        fast_semantic_mean_delta_pp=float(fast_delta.mean()),
        fast_neighborhood_mean_delta_pp=float(near_delta.mean()),
        F_only_per_horizon_delta_pp=fonly_delta.tolist(), acceptance_checks=checks,
        joint_target_passed=passed, source_sha256=source_hashes,
        scope="Fixed checkpoints on a reused development set; no seed or repeated-selection uncertainty."))
    rows = []
    for h in range(7):
        for c, cls in enumerate(CLASS_NAMES):
            row = dict(seconds=h*.5, class_index=c, class_name=cls)
            for (name, _, _), m in zip(SOURCES, metrics):
                row[name + "_iou_percent"] = m["horizons"][h]["class_iou"][c]
            row["N_native_minus_T_A0_pp"] = row["N-native_iou_percent"] - row["T-A0_iou_percent"]
            row["N_native_minus_M_repair_pp"] = row["N-native_iou_percent"] - row["M-repair_iou_percent"]
            rows.append(row)
    with (ROOT / "per_horizon_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    class_lines = ["# 七时刻逐类别 IoU", "",
        "单位为百分数；差值为百分点。全空间类别结果不等同于速度分组结果。", ""]
    for h in range(7):
        class_lines += [f"## {h*.5:g} 秒", "",
            "| 类别 | T-A0 | M-current | M-repair | N-native | N-native−T-A0 |",
            "|---|---:|---:|---:|---:|---:|"]
        for row in rows[h*len(CLASS_NAMES):(h+1)*len(CLASS_NAMES)]:
            values = [row[name + "_iou_percent"] for name, _, _ in SOURCES]
            class_lines.append("| " + row["class_name"] + " | " + " | ".join(f"{v:.4f}" for v in values) +
                               f" | {row['N_native_minus_T_A0_pp']:+.4f} |")
        class_lines.append("")
    (ROOT / "per_class_comparison.md").write_text("\n".join(class_lines) + "\n")
    completed = json.loads((WORK / "completed.json").read_text())
    protocol = json.loads((WORK / "protocol.json").read_text())
    lines = ["# N-native：原生预测器适配结果", "",
        f"**预设联合筛选目标：{'通过，仍需进一步确认' if passed else '未通过'}。**", "",
        "本轮固定感知器，训练原生未来预测分支，不增加模型参数。目标是提高快速目标语义准确率，同时维持固定 F-only 区域准确率。", "",
        "## 实际训练与对照", "",
        f"从同一个 T-A0 第 10 轮检查点开始，完成 512 次更新、2048 个不同训练样本和完整 1024 样本验证（150 场景）。可训练参数 {protocol['trainable_parameters']:,}，新增参数 0。",
        "样本顺序、LR 计划与损失处理匹配 M-repair。可训练模块和参数量不同，不能将收益单独归因于某一种机制。", "",
        f"训练 {completed['training_seconds']:.1f}s，验证 {completed['evaluation_seconds']:.1f}s，训练 PyTorch 峰值已分配显存 {completed['peak_training_cuda_mib']/1024:.3f} GiB；该值不等于整个 GPU 进程占用。",
        "只评估最终第 512 更新权重，没有挑选中间检查点。它不是一个完整子集 epoch，也不是 15 轮从头训练。", "",
        "## 全域 mIoU", "",
        "| 条件 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 0/1/2/3s 均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name, _, _), m in zip(SOURCES, metrics):
        lines.append("| " + name + " | " + " | ".join(f"{h['miou']:.4f}" for h in m["horizons"]) +
                     f" | {m['selection_mean_miou']:.4f} |")
    lines += ["", "## 快速语义预测与 F-only", "",
        "快速语义准确率在 GT 快速占据体素上同时检查非空和正确类别；F-only 使用固定旧参考区域。两者均为百分数，不能当作 IoU。", "",
        "| 条件 | 时距 | 快速语义准确率 | F-only 准确率 | 快速邻域二值 IoU | 邻域 precision | 邻域 recall |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for (name, _, _), m in zip(SOURCES, metrics):
        for h in range(3):
            r = m["regions"][h]["metrics"]
            n = r["fast_neighborhood"]
            values = [r["fast"]["occupied_semantic_accuracy"], r["F_only"]["occupied_semantic_accuracy"],
                n["occupied_iou"], n["occupied_precision"], n["occupied_recall"]]
            lines.append(f"| {name} | {h+1}s | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 相对 T-A0：是否同时满足两个目标", "",
        f"平均快速语义准确率变化 **{fast_delta.mean():+.4f} pp**，场景配对区间 **{interval(ci['fast_accuracy']['mean_delta_95ci_pp'])}**。", "",
        "| 时距 | 快速语义准确率变化 pp | 95% 区间 | F-only 变化 pp | 95% 区间 | 邻域 FP 变化 | 邻域 FN 变化 |",
        "|---|---:|---|---:|---|---:|---:|"]
    for h in range(3):
        a = native["regions"][h]["metrics"]["fast_neighborhood"]
        b = base["regions"][h]["metrics"]["fast_neighborhood"]
        lines.append(f"| {h+1}s | {fast_delta[h]:+.4f} | " +
            interval(ci["fast_accuracy"]["per_horizon_delta_95ci_pp"][h]) + f" | {fonly_delta[h]:+.4f} | " +
            interval(ci["F_only"]["per_horizon_delta_95ci_pp"][h]) +
            f" | {a['false_positive']-b['false_positive']:+d} | {a['false_negative']-b['false_negative']:+d} |")
    labels = dict(fast_semantic_mean_gain_at_least_1pp="平均快速语义提升至少 1 pp",
        fast_semantic_mean_ci_lower_positive="快速语义平均差值区间下界大于 0",
        F_only_each_horizon_noninferior="F-only 各时距区间下界不低于 −0.5 pp",
        global_primary_noninferior="全域主 mIoU 区间下界不低于 −0.2 pp",
        current_confusion_exact="当前帧完整混淆矩阵保持一致（隔离检查）")
    lines += ["", "| 预设条件 | 结果 |", "|---|---|"]
    lines += [f"| {labels[k]} | {'通过' if v else '未通过'} |" for k, v in checks.items()]
    overlap = json.loads((ROOT / "fixed_region_overlap.json").read_text())
    lines += ["", "## 两个目标是否在评估区域上直接冲突", "",
        "额外复用固定参考 mask 统计集合重叠，没有新增推理，也没有把这些 mask 用于训练。", "",
        "| 时距 | 快速 GT 中属于 F-only 的比例 | F-only 中属于已知快速 GT 的比例 |",
        "|---|---:|---:|"]
    for r in overlap["horizons"]:
        lines.append(f"| {r['seconds']:g}s | {r['percent_fast_in_F_only']:.4f}% | {r['percent_F_only_that_is_fast']:.4f}% |")
    lines += ["", "两个集合在体素层面重叠很少，因此不能预设两个目标必然零和。它们仍可能受到相同 query、Gaussian 或参数的影响；集合统计没有证明共享表示干扰是主因。已知快速 mask 不覆盖速度未知目标，非快速部分也不能全部称为静态。", "",
        "## 结论的边界", "",
        "快速邻域包含背景，二值 IoU 与快速 GT 语义准确率衡量不同方面。需同时检查误报、漏检和全域指标，不能只按一个数值宣布解决快速目标预测。",
        "F-only 是旧参考模型定义的固定错误分区，适合作为回归保护指标；保持它不等于已经证明某种历史记忆机制。",
        "本轮是一个初始化、一个种子、短程适配；场景区间不覆盖训练随机性或开发集反复使用的影响。一次对照不能定位完整根因，结果不能替代完整数据收敛验证。",
        "是否继续应依据联合目标和误报漏检取舍决定，不自动增加消融矩阵或补满 15 轮。", "",
        "## 文件", "",
        "- [预先写定的实验设计](experiment_design.md)",
        "- [逐时刻逐类别对比](per_class_comparison.md)",
        "- [逐类别 CSV](per_horizon_class_iou.csv)",
        "- [配对统计与验收](comparison_statistics.json)",
        "- [固定区域重叠计数](fixed_region_overlap.json)",
        "- [最终完整指标](final_512updates_metrics.json)",
        "- [运行记录](../../out/pmg_world_native_adapt_20261024/N_native)"]
    (ROOT / "results_summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "README.md").write_text("# N-native 原生预测器适配\n\n"
        "[实验结果](results_summary.md)；[实验设计](experiment_design.md)；[逐类对比](per_class_comparison.md)。\n")
    print(json.dumps(dict(joint_target_passed=passed, fast_semantic_mean_delta_pp=float(fast_delta.mean()),
        fast_neighborhood_mean_delta_pp=float(near_delta.mean()), F_only_delta_pp=fonly_delta.tolist(),
        primary=native["selection_mean_miou"], acceptance_checks=checks), indent=2))


if __name__ == "__main__":
    main()
