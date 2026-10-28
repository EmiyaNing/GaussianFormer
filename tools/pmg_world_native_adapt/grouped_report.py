"""Paired report for the one fixed grouped-update evaluation."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world_native_adapt.report import regional, interval
from tools.pmg_world_stage3.summarize_experiments import regional_sufficient, paired_intervals
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


ROOT = Path("analysis/pmg_world_native_adapt_20261024/grouped_update")
WORK = Path("out/pmg_world_native_adapt_20261024/N_group_step")
SOURCES = [
    ("T-A0", Path("analysis/pmg_world_stage2/T_A0_dense"), "epoch_10"),
    ("N-native", ROOT.parent, "epoch_01"),
    ("N-half", ROOT.parent / "fixed_half_step", "epoch_01"),
    ("N-group", ROOT, "epoch_01"),
]


def main():
    assert (WORK / "completed.json").exists(), "Evaluation must finish first"
    metrics, globals_, regions, hashes = [], [], [], {}
    scenes = samples = current = None
    for name, folder, prefix in SOURCES:
        mp, cp = folder / f"{prefix}_metrics.json", folder / f"{prefix}_counts.npz"
        metrics.append(json.loads(mp.read_text()))
        hashes[name] = {str(p): runtime.digest(p) for p in (mp, cp)}
        with np.load(cp) as a:
            if scenes is None:
                scenes, samples = a["scenes"].copy(), a["sample_counts"].copy()
                current = a["confusion"][0].copy()
            assert np.array_equal(scenes, a["scenes"]), name
            assert np.array_equal(samples, a["sample_counts"]), name
            assert np.array_equal(current, a["confusion"][0]), name
            globals_.append(a["scene_confusion"].copy())
            regions.append(regional_sufficient(a["scene_region_confusion"]))
    assert int(samples.sum()) == 1024 and len(scenes) == 150
    globals_, regions = np.stack(globals_, 1), np.stack(regions, 1)
    weights = np.random.default_rng(20261023).multinomial(
        len(scenes), np.full(len(scenes), 1/len(scenes)), size=2000)
    boot = (weights.astype(float) @ globals_.reshape(len(scenes), -1)).reshape(2000, 4, 7, 18, 18)
    diagonal = np.diagonal(boot, axis1=-2, axis2=-1)
    union = boot.sum(-1) + boot.sum(-2) - diagonal
    miou = (100*np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union>0))[..., :17].mean(-1)
    comparisons = {}
    for right in range(3):
        c = paired_intervals(regions, 3, right, weights)
        delta = miou[:, 3] - miou[:, right]
        c["global_miou"] = dict(
            per_horizon_delta_95ci_pp=np.quantile(delta, [.025,.975], axis=0).T.tolist(),
            primary_delta_95ci_pp=np.quantile(delta[:,[0,2,4,6]].mean(-1), [.025,.975]).tolist())
        comparisons["N-group_minus_"+SOURCES[right][0]] = c
    base, _, half, candidate = metrics
    fast = regional(candidate,"fast","occupied_semantic_accuracy") - regional(base,"fast","occupied_semantic_accuracy")
    fonly = regional(candidate,"F_only","occupied_semantic_accuracy") - regional(base,"F_only","occupied_semantic_accuracy")
    near = regional(candidate,"fast_neighborhood","occupied_iou") - regional(base,"fast_neighborhood","occupied_iou")
    half_fast = regional(candidate,"fast","occupied_semantic_accuracy") - regional(half,"fast","occupied_semantic_accuracy")
    half_fonly = regional(candidate,"F_only","occupied_semantic_accuracy") - regional(half,"F_only","occupied_semantic_accuracy")
    ci = comparisons["N-group_minus_T-A0"]
    checks = dict(fast_semantic_mean_gain_at_least_1pp=float(fast.mean())>=1.,
        fast_semantic_mean_ci_lower_positive=ci["fast_accuracy"]["mean_delta_95ci_pp"][0]>0,
        F_only_each_horizon_noninferior=all(c[0]>=-.5 for c in ci["F_only"]["per_horizon_delta_95ci_pp"]),
        global_primary_noninferior=ci["global_miou"]["primary_delta_95ci_pp"][0]>=-.2,
        current_confusion_exact=True)
    passed = all(checks.values())
    incremental = comparisons["N-group_minus_N-half"]["fast_accuracy"]["mean_delta_95ci_pp"][0]>0
    runtime.atomic_json(ROOT/"comparison_statistics.json", dict(
        comparisons=comparisons, acceptance_checks=checks, joint_target_passed=passed,
        fast_semantic_mean_delta_vs_T_A0_pp=float(fast.mean()),
        fast_semantic_mean_delta_vs_N_half_pp=float(half_fast.mean()),
        F_only_delta_vs_T_A0_pp=fonly.tolist(), F_only_delta_vs_N_half_pp=half_fonly.tolist(),
        fast_neighborhood_mean_delta_vs_T_A0_pp=float(near.mean()),
        positive_fast_increment_vs_N_half_CI=incremental,
        source_sha256=hashes, bootstrap_replicates=2000, bootstrap_seed=20261023,
        scenes=150, samples=1024, additional_training_updates=0,
        scope="One predeclared grouped parameter intervention on reused development data; no training-seed uncertainty."))
    rows, class_lines = [], ["# N-group 七时刻逐类别 IoU", "", "数值为百分数，差值为百分点。全空间类别结果不同于快速 GT 分组。", ""]
    for h in range(7):
        class_lines += [f"## {h*.5:g} 秒", "",
            "| 类别 | T-A0 | N-native | N-half | N-group | N-group−T-A0 | N-group−N-half |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for c,name in enumerate(CLASS_NAMES):
            values=[m["horizons"][h]["class_iou"][c] for m in metrics]
            row=dict(seconds=h*.5,class_index=c,class_name=name,
                **{n+"_iou_percent":v for (n,_,_),v in zip(SOURCES,values)},
                N_group_minus_T_A0_pp=values[3]-values[0],N_group_minus_N_half_pp=values[3]-values[2])
            rows.append(row)
            class_lines.append("| "+name+" | "+" | ".join(f"{v:.4f}" for v in values)+
                f" | {values[3]-values[0]:+.4f} | {values[3]-values[2]:+.4f} |")
        class_lines.append("")
    with (ROOT/"per_horizon_class_iou.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/"per_class_comparison.md").write_text("\n".join(class_lines)+"\n")
    complete=json.loads((WORK/"completed.json").read_text())
    lines=["# N-group：固定分组更新验证结果", "",
        f"**原定联合目标：{'通过' if passed else '未通过'}。**", "",
        "本组把位置/速度分支更新从 N-half 的 50% 改为 100%，其他预测参数与 N-half 精确一致。没有新增训练，只完成一个固定组合的 1024 样本验证（150 场景）。",
        f"验证耗时 {complete['evaluation_seconds']:.1f}s。工程检查包括完整参数拼接检查、固定状态和 query 分配检查；当前帧全部混淆计数与 T-A0 相同。", "",
        "## 快速语义与 F-only", "",
        "以下为准确率百分数，不是 IoU。F-only 使用同一个固定旧参考区域。", "",
        "| 条件 | 快速语义 1s | 2s | 3s | F-only 1s | 2s | 3s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for (name,_,_),m in zip(SOURCES,metrics):
        values=[*regional(m,"fast","occupied_semantic_accuracy"),*regional(m,"F_only","occupied_semantic_accuracy")]
        lines.append("| "+name+" | "+" | ".join(f"{v:.4f}" for v in values)+" |")
    lines += ["", "## 相对 N-half：这个组合是否值得", "",
        f"平均快速语义变化 **{half_fast.mean():+.4f} pp**，场景配对 95% 区间 **{interval(comparisons['N-group_minus_N-half']['fast_accuracy']['mean_delta_95ci_pp'])}**。", "",
        "| 时距 | 快速语义变化 pp | 95% 区间 | F-only 变化 pp | 95% 区间 |",
        "|---|---:|---|---:|---|"]
    for h in range(3):
        c=comparisons["N-group_minus_N-half"]
        lines.append(f"| {h+1}s | {half_fast[h]:+.4f} | "+interval(c["fast_accuracy"]["per_horizon_delta_95ci_pp"][h])+
            f" | {half_fonly[h]:+.4f} | "+interval(c["F_only"]["per_horizon_delta_95ci_pp"][h])+" |")
    lines += ["", "## 相对 T-A0：联合目标", "",
        f"平均快速语义变化 **{fast.mean():+.4f} pp**，区间 **{interval(ci['fast_accuracy']['mean_delta_95ci_pp'])}**。", "",
        "| 时距 | 快速语义变化 pp | F-only 变化 pp | F-only 差值区间 |",
        "|---|---:|---:|---|"]
    for h in range(3):
        lines.append(f"| {h+1}s | {fast[h]:+.4f} | {fonly[h]:+.4f} | "+interval(ci["F_only"]["per_horizon_delta_95ci_pp"][h])+" |")
    labels=["快速语义平均至少 +1 pp","快速语义平均区间下界大于 0","F-only 各时距区间下界至少 −0.5 pp","全域主指标区间下界至少 −0.2 pp","当前帧完整计数相同（隔离条件）"]
    lines += ["", "| 预设条件 | 结果 |", "|---|---|"]
    lines += [f"| {label} | {'通过' if value else '未通过'} |" for label,value in zip(labels,checks.values())]
    lines += ["", "## 全域 mIoU", "",
        "| 条件 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 0/1/2/3s 均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name,_,_),m in zip(SOURCES,metrics):
        lines.append("| "+name+" | "+" | ".join(f"{h['miou']:.4f}" for h in m["horizons"])+f" | {m['selection_mean_miou']:.4f} |")
    lines += ["", "## 邻域覆盖、误报与快速 GT", "",
        "邻域包含背景；快速 GT 分组只包含已关联的快速占据体素。", "",
        "| 条件 | 时距 | 邻域 IoU | precision | recall | FP | FN | 快速 GT 召回 | 已召回快速 GT 类别正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index in (0,2,3):
        for h in range(3):
            n=metrics[index]["regions"][h]["metrics"]["fast_neighborhood"]
            f=metrics[index]["regions"][h]["metrics"]["fast"]
            lines.append(f"| {SOURCES[index][0]} | {h+1}s | {n['occupied_iou']:.4f} | {n['occupied_precision']:.4f} | {n['occupied_recall']:.4f} | {n['false_positive']} | {n['false_negative']} | {f['occupied_recall']:.4f} | {f['conditional_semantic_accuracy']:.4f} |")
    lines += ["", "## 解释边界", "",
        "本组对 reg/vel 两块参数联合干预；其余参数固定为 N-half。位置变化仍会影响后续状态、类别和 Gaussian 属性，因此不能称作固定语义的纯几何消融，也不能分别归功于某一个分支。",
        "这是同一训练方向上的一个固定分组设置。结果不等于按 query 门控已经有效，更不能用反复开发集上的一次通过替代独立数据或充分训练验证。场景区间不包括训练种子与反复选择的不确定性。", "",
        "## 文件", "",
        "- [预设实验设计](experiment_design.md)",
        "- [完整最终指标](fixed_grouped_update_metrics.json)",
        "- [配对统计](comparison_statistics.json)",
        "- [全部七时刻逐类 IoU](per_class_comparison.md)",
        "- [逐类 CSV](per_horizon_class_iou.csv)",
        "- [运行记录](../../../out/pmg_world_native_adapt_20261024/N_group_step)"]
    (ROOT/"results_summary.md").write_text("\n".join(lines)+"\n")
    (ROOT/"README.md").write_text("# N-group 固定分组更新\n\n[结果汇总](results_summary.md)；[实验设计](experiment_design.md)；[逐类对比](per_class_comparison.md)。\n")
    print(json.dumps(dict(joint_target_passed=passed,fast_mean_delta_vs_T_A0_pp=float(fast.mean()),
        fast_mean_delta_vs_N_half_pp=float(half_fast.mean()),F_only_delta_vs_T_A0_pp=fonly.tolist(),
        primary=candidate["selection_mean_miou"],positive_increment_vs_N_half=incremental,checks=checks),indent=2))


if __name__ == "__main__":
    main()
