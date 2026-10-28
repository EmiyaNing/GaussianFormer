"""Report the native adaptation and its single fixed half-step supplement."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world_native_adapt.report import regional, interval
from tools.pmg_world_stage3.summarize_experiments import regional_sufficient, paired_intervals
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


ROOT = Path("analysis/pmg_world_native_adapt_20261024")
HALF = ROOT / "fixed_half_step"
WORK = Path("out/pmg_world_native_adapt_20261024/N_half_step")
SOURCES = [
    ("T-A0", Path("analysis/pmg_world_stage2/T_A0_dense"), "epoch_10"),
    ("M-current", Path("analysis/pmg_world_stage3_20261023/frozen_M_current"), "epoch_01"),
    ("M-repair", Path("analysis/pmg_world_repair_20261024"), "epoch_01"),
    ("N-native", ROOT, "epoch_01"),
    ("N-half", HALF, "epoch_01"),
]


def main():
    assert (WORK / "completed.json").exists()
    metrics, globals_, regions, hashes = [], [], [], {}
    scenes = samples = current = None
    for name, folder, prefix in SOURCES:
        mp, cp = folder / f"{prefix}_metrics.json", folder / f"{prefix}_counts.npz"
        metrics.append(json.loads(mp.read_text()))
        hashes[name] = {str(p):runtime.digest(p) for p in (mp, cp)}
        with np.load(cp) as a:
            if scenes is None:
                scenes, samples, current = a["scenes"].copy(), a["sample_counts"].copy(), a["confusion"][0].copy()
            assert np.array_equal(scenes, a["scenes"]) and np.array_equal(samples, a["sample_counts"])
            assert np.array_equal(current, a["confusion"][0]), name
            globals_.append(a["scene_confusion"].copy())
            regions.append(regional_sufficient(a["scene_region_confusion"]))
    assert samples.sum() == 1024 and len(scenes) == 150
    globals_, regions = np.stack(globals_, 1), np.stack(regions, 1)
    weights = np.random.default_rng(20261023).multinomial(
        len(scenes), np.full(len(scenes), 1/len(scenes)), size=2000)
    boot = (weights.astype(float) @ globals_.reshape(len(scenes), -1)).reshape(2000, 5, 7, 18, 18)
    diagonal = np.diagonal(boot, axis1=-2, axis2=-1)
    union = boot.sum(-1) + boot.sum(-2) - diagonal
    miou = (100*np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union>0))[..., :17].mean(-1)
    comparisons = {}
    for right in (0, 2, 3):
        value = paired_intervals(regions, 4, right, weights)
        delta = miou[:, 4] - miou[:, right]
        value["global_miou"] = dict(
            per_horizon_delta_95ci_pp=np.quantile(delta,[.025,.975],axis=0).T.tolist(),
            primary_delta_95ci_pp=np.quantile(delta[:,[0,2,4,6]].mean(-1),[.025,.975]).tolist())
        comparisons["N-half_minus_"+SOURCES[right][0]] = value
    base, candidate = metrics[0], metrics[-1]
    fast = regional(candidate,"fast","occupied_semantic_accuracy") - regional(base,"fast","occupied_semantic_accuracy")
    fonly = regional(candidate,"F_only","occupied_semantic_accuracy") - regional(base,"F_only","occupied_semantic_accuracy")
    near = regional(candidate,"fast_neighborhood","occupied_iou") - regional(base,"fast_neighborhood","occupied_iou")
    ci = comparisons["N-half_minus_T-A0"]
    checks = dict(fast_semantic_mean_gain_at_least_1pp=float(fast.mean())>=1.,
        fast_semantic_mean_ci_lower_positive=ci["fast_accuracy"]["mean_delta_95ci_pp"][0]>0,
        F_only_each_horizon_noninferior=all(c[0]>=-.5 for c in ci["F_only"]["per_horizon_delta_95ci_pp"]),
        global_primary_noninferior=ci["global_miou"]["primary_delta_95ci_pp"][0]>=-.2,
        current_confusion_exact=True)
    passed = all(checks.values())
    runtime.atomic_json(HALF/"comparison_statistics.json", dict(
        comparisons=comparisons, fast_semantic_mean_delta_pp=float(fast.mean()),
        fast_neighborhood_mean_delta_pp=float(near.mean()), F_only_delta_pp=fonly.tolist(),
        acceptance_checks=checks, joint_target_passed=passed, source_sha256=hashes,
        bootstrap_replicates=2000, bootstrap_seed=20261023, scenes=150, samples=1024,
        supplementary_post_run_probe=True, update_scale=.5, additional_training_updates=0,
        selection_scope="One fixed half step after N-native; reused development data, not an independent confirmation."))
    rows=[]
    class_lines=["# 全部时刻与类别的 IoU 对比", "", "各组均为同一 1024 验证样本；IoU 单位为百分数，差值为百分点。", ""]
    for h in range(7):
        class_lines += [f"## {h*.5:g} 秒", "", "| 类别 | "+" | ".join(n for n,_,_ in SOURCES)+" | N-half−T-A0 |",
            "|---|"+"---:|"*6]
        for c,name in enumerate(CLASS_NAMES):
            values=[m["horizons"][h]["class_iou"][c] for m in metrics]
            row=dict(seconds=h*.5,class_index=c,class_name=name,
                **{n+"_iou_percent":v for (n,_,_),v in zip(SOURCES,values)},
                N_half_minus_T_A0_pp=values[-1]-values[0])
            rows.append(row)
            class_lines.append("| "+name+" | "+" | ".join(f"{v:.4f}" for v in values)+f" | {values[-1]-values[0]:+.4f} |")
        class_lines.append("")
    with (ROOT/"joint_per_horizon_class_iou.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/"joint_per_class_comparison.md").write_text("\n".join(class_lines)+"\n")
    lines=["# 快速语义预测与 F-only：本轮联合结果", "",
        f"**固定半步 N-half 的原定联合目标：{'通过短程筛选' if passed else '未通过'}。**", "",
        f"实际结果：平均快速语义变化 {fast.mean():+.4f} pp；F-only 各时距保持条件{'通过' if checks['F_only_each_horizon_noninferior'] else '未通过'}。幅度条件仍严格按至少 +1 pp 判断，不通过四舍五入修改验收结果。", "",
        "本轮实际只有一次训练：N-native 完成 512 更新、2048 个不同训练样本。它的快速语义收益明显，但 F-only 未达保护条件；随后固定将预测器参数更新量减半，增加一次完整验证，不增加训练、不扫描比例。两组均在 1024 样本、150 场景上评估。", "",
        "N-half 的部署模型只有一套参数、没有新增模块。它是结果出现后提出的补充检验，不能作为独立确认或新架构贡献。原 N-native 失败结论保留。", "",
        "## 快速语义与 F-only 的绝对表现", "",
        "快速语义准确率同时要求非空和类别正确；F-only 是固定旧参考区域准确率。两者单位为百分数，不是 IoU。", "",
        "| 条件 | 快速语义 1s | 2s | 3s | F-only 1s | 2s | 3s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for (name,_,_),m in zip(SOURCES,metrics):
        values=[*regional(m,"fast","occupied_semantic_accuracy"),*regional(m,"F_only","occupied_semantic_accuracy")]
        lines.append("| "+name+" | "+" | ".join(f"{v:.4f}" for v in values)+" |")
    lines += ["", "## N-half 相对 T-A0 的联合判断", "",
        f"快速语义平均提升 **{fast.mean():+.4f} pp**，95% 场景配对区间 **{interval(ci['fast_accuracy']['mean_delta_95ci_pp'])}**。", "",
        "| 时距 | 快速语义变化 pp | 95% 区间 | F-only 变化 pp | 95% 区间 |",
        "|---|---:|---|---:|---|"]
    for h in range(3):
        lines.append(f"| {h+1}s | {fast[h]:+.4f} | "+interval(ci["fast_accuracy"]["per_horizon_delta_95ci_pp"][h])+
            f" | {fonly[h]:+.4f} | "+interval(ci["F_only"]["per_horizon_delta_95ci_pp"][h])+" |")
    labels=["平均快速语义至少 +1 pp","快速语义区间下界大于零","F-only 每时距区间下界至少 −0.5 pp","全域主指标区间下界至少 −0.2 pp","当前帧完整计数相同（隔离条件）"]
    lines += ["", "| 预设条件 | 结果 |", "|---|---|"]
    lines += [f"| {label} | {'通过' if value else '未通过'} |" for label,value in zip(labels,checks.values())]
    lines += ["", "## 全域 mIoU", "", "| 条件 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 0/1/2/3s 均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (name,_,_),m in zip(SOURCES,metrics):
        lines.append("| "+name+" | "+" | ".join(f"{h['miou']:.4f}" for h in m["horizons"])+f" | {m['selection_mean_miou']:.4f} |")
    lines += ["", "## 邻域覆盖与误报", "", "| 条件 | 时距 | 二值 IoU | precision | recall | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for index in (0,3,4):
        for h in range(3):
            n=metrics[index]["regions"][h]["metrics"]["fast_neighborhood"]
            lines.append(f"| {SOURCES[index][0]} | {h+1}s | {n['occupied_iou']:.4f} | {n['occupied_precision']:.4f} | {n['occupied_recall']:.4f} | {n['false_positive']} | {n['false_negative']} |")
    lines += ["", "## 快速语义收益来自哪一层表现", "",
        "快速语义准确率等于快速 GT 占据召回率乘以已召回体素中的类别正确率。下面是同一快速 GT 分组的指标，不能与含背景的邻域召回混用。", "",
        "| 条件 | 时距 | 快速 GT 占据召回率 | 已召回快速 GT 的类别正确率 |",
        "|---|---:|---:|---:|"]
    for index in (0,3,4):
        for h in range(3):
            f=metrics[index]["regions"][h]["metrics"]["fast"]
            lines.append(f"| {SOURCES[index][0]} | {h+1}s | {f['occupied_recall']:.4f} | {f['conditional_semantic_accuracy']:.4f} |")
    lines += ["", "原生适配的快速 GT 占据召回提高约 2.4–2.7 pp，已召回部分的类别正确率仅提高约 0.36–0.61 pp。因此当前快速语义收益更突出地体现在覆盖恢复。召回变化仍可能由位置、形状、opacity 或类别峰值共同造成，这项统计分解不是因果归因。", "",
        "## 解释边界", "",
        "原生适配与修正头的可训练模块、参数量和优化难度不同，不能凭一个对照定位唯一根因。固定半步检验的是参数更新幅度的取舍，不证明具备选择性保留历史信息的新机制。",
        "F-only 保护条件是容许有限损失的非劣检验，通过也不等于完全无损。快速语义需要结合邻域 FP/FN 与全域质量解释。所有区间来自固定检查点的场景重采样，不包含训练种子和反复开发选择的不确定性。", "",
        "## 记录与完整类别结果", "",
        "- [原生适配的原定设计](experiment_design.md)",
        "- [原生适配完整结果](results_summary.md)",
        "- [固定半步的事后补充设计](fixed_half_step/experiment_design.md)",
        "- [固定半步配对统计](fixed_half_step/comparison_statistics.json)",
        "- [全部七时刻逐类 IoU](joint_per_class_comparison.md)",
        "- [逐类 CSV](joint_per_horizon_class_iou.csv)",
        "- [区域重叠统计](fixed_region_overlap.json)"]
    if (ROOT/"research_findings.md").exists():
        lines.append("- [本轮发现与下一步研究判断](research_findings.md)")
    (ROOT/"joint_results_summary.md").write_text("\n".join(lines)+"\n")
    (ROOT/"README.md").write_text("# 快速语义预测与 F-only 联合改进\n\n"
        "[本轮联合结果](joint_results_summary.md)；[原生适配原组结果](results_summary.md)；[逐类对比](joint_per_class_comparison.md)。\n")
    print(json.dumps(dict(joint_target_passed=passed,fast_semantic_mean_delta_pp=float(fast.mean()),
        F_only_delta_pp=fonly.tolist(),near_delta_pp=float(near.mean()),primary=candidate["selection_mean_miou"],checks=checks),indent=2))


if __name__ == "__main__":
    main()
