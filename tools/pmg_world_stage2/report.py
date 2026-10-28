"""Readable stage2 metrics and fixed-budget comparisons; no stage1 writes."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES
from tools.pmg_world.regions import region_result
from tools.pmg_world.summarize import miou


ROOT = Path("analysis/pmg_world_stage2")
ARMS = ("A0_control", "M1_motion", "T_A0_dense", "T_C1_dense")


def records(directory):
    return [
        json.loads(p.read_text())
        for p in sorted(Path(directory).glob("epoch_*_metrics.json"))
    ]


def refresh(cfg):
    directory = Path(cfg.report_directory)
    rows = records(directory)
    if not rows:
        return
    best = max(rows, key=lambda r: r["selection_mean_miou"])
    runtime.atomic_json(
        directory / "summary.json",
        dict(
            completed_epochs=len(rows),
            planned_epochs=10,
            best_epoch=best["epoch"],
            best_metrics=best,
        ),
    )
    output = [
        dict(
            epoch=r["epoch"],
            seconds=h["seconds"],
            class_index=i,
            class_name=name,
            iou=h["class_iou"][i],
            miou=h["miou"],
        )
        for r in rows
        for h in r["horizons"]
        for i, name in enumerate(CLASS_NAMES)
    ]
    with (directory / "epoch_horizon_class_iou.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    lines = [
        f"# Stage2 {cfg.pmg_experiment.arm}",
        "",
        "实际训练上限10轮；沿用原15轮学习率曲线的前10轮。固定4096/1024样本，每轮完整验证。",
        "",
        "| 轮次 | 当前 | 0.5秒 | 1秒 | 1.5秒 | 2秒 | 2.5秒 | 3秒 | 主指标 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        values = " | ".join(f"{h['miou']:.4f}" for h in r["horizons"])
        lines.append(f"| {r['epoch']} | {values} | {r['selection_mean_miou']:.4f} |")
    lines += [
        "",
        f"本组前{len(rows)}轮按原主指标选中第{best['epoch']}轮；其他指标统一采用该轮。",
        "",
        "逐类结果：[CSV](epoch_horizon_class_iou.csv)。每轮JSON含快速区域、F-only与邻域FP。",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    overview()


def overview():
    entries = []
    for arm in ARMS:
        rows = records(ROOT / arm)
        if rows:
            best = max(rows, key=lambda r: r["selection_mean_miou"])
            entries.append(
                dict(
                    arm=arm,
                    completed_epochs=len(rows),
                    best_epoch=best["epoch"],
                    best_metrics=best,
                )
            )
    runtime.atomic_json(ROOT / "comparison.json", dict(planned_epochs=10, arms=entries))
    lines = [
        "# Stage2 训练结果",
        "",
        "各组最佳值来自各自一个checkpoint；完成轮数不同的结果不作同预算胜负判断。机制诊断见[诊断报告](diagnosis_report.md)。",
        "",
        "| 组别 | 完成轮数 | 最佳轮次 | 0/1/2/3秒mIoU | 七时刻均值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in entries:
        b = r["best_metrics"]
        lines.append(
            f"| {r['arm']} | {r['completed_epochs']} | {r['best_epoch']} | {b['selection_mean_miou']:.4f} | {b['all_horizons_mean_miou']:.4f} |"
        )
    if not entries:
        lines.append("\n尚无完成整轮的训练结果。")
    dense = {r["arm"]: r for r in entries}
    if all(a in dense for a in ("T_A0_dense", "T_C1_dense")):
        common = min(dense[a]["completed_epochs"] for a in ("T_A0_dense", "T_C1_dense"))
        conditions = {
            "A0_sparse": Path("analysis/pmg_world_stage1/A0"),
            "C1_sparse": Path("analysis/pmg_world_stage1/C1"),
            "A0_dense": ROOT / "T_A0_dense",
            "C1_dense": ROOT / "T_C1_dense",
        }
        selected = {
            name: max(
                (r for r in records(path) if r["epoch"] <= common),
                key=lambda r: r["selection_mean_miou"],
            )
            for name, path in conditions.items()
        }
        lines += [
            "",
            f"## 时间监督2×2对照：统一前{common}轮",
            "",
            "| 条件 | 选中轮次 | 主指标 | 七时刻均值 |",
            "|---|---:|---:|---:|",
        ]
        for name, r in selected.items():
            lines.append(
                f"| {name} | {r['epoch']} | {r['selection_mean_miou']:.4f} | {r['all_horizons_mean_miou']:.4f} |"
            )
        diff = []
        lines += [
            "",
            "| 秒 | 稀疏监督C1−A0 | 全时刻C1−A0 | 差值变化 |",
            "|---|---:|---:|---:|",
        ]
        for h in range(7):
            sparse = (
                selected["C1_sparse"]["horizons"][h]["miou"]
                - selected["A0_sparse"]["horizons"][h]["miou"]
            )
            dense_delta = (
                selected["C1_dense"]["horizons"][h]["miou"]
                - selected["A0_dense"]["horizons"][h]["miou"]
            )
            diff.append(
                dict(
                    seconds=h * 0.5,
                    sparse_c1_minus_a0=sparse,
                    dense_c1_minus_a0=dense_delta,
                    difference_in_differences=dense_delta - sparse,
                )
            )
            lines.append(
                f"| {h*.5:g} | {sparse:+.4f} | {dense_delta:+.4f} | {dense_delta-sparse:+.4f} |"
            )
        runtime.atomic_json(
            ROOT / "time_supervision_comparison.json",
            dict(
                common_epoch_budget=common,
                selected=selected,
                per_horizon=diff,
                limitation="Historical sparse arms share data, initial parameter hashes, LR prefix and total voxel weights. Per-time supervision allocation changes by design; scene intervals are conditional on checkpoint selection.",
            ),
        )
        lines += [
            "",
            "第1轮所有组的未来体素预算为0；时间监督干预从第2轮起生效。完整时距从第6轮起展开，早期结果不作为最终机制结论。",
        ]
    (ROOT / "training_report.md").write_text("\n".join(lines) + "\n")


def diagnosis_report():
    lines = [
        "# Stage2 机制诊断",
        "",
        "先区分现象与原因：完整1024样本的[快速错误分解](offline_diagnosis.md)已经完成；下述干预固定权重与预测中心，但不替代重新训练。",
        "",
    ]
    for arm in ("A0", "A4", "C1"):
        path = ROOT / f"diagnosis_{arm}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines += [
            f"## {arm}：实际第{data['checkpoint_epoch']}轮权重，{len(data['sample_indices'])}个固定验证样本",
            "",
            "各组权重轮次不同，跨组绝对值仅作探索。同一行内的读出替换才保持权重和中心相同；替换可能有分布不匹配。",
            "",
            "| 读出方式 | 0s mIoU | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        with np.load(ROOT / f"diagnosis_{arm}_counts.npz") as counts:
            for variant in data["variants"]:
                values = " | ".join(
                    f"{miou(h):.3f}" for h in counts[f"global_{variant}"]
                )
                lines.append(f"| {variant} | {values} |")
            lines += [
                "",
                "| 读出方式 | 1s快速正确% | 2s | 3s | 1s召回% | 2s | 3s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for variant in data["variants"]:
                regions = [region_result(h[0]) for h in counts[f"regions_{variant}"]]
                values = [r["occupied_semantic_accuracy"] for r in regions] + [
                    r["occupied_recall"] for r in regions
                ]
                lines.append(
                    f"| {variant} | "
                    + " | ".join(f"{v:.3f}" if v is not None else "NA" for v in values)
                    + " |"
                )
        lines += [
            "",
            "距离仅为最近有效Gaussian中心的代理，不代表精确支持、正确类别或可靠实例关联。",
            "",
            "| 秒 | 快速GT数量 | 最近中心均距m | 漏检中中心距离>0.8m% | 最近中心速度开关关闭% |",
            "|---|---:|---:|---:|---:|",
        ]
        for seconds, row in data["coverage_summary"].items():
            lines.append(
                f"| {seconds} | {row['fast_voxels']} | {row['nearest_distance_mean_m']:.3f} | {row['missed_farther_than_0_8m_pct']:.3f} | {row['nearest_gate_closed_pct']:.3f} |"
            )
        associated = sum(r["associated_children"] for r in data["association"])
        total = sum(r["total_children"] for r in data["association"])
        fast = sum(r["fast_associated_children"] for r in data["association"])
        closed = sum(r["fast_gate_closed_children"] for r in data["association"])
        eligible = sum(r["eligible_queries"] for r in data["association"])
        mixed = sum(r["multi_instance_queries"] for r in data["association"])
        lines += [
            "",
            f"预测位置的GT框关联覆盖{100*associated/total:.2f}%；关联到快速实例的子Gaussian中速度开关关闭{100*closed/fast if fast else 0:.2f}%；有效query中多实例占{100*mixed/eligible if eligible else 0:.2f}%。关联仅覆盖已落入GT框的点，存在选择偏差；多实例也不必然等于不同运动。",
            "",
        ]
    path = ROOT / "gradient_diagnosis.json"
    if path.exists():
        data = json.loads(path.read_text())
        lines += [
            "## 当前／未来体素梯度",
            "",
            "两份固定训练样本、无优化器更新，仅用于筛查。负余弦不证明有害冲突。",
            "",
            "| 组别 | 样本 | 参数组 | 当前梯度范数 | 未来合计范数 | 余弦 |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for r in data["records"]:
            if r["future_horizon"] != "sum":
                continue
            cosine = (
                f"{r['cosine']:.4f}" if r["cosine"] is not None else "无共同非零梯度"
            )
            lines.append(
                f"| {r['arm']} | {r['train_index']} | {r['group']} | {r['current_norm']:.4f} | {r['future_norm']:.4f} | {cosine} |"
            )
        lines += [
            "",
            "这两份样本中，A0/C1的属性梯度余弦均为正，C1当前语义读出也收到同向的未来梯度。当前证据不支持把退化直接归因于有害梯度冲突；该结论只覆盖这些参数组和样本。",
        ]
    for arm in ("A0", "C1"):
        path = ROOT / f"gates_{arm}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines += [
            f"## {arm}：固定权重的运动开关干预",
            "",
            "原开关、连续softmax门、全部打开三种推理；原开关复现已逐体素核对一致。soft/open属于分布外干预，不代替训练。",
            "",
            "| 开关 | 主指标 | 1s快速正确% | 2s | 3s | 1s邻域IoU | 2s | 3s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        with np.load(ROOT / f"gates_{arm}_counts.npz") as counts:
            for variant in data["variants"]:
                primary = np.mean(
                    [miou(counts[f"global_{variant}"][h]) for h in (0, 2, 4, 6)]
                )
                regions = counts[f"regions_{variant}"]
                values = [
                    region_result(r[0])["occupied_semantic_accuracy"] for r in regions
                ] + [region_result(r[5])["occupied_iou"] for r in regions]
                lines.append(
                    f"| {variant} | {primary:.4f} | "
                    + " | ".join(f"{v:.3f}" for v in values)
                    + " |"
                )
        if arm == "C1":
            lines += [
                "",
                "| 秒 | GT框关联且渲染类别正确的子Gaussian数 | 其中速度开关关闭% |",
                "|---|---:|---:|",
            ]
            for h in (2, 4, 6):
                rows = [
                    r for r in data["semantic_motion_disagreement"] if r["horizon"] == h
                ]
                n = sum(r["fast_render_correct"] for r in rows)
                off = sum(r["fast_render_correct_gate_closed"] for r in rows)
                lines.append(f"| {h*.5:g} | {n} | {100*off/n if n else 0:.3f} |")
        lines += [
            "",
            "全部打开没有同时改善三个时刻的快速准确率与邻域IoU。因此，语义／运动不一致虽可观测，单纯解除硬开关不能解释或修复全部退化。",
        ]
    for arm in ("A0", "A4", "C1"):
        path = ROOT / f"readout_audit_{arm}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines += [
            f"## {arm}：实际渲染分数中的漏检",
            "",
            "前景指17个非空类内部的最大分数；下表只考察GT快速体素。前景类别正确但空类获胜，描述的是实际分数竞争，不证明阈值修改能带来公平的整体收益。",
            "",
            "| 秒 | 正确非空% | 错误非空% | 前景类别正确但预测为空% | 前景类别也错且预测为空% |",
            "|---|---:|---:|---:|---:|",
        ]
        for seconds, r in data["summary"].items():
            n = r["fast_voxels"]
            values = [
                r[k] * 100 / n
                for k in (
                    "correct",
                    "wrong_nonempty",
                    "empty_foreground_correct",
                    "empty_foreground_wrong",
                )
            ]
            lines.append(
                f"| {seconds} | " + " | ".join(f"{v:.3f}" for v in values) + " |"
            )
    lines += [
        "",
        "## 如何解读",
        "",
        "固定中心时，若读出替换能改变半秒时刻表现，则该部分变化不需要位置变化就能产生；仍不能把分布外替换等同于训练后的属性消融。",
        "",
        "最近中心接近GT但预测为空，说明单凭中心距离不足以解释漏检；需要继续检查精确渲染贡献和类别。运动开关关闭的相关性不证明打开即可改善。",
        "",
        "下一项因果训练对照为T：A0/C1共同增加直接体素监督时刻，同时保持当前权重和未来总权重。比较两个监督条件下的C1−A0，而不只比较T-C1与旧C1。",
    ]
    (ROOT / "diagnosis_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    diagnosis_report()
    overview()
