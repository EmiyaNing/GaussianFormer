"""Joint-training reports allow current-frame perception to change."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world_native_adapt import gate_report
from tools.pmg_world_native_adapt.gate_optimization_report import (
    CONTROLS,
    build_statistics,
    validate_metric_fields,
    write_class_results,
    write_regional_results,
)
from tools.pmg_world_native_adapt.report import regional
from tools.pmg_world_stage3.summarize_experiments import regional_sufficient
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


def load_joint_sources(sources):
    """Pair scenes/GT counts, without requiring identical predictions at 0s."""
    metrics, global_counts, parts, regions, hashes = [], [], [], [], {}
    reference = None
    for name, folder, prefix in sources:
        metric_path = folder / f"{prefix}_metrics.json"
        counts_path = folder / f"{prefix}_counts.npz"
        metric = json.loads(metric_path.read_text())
        if metric["num_samples"] != 1024 or metric.get("partial", False):
            raise ValueError(f"{name}: incomplete diagnostic subset")
        hashes[name] = {str(p): runtime.digest(p) for p in (metric_path, counts_path)}
        with np.load(counts_path) as archive:
            paired = {key: archive[key].copy() for key in ("scenes", "sample_counts")}
            paired["global_gt"] = archive["scene_confusion"].sum(-1)
            paired["region_gt"] = archive["scene_region_confusion"].sum(-1)
            if reference is None:
                reference = paired
            else:
                for key, values in paired.items():
                    np.testing.assert_array_equal(
                        values, reference[key], err_msg=f"{name}/{key}"
                    )
            np.testing.assert_array_equal(
                archive["scene_confusion"].sum(0), archive["confusion"]
            )
            np.testing.assert_array_equal(
                archive["scene_region_confusion"].sum(0), archive["region_confusion"]
            )
            counts = archive["confusion"]
            diagonal = np.diagonal(counts, axis1=-2, axis2=-1)
            union = counts.sum(-1) + counts.sum(-2) - diagonal
            iou = 100 * np.divide(
                diagonal,
                union,
                out=np.zeros_like(diagonal, dtype=float),
                where=union > 0,
            )
            np.testing.assert_allclose(
                iou, [h["class_iou"] for h in metric["horizons"]], atol=1e-10
            )
            global_counts.append(archive["scene_confusion"].copy())
            parts.append(regional_sufficient(archive["scene_region_confusion"]))
            regions.append(archive["scene_region_confusion"].copy())
        metrics.append(metric)
    assert len(reference["scenes"]) == 150
    assert int(reference["sample_counts"].sum()) == 1024
    return (
        metrics,
        np.stack(global_counts, 1),
        np.stack(parts, 1),
        np.stack(regions, 1),
        hashes,
    )


def diagnostic_report(folder, epoch, updates):
    folder = Path(folder)
    sources = (*CONTROLS, ("S3-joint", folder, f"epoch_{epoch:02d}"))
    metrics, global_counts, parts, regions, hashes = load_joint_sources(sources)
    validate_metric_fields(metrics, global_counts, regions, sources)
    statistics = build_statistics(
        metrics, global_counts, parts, regions, sources, hashes, "full-network-training"
    )
    # The adaptation helper's identical-current check is inapplicable once
    # perception is trainable. Keep its four numerical performance thresholds.
    statistics["acceptance_checks"].pop("current_confusion_exact")
    statistics["joint_target_passed"] = all(statistics["acceptance_checks"].values())
    statistics["paired_count_checks"].pop("scene_current_confusion_exact")
    statistics["paired_count_checks"]["current_prediction_changes_allowed"] = True
    statistics.update(
        additional_training_updates=updates,
        full_training_epochs=epoch,
        diagnostic_scope="Fixed 1024 samples within full 4219 validation",
        scope="Per-epoch development monitoring; scene intervals exclude training-seed and repeated-selection uncertainty.",
    )
    runtime.atomic_json(folder / "comparison_statistics.json", statistics)
    write_class_results(folder, sources, metrics)
    write_regional_results(folder, sources, metrics, regions)
    lines = [
        f"# S3完整训练第{epoch}轮：固定1024样本诊断",
        "",
        "感知器、两条预测分支、Gaussian属性解码和渲染参数共同训练。当前帧允许改变。",
        "",
        "区域计数只覆盖1024样本、150场景；完整4219样本指标见上层报告。F-only不是静态区域。",
        "",
        "| 模型 | 当前mIoU | 主mIoU | fast均值 | F-only 1s | 2s | 3s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (name, _, _), metric in zip(sources, metrics):
        values = [
            metric["horizons"][0]["miou"],
            metric["selection_mean_miou"],
            regional(metric, "fast", "occupied_semantic_accuracy").mean(),
            *regional(metric, "F_only", "occupied_semantic_accuracy"),
        ]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "逐类IoU和区域空预测/错类见同目录CSV及Markdown；配对场景区间见comparison_statistics.json。",
        "旧适配实验的训练预算不同，这些比较用于观察变化，不能隔离架构收益。",
        "每轮均评估，不按区域门槛提前终止15轮训练。",
    ]
    (folder / "results_summary.md").write_text("\n".join(lines) + "\n")


def refresh(directory):
    directory = Path(directory)
    epochs = [
        json.loads(p.read_text())
        for p in sorted(directory.glob("epoch_*_metrics.json"))
    ]
    if not epochs:
        return
    best = max(epochs, key=lambda metric: metric["selection_mean_miou"])
    runtime.atomic_json(
        directory / "training_summary.json",
        dict(
            model="S3-semantic-route-joint-full15e",
            completed_epochs=len(epochs),
            planned_epochs=15,
            best_epoch_by_full_primary=best["epoch"],
            best_selection_mean_miou=best["selection_mean_miou"],
            validation_samples=best["num_samples"],
            initialization="T-A0 subset epoch10",
            training_scope="All parameters; both forecast branches and shared perception",
            perception_lr_policy="epochs 1-5: 1x; epochs 6-15: 0.1x prediction LR",
            best_metrics=best,
        ),
    )
    lines = [
        "# 语义Gaussian路由：完整网络15轮训练",
        "",
        "从T-A0子集第10轮权重重新开始15轮全量联合训练，不接续冻结感知器的S2断点。",
        "训练19,730样本，完整验证4,219样本；主mIoU平均0/1/2/3秒，单时刻mIoU平均17个非空类。",
        "",
        "第1—5轮感知与预测学习率相等；第6—15轮感知学习率为预测的1/10。",
        "",
        "| Epoch | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 主mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for metric in epochs:
        lines.append(
            f"| {metric['epoch']} | "
            + " | ".join(f"{h['miou']:.4f}" for h in metric["horizons"])
            + f" | {metric['selection_mean_miou']:.4f} |"
        )
        for horizon in metric["horizons"]:
            for index, name in enumerate(CLASS_NAMES):
                rows.append(
                    dict(
                        epoch=metric["epoch"],
                        seconds=horizon["seconds"],
                        class_index=index,
                        class_name=name,
                        iou_percent=horizon["class_iou"][index],
                        miou_percent=horizon["miou"],
                        occupied_iou_percent=horizon["occupied_iou"],
                    )
                )
    with (directory / "epoch_horizon_class_iou.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines += [
        "",
        f"当前全量主mIoU最佳：第{best['epoch']}轮，{best['selection_mean_miou']:.4f}%。",
        "",
        "固定1024样本的逐类、fast和F-only诊断在diagnostics目录。每轮诊断不触发提前停止。",
        "尚未完成的轮次不预填性能。",
    ]
    (directory / "training_report.md").write_text("\n".join(lines) + "\n")
