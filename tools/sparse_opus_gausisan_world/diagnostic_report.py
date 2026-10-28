"""Serialize complete per-epoch diagnostic evidence and a readable report."""

import csv
import json
from pathlib import Path
import numpy as np
from . import runtime as base_runner
from .metrics import CLASS_NAMES


def save_diagnostic(output, epoch, payload, formal, reference_directory=None):
    output.mkdir(parents=True, exist_ok=True)
    arrays = payload.pop("_visibility_arrays")
    np.savez_compressed(
        output / f"epoch_{epoch:02d}_visibility_confusion.npz", **arrays
    )
    # Keep the JSON free of NumPy arrays; the NPZ carries exact counts.
    base_runner.atomic_json(output / f"epoch_{epoch:02d}_diagnostics.json", payload)
    baseline_path = (
        Path(reference_directory) / f"epoch_{epoch:02d}_metrics.json"
        if reference_directory
        else None
    )
    baseline = (
        json.loads(baseline_path.read_text())
        if baseline_path is not None and baseline_path.exists()
        else None
    )
    rows = []
    for h, value in enumerate(formal["horizons"]):
        for c, name in enumerate(CLASS_NAMES):
            reference = baseline["horizons"][h]["class_iou"][c] if baseline else None
            rows.append(
                dict(
                    seconds=h * 0.5,
                    class_index=c,
                    class_name=name,
                    fullspace_iou=value["class_iou"][c],
                    strict_iou=reference,
                    delta_percentage_points=(
                        value["class_iou"][c] - reference
                        if reference is not None
                        else None
                    ),
                    gt_voxels=value["class_gt_count"][c],
                )
            )
    with (output / f"epoch_{epoch:02d}_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# sparse_opus_gausisan_world：第{epoch}轮诊断",
        "",
        f"评估样本数：{formal['num_samples']}；mIoU固定平均17个非空类。"
        f"当前/1/2/3秒平均mIoU：{formal['selection_mean_miou']:.4f}%。",
        "",
        "## 本轮观察",
        "",
    ]
    full = payload["visibility"]["h0/full"]
    hidden = payload["visibility"]["h0/noncamera"]
    hidden_fp_share = 100 * hidden["occupied_fp"] / max(full["occupied_fp"], 1)
    hidden_gt_percent = 100 * hidden["gt_occupied"] / max(hidden["voxel_count"], 1)
    lines += [
        f"当前误报中不可见区域占{hidden_fp_share:.3f}%；不可见区域GT占据比例为{hidden_gt_percent:.3f}%，"
        f"模型预测占据比例为{hidden['predicted_occupied_percent']:.3f}%。",
        "",
        f"当前训练阶段：{'时间扩展预训练' if epoch <= 5 else '当前与未来逐步直接监督'}。"
        f"本轮实际Gaussian体素监督时刻：{[h * .5 for h in payload.get('supervised_gaussian_horizons', [0])]}秒。",
        "",
    ]
    if baseline is not None:
        for h in (0, 2, 4, 6):
            delta = np.asarray(formal["horizons"][h]["class_iou"][:17]) - np.asarray(
                baseline["horizons"][h]["class_iou"][:17]
            )
            smallest = np.argsort(delta)[:3]
            details = "、".join(
                f"{CLASS_NAMES[c]} {delta[c]:+.3f} pp" for c in smallest
            )
            lines += [
                f"{h * .5:g}秒与Strict同轮相比差值最小的三个类别：{details}。",
                "",
            ]
    lines += [
        "## 监督范围与可见性",
        "",
        "Gaussian体素监督mask=none，包含不可见区域及空体素。未来直接监督仍按原训练时域逐步启用。",
        "",
        "| 时刻 | 区域 | mIoU (%) | 占据IoU (%) | 精度 (%) | 召回 (%) | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h in range(7):
        for region in ("full", "camera", "noncamera"):
            m = payload["visibility"][f"h{h}/{region}"]
            lines.append(
                f"| {h * .5:g}s | {region} | {m['miou']:.4f} | {m['occupied_iou']:.4f} | {m['occupied_precision']:.4f} | {m['occupied_recall']:.4f} | {m['occupied_fp']} | {m['occupied_fn']} |"
            )
    if baseline is not None:
        lines += [
            "",
            "## 与同轮Strict的完整子集对比",
            "",
            "| 时刻 | Strict mIoU (%) | 全空间Gaussian mIoU (%) | Δ pp | Strict占据IoU (%) | 全空间Gaussian占据IoU (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for h in range(7):
            a, b = baseline["horizons"][h], formal["horizons"][h]
            lines.append(
                f"| {h * .5:g}s | {a['miou']:.4f} | {b['miou']:.4f} | {b['miou']-a['miou']:+.4f} | {a['occupied_iou']:.4f} | {b['occupied_iou']:.4f} |"
            )
    lines += [
        "",
        "## Gaussian属性与未来特征",
        "",
        f"固定属性/渲染样本：{payload['attribute_sample_indices']}；梯度探针只覆盖其中前2个样本的实际监督时刻。",
        "",
        "| 时刻 | 尺度>0.395m (%) | opacity>0.99 (%) | ROI内child (%) | feature范数中位统计 |",
        "|---|---:|---:|---:|---:|",
    ]
    for h in range(7):
        records = [r for r in payload["attributes"] if r["horizon"] == h]
        if records:
            lines.append(
                f"| {h * .5:g}s | {np.mean([r['scale_upper_saturation_percent'] for r in records]):.3f} | {np.mean([r['opacity_upper_saturation_percent'] for r in records]):.3f} | {np.mean([r['in_range_percent'] for r in records]):.3f} | {np.median([r['feature_norm_quantiles'][2] for r in records]):.3f} |"
            )
    lines += [
        "",
        "## 渲染探针",
        "",
        "以下固定小样本变体只用于诊断，不是正式模型成绩，不用于选择checkpoint或阈值。",
        "",
        "| 时刻 | 方式 | 区域 | mIoU (%) | 占据IoU (%) |",
        "|---|---|---|---:|---:|",
    ]
    for key, value in payload["render_probes"].items():
        h, mode, region = key.split("/")
        lines.append(
            f"| {int(h[1:]) * .5:g}s | {mode} | {region} | {value['miou']:.4f} | {value['occupied_iou']:.4f} |"
        )
    lines += [
        "",
        "## 输出层梯度探针",
        "",
        "只比较实际启用的集合/体素损失对最后层raw logits与world centers的梯度；features、opacity、covariance固定，不代表整个模型参数梯度。",
        "",
        "| 样本 | 时刻 | 体素/集合语义范数比 | 语义余弦 | 中心范数比 | 中心余弦 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for p in payload["gradient_probes"]:
        lines.append(
            f"| {p['index']} | {p['seconds']:g}s | {p['semantic']['voxel_to_set_norm_ratio']:.3f} | {p['semantic']['cosine']:.3f} | {p['geometry']['voxel_to_set_norm_ratio']:.3f} | {p['geometry']['cosine']:.3f} |"
        )
    lines += [
        "",
        f"- [每时刻每类别IoU与同轮Strict对比](epoch_{epoch:02d}_class_iou.csv)",
        f"- [完整诊断JSON](epoch_{epoch:02d}_diagnostics.json)",
        f"- [可见性混淆矩阵](epoch_{epoch:02d}_visibility_confusion.npz)",
        "",
        "渲染/梯度探针没有更新参数；诊断前后保留随机数状态。",
    ]
    (output / f"epoch_{epoch:02d}_diagnostics.md").write_text("\n".join(lines) + "\n")
