"""Refresh per-epoch tables from formal metrics, without touching checkpoints."""

import csv
import json
from pathlib import Path

from .metrics import CLASS_NAMES
from .protocol import MODEL_NAME
from .runtime import atomic_json


def refresh(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [
        json.loads(path.read_text())
        for path in sorted(directory.glob("epoch_*_metrics.json"))
    ]
    if not epochs:
        return
    best = max(epochs, key=lambda item: item["selection_mean_miou"])
    atomic_json(
        output / "training_summary.json",
        dict(
            model_name=MODEL_NAME,
            completed_epochs=len(epochs),
            planned_epochs=15,
            best_epoch=best["epoch"],
            best_selection_mean_miou=best["selection_mean_miou"],
            selection_horizons_seconds=[0, 1, 2, 3],
            best_metrics=best,
        ),
    )
    rows = [
        dict(
            epoch=item["epoch"],
            seconds=horizon["seconds"],
            class_index=index,
            class_name=name,
            iou_percent=horizon["class_iou"][index],
            miou_percent=horizon["miou"],
            occupied_iou_percent=horizon["occupied_iou"],
        )
        for item in epochs
        for horizon in item["horizons"]
        for index, name in enumerate(CLASS_NAMES)
    ]
    with (output / "epoch_horizon_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {MODEL_NAME}",
        "",
        "mIoU 固定平均 17 个非空类，选择指标为当前、1、2、3 秒 mIoU 的平均值。",
        "",
        "| Epoch | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | 选择指标 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in epochs:
        values = " | ".join(f"{h['miou']:.4f}" for h in item["horizons"])
        lines.append(
            f"| {item['epoch']} | {values} | {item['selection_mean_miou']:.4f} |"
        )
    lines += [
        "",
        f"最佳轮次：{best['epoch']}；选择指标：{best['selection_mean_miou']:.4f}%。",
        "",
        "[逐轮、逐时刻、逐类别 IoU](epoch_horizon_class_iou.csv)",
        "[每轮详细诊断](epoch_diagnostics/)",
    ]
    (output / "training_report.md").write_text("\n".join(lines) + "\n")
