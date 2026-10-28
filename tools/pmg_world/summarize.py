"""Live/final PMG comparison and scene-paired bootstrap against A0."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.sparse_opus_gausisan_world.runtime import atomic_json
from .regions import HORIZONS, REGIONS, region_result


ROOT = Path("analysis/pmg_world_stage1")
ARMS = ("A0", "A1", "A3", "A4", "C1")


def paired_interval(candidate, baseline, transform, iterations=2000):
    generator = np.random.default_rng(20261019)
    changes = []
    for _ in range(iterations):
        indices = generator.integers(len(candidate), size=len(candidate))
        left, right = transform(candidate[indices].sum(0)), transform(baseline[indices].sum(0))
        if left is not None and right is not None:
            changes.append(left - right)
    return np.quantile(changes, [.025, .975]).tolist() if changes else None


def miou(hist):
    diagonal = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - diagonal
    return float(np.divide(diagonal[:17], union[:17], out=np.zeros(17),
                           where=union[:17] > 0).mean() * 100)


def static_miou(hist):
    diagonal = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - diagonal
    values = np.divide(diagonal, union, out=np.zeros(18), where=union > 0)
    return float(values[[0, 1, 11, 12, 13, 14, 15, 16]].mean() * 100)


def refresh():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, epochs, curves = [], {}, []
    for arm in ARMS:
        files = sorted((ROOT / arm).glob("epoch_*_metrics.json"))
        records = [json.loads(p.read_text()) for p in files]
        if not records:
            continue
        epochs[arm] = len(records)
        best = max(records, key=lambda e: e["selection_mean_miou"])
        for record in records:
            for horizon in record["horizons"]:
                curves.append(dict(arm=arm, epoch=record["epoch"], seconds=horizon["seconds"],
                                   miou=horizon["miou"], occupied_iou=horizon["occupied_iou"]))
        rows.append(dict(arm=arm, completed_epochs=len(records), best_epoch=best["epoch"],
                         selection_mean_miou=best["selection_mean_miou"],
                         future_mean_miou=best["future_123_mean_miou"],
                         horizons=best["horizons"], regions=best["regions"]))
    result = dict(planned_epochs=15, training_samples=4096, validation_samples=1024,
                  complete=bool(len(rows) == len(ARMS) and all(e == 15 for e in epochs.values())),
                  arms=rows, matched_epoch_intervals=[])
    common = min(epochs.values()) if len(epochs) == len(ARMS) else 0
    previous_path = ROOT / "comparison.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if common and previous.get("interval_epoch") == common:
        result["matched_epoch_intervals"] = previous["matched_epoch_intervals"]
        result["interval_epoch"] = common
    elif common:
        with np.load(ROOT / "A0" / f"epoch_{common:02d}_counts.npz") as saved:
            baseline = {key: saved[key] for key in saved.files}
        for arm in ARMS[1:]:
            with np.load(ROOT / arm / f"epoch_{common:02d}_counts.npz") as saved:
                if not np.array_equal(saved["scenes"], baseline["scenes"]):
                    raise ValueError("scene mismatch in paired evaluation")
                entry = dict(arm=arm, epoch=common, horizon_results=[])
                for slot, horizon in enumerate(HORIZONS):
                    item = dict(seconds=horizon * .5,
                                global_miou_delta_95ci=paired_interval(
                                    saved["scene_confusion"][:, horizon],
                                    baseline["scene_confusion"][:, horizon], miou),
                                static_class_miou_delta_95ci=paired_interval(
                                    saved["scene_confusion"][:, horizon],
                                    baseline["scene_confusion"][:, horizon], static_miou), regions={})
                    for region in ("fast", "F_only", "fast_neighborhood"):
                        i = REGIONS.index(region)
                        key = "occupied_semantic_accuracy" if region != "fast_neighborhood" else "occupied_iou"
                        item["regions"][region] = paired_interval(
                            saved["scene_region_confusion"][:, slot, i],
                            baseline["scene_region_confusion"][:, slot, i],
                            lambda hist, name=key: region_result(hist)[name])
                    entry["horizon_results"].append(item)
                result["matched_epoch_intervals"].append(entry)
        result["interval_epoch"] = common
    atomic_json(ROOT / "comparison.json", result)
    if curves:
        with (ROOT / "learning_curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
    lines = ["# PMG 架构实验：执行与结果", "",
             "固定 4096 训练 / 1024 验证；各组从相同图像初始化开始，15 轮正式训练，无五轮特殊预训练。", "",
             "| 组别 | 架构 | 已完成轮数 | 当前共同最优 checkpoint | 0/1/2/3s 平均 mIoU | future 1/2/3s 平均 mIoU |",
             "|---|---|---:|---:|---:|---:|"]
    descriptions = dict(A0="Gaussian 原传播基线", A1="局部形状保持＋共享平移", A3="显式局部坐标加性更新（自由残差）",
                        A4="刚体位姿＋正交有界残差", C1="A4＋Gaussian内容持久化")
    by_arm = {r["arm"]: r for r in rows}
    for arm in ARMS:
        row = by_arm.get(arm)
        values = f"{row['completed_epochs']} | {row['best_epoch']} | {row['selection_mean_miou']:.4f} | {row['future_mean_miou']:.4f}" if row else "0 | 待产生 | — | —"
        lines.append(f"| {arm} | {descriptions[arm]} | {values} |")
    lines += ["", "状态：" + ("全部完成。" if result["complete"] else "实验执行中；表内仅列已完成的正式整轮验证。"), "",
              "各组逐时刻逐类别 IoU 位于各目录的 epoch_horizon_class_iou.csv；comparison.json 包含相同已完成轮次的场景配对区间。",
              "选共同最佳 checkpoint 的主表与同轮配对区间回答不同问题，不拼接各时刻最优。第一轮尚未训练完整未来范围，不能据其结果判定三秒预测优劣。", "",
              "固定优势区域和真实速度由 reference/ 建立，仅参与评测。该子集已有反复选方案历史，属于探索实验。"]
    if (ROOT / "reference" / "results.json").exists():
        reference = json.loads((ROOT / "reference" / "results.json").read_text())
        lines += ["", "## 原 D 模型的机制复核", "",
                  "| 时距 | 快速 F 准确率 | 快速 P 准确率 | P−F pp | F 仅正确体素 |", "|---|---:|---:|---:|---:|"]
        for r in reference["regions"]:
            f, p = r["fast"]["F"]["occupied_semantic_accuracy"], r["fast"]["P"]["occupied_semantic_accuracy"]
            lines.append(f"| {r['seconds']:g}s | {f:.4f} | {p:.4f} | {p-f:+.4f} | {r['F_only']['F']['occupied_gt']} |")
        lines += ["", "此表使用旧 D 第15轮同权重 F/P；不与本轮从头训练的 A0 混为同一基线。P 使用未来窗口，仅作诊断。"]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    refresh()
