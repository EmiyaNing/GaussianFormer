"""Aggregate exact intervention counts, paired scene intervals and readable reports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world.regions import REGIONS, region_result
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.diagnostic_statistics import summarize as metric_summary
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES
from .interventions import VARIANTS


def summarize(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    scene_global, scene_regions, coverage, records = {}, {}, [], []
    for index in protocol["sample_indices"]:
        metadata = json.loads((output / "samples" / f"{index:04d}.json").read_text())
        with np.load(output / "samples" / f"{index:04d}.npz") as arrays:
            scene = metadata["scene"]
            scene_global.setdefault(scene, np.zeros_like(arrays["global_confusion"]))
            scene_regions.setdefault(scene, np.zeros_like(arrays["region_confusion"]))
            scene_global[scene] += arrays["global_confusion"]
            scene_regions[scene] += arrays["region_confusion"]
            records.append(arrays["readout_partitions"].copy())
        coverage.extend(metadata["coverage"])
    names = sorted(scene_global)
    global_by_scene = np.stack([scene_global[s] for s in names])
    regions_by_scene = np.stack([scene_regions[s] for s in names])
    global_counts, region_counts = global_by_scene.sum(0), regions_by_scene.sum(0)
    results = {}
    csv_rows = []
    for v, name in enumerate(VARIANTS):
        horizons = [dict(seconds=h * .5, **metric_summary(global_counts[v, h])) for h in range(7)]
        regions = [{region:region_result(region_counts[v, h, j]) for j, region in enumerate(REGIONS)} for h in range(3)]
        results[name] = dict(horizons=horizons, regions=regions,
            primary_miou=float(np.mean([horizons[h]["miou"] for h in (0,2,4,6)])),
            future_123_miou=float(np.mean([horizons[h]["miou"] for h in (2,4,6)])))
        for h, row in enumerate(horizons):
            for c, cls in enumerate(CLASS_NAMES):
                csv_rows.append(dict(variant=name, seconds=h * .5, class_index=c, class_name=cls,
                    iou_percent=row["class_iou"][c], miou_percent=row["miou"],
                    occupied_iou_percent=row["occupied_iou"]))
    iterations = 2000
    weights = np.random.default_rng(20261023).multinomial(len(names), np.full(len(names), 1 / len(names)), size=iterations)
    statistics = {}
    # Bootstrap only compact sufficient counts for regional accuracy and binary IoU.
    fast = regions_by_scene[:, :, :, REGIONS.index("fast")]
    near = regions_by_scene[:, :, :, REGIONS.index("fast_neighborhood")]
    fonly = regions_by_scene[:, :, :, REGIONS.index("F_only")]
    features = np.stack((np.trace(fast[..., :17,:17], axis1=-2,axis2=-1), fast[..., :17,:].sum((-1,-2)),
                         near[..., :17,:17].sum((-1,-2)), near[..., :17,:].sum((-1,-2)), near[..., :,:17].sum((-1,-2)),
                         np.trace(fonly[..., :17,:17],axis1=-2,axis2=-1), fonly[..., :17,:].sum((-1,-2))), -1)
    boot = (weights @ features.reshape(len(names), -1)).reshape(iterations, len(VARIANTS), 3, 7)
    boot_metrics = dict(fast_accuracy=100 * boot[...,0] / np.maximum(boot[...,1],1),
        fast_neighborhood_iou=100 * boot[...,2] / np.maximum(boot[...,3]+boot[...,4]-boot[...,2],1),
        F_only=100 * boot[...,5] / np.maximum(boot[...,6],1))
    for v, name in enumerate(VARIANTS):
        if v == 0:
            continue
        statistics[name] = {metric:dict(
            per_horizon_delta_95ci_pp=np.quantile(values[:,v]-values[:,0], [.025,.975],axis=0).T.tolist(),
            mean_delta_95ci_pp=np.quantile((values[:,v]-values[:,0]).mean(-1), [.025,.975]).tolist())
            for metric, values in boot_metrics.items()}
    runtime.atomic_json(output / "metrics.json", dict(num_samples=len(protocol["sample_indices"]),
        num_scenes=len(names), variants=results, readout_partitions=np.sum(records,axis=0).tolist(),
        region_order=list(REGIONS), coverage=coverage))
    runtime.atomic_json(output / "scene_statistics.json", dict(iterations=iterations, seed=20261023,
        comparisons_against_D00=statistics, note="Fixed checkpoint and reused development set; excludes training-seed and multiple-comparison uncertainty."))
    np.savez_compressed(output / "scene_counts.npz", scenes=np.asarray(names),
                        scene_global_confusion=global_by_scene, scene_region_confusion=regions_by_scene)
    with (output / "per_horizon_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# 固定检查点输出干预结果", "",
        f"样本数：{len(protocol['sample_indices'])}；场景数：{len(names)}；检查点：T-A0第{protocol['checkpoint_epoch']}轮。",
        "GT辅助输出诊断，没有训练模型；不代表可部署方法成绩。干预仅发生于1/2/3秒，其余时刻保持原生输出。", "",
        "| 干预 | 全域0/1/2/3秒mIoU | 1s快区正确率 | 2s | 3s | 1s邻域IoU | 2s | 3s | 3s F-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        fast_values=[r["fast"]["occupied_semantic_accuracy"] for r in row["regions"]]
        near_values=[r["fast_neighborhood"]["occupied_iou"] for r in row["regions"]]
        values=[row["primary_miou"], *fast_values, *near_values, row["regions"][2]["F_only"]["occupied_semantic_accuracy"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" if v is not None else "NA" for v in values) + " |")
    lines += ["", "| 干预 | 平均快速邻域IoU变化pp | 场景配对95%区间 |", "|---|---:|---|"]
    baseline = results["D00"]
    for name in VARIANTS[1:]:
        delta=np.mean([results[name]["regions"][h]["fast_neighborhood"]["occupied_iou"]-baseline["regions"][h]["fast_neighborhood"]["occupied_iou"] for h in range(3)])
        interval=statistics[name]["fast_neighborhood_iou"]["mean_delta_95ci_pp"]
        lines.append(f"| {name} | {delta:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] |")
    lines += ["", "匹配覆盖、逐时刻全部区域和实际分数分类见 metrics.json；逐时刻逐类别IoU见 per_horizon_class_iou.csv。",
              "覆盖有限时，阴性结果不能排除未匹配目标或更大位置误差。单类别语义注入并非实际可学习收益。"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output")
    summarize(parser.parse_args().output)
