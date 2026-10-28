"""One-pass seven-horizon metrics, motion regions and shape diagnostics."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from model.encoder.opus_encoder import _opus_decode_points
from model.encoder.pmg_shape_projection import centered, fit_yaw, apply_rotation, yaw_matrix
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics, CLASS_NAMES
from .regions import HORIZONS, REGIONS, histogram, load_masks, region_result, token_at


@torch.no_grad()
def shape_statistics(model, result, sample_index):
    current = result["query_stamps"] == 0
    reference = centered(_opus_decode_points(result["all_refine_pts"][-1][:, current], model.pc_range))
    indices = torch.triu_indices(reference.shape[-2], reference.shape[-2], offset=1,
                                device=reference.device)
    reference_distances = torch.cdist(reference, reference)[..., indices[0], indices[1]].sort(-1).values
    records = []
    for horizon, points in enumerate(result["forecast_points_list"], 1):
        local = centered(_opus_decode_points(points[:, :reference.shape[1]], model.pc_range))
        rotation = yaw_matrix(fit_yaw(reference, local))
        residual = apply_rotation(local, rotation.transpose(-1, -2)) - reference
        rms = residual.square().sum(-1).mean(-1).sqrt()
        distances = torch.cdist(local, local)[..., indices[0], indices[1]].sort(-1).values
        invariant_change = (distances - reference_distances).abs().mean(-1)
        records.append(dict(index=sample_index, seconds=horizon * .5,
                            scope="all_current_query_patches_not_GT_instances",
                            canonical_shape_rms_mean_m=float(rms.mean()),
                            canonical_shape_rms_p95_m=float(torch.quantile(rms, .95)),
                            canonical_shape_rms_max_m=float(rms.max()),
                            rms_note="fixed_child_slot_residual; baseline child slots need not be physical correspondences",
                            pair_distance_distribution_change_mean_m=float(invariant_change.mean()),
                            pair_distance_distribution_change_p95_m=float(torch.quantile(invariant_change, .95)),
                            finite_points=bool(torch.isfinite(points).all())))
    return records


@torch.no_grad()
def evaluate(model, dataset, cfg, directory, epoch, limit=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference = Path(cfg.pmg_experiment.reference_directory)
    if not (reference / "completed.json").exists():
        raise ValueError("frozen region reference must finish before formal evaluation")
    expected = json.loads((reference / "protocol.json").read_text())
    if expected["manifests"] != runtime.audit_manifests(cfg):
        raise ValueError("reference and training manifests differ")
    metric = HorizonMetrics()
    region_counts = np.zeros((3, len(REGIONS), 18, 18), dtype=np.int64)
    scene_counts, scene_region_counts, samples_per_scene = {}, {}, {}
    geometry = []
    selected = set(np.linspace(0, len(dataset) - 1, 16, dtype=int).tolist())
    rng = runtime.capture_rng()
    started = time.monotonic()
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    try:
        for index, data in enumerate(runtime.loader(dataset, cfg.val_loader)):
            if limit and index >= limit:
                break
            images, metas = runtime.move_batch(data, "cuda")
            result = model(imgs=images, metas=metas)
            metric.add(result, metas)
            scene = str(dataset.keyframes[index][0])
            scene_counts.setdefault(scene, np.zeros((7, 18, 18), dtype=np.int64))
            scene_region_counts.setdefault(scene, np.zeros_like(region_counts))
            samples_per_scene[scene] = samples_per_scene.get(scene, 0) + 1
            token = token_at(dataset, index, 0)
            masks_path = reference / "regions" / (token + ".npz")
            grids = [result["final_occ_grid"]] + result["future_occ_grids"]
            for horizon, prediction in enumerate(grids):
                target = metas["occ_label"] if horizon == 0 else metas["future_occ_labels"][:, horizon - 1]
                target = target.cpu().numpy().reshape(-1)
                prediction = prediction.cpu().numpy().reshape(-1)
                scene_counts[scene][horizon] += histogram(prediction, target)
                if horizon in HORIZONS:
                    slot = HORIZONS.index(horizon)
                    masks = load_masks(masks_path, horizon)
                    for region_index, name in enumerate(REGIONS):
                        counts = histogram(prediction, target, masks[name])
                        region_counts[slot, region_index] += counts
                        scene_region_counts[scene][slot, region_index] += counts
            if index in selected:
                geometry.extend(shape_statistics(model, result, index))
            del result, images, metas, data
            if (index + 1) % 100 == 0:
                print("EVAL", epoch, index + 1, "seconds", round(time.monotonic() - started, 1), flush=True)
    finally:
        model.memory.clear()
        model.memory_queue.clear()
        runtime.restore_rng(rng)
    results = metric.results()
    for horizon in results["horizons"]:
        horizon["static_class_miou_fullspace"] = float(np.mean(
            [horizon["class_iou"][c] for c in (0, 1, 11, 12, 13, 14, 15, 16)]))
    results.update(epoch=epoch, arm=cfg.pmg_experiment.arm,
                   partial=limit is not None, planned_epochs=cfg.max_epochs,
                   evaluation_seconds=time.monotonic() - started,
                   future_123_mean_miou=float(np.mean([results["horizons"][h]["miou"] for h in HORIZONS])),
                   regions=[dict(seconds=horizon * .5, metrics={
                       name: region_result(region_counts[slot, i]) for i, name in enumerate(REGIONS)
                   }) for slot, horizon in enumerate(HORIZONS)],
                   geometry=geometry,
                   region_reference=expected["checkpoint"],
                   speed_region_note="GT occupied slice; FP assessed separately in the full grid and fast_neighborhood")
    runtime.atomic_json(directory / f"epoch_{epoch:02d}_metrics.json", results)
    names = sorted(scene_counts)
    np.savez_compressed(directory / f"epoch_{epoch:02d}_counts.npz",
                        confusion=metric.confusion, region_confusion=region_counts,
                        scenes=np.asarray(names), scene_confusion=np.stack([scene_counts[s] for s in names]),
                        scene_region_confusion=np.stack([scene_region_counts[s] for s in names]),
                        sample_counts=np.asarray([samples_per_scene[s] for s in names]))
    return results


def refresh(cfg):
    directory = Path(cfg.report_directory)
    epochs = [json.loads(p.read_text()) for p in sorted(directory.glob("epoch_*_metrics.json"))]
    if not epochs:
        return
    best = max(epochs, key=lambda x: x["selection_mean_miou"])
    runtime.atomic_json(directory / "summary.json", dict(completed_epochs=len(epochs), planned_epochs=15,
                         best_epoch=best["epoch"], best_metrics=best, arm=cfg.pmg_experiment.arm))
    rows = [dict(epoch=e["epoch"], seconds=h["seconds"], class_index=c, class_name=name,
                 iou=h["class_iou"][c], miou=h["miou"], occupied_iou=h["occupied_iou"])
            for e in epochs for h in e["horizons"] for c, name in enumerate(CLASS_NAMES)]
    with (directory / "epoch_horizon_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [f"# PMG {cfg.pmg_experiment.arm}", "", "固定 4096/1024 探索子集；15 轮正式训练，每轮验证。", "",
            "| Epoch | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s | future Avg |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in epochs:
        values = " | ".join(f"{h['miou']:.4f}" for h in e["horizons"])
        text.append(f"| {e['epoch']} | {values} | {e['future_123_mean_miou']:.4f} |")
    text += ["", f"按当前及 1/2/3 秒平均 mIoU 选出的共同最佳轮次：{best['epoch']}。", "",
             "逐类 IoU：[CSV](epoch_horizon_class_iou.csv)。每轮 JSON 含速度分组、固定 F/P 优势区域保持率、邻域 FP 与形状诊断。"]
    (directory / "report.md").write_text("\n".join(text) + "\n")
