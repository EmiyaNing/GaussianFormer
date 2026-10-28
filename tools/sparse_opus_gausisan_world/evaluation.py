"""Full evaluation with fixed-sample Gaussian, semantic and gradient diagnostics.

The main loop evaluates all selected validation samples. Only the configured
16 samples get counterfactual renders; only the first two get gradient probes.
Evaluation restores the RNG and clears image caches before returning.
"""

from pathlib import Path
import time
import numpy as np
import torch
from loss import OPENOCC_LOSS
from . import runtime as runner
from .diagnostic_statistics import (
    add_evaluation_batch,
    attribute_record,
    classification_statistics,
    conditional_confusion,
    conditional_statistics,
    gradient_probe,
    partition_histograms,
    region_metrics,
)
from .diagnostic_report import save_diagnostic
from .metrics import CLASS_NAMES, HorizonMetrics
from .protocol import MODEL_NAME


def scene_name(dataset, index):
    (scene_index, _) = dataset.keyframes[index]
    return str(scene_index)


def save_scene_counts(path, scenes, sample_counts):
    names = sorted(scenes)
    np.savez_compressed(
        path,
        scenes=np.asarray(names),
        confusion=np.stack([scenes[name] for name in names]),
        sample_counts=np.asarray(
            [sample_counts[name] for name in names], dtype=np.int64
        ),
    )


def render_variants(model, renderer, gaussian, xyz, logits, points, classes, target):
    """Construct D's fixed counterfactual probes; the default prediction is unchanged.

    Bias deltas, fixed scale and opacity changes only measure sensitivity on
    selected samples. They never choose an inference threshold or alter weights.
    """
    modes = dict(
        default=logits.argmax(1).view_as(target),
        point_rasterizer=model.pts_bbox_head.get_occ(
            dict(cls_scores=classes, refine_pts=points)
        ),
    )
    for delta in (-2.0, -1.0, 1.0, 2.0):
        changed = logits.clone()
        changed[:, 17] -= delta
        modes[f"empty_bias_delta_{delta:+g}"] = changed.argmax(1).view_as(target)
    fixed_scale = gaussian._replace(scales=torch.full_like(gaussian.scales, 0.2))
    modes["fixed_scale_0_2"] = renderer(fixed_scale, xyz).argmax(1).view_as(target)
    reduced_opacity = gaussian._replace(opacities=gaussian.opacities * 0.5)
    modes["opacity_times_0_5"] = (
        renderer(reduced_opacity, xyz).argmax(1).view_as(target)
    )
    return modes


@torch.no_grad()
def evaluate(model, dataset, cfg, directory, epoch, limit=None):
    if cfg.loss.mask_type != "none" or cfg.val_loader.batch_size != 1:
        raise ValueError(
            "diagnostics require full-space supervision and batch-one validation"
        )
    indices = np.linspace(
        0, len(dataset) - 1, cfg.epoch_diagnostics.attribute_samples, dtype=int
    ).tolist()
    gradient_indices = set(indices[: cfg.epoch_diagnostics.gradient_samples])
    metric = HorizonMetrics()
    visibility = {
        region: np.zeros((7, 18, 18), dtype=np.int64)
        for region in ("full", "camera", "noncamera")
    }
    conditional = {region: np.zeros((17, 17), dtype=np.int64) for region in visibility}
    scenes, sample_counts = {}, {}
    attributes, gradients, probes = [], [], {}
    actual_indices, classification, semantics = [], [], []
    (captured, capture_active) = ([], False)
    original_render = model.gaussian_head.render_state
    rng = runner.capture_rng()

    def record_render(gaussian, xyz):
        logits = original_render(gaussian, xyz)
        if capture_active:
            captured.append((gaussian, logits))
        return logits

    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    model.gaussian_head.render_state = record_render
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    started = time.monotonic()
    try:
        for index, data in enumerate(runner.loader(dataset, cfg.val_loader)):
            capture_active = index in indices
            captured.clear()
            (images, metas) = runner.move_batch(data, "cuda")
            result = model(imgs=images, metas=metas)
            add_evaluation_batch(metric, visibility, result, metas)
            name = scene_name(dataset, index)
            scenes.setdefault(name, np.zeros((7, 18, 18), dtype=np.int64))
            sample_counts[name] = sample_counts.get(name, 0) + 1
            grids = [result["final_occ_grid"]] + result["future_occ_grids"]
            for h, grid in enumerate(grids):
                target = (
                    metas["occ_label"]
                    if h == 0
                    else metas["future_occ_labels"][:, h - 1]
                )
                cam = (
                    metas["occ_cam_mask"]
                    if h == 0
                    else metas["future_occ_cam_masks"][:, h - 1]
                )
                scenes[name][h] += partition_histograms(grid, target, cam)["full"]
            current_logits = result["gaussian_logits_by_horizon"][0]
            for region, hist in conditional_confusion(
                current_logits, metas["occ_label"], metas["occ_cam_mask"]
            ).items():
                conditional[region] += hist
            if capture_active:
                if len(captured) != 7:
                    raise ValueError(
                        "all seven horizons must be exposed to diagnostics"
                    )
                actual_indices.append(index)
                classification.append(
                    classification_statistics(
                        current_logits,
                        metas["occ_label"],
                        metas["occ_cam_mask"],
                        cfg,
                        index,
                    )
                )
                semantics.append(
                    conditional_statistics(
                        current_logits,
                        metas["occ_label"],
                        metas["occ_cam_mask"],
                        cfg,
                        index,
                    )
                )
                current = model.pts_bbox_head.ind_stamps_all == 0
                points_list = [result["all_refine_pts"][-1][:, current]] + result[
                    "forecast_points_list"
                ]
                classes_list = [result["all_cls_scores"][-1][:, current]] + result[
                    "forecast_semantics_list"
                ]
                features_list = [result["query_feat"][:, current]] + result[
                    "forecast_features_list"
                ]
                xyz = metas["occ_xyz"].float().flatten(1, 3)
                trained_horizon = max(
                    1, min(model.curr_epoch - model.finetune_epoch + 1, 6)
                )
                for h, (gaussian, logits) in enumerate(tuple(captured)):
                    (points, classes, features) = (
                        points_list[h],
                        classes_list[h],
                        features_list[h],
                    )
                    attributes.append(
                        attribute_record(
                            index,
                            h,
                            points,
                            classes,
                            features,
                            gaussian,
                            model.gaussian_head,
                        )
                    )
                    target = (
                        metas["occ_label"]
                        if h == 0
                        else metas["future_occ_labels"][:, h - 1]
                    )
                    cam = (
                        metas["occ_cam_mask"]
                        if h == 0
                        else metas["future_occ_cam_masks"][:, h - 1]
                    )
                    modes = render_variants(
                        model,
                        original_render,
                        gaussian,
                        xyz,
                        logits,
                        points,
                        classes,
                        target,
                    )
                    for mode, prediction in modes.items():
                        for region, hist in partition_histograms(
                            prediction, target, cam
                        ).items():
                            key = f"h{h}/{mode}/{region}"
                            probes.setdefault(key, np.zeros((18, 18), dtype=np.int64))
                            probes[key] += hist
                    supervised = h == 0 or (
                        not model.pretrain
                        and h <= trained_horizon
                        and (h in model.gaussian_loss_horizons)
                    )
                    if index in gradient_indices and supervised:
                        model.gaussian_head.render_state = original_render
                        gradients.append(
                            dict(
                                index=index,
                                **gradient_probe(
                                    model, cfg, result, gaussian, xyz, metas, h, loss_fn
                                ),
                            )
                        )
                        model.gaussian_head.render_state = record_render
            if (index + 1) % 128 == 0:
                print(
                    f"EVAL+DIAGNOSTICS model={MODEL_NAME} epoch={epoch} samples={metric.samples}/{len(dataset)} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            if limit is not None and index + 1 >= limit:
                break
    finally:
        model.gaussian_head.render_state = original_render
        captured.clear()
        runner.restore_rng(rng)
        model.memory.clear()
        model.memory_queue.clear()
    if limit is None and metric.samples != 1024:
        raise ValueError("formal evaluation requires all 1024 samples")
    if not np.array_equal(sum(scenes.values()), metric.confusion):
        raise ValueError("scene counts differ from formal confusion")
    if not np.array_equal(
        visibility["full"], visibility["camera"] + visibility["noncamera"]
    ):
        raise ValueError("visibility partition failed")
    if not np.array_equal(conditional["full"].sum(1), metric.confusion[0].sum(1)[:17]):
        raise ValueError("conditional diagnosis omitted GT occupied voxels")
    formal = dict(
        metric.results(), epoch=epoch, evaluation_seconds=time.monotonic() - started
    )
    runner.atomic_json(directory / f"epoch_{epoch:02d}_metrics.json", formal)
    np.savez_compressed(
        directory / f"epoch_{epoch:02d}_confusion.npz", confusion=metric.confusion
    )
    save_scene_counts(
        directory / f"epoch_{epoch:02d}_scene_confusion.npz", scenes, sample_counts
    )
    output = (
        directory / "epoch_diagnostics_smoke"
        if limit is not None
        else Path(cfg.report_directory) / "epoch_diagnostics"
    )
    payload = dict(
        epoch=epoch,
        num_samples=metric.samples,
        is_smoke=limit is not None,
        voxel_supervision_mask="none",
        config_sha256=runner.digest(cfg.filename),
        class_names=CLASS_NAMES,
        attribute_sample_indices=actual_indices,
        supervised_gaussian_horizons=[0]
        + (
            []
            if model.pretrain
            else [h for h in model.gaussian_loss_horizons if 0 < h <= trained_horizon]
        ),
        visibility={
            f"h{h}/{region}": region_metrics(visibility[region][h])
            for h in range(7)
            for region in visibility
        },
        attributes=attributes,
        gradient_probes=gradients,
        render_probes={key: region_metrics(hist) for (key, hist) in probes.items()},
        selection_mean_miou=formal["selection_mean_miou"],
        model_name=MODEL_NAME,
        decision_rule="18-class argmax",
        _visibility_arrays=visibility,
    )
    save_diagnostic(
        output, epoch, payload, formal, cfg.get("strict_reference_directory")
    )
    runner.atomic_json(
        output / f"epoch_{epoch:02d}_classification_components.json",
        dict(
            epoch=epoch,
            num_samples=len(classification),
            formal_sample_count=metric.samples,
            is_smoke=limit is not None,
            records=classification,
            configuration=cfg.loss.occupancy_loss.to_dict(),
            terms_included=["weighted_18class_joint_ce"],
        ),
    )
    runner.atomic_json(
        output / f"epoch_{epoch:02d}_conditional_semantics.json",
        dict(
            epoch=epoch,
            num_samples=len(semantics),
            formal_sample_count=metric.samples,
            is_smoke=limit is not None,
            records=semantics,
            formal_current_conditional_class_accuracy_percent=100
            * float(np.trace(conditional["full"]))
            / max(int(conditional["full"].sum()), 1),
            formal_current_conditional_macro_accuracy_percent=100
            * float(
                np.mean(
                    np.divide(
                        conditional["full"].diagonal(),
                        conditional["full"].sum(1),
                        out=np.zeros(17),
                        where=conditional["full"].sum(1) > 0,
                    )
                )
            ),
        ),
    )
    np.savez_compressed(
        output / f"epoch_{epoch:02d}_conditional_semantic_confusion.npz", **conditional
    )
    print(
        f"DIAGNOSTIC_COMPLETE model={MODEL_NAME} epoch={epoch} samples={metric.samples}",
        flush=True,
    )
    return formal


@torch.no_grad()
def evaluate_basic(model, dataset, cfg, directory, epoch, limit=None):
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    metric = HorizonMetrics()
    started = time.monotonic()
    for index, data in enumerate(runner.loader(dataset, cfg.val_loader)):
        images, metas = runner.move_batch(data, "cuda")
        result = model(imgs=images, metas=metas)
        metric.add(result, metas)
        if (index + 1) % 128 == 0:
            print(
                f"EVAL epoch={epoch} samples={metric.samples}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
        if limit is not None and index + 1 >= limit:
            break
    metrics = metric.results()
    metrics.update(epoch=epoch, evaluation_seconds=time.monotonic() - started)
    if limit is None and metrics["num_samples"] != 1024:
        raise ValueError("validation must contain exactly 1024 samples")
    runner.atomic_json(directory / f"epoch_{epoch:02d}_metrics.json", metrics)
    np.savez_compressed(
        directory / f"epoch_{epoch:02d}_confusion.npz", confusion=metric.confusion
    )
    model.memory.clear()
    model.memory_queue.clear()
    return metrics
