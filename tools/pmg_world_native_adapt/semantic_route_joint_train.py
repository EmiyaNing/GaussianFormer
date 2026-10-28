"""Resumable end-to-end semantic routing: 15 full-data epochs, 5+10 LR policy."""

import argparse
import fcntl
import gc
import json
import math
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from mmengine.config import Config

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.pmg_world.regions import (
    HORIZONS,
    REGIONS,
    histogram,
    load_masks,
    region_result,
    token_at,
)
from tools.pmg_world_full.train import accumulation_divisor, PauseRequested
from tools.pmg_world_native_adapt.semantic_route_joint_report import (
    diagnostic_report,
    refresh,
)
from tools.pmg_world_native_adapt.joint_checks import (
    initialize,
    state_audit,
    engineering_check,
)
from tools.pmg_world_native_adapt.joint_optimization import (
    make_optimizer,
    set_learning_rates,
    PERCEPTION_MODULES,
    PREDICTION_MODULES,
)
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics


STATIC = (0, 1, 11, 12, 13, 14, 15, 16)


def add_static(metric):
    for horizon in metric["horizons"]:
        horizon["static_class_miou_fullspace"] = float(
            np.asarray(horizon["class_iou"])[list(STATIC)].mean()
        )
    return metric


def write_counts(path, metric, scenes, samples, regions=None, scene_regions=None):
    names = sorted(scenes)
    data = dict(
        confusion=metric.confusion,
        scenes=np.asarray(names),
        scene_confusion=np.stack([scenes[name] for name in names]),
        sample_counts=np.asarray([samples[name] for name in names]),
    )
    if regions is not None:
        data.update(
            region_confusion=regions,
            scene_region_confusion=np.stack([scene_regions[name] for name in names]),
        )
    np.savez_compressed(path, **data)


@torch.no_grad()
def evaluate(
    model, dataset, cfg, report, epoch, updates, should_stop, progress, limit=None
):
    full, subset = HorizonMetrics(), HorizonMetrics()
    full_scenes, full_samples, sub_scenes, sub_samples = {}, {}, {}, {}
    region_counts = np.zeros((3, len(REGIONS), 18, 18), dtype=np.int64)
    sub_regions = {}
    manifest = json.loads(Path(cfg.subset_diagnostics.selection_manifest).read_text())
    subset_tokens = {item["token"] for item in manifest["samples"]}
    region_root = Path(cfg.subset_diagnostics.reference_directory) / "regions"
    seen = set()
    rng = runtime.capture_rng()
    started = time.monotonic()
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    model.reset_route_statistics(enabled=True)
    model.reset_gate_statistics(enabled=True)
    try:
        for index, data in enumerate(runtime.loader(dataset, cfg.val_loader)):
            if should_stop():
                raise PauseRequested()
            images, metas = runtime.move_batch(data, "cuda")
            result = model(imgs=images, metas=metas)
            before = full.confusion.copy()
            full.add(result, metas)
            delta = full.confusion - before
            scene = str(dataset.keyframes[index][0])
            full_scenes.setdefault(scene, np.zeros_like(delta))
            full_scenes[scene] += delta
            full_samples[scene] = full_samples.get(scene, 0) + 1
            token = token_at(dataset, index, 0)
            if token in subset_tokens:
                if token in seen:
                    raise ValueError("Duplicate diagnostic token")
                seen.add(token)
                before = subset.confusion.copy()
                subset.add(result, metas)
                sub_scenes.setdefault(scene, np.zeros_like(delta))
                sub_scenes[scene] += subset.confusion - before
                sub_samples[scene] = sub_samples.get(scene, 0) + 1
                sub_regions.setdefault(scene, np.zeros_like(region_counts))
                for slot, horizon in enumerate(HORIZONS):
                    pred = (
                        result["future_occ_grids"][horizon - 1]
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    target = (
                        metas["future_occ_labels"][:, horizon - 1]
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    masks = load_masks(region_root / (token + ".npz"), horizon)
                    for r, name in enumerate(REGIONS):
                        count = histogram(pred, target, masks[name])
                        region_counts[slot, r] += count
                        sub_regions[scene][slot, r] += count
            del result, images, metas, data
            if (index + 1) % 100 == 0:
                progress(
                    "evaluating",
                    phase="evaluate",
                    validation_processed=index + 1,
                    diagnostic_processed=len(seen),
                    evaluation_seconds=time.monotonic() - started,
                )
            if limit is not None and index + 1 >= limit:
                break
    finally:
        model.collect_route_statistics = False
        model.collect_gate_statistics = False
        model.memory.clear()
        model.memory_queue.clear()
        runtime.restore_rng(rng)
    assert full.samples == (
        min(limit, len(dataset)) if limit is not None else len(dataset)
    )
    assert np.array_equal(sum(full_scenes.values()), full.confusion)
    metric = add_static(
        dict(
            full.results(),
            epoch=epoch,
            is_smoke=limit is not None,
            evaluation_seconds=time.monotonic() - started,
            planned_epochs=15,
            dataset_scope="all_eligible_official_validation_samples",
            training_updates=updates,
        )
    )
    runtime.atomic_json(report / f"epoch_{epoch:02d}_metrics.json", metric)
    write_counts(
        report / f"epoch_{epoch:02d}_counts.npz", full, full_scenes, full_samples
    )
    runtime.atomic_json(
        report / f"epoch_{epoch:02d}_route_statistics.json", model.route_statistics()
    )
    runtime.atomic_json(
        report / f"epoch_{epoch:02d}_gate_statistics.json", model.gate_statistics()
    )
    if limit is None:
        with np.load(cfg.full_baseline_counts) as initial:
            assert np.array_equal(initial["scenes"], np.asarray(sorted(full_scenes)))
            assert np.array_equal(
                initial["sample_counts"],
                np.asarray([full_samples[s] for s in sorted(full_scenes)]),
            )
            assert np.array_equal(
                initial["scene_confusion"].sum(-1),
                np.stack([full_scenes[s] for s in sorted(full_scenes)]).sum(-1),
            )
        assert (
            seen == subset_tokens and subset.samples == 1024 and len(sub_scenes) == 150
        )
        with np.load(cfg.subset_diagnostics.baseline_counts) as old:
            assert np.array_equal(old["scenes"], np.asarray(sorted(sub_scenes)))
            assert np.array_equal(
                old["sample_counts"],
                np.asarray([sub_samples[s] for s in sorted(sub_scenes)]),
            )
            assert np.array_equal(
                old["scene_confusion"].sum(-1),
                np.stack([sub_scenes[s] for s in sorted(sub_scenes)]).sum(-1),
            )
            assert np.array_equal(
                old["scene_region_confusion"].sum(-1),
                np.stack([sub_regions[s] for s in sorted(sub_regions)]).sum(-1),
            )
        folder = report / "diagnostics" / f"epoch_{epoch:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        sub = add_static(
            dict(
                subset.results(),
                epoch=epoch,
                partial=False,
                planned_epochs=15,
                future_123_mean_miou=float(
                    np.mean([subset.results()["horizons"][h]["miou"] for h in HORIZONS])
                ),
                evaluation_seconds=time.monotonic() - started,
                training_updates=updates,
                regions=[
                    dict(
                        seconds=h * 0.5,
                        metrics={
                            name: region_result(region_counts[slot, r])
                            for r, name in enumerate(REGIONS)
                        },
                    )
                    for slot, h in enumerate(HORIZONS)
                ],
                region_reference=cfg.subset_diagnostics.reference_directory,
                scope="Fixed 1024 sample monitor within full validation, not a full-split regional result",
            )
        )
        runtime.atomic_json(folder / f"epoch_{epoch:02d}_metrics.json", sub)
        write_counts(
            folder / f"epoch_{epoch:02d}_counts.npz",
            subset,
            sub_scenes,
            sub_samples,
            region_counts,
            sub_regions,
        )
        diagnostic_report(folder, epoch, updates)
    return metric


def run(args):
    cfg = Config.fromfile(args.config)
    assert "_base_" not in Path(args.config).read_text()
    assert cfg.max_epochs == cfg.schedule_epochs == 15 and cfg.model.finetune_epoch == 0
    assert list(cfg.model.gaussian_loss_horizons) == list(range(7))
    assert cfg.perception_lr.equal_epochs == 5 and cfg.perception_lr.late_ratio == 0.1
    assert (
        not cfg.semantic_route_training.frozen_perception
        and not cfg.semantic_route_training.frozen_reference
    )
    runtime.configure_runtime(cfg)
    output = Path(args.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = output / "smoke_report" if args.smoke else Path(cfg.report_directory)
    report.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.smoke and (output / "latest.pth").exists():
        raise FileExistsError("Smoke directory contains training state")
    train = OPENOCC_DATASET.build(cfg.train_dataset_config)
    val = OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert (len(train), len(val)) == (19730, 4219)
    manifest = json.loads(Path(cfg.subset_diagnostics.selection_manifest).read_text())
    full_tokens = {token_at(val, i, 0) for i in range(len(val))}
    assert {r["token"] for r in manifest["samples"]} <= full_tokens
    assert all(
        (
            Path(cfg.subset_diagnostics.reference_directory)
            / "regions"
            / (r["token"] + ".npz")
        ).exists()
        for r in manifest["samples"]
    )
    from tools.pmg_world_stage2.train import implementation_identity

    files = [
        args.config,
        __file__,
        "model/segmentor/pmg_semantic_route_joint_world.py",
        "model/segmentor/pmg_query_update_gate_world.py",
        "model/segmentor/pmg_semantic_gaussian_route_world.py",
        "tools/pmg_world_native_adapt/semantic_route_joint_report.py",
        "tools/pmg_world_native_adapt/semantic_route_joint_launch.py",
        "tools/pmg_world_native_adapt/joint_checks.py",
        "tools/pmg_world_native_adapt/joint_optimization.py",
        "tools/pmg_world_native_adapt/gate_optimization_report.py",
        "tools/pmg_world_native_adapt/gate_report.py",
        "tools/pmg_world_full/train.py",
        "loss/pmg_repair_distance_weight.py",
    ]
    sources = implementation_identity()
    sources.update({str(p): runtime.digest(p) for p in files})
    identity = dict(
        config_sha256=runtime.digest(args.config),
        source_sha256=sources,
        source_checkpoint_sha256=runtime.digest(cfg.adaptation_checkpoint),
        manifests=runtime.audit_manifests(cfg),
        diagnostic_manifest_sha256=runtime.digest(
            cfg.subset_diagnostics.selection_manifest
        ),
        full_baseline_counts_sha256=runtime.digest(cfg.full_baseline_counts),
    )
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run identity changed")
    runtime.atomic_json(identity_path, identity)
    cfg.dump(str(output / "resolved_config.py"))
    model, source = initialize(cfg)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, cfg)
    accumulation = cfg.gradient_accumulation
    updates_per_epoch = math.ceil(len(train) / accumulation)
    planned = updates_per_epoch * 15
    assert cfg.train_loader.batch_size == 1 and planned == 73995
    protocol = dict(
        arm=cfg.pmg_experiment.arm,
        train_samples=len(train),
        validation_samples=len(val),
        planned_epochs=15,
        updates_per_epoch=updates_per_epoch,
        planned_updates=planned,
        initialization_checkpoint=cfg.adaptation_checkpoint,
        initialization_scope="T-A0 subset epoch10; restart 15 full epochs, not S2 resume",
        trainable_modules="all",
        trainable_parameters=sum(p.numel() for p in parameters),
        frozen_parameter_names=[
            name for name, p in model.named_parameters() if not p.requires_grad
        ],
        perception_modules=list(PERCEPTION_MODULES),
        prediction_modules=list(PREDICTION_MODULES),
        frozen_perception=False,
        frozen_ungated_forecast=False,
        frozen_attribute_decoder=False,
        backbone_frozen_stages=cfg.model.img_backbone.frozen_stages,
        backbone_BN_running_stats_fixed=cfg.model.img_backbone.norm_eval,
        backbone_BN_affine_trainable=True,
        training_dropout_enabled=True,
        query_assignment_fixed=True,
        gradient_accumulation=accumulation,
        microbatch_size=1,
        learning_rate=cfg.base_lr,
        warmup_updates=cfg.warmup_iters,
        min_lr_ratio=cfg.min_lr_ratio,
        perception_lr=dict(equal_epochs=5, late_ratio=0.1),
        additional_backbone_or_sampling_lr_multiplier=False,
        trajectory_head_shared=True,
        route_class_ids=list(model.route_class_ids),
        route_source="Trainable ungated branch predicted semantics; hard labels detached, no GT routing",
        native_set_and_trajectory_loss="Candidate all children; original definition retained",
        voxel_loss="Routed complete Gaussians; both forecast paths differentiable; seven horizons",
        future_focal_distance_weight_detached=True,
        distillation=False,
        first_five_epochs_are_formal=True,
        full_validation_every_epoch=True,
        fixed_subset_diagnostics=1024,
        wall_time_limit_hours=0,
        resume_note="Model/optimizer/RNG/epoch/sample/allocation/phase restored; prefetched worker augmentation is not bitwise replayed",
    )
    runtime.atomic_json(output / "protocol.json", protocol)
    print("S3_JOINT_PROTOCOL " + json.dumps(protocol), flush=True)
    latest = output / "latest.pth"
    epoch, next_sample, updates, best, phase = 0, 0, 0, -1.0, "train"
    resume_rng = None
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
        assert saved["identity"] == identity
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        epoch, next_sample, updates, best, phase = (
            saved[k] for k in ("epoch", "next_sample", "updates", "best", "phase")
        )
        model.pts_bbox_head.ind_stamps_all = saved["query_stamps"].cuda()
        model.pts_bbox_head.reset_mask()
        resume_rng = saved["rng"]
        del saved
    else:
        engineering_check(model, objective, train, val, cfg, output)
        state_audit(model, source)
    requested = False

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def progress(status, **values):
        value = dict(
            status=status,
            epoch=min(epoch + 1, 15),
            completed_epochs=(epoch + 1 if status == "epoch_completed" else epoch),
            updates=updates,
            planned_epochs=15,
            planned_updates=planned,
            train_samples=len(train),
            validation_samples=len(val),
            **values,
        )
        runtime.atomic_json(output / "progress.json", value)
        print("PROGRESS " + json.dumps(value), flush=True)

    def save(save_epoch, sample, save_phase):
        runtime.atomic_save(
            latest,
            dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=save_epoch,
                next_sample=sample,
                updates=updates,
                best=best,
                phase=save_phase,
                query_stamps=model.pts_bbox_head.ind_stamps_all,
                rng=runtime.capture_rng(),
                identity=identity,
            ),
        )

    if not args.smoke and not latest.exists():
        save(0, 0, "train")
    while epoch < 15:
        model.train()
        model.set_epoch(epoch)
        train.set_epoch(epoch)
        rates = set_learning_rates(optimizer, updates, planned, epoch, cfg)
        runtime.atomic_json(
            output / f"epoch_{epoch+1:02d}_lr.json",
            dict(epoch=epoch + 1, updates=updates, **rates),
        )
        if resume_rng is not None:
            runtime.restore_rng(resume_rng)
            resume_rng = None
        model.memory.clear()
        model.memory_queue.clear()
        optimizer.zero_grad(set_to_none=True)
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train)).tolist()
        start = time.monotonic()
        losses = []
        torch.cuda.reset_peak_memory_stats()
        if phase == "train":
            for local_index, data in enumerate(
                runtime.loader(
                    train, cfg.train_loader, order[next_sample:], cfg.seed + epoch
                )
            ):
                sample = next_sample + local_index
                processed = sample + 1
                images, metas = runtime.move_batch(data, "cuda")
                result = model(imgs=images, metas=metas)
                if set(result["gaussian_logits_by_horizon"]) != set(range(7)):
                    raise ValueError("Missing voxel horizon")
                loss, details = objective(
                    dict(
                        result,
                        metas=metas,
                        update_stamp_statistics=False,
                        global_iter=epoch * len(train) + sample,
                    )
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (
                    loss / accumulation_divisor(sample, len(train), accumulation)
                ).backward()
                losses.append(float(loss.detach()))
                boundary = processed % accumulation == 0 or processed == len(train)
                norm = None
                if boundary:
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            parameters, cfg.grad_max_norm, error_if_nonfinite=True
                        )
                    )
                    rates = set_learning_rates(optimizer, updates, planned, epoch, cfg)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                details = (
                    {key: float(value) for key, value in details.items()}
                    if processed % 100 == 0 or args.smoke
                    else {}
                )
                del result, loss, images, metas, data
                if boundary and (
                    updates == 1 or updates % cfg.print_freq == 0 or args.smoke
                ):
                    seconds = (time.monotonic() - start) / (local_index + 1)
                    progress(
                        "smoke_training" if args.smoke else "training",
                        phase="train",
                        samples=processed,
                        mean_loss=float(np.mean(losses[-100:])),
                        gradient_norm=norm,
                        **rates,
                        seconds_per_microbatch=seconds,
                        estimated_remaining_train_hours=(len(train) - processed)
                        * seconds
                        / 3600,
                        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                        losses=details,
                    )
                if args.smoke and boundary:
                    audit = state_audit(model, source, require_updates=True)
                    metric = evaluate(
                        model,
                        val,
                        cfg,
                        report,
                        1,
                        updates,
                        lambda: False,
                        progress,
                        limit=1,
                    )
                    runtime.atomic_json(
                        output / "smoke.json",
                        dict(
                            passed=True,
                            config_sha256=identity["config_sha256"],
                            identity=identity,
                            actual_optimizer_updates=updates,
                            seven_voxel_horizons=list(range(7)),
                            finite_gradient_norm=norm,
                            all_parameter_update_audit=audit,
                            learning_rates=rates,
                            validation_samples=metric["num_samples"],
                        ),
                    )
                    progress("smoke_completed", samples=processed, phase="train")
                    return
                if boundary and (
                    requested
                    or updates == 1
                    or updates % cfg.checkpoint_interval_steps == 0
                ):
                    save(epoch, processed, "train")
                    if requested:
                        progress(
                            "paused_at_optimizer_boundary",
                            samples=processed,
                            phase="train",
                        )
                        return
            save(epoch, len(train), "evaluate")
            phase = "evaluate"
        audit = state_audit(model, source, require_updates=True)
        runtime.atomic_json(report / f"epoch_{epoch+1:02d}_state_audit.json", audit)
        progress("evaluating", phase="evaluate", validation_processed=0)
        try:
            metric = evaluate(
                model, val, cfg, report, epoch + 1, updates, lambda: requested, progress
            )
        except PauseRequested:
            progress("paused_before_validation_completed", phase="evaluate")
            return
        score = metric["selection_mean_miou"]
        if score > best:
            best = score
            runtime.atomic_save(
                output / "best.pth",
                dict(
                    state_dict=model.state_dict(),
                    epoch_index=epoch,
                    updates=updates,
                    query_stamps=model.pts_bbox_head.ind_stamps_all,
                    metrics=metric,
                    identity=identity,
                ),
            )
        save(epoch + 1, 0, "train")
        refresh(report)
        progress(
            "epoch_completed",
            selection_mean_miou=score,
            best=best,
            phase="train",
            samples=len(train),
        )
        epoch, next_sample, phase = epoch + 1, 0, "train"
        if requested:
            progress("paused_after_epoch", phase="train")
            return
    assert updates == planned
    progress("completed", phase="complete", best=best)
    runtime.atomic_json(
        output / "completed.json",
        dict(status="completed", epochs=15, updates=updates, best=best),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as error:
        runtime.atomic_json(
            output / "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            ),
        )
        progress = output / "progress.json"
        previous = json.loads(progress.read_text()) if progress.exists() else {}
        runtime.atomic_json(
            progress,
            dict(
                previous,
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
