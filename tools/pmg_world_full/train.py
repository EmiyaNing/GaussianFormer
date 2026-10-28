"""Full-split training with bounded runs, complete state resume and seven-time evaluation."""

import argparse
import fcntl
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.diagnostic_statistics import partition_histograms
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics
from tools.sparse_opus_gausisan_world.protocol import training_contract_sha256
from tools.sparse_opus_gausisan_world.report import refresh


class PauseRequested(Exception):
    """Exit evaluation without treating a requested pause as an experiment failure."""


def accumulation_divisor(batch_index, batches, accumulation):
    """Normalize the final incomplete accumulation group by its actual size."""
    return min(accumulation, batches - (batch_index // accumulation) * accumulation)


@torch.no_grad()
def evaluate(model, dataset, cfg, directory, epoch, should_stop, limit=None):
    metric = HorizonMetrics()
    visibility = {name: np.zeros((7, 18, 18), dtype=np.int64)
                  for name in ("camera", "noncamera")}
    scene_counts, scene_samples, attributes = {}, {}, []
    selected = set(np.linspace(0, len(dataset) - 1, 16, dtype=int))
    rng, started = runtime.capture_rng(), time.monotonic()
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    original_render = model.gaussian_head.render_state
    capture = False
    captured = []

    def record_render(gaussian, xyz):
        logits = original_render(gaussian, xyz)
        if capture:
            captured.append(dict(
                scale_upper_saturation_percent=float((gaussian.scales > .395).float().mean()) * 100,
                opacity_upper_saturation_percent=float((gaussian.opacities > .99).float().mean()) * 100,
                mean_scale_m=float(gaussian.scales.mean()),
                mean_opacity=float(gaussian.opacities.mean()),
                num_gaussians=int(gaussian.means.numel() // 3)))
        return logits

    model.gaussian_head.render_state = record_render
    try:
        for index, data in enumerate(runtime.loader(dataset, cfg.val_loader)):
            if should_stop():
                raise PauseRequested()
            capture = index in selected
            captured.clear()
            images, metas = runtime.move_batch(data, "cuda")
            result = model(imgs=images, metas=metas)
            previous = metric.confusion.copy()
            metric.add(result, metas)
            scene = str(dataset.keyframes[index][0])
            scene_counts.setdefault(scene, np.zeros_like(metric.confusion))
            scene_counts[scene] += metric.confusion - previous
            scene_samples[scene] = scene_samples.get(scene, 0) + images.shape[0]
            grids = [result["final_occ_grid"], *result["future_occ_grids"]]
            for horizon, prediction in enumerate(grids):
                target = metas["occ_label"] if horizon == 0 else metas["future_occ_labels"][:, horizon - 1]
                camera = metas["occ_cam_mask"] if horizon == 0 else metas["future_occ_cam_masks"][:, horizon - 1]
                counts = partition_histograms(prediction, target, camera)
                for region in visibility:
                    visibility[region][horizon] += counts[region]
            if capture:
                if len(captured) != 7:
                    raise ValueError("attribute capture must cover all seven horizons")
                attributes.extend(dict(index=index, horizon=h, **record)
                                  for h, record in enumerate(captured))
            del result, images, metas, grids, data
            if (index + 1) % 100 == 0:
                print("EVAL", epoch, index + 1, len(dataset), flush=True)
            if limit is not None and index + 1 >= limit:
                break
    finally:
        model.gaussian_head.render_state = original_render
        model.memory.clear()
        model.memory_queue.clear()
        runtime.restore_rng(rng)
    expected_samples = min(limit, len(dataset)) if limit is not None else len(dataset)
    if metric.samples != expected_samples:
        raise ValueError("validation omitted full-split samples")
    if not np.array_equal(sum(scene_counts.values()), metric.confusion):
        raise ValueError("scene confusion does not match global confusion")
    if not np.array_equal(visibility["camera"] + visibility["noncamera"], metric.confusion):
        raise ValueError("visibility counts do not partition full-space counts")
    result = dict(metric.results(), epoch=epoch, is_smoke=limit is not None,
                  evaluation_seconds=time.monotonic() - started,
                  supervised_gaussian_horizons=list(range(7)),
                  dataset_scope="all_eligible_official_validation_samples")
    runtime.atomic_json(directory / f"epoch_{epoch:02d}_metrics.json", result)
    runtime.atomic_json(directory / f"epoch_{epoch:02d}_attributes.json", attributes)
    names = sorted(scene_counts)
    np.savez_compressed(directory / f"epoch_{epoch:02d}_counts.npz",
                        confusion=metric.confusion, **visibility,
                        scenes=np.asarray(names),
                        scene_confusion=np.stack([scene_counts[s] for s in names]),
                        sample_counts=np.asarray([scene_samples[s] for s in names]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--stop-after-epoch", type=int, default=3)
    parser.add_argument("--max-hours", type=float, default=5.0,
                        help="Pause at an optimizer boundary; zero disables the wall-time limit")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    if args.max_hours < 0 or not 1 <= args.stop_after_epoch <= 15:
        parser.error("invalid duration or epoch limit")
    if "_base_" in Path(args.config).read_text():
        raise ValueError("standalone configuration required")
    cfg = Config.fromfile(args.config)
    if (cfg.max_epochs != 15 or cfg.schedule_epochs != 15
            or cfg.model.finetune_epoch != 0
            or list(cfg.model.gaussian_loss_horizons) != list(range(7))):
        raise ValueError("full-horizon baseline training contract changed")
    runtime.configure_runtime(cfg)
    directory = Path(args.work_dir)
    report = directory if args.smoke else Path(cfg.report_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    latest = directory / "latest.pth"
    if args.smoke and latest.exists():
        raise ValueError("smoke requires an isolated work directory")
    requested = False

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    def should_stop():
        return requested or (args.max_hours > 0 and time.monotonic() - started >= args.max_hours * 3600)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    train_data = OPENOCC_DATASET.build(cfg.train_dataset_config)
    val_data = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if len(train_data) <= 4096 or len(val_data) <= 1024:
        raise ValueError("full-split baseline must not use the exploratory subsets")
    batches = math.ceil(len(train_data) / cfg.train_loader.batch_size)
    updates_per_epoch = math.ceil(batches / cfg.gradient_accumulation)
    total_updates = updates_per_epoch * cfg.schedule_epochs
    from tools.pmg_world_stage2.train import implementation_identity
    sources = implementation_identity()
    extra = [Path(__file__), Path("model/segmentor/sparse_opus_full_horizon_baseline.py")]
    sources.update({str(path): runtime.digest(path) for path in extra})
    identity = dict(training_contract=training_contract_sha256(cfg),
                    manifests=runtime.audit_manifests(cfg), source_sha256=sources)
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = runtime.load_image_initialization(model, cfg.train_load_from)
    initialization["initial_model_sha256"] = runtime.model_fingerprint(model)
    model.cuda()
    optimizer = runtime.make_optimizer(model, cfg)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    epoch, next_sample, updates, best, phase = 0, 0, 0, -1.0, "train"
    resume_rng, resume_stamps = None, None
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("resume implementation, configuration or dataset identity changed")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        epoch, next_sample, updates, best, phase = (
            saved[k] for k in ("epoch", "next_sample", "updates", "best", "phase"))
        resume_rng, resume_stamps = saved["rng"], saved["query_stamps"]
        initialization = saved["initialization"]
        del saved
    cfg.dump(str(directory / "resolved_config.py"))
    runtime.atomic_json(directory / "initialization.json", initialization)
    runtime.atomic_json(directory / "protocol.json", dict(
        identity=identity, train_samples=len(train_data), validation_samples=len(val_data),
        planned_epochs=15, schedule_epochs=15, total_updates=total_updates,
        updates_per_epoch=updates_per_epoch, microbatch_size=cfg.train_loader.batch_size,
        gradient_accumulation=cfg.gradient_accumulation,
        supervised_horizons_seconds=[h * .5 for h in range(7)],
        future_voxel_weight_per_horizon=.25, future_voxel_total_weight=1.5,
        prediction_inputs="five historical image frames and native ego state",
        initialization_scope="image_backbone_and_neck_only", distillation=False))

    def save(progress_epoch, sample, progress_phase):
        runtime.atomic_save(latest, dict(
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=progress_epoch, next_sample=sample, updates=updates, best=best,
            phase=progress_phase, rng=runtime.capture_rng(),
            query_stamps=model.pts_bbox_head.ind_stamps_all,
            initialization=initialization, identity=identity))

    def progress(status, **values):
        row = dict(status=status, epoch=epoch + 1, updates=updates,
                   train_samples=len(train_data), validation_samples=len(val_data),
                   planned_epochs=15, **values)
        runtime.atomic_json(directory / "progress.json", row)
        print("PROGRESS " + json.dumps(row), flush=True)

    print("FULL_SPLIT", len(train_data), len(val_data), "UPDATES", total_updates, flush=True)
    while epoch < args.stop_after_epoch:
        model.train()
        within_epoch = runtime.should_restore_query_statistics(resume_rng, next_sample, phase)
        statistics = model.pts_bbox_head.num_stamps_all.clone() if within_epoch else None
        model.set_epoch(epoch)
        if statistics is not None:
            model.pts_bbox_head.num_stamps_all.copy_(statistics)
            model.pts_bbox_head.ind_stamps_all = resume_stamps.cuda()
            model.pts_bbox_head.reset_mask()
        train_data.set_epoch(epoch)
        if resume_rng is not None:
            runtime.restore_rng(resume_rng)
            resume_rng = None
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train_data)).tolist()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        train_started, losses = time.monotonic(), []
        if phase != "evaluate":
            for local_index, data in enumerate(runtime.loader(
                    train_data, cfg.train_loader, order[next_sample:], cfg.seed + epoch)):
                sample = next_sample + local_index * cfg.train_loader.batch_size
                processed = min(sample + cfg.train_loader.batch_size, len(train_data))
                batch_index = sample // cfg.train_loader.batch_size
                images, metas = runtime.move_batch(data, "cuda")
                result = model(imgs=images, metas=metas)
                if len(result["forecast_points_list"]) != 6 or set(result["gaussian_logits_by_horizon"]) != set(range(7)):
                    raise ValueError("all forecast and voxel horizons must be active from epoch one")
                loss, details = objective(dict(result, metas=metas,
                    global_iter=epoch * batches + batch_index))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                divisor = accumulation_divisor(batch_index, batches, cfg.gradient_accumulation)
                (loss / divisor).backward()
                losses.append(float(loss.detach()))
                boundary = (batch_index + 1) % cfg.gradient_accumulation == 0 or processed == len(train_data)
                norm = None
                if boundary or args.smoke:
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm,
                                                               error_if_nonfinite=True))
                    runtime.set_lr(optimizer, updates, total_updates, cfg.warmup_iters, cfg.min_lr_ratio)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                breakdown = {k: float(v) for k, v in details.items()} if (batch_index + 1) % cfg.print_freq == 0 or args.smoke else {}
                del result, loss, details, images, metas, data
                if (batch_index + 1) % cfg.print_freq == 0 or args.smoke:
                    seconds_per_batch = (time.monotonic() - train_started) / (local_index + 1)
                    progress("smoke_training" if args.smoke else "training", samples=processed,
                        mean_loss=float(np.mean(losses[-cfg.print_freq:])), gradient_norm=norm,
                        seconds_per_microbatch=seconds_per_batch,
                        estimated_remaining_train_hours=(batches - batch_index - 1) * seconds_per_batch / 3600,
                        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                        current_lr=optimizer.param_groups[0]["lr"], losses=breakdown)
                if args.smoke:
                    metrics = evaluate(model, val_data, cfg, report, 1, lambda: False, limit=1)
                    runtime.atomic_json(directory / "smoke.json", dict(
                        passed=True, batch_size=cfg.train_loader.batch_size,
                        config_sha256=runtime.digest(args.config),
                        active_horizons=list(range(7)), gradient_norm=norm,
                        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                        initial_model_sha256=initialization["initial_model_sha256"],
                        finite_loss=bool(np.isfinite(losses[-1])), metrics=metrics))
                    return
                if boundary and should_stop():
                    save(epoch, processed, "train")
                    progress("paused_at_optimizer_boundary", samples=processed, phase="train")
                    return
                if boundary and updates % cfg.checkpoint_interval_steps == 0:
                    save(epoch, processed, "train")
            save(epoch, len(train_data), "evaluate")
        try:
            metrics = evaluate(model, val_data, cfg, report, epoch + 1, should_stop)
        except PauseRequested:
            progress("paused_before_validation_completed", samples=len(train_data), phase="evaluate")
            return
        score = metrics["selection_mean_miou"]
        if score > best:
            best = score
            runtime.atomic_save(directory / "best.pth", dict(
                state_dict=model.state_dict(), epoch_index=epoch, metrics=metrics,
                query_stamps=model.pts_bbox_head.ind_stamps_all, identity=identity))
        save(epoch + 1, 0, "train")
        refresh(report, report)
        progress("epoch_completed", samples=len(train_data), completed_epochs=epoch + 1,
                 selection_mean_miou=score, best=best)
        epoch, next_sample, phase = epoch + 1, 0, "train"
        if should_stop():
            break
    if epoch == cfg.max_epochs and updates != total_updates:
        raise ValueError("completed epochs without the scheduled optimizer updates")
    progress("completed" if epoch == cfg.max_epochs else "paused_after_epoch_limit",
             completed_epochs=epoch, best=best, next_sample=next_sample)


if __name__ == "__main__":
    main()
