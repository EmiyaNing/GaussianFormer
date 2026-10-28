"""Isolated ten-epoch stage2 experiments with the original 15-epoch LR prefix.

Only newly created experiment directories are written. Stateful resume checks
configuration, immutable manifests, implementation hashes and optimizer state.
"""

import argparse
import fcntl
import json
import math
import signal
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.protocol import training_contract_sha256
from tools.pmg_world.evaluate import evaluate
from .report import refresh


def implementation_identity():
    from tools.pmg_world.train import implementation_identity as original_identity

    identity = original_identity()
    paths = list(Path("tools/pmg_world_stage2").glob("*.py"))
    paths += [
        Path("model/segmentor/pmg_motion_residual_world.py"),
        Path("loss/pmg_time_supervision_loss.py"),
    ]
    paths += list(Path("tools/sparse_opus_gausisan_world").glob("*.py"))
    paths += [
        Path("model/encoder/opus_v1_gaussian_temporal_encoder.py"),
        Path("model/encoder/opus_v1_gaussian_softplus_encoder.py"),
        Path("model/head/gaussian_opus_head.py"),
        Path("model/head/gaussian_head.py"),
        Path("model/head/opus_v1_gaussian_head.py"),
        Path("model/head/opus_v1_gaussian_trainable_bias_head.py"),
        Path("loss/occupancy_loss.py"),
    ]
    return dict(identity, **{str(p): runtime.digest(p) for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--stop-after-epoch", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.stop_after_epoch <= 10:
        parser.error("stop-after-epoch must be in 1..10")
    if "_base_" in Path(args.config).read_text():
        raise ValueError("standalone configuration required")
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    if (
        cfg.max_epochs != 10
        or cfg.schedule_epochs != 15
        or cfg.model.finetune_epoch != 0
    ):
        raise ValueError(
            "stage2 uses 10 actual epochs, the original 15-epoch LR prefix, and formal training from epoch 1"
        )
    directory, report = Path(args.work_dir), Path(cfg.report_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = runtime.load_image_initialization(model, cfg.train_load_from)
    initialization["initial_model_sha256"] = runtime.model_fingerprint(model)
    model.cuda()
    optimizer = runtime.make_optimizer(model, cfg)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train_data = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation_data = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if (len(train_data), len(validation_data)) != (4096, 1024):
        raise ValueError("the stage-1 comparison must retain 4096/1024 samples")
    batches = math.ceil(len(train_data) / cfg.train_loader.batch_size)
    total_updates = math.ceil(batches / cfg.gradient_accumulation) * cfg.schedule_epochs
    if total_updates != 15360:
        raise ValueError("optimizer-update budget changed")
    identity = dict(
        training_contract=training_contract_sha256(cfg),
        manifests=runtime.audit_manifests(cfg),
        source_sha256=implementation_identity(),
    )
    latest = directory / "latest.pth"
    epoch, next_sample, updates, best, phase = 0, 0, 0, -1.0, "train"
    resume_rng, resume_stamps = None, None
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError(
                "resume implementation/configuration/data identity changed"
            )
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        epoch, next_sample, updates, best, phase = (
            saved[k] for k in ("epoch", "next_sample", "updates", "best", "phase")
        )
        resume_rng, resume_stamps = saved["rng"], saved["query_stamps"]
        initialization = saved["initialization"]
        del saved
    else:
        # An independent generator makes sampling order unrelated to model construction.
        # Original tensors are matched; the optional residual adds zero-output parameters.
        runtime.atomic_json(directory / "initialization.json", initialization)
    cfg.dump(str(directory / "resolved_config.py"))
    runtime.atomic_json(
        directory / "protocol.json",
        dict(
            identity,
            arm=cfg.pmg_experiment.arm,
            actual_planned_updates=10240,
            schedule_updates=total_updates,
            planned_epochs=10,
            schedule_epochs=15,
            prediction_inputs="five historical frames only",
            distillation=False,
        ),
    )

    def save(progress_epoch, sample, progress_phase):
        runtime.atomic_save(
            latest,
            dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=progress_epoch,
                next_sample=sample,
                updates=updates,
                best=best,
                phase=progress_phase,
                rng=runtime.capture_rng(),
                query_stamps=model.pts_bbox_head.ind_stamps_all,
                initialization=initialization,
                identity=identity,
            ),
        )

    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while epoch < min(args.stop_after_epoch, 10):
        model.train()
        within_epoch = runtime.should_restore_query_statistics(
            resume_rng, next_sample, phase
        )
        statistics = (
            model.pts_bbox_head.num_stamps_all.clone() if within_epoch else None
        )
        model.set_epoch(epoch)
        if statistics is not None:
            model.pts_bbox_head.num_stamps_all.copy_(statistics)
            model.pts_bbox_head.ind_stamps_all = resume_stamps.cuda()
            model.pts_bbox_head.reset_mask()
        train_data.set_epoch(epoch)
        if resume_rng is not None:
            runtime.restore_rng(resume_rng)
            resume_rng = None
        order = (
            np.random.default_rng(cfg.seed + epoch)
            .permutation(len(train_data))
            .tolist()
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses = []
        if phase != "evaluate":
            for local_index, data in enumerate(
                runtime.loader(
                    train_data, cfg.train_loader, order[next_sample:], cfg.seed + epoch
                )
            ):
                sample = next_sample + local_index * cfg.train_loader.batch_size
                processed = min(sample + cfg.train_loader.batch_size, len(train_data))
                batch_index = sample // cfg.train_loader.batch_size
                images, metas = runtime.move_batch(data, "cuda")
                result = model(imgs=images, metas=metas)
                inputs = dict(
                    result, metas=metas, global_iter=epoch * batches + batch_index
                )
                loss, details = objective(inputs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss in epoch {epoch+1}, sample {sample}"
                    )
                (loss / cfg.gradient_accumulation).backward()
                losses.append(float(loss.detach()))
                boundary = (batch_index + 1) % cfg.gradient_accumulation == 0
                gradient_norm = None
                if boundary:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True
                    )
                    gradient_norm = float(norm)
                    runtime.set_lr(
                        optimizer,
                        updates,
                        total_updates,
                        cfg.warmup_iters,
                        cfg.min_lr_ratio,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                # Tensor values never escape their batch graph; logs contain scalars.
                del result, inputs, loss, details, images, metas, data
                if (batch_index + 1) % cfg.print_freq == 0:
                    row = dict(
                        status="training",
                        arm=cfg.pmg_experiment.arm,
                        epoch=epoch + 1,
                        samples=processed,
                        updates=updates,
                        planned_epochs=10,
                        schedule_epochs=15,
                        mean_loss=float(np.mean(losses[-cfg.print_freq :])),
                        gradient_norm=gradient_norm,
                        seconds_per_microbatch=(time.monotonic() - started)
                        / (local_index + 1),
                        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                        allocated_after_cleanup_mib=torch.cuda.memory_allocated()
                        / 2**20,
                    )
                    print("TRAIN " + json.dumps(row), flush=True)
                    runtime.atomic_json(directory / "progress.json", row)
                if boundary and stopped:
                    save(epoch, processed, "train")
                    runtime.atomic_json(
                        directory / "progress.json",
                        dict(
                            status="stopped_by_request",
                            epoch=epoch + 1,
                            samples=processed,
                            updates=updates,
                            planned_epochs=10,
                        ),
                    )
                    return
                if boundary and updates % cfg.checkpoint_interval_steps == 0:
                    save(epoch, processed, "train")
            save(epoch, len(train_data), "evaluate")
        metrics = evaluate(model, validation_data, cfg, report, epoch + 1)
        score = metrics["selection_mean_miou"]
        if score > best:
            best = score
            runtime.atomic_save(
                directory / "best.pth",
                dict(
                    state_dict=model.state_dict(),
                    epoch_index=epoch,
                    metrics=metrics,
                    query_stamps=model.pts_bbox_head.ind_stamps_all,
                    identity=identity,
                ),
            )
        save(epoch + 1, 0, "train")
        refresh(cfg)
        print("EPOCH", epoch + 1, "selection_mean_miou", score, flush=True)
        epoch, next_sample, phase = epoch + 1, 0, "train"
        runtime.atomic_json(
            directory / "progress.json",
            dict(
                status="epoch_completed",
                epoch=epoch,
                arm=cfg.pmg_experiment.arm,
                updates=updates,
                best=best,
                planned_epochs=10,
                schedule_epochs=15,
            ),
        )
        if stopped:
            runtime.atomic_json(
                directory / "progress.json",
                dict(status="stopped_by_request", epoch=epoch, updates=updates),
            )
            return
    if epoch == 10:
        if updates != 10240:
            raise ValueError("completed epochs without the required updates")
        runtime.atomic_json(
            directory / "progress.json",
            dict(
                status="verifying_best_checkpoint",
                epoch=10,
                arm=cfg.pmg_experiment.arm,
                updates=updates,
            ),
        )
        del model, optimizer, objective
        torch.cuda.empty_cache()
        from .verify import verify_best

        verification = verify_best(cfg, directory, validation_data)
        runtime.atomic_json(
            directory / "completed.json",
            dict(
                status="completed",
                epochs=10,
                updates=updates,
                best=best,
                arm=cfg.pmg_experiment.arm,
                best_checkpoint_verified=verification["passed"],
            ),
        )


if __name__ == "__main__":
    main()
