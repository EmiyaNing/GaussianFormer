"""Train the single predeclared M-repair candidate and evaluate update 512 once."""

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from loss.sparseworld_strict_loss import SparseWorldStrictLoss
from loss.pmg_repair_distance_weight import PMGRepairStrictLoss
from tools.pmg_world.evaluate import evaluate
from tools.pmg_world.regions import histogram
from tools.sparse_opus_gausisan_world import runtime


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.original.write(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def native_training_mode(model):
    model.train()
    for name, child in model.named_children():
        if name != "motion_correction":
            child.eval()
    model.memory.clear()
    model.memory_queue.clear()


def loss_control():
    """Actual loss calls: same values, same logit/point gradients, removed weight route."""
    original, repair = SparseWorldStrictLoss(), PMGRepairStrictLoss()
    results = []
    for objective in (original, repair):
        point = torch.tensor([[1.5, 0., 0.]], requires_grad=True)
        logits = torch.zeros(1, 17, requires_grad=True)
        cls, pts = objective._set_loss(point, logits, torch.zeros(1, 3),
                                      torch.tensor([4]), prediction_mask=torch.ones(1))
        gp, gl = torch.autograd.grad(cls, (point, logits), allow_unused=True, retain_graph=True)
        gt = torch.autograd.grad(pts, point)[0]
        results.append((cls.detach(), pts.detach(), gp, gl, gt))
    for i in (0, 1, 3, 4):
        torch.testing.assert_close(results[0][i], results[1][i], rtol=0., atol=0.)
    assert results[0][2][0, 0] < 0 and results[1][2] is None
    return dict(forward_loss_exact=True, classification_logit_gradient_exact=True,
                point_regression_gradient_exact=True,
                original_classification_position_gradient=results[0][2].tolist(),
                repaired_classification_position_gradient="no graph connection")


def engineering_check(model, objective, train, validation, cfg, order, directory):
    rng = runtime.capture_rng()
    initial = {k: v.detach().clone() for k, v in model.motion_correction.state_dict().items()}
    captured = []
    render = model.gaussian_head.render_state

    def capture(state, xyz):
        captured.append(state)
        return render(state, xyz)

    model.gaussian_head.render_state = capture
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    data = next(iter(runtime.loader(validation, cfg.val_loader, [0], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    try:
        with torch.no_grad():
            zero = model(imgs=images, metas=metas)
            zero_states = list(captured)
            captured.clear()
            grids = [zero["final_occ_grid"], *zero["future_occ_grids"]]
            targets = [metas["occ_label"], *metas["future_occ_labels"].unbind(1)]
            counts = np.stack([histogram(p.cpu().numpy(), t.cpu().numpy())
                               for p, t in zip(grids, targets)])
            with np.load("analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz") as source:
                assert np.array_equal(counts, source["global_confusion"][0])
            model.repair_disabled = True
            native = model(imgs=images, metas=metas)
            assert all(torch.equal(a, b) for a, b in zip(
                grids, [native["final_occ_grid"], *native["future_occ_grids"]]))
            captured.clear()
            model.repair_disabled = False
            model.motion_correction.child[-1].bias.copy_(images.new_tensor([.2, -.1, 0.]))
            moved = model(imgs=images, metas=metas)
            assert torch.equal(zero["final_occ_grid"], moved["final_occ_grid"])
            content_exact = all(torch.equal(getattr(a, field), getattr(b, field))
                for a, b in zip(zero_states, captured)
                for field in ("semantics", "scales", "rotations"))
            births_exact = all(torch.equal(getattr(a, field)[:, model.num_query:],
                                          getattr(b, field)[:, model.num_query:])
                for a, b in zip(zero_states[1:], captured[1:])
                for field in ("means", "semantics", "scales", "rotations", "opacities"))
            assert content_exact and births_exact
            assert any(not torch.equal(a, b) for a, b in zip(
                zero["forecast_points_list"], moved["forecast_points_list"]))
            model.motion_correction.load_state_dict(initial)
        del zero, native, moved, grids, targets, images, metas, data, zero_states
        captured.clear()
    finally:
        model.gaussian_head.render_state = render
        model.repair_disabled = False
        model.motion_correction.load_state_dict(initial)
    native_training_mode(model)
    data = next(iter(runtime.loader(train, cfg.train_loader, order[:1], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    result = model(imgs=images, metas=metas)
    assert set(result["gaussian_logits_by_horizon"]) == set(range(7))
    loss, _ = objective(dict(result, metas=metas, update_stamp_statistics=False))
    loss.backward()
    gradients = [p.grad for p in model.motion_correction.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    norm = math.sqrt(sum(float(g.square().sum()) for g in gradients))
    assert norm > 0
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith("motion_correction."))
    model.zero_grad(set_to_none=True)
    for key, value in model.motion_correction.state_dict().items():
        assert torch.equal(value, initial[key])
    audit = dict(status="passed", validation_sample=0, training_sample=order[0],
                 zero_head_original_seven_horizon_counts_exact=True,
                 zero_head_vs_native_grids_exact=True,
                 nonzero_repair_semantics_and_shape_exact=content_exact,
                 nonzero_repair_births_exact=births_exact,
                 finite_nonzero_gradient_norm=norm, native_gradients_absent=True,
                 focal_control=loss_control())
    runtime.atomic_json(directory / "engineering_check.json", audit)
    print("ENGINEERING_CHECK " + json.dumps(audit), flush=True)
    del loss, result, images, metas, data, gradients, initial
    gc.collect()
    torch.cuda.empty_cache()
    runtime.restore_rng(rng)
    native_training_mode(model)


def run(args):
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output, report = Path(args.work_dir), Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise FileExistsError("The declared experiment is already completed")
    source_files = [args.config, __file__, "model/segmentor/pmg_child_state_repair_world.py",
                    "loss/pmg_repair_distance_weight.py", "loss/sparseworld_strict_loss.py",
                    "model/segmentor/pmg_sparse_opus_world.py",
                    "model/segmentor/sparse_opus_gausisan_world.py",
                    "tools/pmg_world/evaluate.py"]
    identity = dict(checkpoint_sha256=runtime.digest(cfg.adaptation_checkpoint),
                    source_sha256={str(p): runtime.digest(p) for p in source_files},
                    manifests=runtime.audit_manifests(cfg))
    old_identity = output / "identity.json"
    if old_identity.exists() and json.loads(old_identity.read_text()) != identity:
        raise ValueError("Source/config/data identity changed in the same experiment")
    runtime.atomic_json(old_identity, identity)
    model = build_segmentor(cfg.model)
    source = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    incompatible = model.load_state_dict(source["state_dict"], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("motion_correction.")
                                          for k in incompatible.missing_keys):
        raise ValueError("Native parameter namespace changed")
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(source["epoch_index"]))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = source["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    del source, counts
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("motion_correction."))
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    accumulation = int(cfg.gradient_accumulation)
    planned_updates = int(cfg.adaptation_updates)
    order = np.random.default_rng(20261023).permutation(len(train)).tolist()
    order = order[:planned_updates * accumulation]
    reference_protocol = json.loads(Path("out/pmg_world_stage3_20261023/frozen_M_current/protocol.json").read_text())
    if order != reference_protocol["training_sample_indices"]:
        raise ValueError("Training order differs from M-current")
    if len(order) != 2048 or planned_updates != 512 or accumulation != 4:
        raise ValueError("The single declared budget must remain 2048 samples / 512 updates")
    cfg.dump(str(output / "resolved_config.py"))
    protocol = dict(arm="M-repair", actual_updates=512, training_samples=2048,
        validation_samples=len(validation), training_sample_indices=order, sample_order_seed=20261023,
        source_checkpoint=cfg.adaptation_checkpoint, source_sha256=identity["checkpoint_sha256"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        previous_M_current_parameters=reference_protocol["trainable_parameters"],
        microbatch_size=1, accumulation=4, frozen_native=True, native_dropout_disabled=True,
        distillation=False, final_checkpoint_only=True, two_changes_combined=True,
        changes=["child-conditioned shared XY repair", "stop future focal distance-weight position gradient"],
        local_state="absolute/relative positions, 17 semantic proportions, log mass, 3 scales, ROI opacity, time",
        correction_radius_m=2., native_query_count=720, gaussian_budget_unchanged=True)
    runtime.atomic_json(output / "protocol.json", protocol)
    optimizer = torch.optim.AdamW(model.motion_correction.parameters(), lr=2e-4, weight_decay=.01)
    updates = 0
    resume = output / "latest.pth"
    if resume.exists():
        checkpoint = torch.load(resume, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        updates = int(checkpoint["updates"])
        runtime.restore_rng(checkpoint["rng"])
        del checkpoint
    else:
        engineering_check(model, objective, train, validation, cfg, order, output)
    if args.check_only:
        return
    native_training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses, started = [], time.monotonic()

    def save():
        runtime.atomic_save(output / "latest.pth", dict(state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(), epoch_index=model.curr_epoch, updates=updates,
            query_stamps=model.pts_bbox_head.ind_stamps_all, rng=runtime.capture_rng()))

    for microbatch, data in enumerate(runtime.loader(train, cfg.train_loader,
                         order[updates * accumulation:], cfg.seed)):
        images, metas = runtime.move_batch(data, "cuda")
        result = model(imgs=images, metas=metas)
        loss, details = objective(dict(result, metas=metas, update_stamp_statistics=False))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        (loss / accumulation).backward()
        losses.append(float(loss.detach()))
        latest_losses = {k: float(v) for k, v in details.items()}
        del result, loss, details, images, metas, data
        if (microbatch + 1) % accumulation:
            continue
        norm = float(torch.nn.utils.clip_grad_norm_(model.motion_correction.parameters(),
                                                    5., error_if_nonfinite=True))
        factor = (1/3 + 2/3 * updates/32) if updates < 32 else (
            .001 + .999 * .5 * (1 + np.cos(np.pi * (updates - 32) / (planned_updates - 32))))
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * float(factor)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if updates % 16 == 0:
            progress = dict(status="training", updates=updates, planned_updates=planned_updates,
                samples_consumed=updates * accumulation, mean_loss=float(np.mean(losses[-64:])),
                gradient_norm=norm, lr=optimizer.param_groups[0]["lr"],
                elapsed_seconds=time.monotonic() - started,
                peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                correction=model.repair_diagnostics, latest_losses=latest_losses)
            runtime.atomic_json(output / "progress.json", progress)
            print("REPAIR " + json.dumps(progress), flush=True)
        if updates % 128 == 0:
            save()
    if updates != planned_updates:
        raise ValueError("Training sample count did not satisfy the declared budget")
    save()
    training_seconds = time.monotonic() - started
    peak_training_mib = torch.cuda.max_memory_allocated() / 2**20
    source = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = model.state_dict()
    changed = [k for k, v in source["state_dict"].items() if not torch.equal(v, state[k].detach().cpu())]
    query_exact = torch.equal(source["query_stamps"], model.pts_bbox_head.ind_stamps_all.cpu())
    invariant = dict(native_state_changed_keys=changed, native_state_exact=not changed,
                     query_assignment_exact=bool(query_exact))
    runtime.atomic_json(output / "frozen_state_invariants.json", invariant)
    if changed or not query_exact:
        raise ValueError("Native frozen-state invariant failed")
    del source, state, optimizer, objective
    gc.collect()
    torch.cuda.empty_cache()
    runtime.atomic_json(output / "progress.json", dict(status="evaluating", updates=updates,
                        validation_samples=len(validation), training_seconds=training_seconds))
    metrics = evaluate(model, validation, cfg, report, 1)
    runtime.atomic_json(report / "final_512updates_metrics.json", dict(metrics,
        actual_training_updates=512, training_samples_consumed=2048, evaluation_round=1,
        budget_note="Frozen T-A0 adaptation; epoch_01 denotes the single final evaluation."))
    runtime.atomic_json(output / "completed.json", dict(status="completed", updates=updates,
        training_seconds=training_seconds, evaluation_seconds=metrics["evaluation_seconds"],
        total_training_and_evaluation_seconds=time.monotonic() - started,
        peak_training_cuda_mib=peak_training_mib,
        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
        selection_mean_miou=metrics["selection_mean_miou"]))
    runtime.atomic_json(output / "progress.json", dict(status="completed", updates=updates,
                        selection_mean_miou=metrics["selection_mean_miou"]))
    print("M_REPAIR_COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    directory = Path(args.work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "train.log").open("a", buffering=1) as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)
        try:
            run(args)
        except Exception as error:
            runtime.atomic_json(directory / "failure.json", dict(type=type(error).__name__,
                                message=str(error), time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))
            traceback.print_exc(file=sys.stdout)
            raise
        finally:
            sys.stdout = original


if __name__ == "__main__":
    main()
