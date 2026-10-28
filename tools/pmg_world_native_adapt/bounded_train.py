"""Joint forecast adaptation with a fixed cumulative parameter displacement budget."""

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
from tools.pmg_world.evaluate import evaluate
from tools.pmg_world.regions import histogram
from tools.pmg_world_repair.train import Tee, loss_control
from tools.pmg_world_native_adapt.bounded_projection import (
    JointDisplacementBudget, projection_self_check)
from tools.sparse_opus_gausisan_world import runtime


FORECAST_MODULES = ("ego_cross_attn", "position_encoder", "reg_branch",
                    "vel_branch", "cls_branch", "traj_head")


def is_forecast_parameter(name):
    return name.split(".", 1)[0] in FORECAST_MODULES


def adaptation_mode(model):
    """Keep native dropout/BN off, while retaining forecast autograd and losses."""
    model.train()
    for child in model.children():
        child.eval()
    model.memory.clear()
    model.memory_queue.clear()


def engineering_check(model, objective, train, validation, cfg, order, directory):
    rng = runtime.capture_rng()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()
               if is_forecast_parameter(k)}
    model.eval()
    data = next(iter(runtime.loader(validation, cfg.val_loader, [0], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    try:
        with torch.no_grad():
            native = model(imgs=images, metas=metas)
            grids = [native["final_occ_grid"], *native["future_occ_grids"]]
            targets = [metas["occ_label"], *metas["future_occ_labels"].unbind(1)]
            counts = np.stack([histogram(p.cpu().numpy(), t.cpu().numpy())
                               for p, t in zip(grids, targets)])
            with np.load("analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz") as old:
                assert np.array_equal(counts, old["global_confusion"][0])
            model.reg_branch[-1].bias.add_(.02)
            changed = model(imgs=images, metas=metas)
            assert torch.equal(native["final_occ_grid"], changed["final_occ_grid"])
            assert any(not torch.equal(a, b) for a, b in zip(
                native["forecast_points_list"], changed["forecast_points_list"]))
        del native, changed, grids, targets, images, metas, data
    finally:
        model.load_state_dict(initial, strict=False)
        model.memory.clear()
        model.memory_queue.clear()
    adaptation_mode(model)
    data = next(iter(runtime.loader(train, cfg.train_loader, order[:1], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    result = model(imgs=images, metas=metas)
    assert set(result["gaussian_logits_by_horizon"]) == set(range(7))
    loss, _ = objective(dict(result, metas=metas, update_stamp_statistics=False))
    loss.backward()
    norms = {}
    for prefix in FORECAST_MODULES:
        gradients = [p.grad for n, p in model.named_parameters()
                     if n.startswith(prefix + ".") and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), prefix
        norms[prefix] = math.sqrt(sum(float(g.square().sum()) for g in gradients))
        assert norms[prefix] > 0, prefix
    assert all(p.grad is None for n, p in model.named_parameters() if not is_forecast_parameter(n))
    model.zero_grad(set_to_none=True)
    state = model.state_dict()
    assert all(torch.equal(value, state[key]) for key, value in initial.items())
    audit = dict(status="passed", validation_sample=0, training_sample=order[0],
        initial_seven_horizon_counts_exact=True, forecast_perturbation_current_grid_exact=True,
        forecast_perturbation_changes_future_points=True, gradient_norms=norms,
        frozen_gradients_absent=True, preflight_optimizer_updates=0, focal_control=loss_control())
    runtime.atomic_json(directory / "engineering_check.json", audit)
    print("ENGINEERING_CHECK " + json.dumps(audit), flush=True)
    del loss, result, images, metas, data, gradients, initial, state
    gc.collect()
    torch.cuda.empty_cache()
    runtime.restore_rng(rng)
    adaptation_mode(model)


def run(args):
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output, report = Path(args.work_dir), Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise FileExistsError("This declared experiment is already completed")
    source_files = [args.config, __file__, "model/segmentor/sparse_opus_full_horizon_baseline.py",
        "model/segmentor/sparse_opus_gausisan_world.py", "model/segmentor/sparseworld_strict.py",
        "loss/pmg_repair_distance_weight.py", "loss/pmg_time_supervision_loss.py",
        "loss/sparseworld_strict_loss.py", "tools/pmg_world_repair/train.py",
        "tools/pmg_world/evaluate.py",
        "tools/pmg_world_native_adapt/bounded_projection.py"]
    identity = dict(checkpoint_sha256=runtime.digest(cfg.adaptation_checkpoint),
        source_sha256={str(p): runtime.digest(p) for p in source_files},
        manifests=runtime.audit_manifests(cfg))
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Source/config/data identity changed")
    runtime.atomic_json(identity_path, identity)
    model = build_segmentor(cfg.model)
    source = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(source["state_dict"], strict=True)
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(source["epoch_index"]))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = source["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    del source, counts
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_forecast_parameter(name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    radius = float(cfg.joint_displacement_budget.radius)
    reference_budget = json.loads(Path(cfg.joint_displacement_budget.reference_protocol).read_text())
    if radius != reference_budget["half_step_parameter_delta_l2"]:
        raise ValueError("Radius must equal the predeclared N-half budget")
    budget = JointDisplacementBudget(
        ((n, p) for n, p in model.named_parameters() if p.requires_grad), radius)
    if budget.norm() != 0.:
        raise AssertionError("Origins must be the T-A0 initialization")
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    accumulation, planned = int(cfg.gradient_accumulation), int(cfg.adaptation_updates)
    order = np.random.default_rng(20261023).permutation(len(train)).tolist()[:planned * accumulation]
    reference = json.loads(Path("out/pmg_world_repair_20261024/M_repair/protocol.json").read_text())
    assert order == reference["training_sample_indices"]
    assert len(order) == 2048 and planned == 512 and accumulation == 4
    assert len(validation) == 1024
    cfg.dump(str(output / "resolved_config.py"))
    protocol = dict(arm="N-bounded", source_checkpoint=cfg.adaptation_checkpoint,
        source_sha256=identity["checkpoint_sha256"], training_sample_indices=order,
        sample_order_seed=20261023, planned_updates=planned, training_samples=2048,
        validation_samples=1024, trainable_modules=list(FORECAST_MODULES),
        trainable_parameters=sum(p.numel() for p in parameters), new_parameters=0,
        frozen_perception=True, frozen_plan_head=True, frozen_attribute_decoder=True,
        native_dropout_disabled=True, query_assignment_fixed=True,
        microbatch_size=1, accumulation=4, future_focal_distance_weight_detached=True,
        distillation=False, GT_region_training_masks=False, final_checkpoint_only=True,
        joint_displacement_radius=radius, projection_after_each_update=True,
        projection_scope="One concatenated vector across all six forecast modules",
        optimizer_moments="Ordinary AdamW moments retained unchanged after projection",
        fp32_roundoff="If rounding exits the ball, apply a common inward correction of eight dtype epsilons",
        interpretation="Same training as N-native with one cumulative L2 budget; compare against N-half, not an architecture innovation.")
    runtime.atomic_json(output / "protocol.json", protocol)
    print("PROTOCOL " + json.dumps({k: v for k, v in protocol.items() if k != "training_sample_indices"}), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=.01)
    updates, resume = 0, output / "latest.pth"
    projection_count, first_projected_update = 0, None
    maximum_pre, maximum_post = 0., 0.
    trace_path = output / "projection_trace.jsonl"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        updates = int(saved["updates"])
        history = saved["projection_state"]
        projection_count = history["projection_count"]
        first_projected_update = history["first_projected_update"]
        maximum_pre, maximum_post = history["max_pre_delta_l2"], history["max_post_delta_l2"]
        assert budget.norm() <= radius + 1e-6
        # Keep a resumed trace aligned with the last durable checkpoint.
        if trace_path.exists():
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            rows = [row for row in rows if row["updates"] <= updates]
            trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        runtime.restore_rng(saved["rng"])
        del saved
    else:
        engineering_check(model, objective, train, validation, cfg, order, output)
        check = projection_self_check()
        with torch.no_grad():
            first_name = next(iter(budget.parameters))
            budget.parameters[first_name].view(-1)[0].add_(2. * radius)
        actual_projection = budget.project()
        assert actual_projection["projected"] and actual_projection["post_delta_l2"] <= radius + 1e-6
        with torch.no_grad():
            for name, parameter in budget.parameters.items():
                parameter.copy_(budget.origins[name])
        assert budget.norm() == 0.
        check["actual_model_projection"] = actual_projection
        check["original_parameters_restored_exact"] = True
        runtime.atomic_json(output / "projection_engineering_check.json", check)
        print("PROJECTION_CHECK " + json.dumps(check), flush=True)
        if trace_path.exists():
            raise FileExistsError("Projection trace exists without a durable checkpoint")
    if args.check_only:
        return
    adaptation_mode(model)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.monotonic(), []

    def save():
        runtime.atomic_save(output / "latest.pth", dict(state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(), epoch_index=model.curr_epoch, updates=updates,
            query_stamps=model.pts_bbox_head.ind_stamps_all, rng=runtime.capture_rng(),
            projection_state=dict(projection_count=projection_count,
                first_projected_update=first_projected_update,
                max_pre_delta_l2=maximum_pre, max_post_delta_l2=maximum_post,
                radius=radius, final_delta_l2=budget.norm())))

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
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True))
        factor = (1/3 + 2/3 * updates/32) if updates < 32 else (
            .001 + .999 * .5 * (1 + np.cos(np.pi * (updates - 32) / (planned - 32))))
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * float(factor)
        optimizer.step()
        projection = budget.project()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if projection["projected"]:
            projection_count += 1
            if first_projected_update is None:
                first_projected_update = updates
        maximum_pre = max(maximum_pre, projection["pre_delta_l2"])
        maximum_post = max(maximum_post, projection["post_delta_l2"])
        with trace_path.open("a", buffering=1) as trace:
            trace.write(json.dumps(dict(updates=updates, projection_count=projection_count,
                                        **projection)) + "\n")
        if updates % 16 == 0:
            progress = dict(status="training", updates=updates, planned_updates=planned,
                samples_consumed=updates * accumulation, mean_loss=float(np.mean(losses[-64:])),
                gradient_norm=norm, lr=optimizer.param_groups[0]["lr"],
                elapsed_seconds=time.monotonic() - started,
                peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20, latest_losses=latest_losses,
                projection_count=projection_count, first_projected_update=first_projected_update,
                **projection)
            runtime.atomic_json(output / "progress.json", progress)
            print("BOUNDED_ADAPT " + json.dumps(progress), flush=True)
        if updates % 128 == 0:
            save()
    assert updates == planned
    save()
    training_seconds = time.monotonic() - started
    runtime.atomic_json(output / "projection_summary.json", dict(radius=radius,
        steps=updates, projection_count=projection_count,
        first_projected_update=first_projected_update, max_pre_delta_l2=maximum_pre,
        max_post_delta_l2=maximum_post, final_delta_l2=budget.norm(),
        max_projection_excess=max(0., maximum_post - radius)))
    peak_training = torch.cuda.max_memory_allocated() / 2**20
    source = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = model.state_dict()
    changed = [k for k, v in source["state_dict"].items() if not torch.equal(v, state[k].detach().cpu())]
    frozen_changed = [k for k in changed if not is_forecast_parameter(k)]
    query_exact = torch.equal(source["query_stamps"], model.pts_bbox_head.ind_stamps_all.cpu())
    invariant = dict(changed_state_keys=changed, frozen_state_changed_keys=frozen_changed,
        frozen_state_exact=not frozen_changed, query_assignment_exact=bool(query_exact),
        parameter_delta_l2=budget.norm(), parameter_budget_satisfied=budget.norm() <= radius + 1e-6)
    runtime.atomic_json(output / "state_invariants.json", invariant)
    assert changed and not frozen_changed and query_exact
    del source, state, optimizer, objective
    gc.collect()
    torch.cuda.empty_cache()
    runtime.atomic_json(output / "progress.json", dict(status="evaluating", updates=updates,
        validation_samples=len(validation), training_seconds=training_seconds))
    metrics = evaluate(model, validation, cfg, report, 1)
    runtime.atomic_json(report / "final_512updates_metrics.json", dict(metrics,
        actual_training_updates=updates, training_samples_consumed=2048, evaluation_round=1,
        budget_note="Bounded joint predictor adaptation; epoch_01 is the fixed final 512-update evaluation, not a full training epoch."))
    runtime.atomic_json(output / "completed.json", dict(status="completed", updates=updates,
        training_seconds=training_seconds, evaluation_seconds=metrics["evaluation_seconds"],
        total_training_and_evaluation_seconds=time.monotonic() - started,
        peak_training_cuda_mib=peak_training, selection_mean_miou=metrics["selection_mean_miou"]))
    runtime.atomic_json(output / "progress.json", dict(status="completed", updates=updates,
        selection_mean_miou=metrics["selection_mean_miou"]))
    print("N_BOUNDED_COMPLETED", flush=True)


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
