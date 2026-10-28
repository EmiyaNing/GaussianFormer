"""One matched-budget training of the minimal recurrent query gate."""

import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from model.segmentor.sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline
from tools.pmg_world.evaluate import evaluate
from tools.pmg_world.regions import histogram
from tools.pmg_world_native_adapt.train import adaptation_mode, FORECAST_MODULES
from tools.pmg_world_repair.train import Tee, loss_control
from tools.sparse_opus_gausisan_world import runtime


TRAINABLE_MODULES = (*FORECAST_MODULES, "query_update_gate")


def is_trainable(name):
    return name.split(".", 1)[0] in TRAINABLE_MODULES


@contextmanager
def native_rollout(model):
    """Exercise the actual unchanged parent, not a second gated approximation."""
    assert "_forecast" not in model.__dict__
    model._forecast = MethodType(SparseOpusFullHorizonBaseline._forecast, model)
    try:
        yield
    finally:
        del model._forecast


def engineering_check(model, objective, train, validation, cfg, order, directory):
    rng = runtime.capture_rng()
    model.eval()
    data = next(iter(runtime.loader(validation, cfg.val_loader, [0], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    with torch.no_grad():
        with native_rollout(model):
            native = model(imgs=images, metas=metas)
        model.memory.clear()
        model.memory_queue.clear()
        candidate = model(imgs=images, metas=metas)
        for key in ("forecast_points_list", "forecast_features_list", "forecast_semantics_list"):
            assert all(torch.equal(a, b) for a, b in zip(native[key], candidate[key])), key
        assert all(torch.equal(x, torch.ones_like(x)) for x in model.last_gate_values)
        grids = [candidate["final_occ_grid"], *candidate["future_occ_grids"]]
        targets = [metas["occ_label"], *metas["future_occ_labels"].unbind(1)]
        counts = np.stack([histogram(p.cpu().numpy(), t.cpu().numpy())
                           for p, t in zip(grids, targets)])
        with np.load("analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz") as old:
            assert np.array_equal(counts, old["global_confusion"][0])
        model.query_update_gate.linear.bias.fill_(.1)
        perturbed = model(imgs=images, metas=metas)
        assert torch.equal(candidate["final_occ_grid"], perturbed["final_occ_grid"])
        assert any(not torch.equal(a, b) for a, b in zip(
            candidate["forecast_points_list"], perturbed["forecast_points_list"]))
        model.query_update_gate.linear.bias.zero_()
    del native, candidate, perturbed, grids, targets, images, metas, data
    adaptation_mode(model)
    data = next(iter(runtime.loader(train, cfg.train_loader, order[:1], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    forward_rng = runtime.capture_rng()
    with native_rollout(model):
        result = model(imgs=images, metas=metas)
        native_loss, native_details = objective(dict(result, metas=metas, update_stamp_statistics=False))
        native_value = float(native_loss.detach())
        native_terms = {key: float(value) for key, value in native_details.items()}
        native_states = {key: [x.detach().clone() for x in result[key]] for key in
                         ("forecast_points_list", "forecast_features_list", "forecast_semantics_list",
                          "current_pred_points", "current_pred_logits",
                          "temporal_pred_points", "temporal_pred_logits")}
        native_loss.backward()
    native_gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                        if n.split(".", 1)[0] in FORECAST_MODULES and p.grad is not None}
    model.zero_grad(set_to_none=True)
    del result, native_loss
    runtime.restore_rng(forward_rng)
    adaptation_mode(model)
    result = model(imgs=images, metas=metas)
    assert set(result["gaussian_logits_by_horizon"]) == set(range(7))
    candidate_loss, candidate_details = objective(dict(result, metas=metas, update_stamp_statistics=False))
    candidate_value = float(candidate_loss.detach())
    forward_audit = dict(native_loss=native_value, candidate_loss=candidate_value,
        loss_difference=candidate_value - native_value,
        term_differences={key: float(candidate_details[key]) - value for key, value in native_terms.items()},
        state_maximum_absolute_errors={key: max(float((a - b).abs().max())
            for a, b in zip(values, result[key])) for key, values in native_states.items()})
    repeated_loss, repeated_details = objective(dict(result, metas=metas, update_stamp_statistics=False))
    forward_audit["same_output_repeated_loss_difference"] = float(repeated_loss.detach()) - candidate_value
    forward_audit["same_output_repeated_term_differences"] = {
        key: float(repeated_details[key]) - float(candidate_details[key]) for key in candidate_details}
    del repeated_loss, repeated_details
    active_terms = {key: value for key, value in forward_audit["term_differences"].items()
                    if key.startswith(("fu", "loss_traj")) or
                    (key.startswith("gaussian.h") and key != "gaussian.h0.loss_voxel")}
    forward_audit["maximum_active_term_absolute_error"] = max(abs(x) for x in active_terms.values())
    runtime.atomic_json(directory / "preflight_forward_comparison.json", forward_audit)
    print("PREFLIGHT_FORWARD " + json.dumps(forward_audit), flush=True)
    assert all(x == 0 for x in forward_audit["state_maximum_absolute_errors"].values()), forward_audit
    assert forward_audit["maximum_active_term_absolute_error"] < 1e-5, forward_audit
    candidate_loss.backward()
    norms, gradient_errors = {}, {}
    numerator, denominator = 0., 0.
    for prefix in TRAINABLE_MODULES:
        gradients = [p.grad for n, p in model.named_parameters()
                     if n.startswith(prefix + ".") and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), prefix
        norms[prefix] = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        assert norms[prefix] > 0, prefix
    for name, parameter in model.named_parameters():
        if name in native_gradients:
            reference = native_gradients[name]
            difference = float((parameter.grad - reference).double().square().sum())
            magnitude = float(reference.double().square().sum())
            numerator += difference
            denominator += magnitude
            gradient_errors[name] = math.sqrt(difference / max(magnitude, 1e-24))
    relative_error = math.sqrt(numerator / denominator)
    # CUDA local aggregation has nondeterministic gradient summation. Report
    # the observed error, rather than asserting bitwise equality on GPU.
    assert relative_error < .01, relative_error
    assert all(p.grad is None for n, p in model.named_parameters() if not is_trainable(n))
    model.zero_grad(set_to_none=True)
    assert all(torch.count_nonzero(p) == 0 for p in model.query_update_gate.parameters())
    audit = dict(status="passed", validation_sample=0, training_sample=order[0],
        initial_seven_horizon_counts_exact=True, native_future_points_features_logits_exact=True,
        gate_initialization_unit_exact=True, gate_perturbation_current_grid_exact=True,
        gate_perturbation_changes_future_points=True, gradient_norms=norms,
        native_loss=native_value, candidate_loss=candidate_value,
        total_loss_difference=candidate_value - native_value,
        maximum_active_term_absolute_error=forward_audit["maximum_active_term_absolute_error"],
        native_gradient_relative_l2_error=relative_error,
        maximum_native_parameter_gradient_relative_error=max(gradient_errors.values()),
        per_parameter_gradient_relative_error=gradient_errors,
        gradient_tolerance_note="CPU gate gradients exact; CUDA renderer backward sum order may vary",
        frozen_gradients_absent=True, preflight_optimizer_updates=0, focal_control=loss_control())
    runtime.atomic_json(directory / "engineering_check.json", audit)
    print("GATE_ENGINEERING_CHECK " + json.dumps({k: v for k, v in audit.items()
                                               if k != "per_parameter_gradient_relative_error"}), flush=True)
    del result, candidate_loss, images, metas, data, native_gradients, gradients
    model.last_gate_values = []
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
        raise FileExistsError("This declared gate experiment is already complete")
    files = [args.config, __file__, "model/segmentor/pmg_query_update_gate_world.py",
        "model/segmentor/sparse_opus_full_horizon_baseline.py", "model/segmentor/sparse_opus_gausisan_world.py",
        "model/segmentor/sparseworld_strict.py", "loss/pmg_repair_distance_weight.py",
        "loss/pmg_time_supervision_loss.py", "loss/sparseworld_strict_loss.py",
        "tools/pmg_world_native_adapt/train.py", "tools/pmg_world_repair/train.py", "tools/pmg_world/evaluate.py"]
    identity = dict(checkpoint_sha256=runtime.digest(cfg.adaptation_checkpoint),
        source_sha256={str(p): runtime.digest(p) for p in files}, manifests=runtime.audit_manifests(cfg))
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Source/config/data identity changed")
    runtime.atomic_json(identity_path, identity)
    model = build_segmentor(cfg.model)
    source = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    missing = model.load_state_dict(source["state_dict"], strict=False)
    assert set(missing.missing_keys) == {"query_update_gate.linear.weight", "query_update_gate.linear.bias"}
    assert not missing.unexpected_keys
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(source["epoch_index"]))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = source["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    del source, counts
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable(name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in model.query_update_gate.parameters()) == 514
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train, validation = (OPENOCC_DATASET.build(cfg.train_dataset_config),
                         OPENOCC_DATASET.build(cfg.val_dataset_config))
    accumulation, planned = int(cfg.gradient_accumulation), int(cfg.adaptation_updates)
    order = np.random.default_rng(20261023).permutation(len(train)).tolist()[:planned * accumulation]
    reference = json.loads(Path("out/pmg_world_native_adapt_20261024/N_native/protocol.json").read_text())
    assert order == reference["training_sample_indices"]
    assert len(order) == 2048 and planned == 512 and accumulation == 4 and len(validation) == 1024
    cfg.dump(str(output / "resolved_config.py"))
    protocol = dict(arm="G1-query-gate", source_checkpoint=cfg.adaptation_checkpoint,
        source_sha256=identity["checkpoint_sha256"], training_sample_indices=order,
        sample_order_seed=20261023, planned_updates=planned, training_samples=2048,
        validation_samples=1024, trainable_modules=list(TRAINABLE_MODULES),
        trainable_parameters=sum(p.numel() for p in parameters), new_parameters=514,
        frozen_perception=True, frozen_plan_head=True, frozen_attribute_decoder=True,
        native_dropout_disabled=True, query_assignment_fixed=True,
        microbatch_size=1, accumulation=4, future_focal_distance_weight_detached=True,
        distillation=False, GT_region_training_masks=False, final_checkpoint_only=True,
        parameter_interpolation=False, parameter_projection=False,
        gate_amplitude=.5, gate_range=[.5, 1.5], gate_hidden_layers=0,
        resume_note="Sample order preserved; worker augmentation stream is not bitwise replayed after interruption",
        interpretation="One minimal forward-architecture change, otherwise matched to N-native")
    runtime.atomic_json(output / "protocol.json", protocol)
    print("PROTOCOL " + json.dumps({k: v for k, v in protocol.items() if k != "training_sample_indices"}), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=.01)
    updates, resume = 0, output / "latest.pth"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        updates = int(saved["updates"])
        runtime.restore_rng(saved["rng"])
        del saved
    else:
        engineering_check(model, objective, train, validation, cfg, order, output)
    if args.check_only:
        return
    adaptation_mode(model)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.monotonic(), []

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
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True))
        factor = (1/3 + 2/3 * updates/32) if updates < 32 else (
            .001 + .999 * .5 * (1 + np.cos(np.pi * (updates - 32) / (planned - 32))))
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * float(factor)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if updates % 16 == 0:
            gates = [dict(seconds=(i + 1) * .5, mean=float(g.mean()),
                          minimum=float(g.min()), maximum=float(g.max()))
                     for i, g in enumerate(model.last_gate_values)]
            progress = dict(status="training", updates=updates, planned_updates=planned,
                samples_consumed=updates * accumulation, mean_loss=float(np.mean(losses[-64:])),
                gradient_norm=norm, lr=optimizer.param_groups[0]["lr"],
                elapsed_seconds=time.monotonic() - started,
                peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                latest_losses=latest_losses, gate_last_microbatch=gates)
            runtime.atomic_json(output / "progress.json", progress)
            print("QUERY_GATE " + json.dumps(progress), flush=True)
        if updates % 128 == 0:
            save()
    assert updates == planned
    save()
    training_seconds = time.monotonic() - started
    peak_training = torch.cuda.max_memory_allocated() / 2**20
    source = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = model.state_dict()
    changed = [k for k, v in source["state_dict"].items() if not torch.equal(v, state[k].detach().cpu())]
    frozen_changed = [k for k in changed if not is_trainable(k)]
    query_exact = torch.equal(source["query_stamps"], model.pts_bbox_head.ind_stamps_all.cpu())
    new_keys = sorted(set(state) - set(source["state_dict"]))
    invariant = dict(changed_state_keys=changed, new_state_keys=new_keys,
        frozen_state_changed_keys=frozen_changed, frozen_state_exact=not frozen_changed,
        query_assignment_exact=bool(query_exact))
    runtime.atomic_json(output / "state_invariants.json", invariant)
    assert changed and not frozen_changed and query_exact
    assert new_keys == ["query_update_gate.linear.bias", "query_update_gate.linear.weight"]
    del source, state, optimizer, objective
    model.last_gate_values = []
    gc.collect()
    torch.cuda.empty_cache()
    runtime.atomic_json(output / "progress.json", dict(status="evaluating", updates=updates,
        validation_samples=len(validation), training_seconds=training_seconds))
    model.reset_gate_statistics(enabled=True)
    metrics = evaluate(model, validation, cfg, report, 1)
    runtime.atomic_json(report / "gate_statistics.json", model.gate_statistics())
    model.collect_gate_statistics = False
    runtime.atomic_json(report / "final_512updates_metrics.json", dict(metrics,
        actual_training_updates=updates, training_samples_consumed=2048, evaluation_round=1,
        budget_note="One query-gate architecture probe; epoch_01 means final evaluation, not a full epoch"))
    runtime.atomic_json(output / "completed.json", dict(status="completed", updates=updates,
        training_seconds=training_seconds, evaluation_seconds=metrics["evaluation_seconds"],
        total_training_and_evaluation_seconds=time.monotonic() - started,
        peak_training_cuda_mib=peak_training, selection_mean_miou=metrics["selection_mean_miou"]))
    runtime.atomic_json(output / "progress.json", dict(status="completed", updates=updates,
        selection_mean_miou=metrics["selection_mean_miou"]))
    print("G1_QUERY_GATE_COMPLETED", flush=True)


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
