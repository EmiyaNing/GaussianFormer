"""Evaluate one fixed module-wise update: reg/vel full, other forecasts half."""

import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
import traceback

import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from tools.pmg_world.evaluate import evaluate
from tools.pmg_world_native_adapt.train import FORECAST_MODULES, is_forecast_parameter
from tools.pmg_world_repair.train import Tee
from tools.sparse_opus_gausisan_world import runtime


def run(args):
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output, report = Path(args.work_dir), Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise FileExistsError("The one declared grouped-update probe is already complete")
    scales = dict(cfg.grouped_update_probe.module_update_scales)
    expected = {name: 1. if name in ("reg_branch", "vel_branch") else .5
                for name in FORECAST_MODULES}
    assert scales == expected, "Do not silently change the predeclared candidate"
    parent = Path(cfg.grouped_update_probe.parent_run)
    half_run = Path(cfg.grouped_update_probe.half_run)
    assert all((folder / "completed.json").exists() for folder in (parent, half_run))
    base = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    native = torch.load(parent / "latest.pth", map_location="cpu", mmap=True, weights_only=False)
    half = torch.load(half_run / "latest.pth", map_location="cpu", mmap=True, weights_only=False)
    assert native["updates"] == 512
    assert torch.equal(base["query_stamps"], native["query_stamps"].cpu())
    assert torch.equal(base["query_stamps"], half["query_stamps"].cpu())
    mixed, audit = {}, {name: dict(scale=scales[name], tensors=0, parent_delta_squared=0.)
                        for name in FORECAST_MODULES}
    frozen_keys = 0
    for key, value in base["state_dict"].items():
        other, halfway = native["state_dict"][key].cpu(), half["state_dict"][key].cpu()
        if is_forecast_parameter(key):
            module = key.split(".", 1)[0]
            delta = other - value
            mixed[key] = other if scales[module] == 1. else value + .5 * delta
            if scales[module] == 1.:
                assert torch.equal(mixed[key], other), key
            else:
                assert torch.equal(mixed[key], halfway), key
            audit[module]["tensors"] += 1
            audit[module]["parent_delta_squared"] += float(delta.double().square().sum())
        else:
            assert torch.equal(value, other) and torch.equal(value, halfway), key
            mixed[key] = value
            frozen_keys += 1
    for record in audit.values():
        assert record["tensors"] > 0
        record["parent_delta_l2"] = record.pop("parent_delta_squared") ** .5
        record["accepted_delta_l2"] = record["scale"] * record["parent_delta_l2"]
    protocol = dict(arm="N-group", post_run_supplementary_probe=True,
        parent_run=str(parent), reference_half_run=str(half_run), parent_training_updates=512,
        additional_training_updates=0, module_update_scales=scales, module_audit=audit,
        fixed_candidates_evaluated_this_round=1, frozen_tensors_verified=frozen_keys,
        full_reg_vel_parameters_exact=True, remaining_parameters_match_N_half=True,
        frozen_state_exact=True, query_assignment_exact=True, new_parameters=0,
        selection="One fixed grouping predeclared before evaluation; no scale sweep or checkpoint selection.",
        base_sha256=runtime.digest(cfg.adaptation_checkpoint),
        parent_sha256=runtime.digest(parent / "latest.pth"),
        half_sha256=runtime.digest(half_run / "latest.pth"),
        source_sha256={p: runtime.digest(p) for p in (args.config, __file__,
            "tools/pmg_world_native_adapt/train.py", "tools/pmg_world/evaluate.py")},
        manifests=runtime.audit_manifests(cfg), distillation=False,
        future_images=False, GT_region_model_inputs=False, validation_samples=1024)
    runtime.atomic_json(output / "protocol.json", protocol)
    cfg.dump(str(output / "resolved_config.py"))
    print("INTERPOLATION_CHECK " + json.dumps(dict(module_audit=audit,
        frozen_state_exact=True, remaining_parameters_match_N_half=True)), flush=True)
    model = build_segmentor(cfg.model)
    model.load_state_dict(mixed, strict=True)
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(base["epoch_index"]))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = base["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    runtime.atomic_save(output / "latest.pth", dict(state_dict=model.state_dict(),
        epoch_index=model.curr_epoch, query_stamps=model.pts_bbox_head.ind_stamps_all,
        additional_training_updates=0, parent_training_updates=512, module_update_scales=scales))
    del base, native, half, mixed, counts
    gc.collect()
    torch.cuda.empty_cache()
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert len(dataset) == 1024
    runtime.atomic_json(output / "progress.json", dict(status="evaluating",
        validation_samples=1024, additional_training_updates=0, module_update_scales=scales))
    metrics = evaluate(model, dataset, cfg, report, 1)
    runtime.atomic_json(report / "fixed_grouped_update_metrics.json", dict(metrics,
        parent_training_updates=512, additional_training_updates=0, module_update_scales=scales,
        budget_note="One fixed grouped parameter-update evaluation; no new optimization."))
    runtime.atomic_json(output / "completed.json", dict(status="completed",
        evaluation_seconds=metrics["evaluation_seconds"], additional_training_updates=0,
        module_update_scales=scales, selection_mean_miou=metrics["selection_mean_miou"]))
    runtime.atomic_json(output / "progress.json", dict(status="completed", validation_samples=1024,
        selection_mean_miou=metrics["selection_mean_miou"], additional_training_updates=0))
    print("N_GROUP_COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "evaluation.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                run(args)
            except Exception as error:
                runtime.atomic_json(directory / "failure.json", dict(
                    type=type(error).__name__, message=str(error)))
                traceback.print_exc(file=sys.stdout)
                raise


if __name__ == "__main__":
    main()
