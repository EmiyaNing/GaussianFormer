"""One post-run fixed half-step probe; no extra optimization or alpha sweep."""

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
from tools.pmg_world_native_adapt.train import is_forecast_parameter
from tools.pmg_world_repair.train import Tee
from tools.sparse_opus_gausisan_world import runtime


def run(args):
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output, report = Path(args.work_dir), Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise FileExistsError("The single half-step probe is already completed")
    assert float(cfg.half_step_probe.update_scale) == .5
    assert Path(cfg.half_step_probe.parent_run, "completed.json").exists()
    base = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    adapted = torch.load(Path(cfg.half_step_probe.parent_run) / "latest.pth",
                         map_location="cpu", mmap=True, weights_only=False)
    assert adapted["updates"] == 512
    assert torch.equal(base["query_stamps"], adapted["query_stamps"].cpu())
    mixed, changed, squared = {}, [], 0.
    for key, value in base["state_dict"].items():
        other = adapted["state_dict"][key].cpu()
        if is_forecast_parameter(key):
            delta = other - value
            mixed[key] = value + .5 * delta
            squared += float(delta.double().square().sum())
            if not torch.equal(value, other):
                changed.append(key)
        else:
            assert torch.equal(value, other), key
            mixed[key] = value
    protocol = dict(arm="N-half", post_run_supplementary_probe=True,
        parent_run=cfg.half_step_probe.parent_run, parent_updates=512,
        additional_training_updates=0, update_scale=.5, number_of_scales_evaluated=1,
        selection="Fixed at one half before this evaluation; motivated by N-native's measured F-only loss.",
        interpolation="theta_base + 0.5 * (theta_native_adapted - theta_base), forecast modules only",
        base_sha256=runtime.digest(cfg.adaptation_checkpoint),
        parent_sha256=runtime.digest(Path(cfg.half_step_probe.parent_run) / "latest.pth"),
        source_sha256={p:runtime.digest(p) for p in (args.config, __file__,
            "tools/pmg_world_native_adapt/train.py", "tools/pmg_world/evaluate.py")},
        manifests=runtime.audit_manifests(cfg), changed_forecast_tensors=len(changed),
        parent_parameter_delta_l2=squared ** .5, half_step_parameter_delta_l2=.5 * squared ** .5,
        frozen_state_exact=True, query_assignment_exact=True, new_parameters=0,
        future_images=False, distillation=False, validation_samples=1024)
    runtime.atomic_json(output / "protocol.json", protocol)
    cfg.dump(str(output / "resolved_config.py"))
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
        additional_training_updates=0, parent_training_updates=512, update_scale=.5))
    del base, adapted, mixed, counts
    gc.collect()
    torch.cuda.empty_cache()
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert len(validation) == 1024
    runtime.atomic_json(output / "progress.json", dict(status="evaluating", samples=1024,
        additional_training_updates=0, update_scale=.5))
    metrics = evaluate(model, validation, cfg, report, 1)
    runtime.atomic_json(report / "fixed_half_step_metrics.json", dict(metrics,
        parent_training_updates=512, additional_training_updates=0, update_scale=.5,
        budget_note="One supplementary fixed parameter-interpolation evaluation; no new training."))
    runtime.atomic_json(output / "completed.json", dict(status="completed",
        evaluation_seconds=metrics["evaluation_seconds"], additional_training_updates=0,
        update_scale=.5, selection_mean_miou=metrics["selection_mean_miou"]))
    runtime.atomic_json(output / "progress.json", dict(status="completed", samples=1024,
        additional_training_updates=0, update_scale=.5))
    print("N_HALF_COMPLETED " + json.dumps(dict(primary=metrics["selection_mean_miou"])), flush=True)


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
