"""One conditional same-checkpoint intervention: set every learned gate to one."""

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
from tools.pmg_world_repair.train import Tee
from tools.sparse_opus_gausisan_world import runtime


def run(args):
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    source = Path(args.source_run)
    output, report = Path(args.work_dir), Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        raise FileExistsError("The single unit-gate intervention is already complete")
    statistics = json.loads((Path(cfg.report_directory) / "comparison_statistics.json").read_text())
    if not statistics["unit_gate_intervention_recommended"]:
        raise ValueError("Candidate did not meet the predeclared intervention trigger")
    completed = json.loads((source / "completed.json").read_text())
    assert completed["status"] == "completed" and completed["updates"] == 512
    checkpoint = torch.load(source / "latest.pth", map_location="cpu", mmap=True, weights_only=False)
    model = build_segmentor(cfg.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(checkpoint["epoch_index"]))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = checkpoint["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    model.force_unit_gate = True
    model.reset_gate_statistics(enabled=True)
    del checkpoint, counts
    gc.collect()
    torch.cuda.empty_cache()
    cfg.pmg_experiment.arm = "G1-unit-gate-intervention"
    cfg.dump(str(output / "resolved_config.py"))
    runtime.atomic_json(output / "protocol.json", dict(
        additional_training_updates=0, source_run=str(source),
        checkpoint_sha256=runtime.digest(source / "latest.pth"),
        intervention="All query gates forced to 1; every trained parameter retained unchanged",
        trigger=statistics["unit_gate_intervention_recommended"],
        interpretation="Forward component intervention on a jointly trained model, not independent causal identification",
        source_sha256={p: runtime.digest(p) for p in
            (args.config, __file__, "model/segmentor/pmg_query_update_gate_world.py", "tools/pmg_world/evaluate.py")},
        manifests=runtime.audit_manifests(cfg), validation_samples=1024))
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert len(dataset) == 1024
    runtime.atomic_json(output / "progress.json", dict(status="evaluating", additional_training_updates=0))
    metrics = evaluate(model, dataset, cfg, report, 1)
    gate = model.gate_statistics()
    assert all(row["mean"] == 1. and row["minimum"] == row["maximum"] == 1. for row in gate["rows"])
    runtime.atomic_json(report / "gate_statistics.json", gate)
    runtime.atomic_json(output / "completed.json", dict(status="completed", additional_training_updates=0,
        selection_mean_miou=metrics["selection_mean_miou"], evaluation_seconds=metrics["evaluation_seconds"]))
    runtime.atomic_json(output / "progress.json", dict(status="completed", additional_training_updates=0))
    print("UNIT_GATE_INTERVENTION_COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    output = Path(args.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "evaluation.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                run(args)
            except Exception:
                traceback.print_exc(file=sys.stdout)
                raise


if __name__ == "__main__":
    main()
