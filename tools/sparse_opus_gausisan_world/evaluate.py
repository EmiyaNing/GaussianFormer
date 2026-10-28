"""Evaluate a D/integrated checkpoint without updating any model weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from .evaluation import evaluate, evaluate_basic
from .protocol import MODEL_NAME, check_resume_identity
from .runtime import (
    atomic_json,
    audit_manifests,
    configure_runtime,
    digest,
    load_evaluation_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Partial smoke check, never a formal score"
    )
    parser.add_argument(
        "--reference-confusion",
        help="Require exact equality to saved D confusion counts",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit is not None and args.reference_confusion:
        parser.error("reference comparison requires the complete validation set")
    cfg = Config.fromfile(args.config)
    configure_runtime(cfg)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.diagnostics:
        cfg.report_directory = str(output)
    source_hash = digest(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = check_resume_identity(
        saved, cfg, digest(args.config), audit_manifests(cfg)
    )
    model = build_segmentor(cfg.model)
    model.init_weights()
    model.cuda()
    epoch = load_evaluation_checkpoint(model, saved)
    del saved
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if len(dataset) != 1024:
        raise ValueError("the D validation manifest must contain exactly 1024 samples")
    evaluator = evaluate if args.diagnostics else evaluate_basic
    metrics = evaluator(model, dataset, cfg, output, epoch, args.limit)
    parity = None
    if args.reference_confusion:
        with np.load(args.reference_confusion) as source:
            expected = source["confusion"]
        with np.load(output / f"epoch_{epoch:02d}_confusion.npz") as source:
            actual = source["confusion"]
        parity = bool(np.array_equal(actual, expected))
        if not parity:
            raise ValueError(
                "integrated predictions differ from the D confusion matrices"
            )
    if digest(args.checkpoint) != source_hash:
        raise ValueError("source checkpoint changed during read-only evaluation")
    record = dict(
        status="passed",
        model_name=MODEL_NAME,
        checkpoint=args.checkpoint,
        checkpoint_sha256=source_hash,
        checkpoint_identity=identity,
        source_checkpoint_unchanged=True,
        epoch=epoch,
        is_partial=args.limit is not None,
        metrics=metrics,
        exact_reference_confusion=parity,
    )
    atomic_json(output / "evaluation_result.json", record)
    print("EVALUATION_RESULT " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
