"""Actual CUDA forward/backward checks before a long PMG experiment queue."""

import argparse
import gc
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics
from .evaluate import shape_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis/pmg_world_stage1/validation.json")
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(args.output, dict(status="running", records=[], batch_size=args.batch_size))
    configs = [Config.fromfile(f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py")
               for arm in ("A0", "A1", "A2", "A3", "A4", "C1")]
    runtime.configure_runtime(configs[0])
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("memory fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    dataset = OPENOCC_DATASET.build(configs[0].train_dataset_config)
    validation = OPENOCC_DATASET.build(configs[0].val_dataset_config)
    batch = custom_collate_fn_temporal([dataset[index] for index in (0, 1000)[:args.batch_size]])
    val_batch = custom_collate_fn_temporal([validation[0]])
    records = []
    fingerprints = []
    for cfg in configs:
        runtime.configure_runtime(cfg)
        model = build_segmentor(cfg.model)
        model.init_weights()
        runtime.load_image_initialization(model, cfg.train_load_from)
        fingerprints.append(runtime.model_fingerprint(model))
        model.cuda()
        objective = OPENOCC_LOSS.build(cfg.loss).cuda()
        for epoch in (0, 14):
            model.train()
            model.set_epoch(epoch)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            images, metas = runtime.move_batch(batch, "cuda")
            result = model(imgs=images, metas=metas)
            loss, details = objective(dict(result, metas=metas, global_iter=0))
            assert torch.isfinite(loss)
            (loss / 2).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert model.reg_branch[-1].weight.grad is not None
            assert model.reg_branch[-1].weight.grad.norm() > 0
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            horizons = list(result["gaussian_logits_by_horizon"])
            assert horizons == ([0] if epoch == 0 else [0, 2, 4, 6])
            records.append(dict(arm=cfg.pmg_experiment.arm, epoch_index=epoch,
                                loss=float(loss.detach()), gradient_norm=float(norm),
                                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                                supervised_horizons=horizons))
            del result, loss, details, images, metas
            model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            images, metas = runtime.move_batch(val_batch, "cuda")
            result = model(imgs=images, metas=metas)
            metric = HorizonMetrics()
            metric.add(result, metas)
            assert len(result["future_occ_grids"]) == 6
            geometry = shape_statistics(model, result, 0)
            if cfg.model.geometry_mode == "prox":
                assert max(x["canonical_shape_rms_max_m"] for x in geometry) <= .1005
            records.append(dict(arm=cfg.pmg_experiment.arm, seven_grids=True, geometry=geometry))
            if cfg.pmg_experiment.arm == "A0":
                reference_grids = [x.cpu().clone() for x in [result["final_occ_grid"], *result["future_occ_grids"]]]
                state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                native_cfg = cfg.model.to_dict()
                native_cfg["type"] = "sparse_opus_gausisan_world"
                native = build_segmentor(native_cfg).cuda()
                native.load_state_dict(state, strict=True)
                native.eval()
                native.set_epoch(14)
                native.pts_bbox_head.ind_stamps_all = model.pts_bbox_head.ind_stamps_all.clone()
                native.pts_bbox_head.reset_mask()
                native_result = native(imgs=images, metas=metas)
                actual = [native_result["final_occ_grid"], *native_result["future_occ_grids"]]
                assert all(torch.equal(a, b.cpu()) for a, b in zip(reference_grids, actual))
                del native, native_result, state
            del result, images, metas
        del model, objective
        gc.collect()
        torch.cuda.empty_cache()
        print("VALIDATED", cfg.pmg_experiment.arm, flush=True)
        runtime.atomic_json(args.output, dict(status="running", records=records,
                             validated_arms=[c.pmg_experiment.arm for c in configs[:len(fingerprints)]],
                             batch_size=args.batch_size, training_updates=0))
    assert len(set(fingerprints)) == 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(Path(args.output), dict(status="passed", model_fingerprints=fingerprints,
                         records=records, training_updates=0, training_checkpoint_written=False,
                         batch_size=args.batch_size,
                         scope="real forward/backward and one validation sample; not full epoch metrics"))


if __name__ == "__main__":
    main()
