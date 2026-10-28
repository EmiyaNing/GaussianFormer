"""Real batch-2 backward, matched initialization, and synchronized GPU timing."""

import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from mmengine import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics


BASE = "config/sparse_opus_gaussian_world/pmg_A0_4096_1024_direct15e.py"
GF2 = "config/sparse_opus_gaussian_world/pmg_A0_gf2_probability_4096_1024_direct15e.py"
ROOT = Path("analysis/pmg_gf2_probability_mini15e")


def tensor_hashes(model):
    return {name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in model.state_dict().items()}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(GF2)
    runtime.configure_runtime(cfg)
    train = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert (len(train), len(validation)) == (4096, 1024)
    batch = custom_collate_fn_temporal([train[index] for index in (0, 1000)])
    val_batch = custom_collate_fn_temporal([validation[0]])
    images, metas = runtime.move_batch(batch, "cuda")
    val_images, val_metas = runtime.move_batch(val_batch, "cuda")
    records, baseline_hashes = [], None
    for name, config_path in (("A0", BASE), ("A0_GF2", GF2)):
        cfg = Config.fromfile(config_path)
        runtime.configure_runtime(cfg)
        model = build_segmentor(cfg.model)
        model.init_weights()
        runtime.load_image_initialization(model, cfg.train_load_from)
        hashes = tensor_hashes(model)
        if baseline_hashes is None:
            baseline_hashes = hashes
        else:
            shared = hashes.keys() & baseline_hashes.keys()
            assert all(hashes[key] == baseline_hashes[key] for key in shared), "shared initial weights differ"
            removed = sorted(baseline_hashes.keys() - hashes.keys())
            added = sorted(hashes.keys() - baseline_hashes.keys())
            assert not added
            assert all(key.startswith("gaussian_head.empty_") for key in removed)
            assert not any("empty_bias" in key or "empty_scalar" in key for key in hashes)
            runtime.atomic_json(ROOT / "initialization_parity.json", dict(
                shared_tensors=len(shared), all_shared_tensors_identical=True,
                removed_tensors=removed, added_tensors=added))
        model.cuda()
        objective = OPENOCC_LOSS.build(cfg.loss).cuda()
        for epoch in (0, 14):
            model.train()
            model.set_epoch(epoch)
            elapsed, peaks = [], []
            for repeat in range(4):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = model(imgs=images, metas=metas)
                loss, details = objective(dict(result, metas=metas, global_iter=0))
                (loss / cfg.gradient_accumulation).backward()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                assert model.reg_branch[-1].weight.grad.norm() > 0
                assert list(result["gaussian_logits_by_horizon"]) == ([0] if epoch == 0 else [0, 2, 4, 6])
                if name == "A0_GF2":
                    raw = result["all_cls_scores"][-1]
                    torch.testing.assert_close(result["representation"][-1]["gaussian"].semantics, raw)
                    for probability in result["gaussian_logits_by_horizon"].values():
                        assert torch.isfinite(probability).all() and probability.min() >= 0
                        torch.testing.assert_close(probability.sum(1), torch.ones_like(probability[:, 0]), rtol=1e-5, atol=1e-5)
                if repeat:
                    elapsed.append(seconds)
                    peaks.append(torch.cuda.max_memory_allocated() / 2**20)
                value = float(loss.detach())
                del result, loss, details
            records.append(dict(arm=name, mode="train_forward_backward", epoch_index=epoch,
                batch_size=2, seconds=elapsed, median_seconds=statistics.median(elapsed),
                peak_cuda_mib=max(peaks), finite_gradients=True, loss=value))
            print("BENCHMARK " + json.dumps(records[-1]), flush=True)
            model.zero_grad(set_to_none=True)
        model.eval()
        elapsed, peaks = [], []
        with torch.no_grad():
            for repeat in range(6):
                model.memory.clear(); model.memory_queue.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = model(imgs=val_images, metas=val_metas)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                assert len(result["future_occ_grids"]) == 6
                metric = HorizonMetrics(); metric.add(result, val_metas)
                if repeat:
                    elapsed.append(seconds)
                    peaks.append(torch.cuda.max_memory_allocated() / 2**20)
                del result
        records.append(dict(arm=name, mode="seven_horizon_inference", batch_size=1,
            seconds=elapsed, median_seconds=statistics.median(elapsed), peak_cuda_mib=max(peaks)))
        print("BENCHMARK " + json.dumps(records[-1]), flush=True)
        del model, objective
        gc.collect(); torch.cuda.empty_cache()
    comparisons = []
    for old, new in zip(records[:3], records[3:]):
        comparisons.append(dict(mode=old["mode"], epoch_index=old.get("epoch_index"),
            speedup=old["median_seconds"] / new["median_seconds"],
            time_reduction_percent=100 * (1 - new["median_seconds"] / old["median_seconds"]),
            peak_memory_reduction_percent=100 * (1 - new["peak_cuda_mib"] / old["peak_cuda_mib"])))
    runtime.atomic_json(ROOT / "validation.json", dict(status="passed", records=records,
        comparisons=comparisons, training_updates=0, checkpoint_written=False,
        manifests=runtime.audit_manifests(cfg), timing_scope="same GPU; preloaded real data; warmup excluded; synchronized wall time; no optimizer step",
        decision="joint probability argmax", cuda=torch.version.cuda,
        pytorch=torch.__version__, gpu=torch.cuda.get_device_name()))
    print("PASSED real batch-2 training and seven-horizon inference", flush=True)


if __name__ == "__main__":
    main()
