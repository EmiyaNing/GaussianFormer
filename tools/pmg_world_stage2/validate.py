"""Real batch-two checks: inherited initialization, residual parity and dense-time gradients."""

import gc
import hashlib
from pathlib import Path

import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from loss.pmg_time_supervision_loss import future_horizon_weight
from tools.sparse_opus_gausisan_world import runtime


ARMS = ("A0_control", "M1_motion", "T_A0_dense", "T_C1_dense")


def original_fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("motion_correction."):
            continue
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    root = Path("analysis/pmg_world_stage2")
    root.mkdir(exist_ok=True, parents=True)
    cfg0 = Config.fromfile(
        "config/sparse_opus_gaussian_world/pmg_stage2_A0_control_4096_1024_10e.py"
    )
    runtime.configure_runtime(cfg0)
    train = OPENOCC_DATASET.build(cfg0.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg0.val_dataset_config)
    train_batch = custom_collate_fn_temporal([train[0], train[1000]])
    val_batch = custom_collate_fn_temporal([validation[0]])
    expected = __import__("json").loads(
        Path("out/pmg_world_stage1/A0/initialization.json").read_text()
    )["initial_model_sha256"]
    records = []
    reference_points = None
    reference_grids = None
    for arm in ARMS:
        cfg = Config.fromfile(
            f"config/sparse_opus_gaussian_world/pmg_stage2_{arm}_4096_1024_10e.py"
        )
        runtime.configure_runtime(cfg)
        model = build_segmentor(cfg.model)
        model.init_weights()
        runtime.load_image_initialization(model, cfg.train_load_from)
        fingerprint = original_fingerprint(model)
        assert fingerprint == expected, (arm, fingerprint, expected)
        model.cuda()
        objective = OPENOCC_LOSS.build(cfg.loss).cuda()
        for epoch in (0, 5):
            model.set_epoch(epoch)
            model.train()
            model.zero_grad(set_to_none=True)
            images, metas = runtime.move_batch(train_batch, "cuda")
            torch.cuda.reset_peak_memory_stats()
            result = model(imgs=images, metas=metas)
            assert len(result["forecast_points_list"]) == epoch + 1
            actual = list(result["gaussian_logits_by_horizon"])
            expected_horizons = (
                list(range(epoch + 2))
                if arm.startswith("T_")
                else [h for h in (0, 2, 4, 6) if h <= epoch + 1]
            )
            assert actual == expected_horizons, (arm, epoch, actual)
            loss, details = objective(
                dict(result, metas=metas, global_iter=epoch * 2048)
            )
            assert torch.isfinite(loss)
            (loss / 2).backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            if arm == "M1_motion":
                norm = model.motion_correction[-1].weight.grad.norm()
                assert norm > 0, "zero-initialized residual head has no learning signal"
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            records.append(
                dict(
                    arm=arm,
                    epoch_index=epoch,
                    batch_size=2,
                    loss=float(loss.detach()),
                    gradient_norm=float(norm),
                    loss_horizons=actual,
                    peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                    base_initial_fingerprint=fingerprint,
                    dense_future_total_weight=(
                        future_horizon_weight(epoch + 1) * (epoch + 1)
                        if arm.startswith("T_")
                        else None
                    ),
                )
            )
            del result, loss, details, images, metas
            model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            images, metas = runtime.move_batch(val_batch, "cuda")
            result = model(imgs=images, metas=metas)
            points = [p.cpu().clone() for p in result["forecast_points_list"]]
            grids = [
                g.cpu().clone()
                for g in [result["final_occ_grid"], *result["future_occ_grids"]]
            ]
            assert len(grids) == 7
            if arm == "A0_control":
                reference_points, reference_grids = points, grids
            if arm in ("M1_motion", "T_A0_dense"):
                assert all(
                    torch.equal(a, b) for a, b in zip(points, reference_points)
                ), arm
                assert all(
                    torch.equal(a, b) for a, b in zip(grids, reference_grids)
                ), arm
            del result, images, metas
        del model, objective
        gc.collect()
        torch.cuda.empty_cache()
        runtime.atomic_json(
            root / "validation.json",
            dict(status="running", records=records, training_updates=0),
        )
        print("VALIDATED", arm, flush=True)
    for horizon in range(1, 7):
        assert (
            abs(future_horizon_weight(horizon) * horizon - 0.5 * (horizon // 2)) < 1e-12
        )
    runtime.atomic_json(
        root / "validation.json",
        dict(
            status="passed",
            records=records,
            training_updates=0,
            zero_residual_exact_prediction_parity=True,
            sparse_dense_a0_initial_parity=True,
            inherited_initialization_matches_stage1=True,
            future_voxel_total_weight_preserved=True,
            scope="batch=2 training forward/backward at epochs 1 and 6; seven-horizon single-sample evaluation",
        ),
    )


if __name__ == "__main__":
    main()
