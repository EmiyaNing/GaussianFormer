"""Explicit GPU regression against D; never update checkpoints on disk."""

import argparse
import gc
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world.runtime import (
    atomic_json,
    capture_rng,
    digest,
    loader,
    move_batch,
    restore_rng,
)


OLD = "config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_continue_4096_1024_15e.py"
NEW = "config/sparse_opus_gausisan_world/sparse_opus_gausisan_world_4096_1024_15e.py"


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_tree(item) for item in value]
    return value


def compare_tree(left, right, *, gradient=False):
    if torch.is_tensor(left):
        torch.testing.assert_close(
            left,
            right,
            rtol=1e-4 if gradient else 1e-6,
            atol=5e-5 if gradient else 2e-6,
        )
        return (
            float((left - right).abs().max())
            if left.numel() and left.is_floating_point()
            else 0.0
        )
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("output/gradient names differ")
        return max(
            (compare_tree(left[key], right[key], gradient=gradient) for key in left),
            default=0.0,
        )
    if isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("horizon count differs")
        return max(
            (compare_tree(a, b, gradient=gradient) for a, b in zip(left, right)),
            default=0.0,
        )
    if left != right:
        raise ValueError("non-tensor outputs differ")
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="out/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024/D_balanced_conditional_ce/latest.pth",
    )
    parser.add_argument(
        "--output",
        default="analysis/sparse_opus_gausisan_world_integration/gpu_parity.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configs = {"D": Config.fromfile(OLD), "integrated": Config.fromfile(NEW)}
    source_hash = digest(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    models = {}
    for name, cfg in configs.items():
        set_random_seed(17)
        models[name] = build_segmentor(cfg.model)
        models[name].init_weights()
        models[name].load_state_dict(saved["state_dict"], strict=True)
    dataset = OPENOCC_DATASET.build(configs["integrated"].train_dataset_config)
    checks = []
    keys = (
        "current_pred_points",
        "current_pred_logits",
        "temporal_pred_points",
        "temporal_pred_logits",
        "future_pred_points",
        "future_pred_logits",
        "forecast_features_list",
        "pred_traj",
        "query_stamps",
        "gaussian_logits_by_horizon",
        "final_occ_grid",
        "future_occ_grids",
    )
    for epoch_index in (0, 5, 6, 14):
        dataset.set_epoch(epoch_index)
        data = next(
            iter(
                loader(
                    dataset,
                    configs["integrated"].train_loader,
                    order=[0, 1],
                    seed=17 + epoch_index,
                )
            )
        )
        set_random_seed(1700 + epoch_index)
        rng = capture_rng()
        snapshots = {}
        for name in ("D", "D_repeat", "integrated"):
            model = models["D" if name == "D_repeat" else name]
            model.cuda()
            model.load_state_dict(saved["state_dict"], strict=True)
            model.train()
            model.set_epoch(epoch_index)
            restore_rng(rng)
            images, metas = move_batch(data, "cuda")
            loss_fn = OPENOCC_LOSS.build(
                configs["D" if name == "D_repeat" else name].loss
            ).cuda()
            torch.cuda.reset_peak_memory_stats()
            result = model(imgs=images, metas=metas)
            loss, details = loss_fn(
                dict(result, metas=metas, global_iter=epoch_index * 2048)
            )
            loss.backward()
            gradients = {
                key: (
                    parameter.grad.detach().cpu().clone()
                    if parameter.grad is not None
                    else None
                )
                for key, parameter in model.named_parameters()
            }
            snapshots[name] = dict(
                loss=loss.detach().cpu(),
                details=cpu_tree(details),
                outputs=cpu_tree({key: result[key] for key in keys}),
                gradients=gradients,
                supervised_horizons=sorted(result["gaussian_logits_by_horizon"]),
                peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            print(
                f"GPU_PARITY epoch={epoch_index+1} implementation={name} loss={float(loss.detach()):.7f}",
                flush=True,
            )
            model.zero_grad(set_to_none=True)
            del result, loss, details, gradients, images, metas, loss_fn
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        left, right = snapshots["D"], snapshots["integrated"]
        torch.testing.assert_close(left["loss"], right["loss"], rtol=1e-6, atol=2e-6)
        forward_error = compare_tree(left["outputs"], right["outputs"])
        details_error = compare_tree(left["details"], right["details"])
        # Establish a same-implementation CUDA baseline before interpreting
        # BF16/atomic-scatter backward discrepancies as a refactoring change.
        repeat = snapshots["D_repeat"]
        repeat_error = 0.0
        gradient_error = 0.0
        maximum_relative_l2 = 0.0
        gradient_squared_error = 0.0
        repeat_squared_error = 0.0
        gradient_squared_norm = 0.0
        gradient_records = []
        for key, a in left["gradients"].items():
            b = right["gradients"][key]
            c = repeat["gradients"][key]
            if a is None:
                if b is not None or c is not None:
                    raise ValueError(f"gradient presence differs: {key}")
                continue
            difference = float((a - b).abs().max())
            noise = float((a - c).abs().max())
            relative_l2 = float((a - b).norm() / a.norm().clamp_min(1e-12))
            reference_l2 = float((a - c).norm() / a.norm().clamp_min(1e-12))
            if not torch.isfinite(b).all():
                raise ValueError(f"non-finite integrated gradient: {key}")
            gradient_squared_error += float((a.double() - b.double()).square().sum())
            repeat_squared_error += float((a.double() - c.double()).square().sum())
            gradient_squared_norm += float(a.double().square().sum())
            gradient_records.append(
                dict(
                    parameter=key,
                    maximum_absolute_error=difference,
                    relative_l2=relative_l2,
                    repeated_D_relative_l2=reference_l2,
                )
            )
            gradient_error = max(gradient_error, difference)
            repeat_error = max(repeat_error, noise)
            maximum_relative_l2 = max(maximum_relative_l2, relative_l2)
        global_relative_l2 = (
            gradient_squared_error / max(gradient_squared_norm, 1e-24)
        ) ** 0.5
        repeat_global_relative_l2 = (
            repeat_squared_error / max(gradient_squared_norm, 1e-24)
        ) ** 0.5
        if global_relative_l2 > 5e-3 or maximum_relative_l2 > 1e-2:
            raise ValueError(
                f"gradient discrepancy exceeds mixed-precision tolerance: global={global_relative_l2}; per_parameter={maximum_relative_l2}"
            )
        for key in ("final_occ_grid", "future_occ_grids"):
            # Predictions must match exactly even if a CUDA reduction differs by a few ulps.
            def exact(a, b):
                if torch.is_tensor(a):
                    return torch.equal(a, b)
                return all(exact(x, y) for x, y in zip(a, b))

            if not exact(left["outputs"][key], right["outputs"][key]):
                raise ValueError("occupancy predictions changed")
        checks.append(
            dict(
                epoch=epoch_index + 1,
                supervised_horizons=right["supervised_horizons"],
                loss=float(right["loss"]),
                maximum_output_absolute_error=forward_error,
                maximum_loss_component_absolute_error=details_error,
                maximum_gradient_absolute_error=gradient_error,
                repeated_D_maximum_gradient_absolute_error=repeat_error,
                maximum_per_parameter_gradient_relative_l2=maximum_relative_l2,
                global_gradient_relative_l2=global_relative_l2,
                repeated_D_global_gradient_relative_l2=repeat_global_relative_l2,
                largest_relative_gradient_differences=sorted(
                    gradient_records, key=lambda x: x["relative_l2"], reverse=True
                )[:10],
                occupancy_grids_exact=True,
                parameter_gradient_count=len(right["gradients"]),
                peak_cuda_mib=right["peak_cuda_mib"],
            )
        )
        del snapshots, data
        gc.collect()
    if digest(args.checkpoint) != source_hash:
        raise ValueError("read-only validation changed the D checkpoint")
    atomic_json(
        Path(args.output),
        dict(
            status="passed",
            checkpoint=args.checkpoint,
            checkpoint_sha256=source_hash,
            source_checkpoint_unchanged=True,
            stages=checks,
            forward_rtol=1e-6,
            forward_atol=2e-6,
            gradient_rule="Forward/loss atol=2e-6 rtol=1e-6; grids exact. BF16/CUDA parameter gradients: global relative L2 <=0.005 and each parameter <=0.01; repeated-D backward noise reported separately, no claim of bitwise backward identity.",
            no_checkpoint_updates=True,
        ),
    )
    print("GPU_PARITY_ALL_PASSED", flush=True)


if __name__ == "__main__":
    main()
