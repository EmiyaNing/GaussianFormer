"""Measure current/future voxel gradient alignment without optimization."""

import gc
from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime


def main():
    root = Path("analysis/pmg_world_stage2")
    records = []
    for arm in ("A0", "C1"):
        cfg = Config.fromfile(
            f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py"
        )
        runtime.configure_runtime(cfg)
        dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
        model = build_segmentor(cfg.model).cuda()
        saved = torch.load(
            f"out/pmg_world_stage1/{arm}/best.pth",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        epoch = runtime.load_evaluation_checkpoint(model, saved)
        del saved
        objective = OPENOCC_LOSS.build(cfg.loss).cuda()
        groups = {
            "gaussian_attributes": [],
            "current_semantic_readout": [],
            "forecast_semantic_readout": [],
        }
        for name, parameter in model.named_parameters():
            if "attribute_branches.5." in name:
                groups["gaussian_attributes"].append(parameter)
            elif "encoder.layers.5.cls_branch." in name:
                groups["current_semantic_readout"].append(parameter)
            elif name.startswith("cls_branch."):
                groups["forecast_semantic_readout"].append(parameter)
        assert all(groups.values())
        parameters = [p for values in groups.values() for p in values]
        model.train()
        for index, batch in zip(
            (0, 1000),
            runtime.loader(
                dataset, SimpleNamespace(batch_size=1, num_workers=0), (0, 1000)
            ),
        ):
            images, metas = runtime.move_batch(batch, "cuda")
            result = model(imgs=images, metas=metas)
            losses = {}
            for horizon, logits in result["gaussian_logits_by_horizon"].items():
                labels = (
                    metas["occ_label"]
                    if horizon == 0
                    else metas["future_occ_labels"][:, horizon - 1]
                )
                losses[horizon] = objective.occupancy.loss_voxel(
                    [logits], metas["occ_xyz"].flatten(1, 3), labels.flatten(1), None
                ) * (1.0 if horizon == 0 else 0.5)
            current = torch.autograd.grad(
                losses[0], parameters, retain_graph=True, allow_unused=True
            )
            futures = [(h, losses[h]) for h in (2, 4, 6)] + [
                ("sum", sum(losses[h] for h in (2, 4, 6)))
            ]
            for horizon, loss in futures:
                future = torch.autograd.grad(
                    loss, parameters, retain_graph=True, allow_unused=True
                )
                offset = 0
                for group, values in groups.items():
                    n = len(values)
                    a = current[offset : offset + n]
                    b = future[offset : offset + n]
                    offset += n
                    na = sum(
                        (g.float().square().sum() for g in a if g is not None),
                        torch.tensor(0.0, device="cuda"),
                    )
                    nb = sum(
                        (g.float().square().sum() for g in b if g is not None),
                        torch.tensor(0.0, device="cuda"),
                    )
                    dot = sum(
                        (
                            (x.float() * y.float()).sum()
                            for x, y in zip(a, b)
                            if x is not None and y is not None
                        ),
                        torch.tensor(0.0, device="cuda"),
                    )
                    norm_a = float(na.sqrt())
                    norm_b = float(nb.sqrt())
                    records.append(
                        dict(
                            arm=arm,
                            checkpoint_epoch=epoch,
                            train_index=index,
                            future_horizon=horizon,
                            group=group,
                            current_norm=norm_a,
                            future_norm=norm_b,
                            cosine=(
                                float(dot / (na * nb).sqrt())
                                if norm_a > 0 and norm_b > 0
                                else None
                            ),
                            current_voxel_loss=float(losses[0].detach()),
                            future_voxel_loss=float(loss.detach()),
                        )
                    )
            del result, losses, current, future, images, metas, batch, futures, loss
            model.zero_grad(set_to_none=True)
            print("GRADIENTS", arm, index, flush=True)
        del model, objective, dataset
        gc.collect()
        torch.cuda.empty_cache()
    runtime.atomic_json(
        root / "gradient_diagnosis.json",
        dict(
            status="completed",
            records=records,
            training_updates=0,
            note="Two fixed augmented training samples per arm; voxel losses only, dropout in training mode, no optimizer updates. Negative cosine is a diagnostic correlation, not proof of a harmful optimization mechanism.",
        ),
    )


if __name__ == "__main__":
    main()
