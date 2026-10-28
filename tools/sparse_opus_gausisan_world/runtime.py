"""Shared reproducible runtime utilities for the integrated D training protocol."""

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.utils import custom_collate_fn_temporal


def atomic_json(path, value):
    """Replace complete JSON records atomically, keeping real UTF-8 newlines."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temp, path)


def atomic_save(path, value):
    """Keep the previous checkpoint until the replacement is fully serialized."""
    temp = Path(str(path) + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def move_batch(data, device):
    result = {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in data.items()
    }
    return result.pop("img"), result


def load_image_initialization(model, path):
    """Load backbone/FPN tensors only; world and Gaussian modules stay random."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    expected = model.state_dict()
    selected = {}
    for key, value in source.items():
        name = key.removeprefix("module.")
        for before, after in (("backbone.", "img_backbone."), ("neck.", "img_neck.")):
            if name.startswith(before):
                name = after + name[len(before) :]
                break
        if not name.startswith(("img_backbone.", "img_neck.")):
            continue
        if name not in expected or expected[name].shape != value.shape:
            raise ValueError(f"incompatible image initialization tensor: {name}")
        selected[name] = value
    required = {k for k in expected if k.startswith(("img_backbone.", "img_neck."))}
    missing = required - selected.keys()
    if missing:
        raise ValueError(f"missing image initialization: {sorted(missing)[:10]}")
    incompatible = model.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys or any(
        k in required for k in incompatible.missing_keys
    ):
        raise ValueError("initialization audit differs from load result")
    return dict(
        checkpoint=path,
        sha256=digest(path),
        loaded_tensors=len(selected),
        scope="backbone_and_fpn_only",
        randomly_initialized_world_tensors=len(incompatible.missing_keys),
    )


def make_optimizer(model, cfg):
    """Preserve D's parameter order and longest-matching per-parameter options."""
    options = cfg.optimizer["optimizer"]
    keys = cfg.optimizer.get("paramwise_cfg", {}).get("custom_keys", {})
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matched = next(
            (key for key in sorted(keys, key=len, reverse=True) if key in name), None
        )
        setting = keys.get(matched, {})
        lr = options["lr"] * setting.get("lr_mult", 1.0)
        groups.append(
            dict(
                params=[parameter],
                lr=lr,
                base_lr=lr,
                weight_decay=options["weight_decay"] * setting.get("decay_mult", 1.0),
            )
        )
    return torch.optim.AdamW(groups)


def set_lr(optimizer, step, total, warmup, minimum):
    """Apply the original linear warm-up and cosine ratio before each update."""
    if step < warmup:
        factor = 1 / 3 + (2 / 3) * step / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        factor = minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def capture_rng():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def loader(dataset, cfg, order=None, seed=17):
    """Use an explicit sample order and a separate DataLoader RNG generator."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=order,
        shuffle=False,
        collate_fn=custom_collate_fn_temporal,
        pin_memory=True,
        generator=generator,
    )


def configure_runtime(cfg):
    """Use D's original precision, deterministic cuDNN and random seed settings."""
    from mmengine.runner import set_random_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gaussian rendering")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_random_seed(cfg.seed)


def model_fingerprint(model):
    """Hash all initialized tensors, without serializing a temporary checkpoint."""
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        fingerprint.update(name.encode())
        fingerprint.update(str(tuple(tensor.shape)).encode())
        fingerprint.update(str(tensor.dtype).encode())
        fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return fingerprint.hexdigest()


def should_restore_query_statistics(resume_rng, next_iter, phase):
    """Restore within an epoch; reset normally when starting the next epoch."""
    return resume_rng is not None and (next_iter > 0 or phase == "evaluate")


def audit_manifests(cfg):
    """Verify immutable split manifests and the data sources they identify."""
    hashes = {}
    source_hashes = {}
    for split in ("train", "val"):
        path = cfg[f"{split}_dataset_config"]["selection_manifest"]
        hashes[split] = digest(path)
        manifest = json.loads(Path(path).read_text())
        for source_path, expected in manifest["source_files"].items():
            if source_path not in source_hashes:
                source_hashes[source_path] = digest(source_path)
            if source_hashes[source_path] != expected:
                raise ValueError(f"dataset source changed: {source_path}")
    return hashes


def load_evaluation_checkpoint(model, checkpoint):
    """Load original D or integrated weights, including its timestamp assignment."""
    from .protocol import checkpoint_epoch_index

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    epoch_index = checkpoint_epoch_index(checkpoint)
    model.set_epoch(epoch_index)
    model.pts_bbox_head.ind_stamps_all = checkpoint["query_stamps"].to(
        model.pc_range.device
    )
    model.pts_bbox_head.reset_mask()
    return epoch_index + 1
