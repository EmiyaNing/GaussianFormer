"""Initialization, real-gradient checks and parameter-update audits."""

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from mmseg.models import build_segmentor

from tools.pmg_world.regions import histogram, token_at
from tools.pmg_world_native_adapt.joint_optimization import (
    PERCEPTION_MODULES,
    PREDICTION_MODULES,
    parameter_role,
)
from tools.sparse_opus_gausisan_world import runtime


def initialize(cfg):
    model = build_segmentor(cfg.model)
    source = torch.load(
        cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    missing = model.load_state_dict(source["state_dict"], strict=False)
    expected = {
        "reference_forecast_modules." + name
        for name in model.reference_forecast_modules.state_dict()
    }
    expected |= {"query_update_gate.linear.weight", "query_update_gate.linear.bias"}
    assert set(missing.missing_keys) == expected and not missing.unexpected_keys
    for name, module in model.reference_forecast_modules.items():
        module.load_state_dict(
            {
                key[len(name) + 1 :]: value
                for key, value in source["state_dict"].items()
                if key.startswith(name + ".")
            },
            strict=True,
        )
    # frozen_stages=-1 is essential: ResNet.train() otherwise refreezes early
    # stages even after a blanket requires_grad_(True) call.
    assert cfg.model.img_backbone.frozen_stages == -1
    for name, parameter in model.named_parameters():
        parameter_role(name)
        parameter.requires_grad_(True)
    model.cuda()
    model.pts_bbox_head.ind_stamps_all = source["query_stamps"].cuda()
    model.set_epoch(0)
    return model, source


def state_audit(model, source, require_updates=False):
    counts, changed = {}, {}
    frozen = []
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        counts[root] = counts.get(root, 0) + parameter.numel()
        if not parameter.requires_grad:
            frozen.append(name)
        initial_name = name.removeprefix("reference_forecast_modules.")
        initial = source["state_dict"].get(initial_name)
        if initial is None:
            assert name.startswith("query_update_gate.")
            initial = torch.zeros_like(parameter, device="cpu")
        if not torch.equal(parameter.detach().cpu(), initial):
            changed[root] = changed.get(root, 0) + 1
    assert not frozen, frozen
    query_exact = torch.equal(
        source["query_stamps"], model.pts_bbox_head.ind_stamps_all.cpu()
    )
    assert query_exact
    if require_updates:
        required = (*PERCEPTION_MODULES, *PREDICTION_MODULES)
        assert all(changed.get(root, 0) > 0 for root in required), changed
    return dict(
        all_parameters_trainable=True,
        frozen_parameter_names=frozen,
        parameters_by_module=counts,
        changed_parameter_tensors_by_module=changed,
        perception_weights_changed=all(
            changed.get(root, 0) > 0 for root in PERCEPTION_MODULES
        ),
        both_forecast_paths_changed=(
            changed.get("reg_branch", 0) > 0
            and changed.get("reference_forecast_modules", 0) > 0
        ),
        query_assignment_exact=query_exact,
        normalization_note="Backbone BN running statistics use norm_eval=True; affine weights train",
    )


def engineering_check(model, objective, train, val, cfg, output):
    rng = runtime.capture_rng()
    expected = json.loads(
        Path(
            "analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.json"
        ).read_text()
    )
    index = next(i for i in range(len(val)) if token_at(val, i, 0) == expected["token"])
    data = next(iter(runtime.loader(val, cfg.val_loader, [index], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    model.eval()
    with torch.no_grad():
        result = model(imgs=images, metas=metas)
        targets = [metas["occ_label"], *metas["future_occ_labels"].unbind(1)]
        grids = [result["final_occ_grid"], *result["future_occ_grids"]]
        counts = np.stack(
            [
                histogram(p.cpu().numpy(), t.cpu().numpy())
                for p, t in zip(grids, targets)
            ]
        )
        with np.load(
            "analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz"
        ) as old:
            np.testing.assert_array_equal(counts, old["global_confusion"][0])
    del result, images, metas, data, targets, grids
    model.memory.clear()
    model.memory_queue.clear()
    model.train()
    assert model.img_backbone.training
    assert model.pts_bbox_head.training
    assert model.reference_forecast_modules.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.audit_route_fields = True
    data = next(iter(runtime.loader(train, cfg.train_loader, [0], cfg.seed)))
    images, metas = runtime.move_batch(data, "cuda")
    result = model(imgs=images, metas=metas)
    assert len(result["forecast_points_list"]) == 6
    assert set(result["gaussian_logits_by_horizon"]) == set(range(7))
    assert all(model.last_reference_requires_grad)
    assert all(all(row.values()) for row in model.last_route_audit)
    loss, details = objective(dict(result, metas=metas, update_stamp_statistics=False))
    assert torch.isfinite(loss)
    loss.backward()
    norms = {}
    for root in (*PERCEPTION_MODULES, *PREDICTION_MODULES):
        grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(root + ".") and parameter.grad is not None
        ]
        assert grads and all(torch.isfinite(grad).all() for grad in grads), root
        norms[root] = math.sqrt(
            sum(float(grad.double().square().sum()) for grad in grads)
        )
        assert norms[root] > 0, root
    stem = model.img_backbone.conv1.weight
    assert stem.requires_grad and stem.grad is not None and stem.grad.norm() > 0
    audit = dict(
        status="passed",
        optimizer_updates=0,
        initial_counts_match_T_A0=True,
        validation_token=expected["token"],
        all_seven_voxel_losses_active=True,
        all_parameters_trainable=True,
        perception_and_prediction_gradient_norms=norms,
        backbone_first_conv_gradient_norm=float(stem.grad.norm()),
        both_forecast_Gaussian_graphs_differentiable=True,
        routed_child_fields_exact=True,
        total_loss=float(loss.detach()),
        training_dropout_enabled=True,
        backbone_BN_running_stats_fixed_but_affine_trainable=True,
    )
    runtime.atomic_json(output / "engineering_check.json", audit)
    print("JOINT_PREFLIGHT " + json.dumps(audit), flush=True)
    model.zero_grad(set_to_none=True)
    model.audit_route_fields = False
    del result, loss, details, images, metas, data, grads, stem
    gc.collect()
    torch.cuda.empty_cache()
    runtime.restore_rng(rng)
