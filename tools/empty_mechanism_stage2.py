"""Stage-two, common-step loss-source interventions. Original runs stay immutable."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.empty_mechanism_stage1 import (
    Frame, build_network, network_output, read, write_json, file_sha256,
    build_loader, to_cuda, seed_all, CONFIG,
)

RUN = ROOT / "work_dirs/empty_mechanism_stage2"
EPOCHS = [1, 3, 5]
RADII = [1e-4, 1e-3]
SOURCE_NAMES = ["set_positive", "set_negative", "ce_occupied", "ce_empty", "lovasz"]
CONDITIONS = ["combo_" + "".join(map(str, bits)) for bits in itertools.product([0, 1], repeat=3)]
CONDITIONS += SOURCE_NAMES[:4] + ["B2_no_empty_ce", "B3_no_set_no_empty_ce",
                                  "C_half_set", "C_project_set", "C_norm_set"]


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "protocol.json"
    if path.exists():
        return read(path)
    old = read(ROOT / "work_dirs/empty_mechanism_stage1/protocol.json")
    p = dict(config=CONFIG, epochs=EPOCHS, radii=RADII, primary_radius=.001,
             checkpoint=old["checkpoint"], scene_samples=old["scene_samples"],
             pairs=[dict(source=a, heldout=b) for a, b in
                    zip(old["scene_samples"]["train"][:8], old["scene_samples"]["val"][:8])],
             conditions=CONDITIONS, source_names=SOURCE_NAMES,
             step_rule="one eta from joint gradient per frame and radius; never renormalize components",
             ce_rule="shared original weighted denominator; original factor 10",
             set_scope="last stage only unless explicitly no_all_set",
             domain="default camera mask and valid labels; original full-domain set targets",
             projection="only last-stage set gradient; double precision output-amplitude tangent",
             spatial_bins=[0., .25, .5, .75, 1.0000001],
             spatial_strata="target class x train-derived amplitude/scale/support quartiles",
             tolerances=dict(loss_atol=2e-6, loss_rtol=2e-6, grad_atol=2e-7, grad_rtol=2e-5,
                             renderer_atol=2e-4, renderer_rtol=2e-4),
             bootstrap=1000, seed=928,
             adamw=dict(pairs=8, epochs=EPOCHS, accumulation=4, clip_norm=35.,
                        conditions=["B0", "B1", "B2", "B3", "half_set", "project_set",
                                    "norm_set", "zero_current", "no_all_set"],
                        source="training scenes only", modes=["classifier", "full"]),
             short=dict(epochs=[1, 3], optimizer_steps=[1, 16, 64], primary_step=64,
                        accumulation=4, evaluation_scenes=32,
                        conditions=["B0", "B1", "B2", "B3", "half_set",
                                    "project_set", "norm_set", "no_all_set"]),
             full_training_gate="candidate must improve 64-step native mIoU in both branches, "
                                "without merely trading FN for FP; otherwise defer method training",
             plan_sha256=file_sha256(ROOT / "plans/semantic_gaussian_empty_mechanism_stage2_design.md"),
             original_summary_sha256=file_sha256(ROOT / "work_dirs/empty_mechanism_stage1/summary.json"))
    write_json(path, p)
    return p


def ce_parts(scores, labels, class_weights, multiplier=10.):
    """Exactly split weighted CrossEntropyLoss(mean), including a common denominator."""
    if scores.ndim == 3:
        scores = scores[0].T
    valid = (labels >= 0) & (labels < scores.shape[-1])
    z, y = scores[valid], labels[valid]
    w = class_weights.to(z)[y]
    denominator = w.sum()
    terms = -F.log_softmax(z, dim=-1).gather(1, y[:, None])[:, 0] * w
    if not len(y):
        raise ValueError("No valid labels")
    return (multiplier * terms[y < 17].sum() / denominator,
            multiplier * terms[y == 17].sum() / denominator)


def set_parts(frame, raw):
    module = frame.modules[0]
    u = raw[0, frame.point_valid]
    target = F.one_hot(frame.assigned, 17).to(u)
    q = u.sigmoid()
    pt = q * target + (1-q) * (1-target)
    alpha = module.focal_alpha * target + (1-module.focal_alpha) * (1-target)
    terms = alpha * (1-pt).pow(module.focal_gamma) * F.binary_cross_entropy_with_logits(
        u, target, reduction="none")
    terms = terms * frame.set_weights[:, None] * frame.set_multiplier / max(len(u), 1)
    return (terms * target).sum(), (terms * (1-target)).sum()


def project_set(raw, gradient):
    d = raw.double().sigmoid()
    g = gradient.double()
    denominator = d.square().sum(-1, keepdim=True)
    if torch.any(denominator <= 1e-30):
        raise FloatingPointError("Near-zero amplitude derivative; do not silently change projection")
    projected = g - d * (d*g).sum(-1, keepdim=True) / denominator
    error = float((d*projected).sum(-1).abs().max())
    return projected.to(gradient), error


def loss_sources(frame, raw):
    scores = frame.render(F.softplus(raw))
    positive, negative = set_parts(frame, raw)
    ceo, cee = ce_parts(scores, frame.y, frame.modules[1].class_weights,
                        frame.modules[1].loss_voxel_ce_weight)
    _, lov = frame.field_objectives(scores)
    return dict(zip(SOURCE_NAMES, [positive, negative, ceo, cee, lov])), scores


def source_gradients(frame, raw):
    losses, scores = loss_sources(frame, raw)
    gradients = {name: torch.autograd.grad(value, raw, retain_graph=True)[0].detach()
                 for name, value in losses.items()}
    original = frame.objectives(raw)
    checks = {}
    for name, value in [("set", losses["set_positive"] + losses["set_negative"]),
                        ("ce", losses["ce_occupied"] + losses["ce_empty"])]:
        reference = original["set_cls" if name == "set" else "voxel_ce"]
        g = torch.autograd.grad(value, raw, retain_graph=True)[0]
        gr = torch.autograd.grad(reference, raw, retain_graph=True)[0]
        torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(g, gr, atol=2e-7, rtol=2e-5)
        checks[name] = dict(value_error=float((value-reference).detach().abs()),
                            gradient_error=float((g-gr).abs().max()))
    joint = sum(gradients.values())
    reference_grad = torch.autograd.grad(original["joint"], raw)[0]
    torch.testing.assert_close(joint, reference_grad, atol=2e-7, rtol=2e-5)
    checks["joint_gradient_error"] = float((joint-reference_grad).abs().max())
    # Real-empty voxels cannot contribute a Lovasz gradient with ignore=17.
    leaf = frame.base.detach().requires_grad_(True)
    _, lov = frame.field_objectives(leaf)
    glov = torch.autograd.grad(lov, leaf)[0]
    checks["lovasz_empty_voxel_gradient"] = float(glov[:, :, frame.y == 17].abs().max())
    assert checks["lovasz_empty_voxel_gradient"] == 0
    return gradients, {k: float(v.detach()) for k, v in losses.items()}, checks


def directions(raw, g):
    s = g["set_positive"] + g["set_negative"]
    ce = g["ce_occupied"] + g["ce_empty"]
    lov = g["lovasz"]
    result = {}
    for bits in itertools.product([0, 1], repeat=3):
        result["combo_" + "".join(map(str, bits))] = bits[0]*s + bits[1]*ce + bits[2]*lov
    for name in SOURCE_NAMES[:4]:
        result[name] = g[name]
    result["B2_no_empty_ce"] = s + g["ce_occupied"] + lov
    result["B3_no_set_no_empty_ce"] = g["ce_occupied"] + lov
    projection, error = project_set(raw, s)
    ratio = float(projection.double().norm() / s.double().norm().clamp_min(1e-30))
    result["C_half_set"] = .5*s + ce + lov
    result["C_project_set"] = projection + ce + lov
    result["C_norm_set"] = ratio*s + ce + lov
    assert list(result) == CONDITIONS
    return result, dict(projection_orthogonality_error=error, set_norm_ratio=ratio)


@torch.no_grad()
def measure(frame, scores, semantic=None, support=None):
    """Native confusion plus fixed cohorts; avoid re-sorting Lovasz for every counterfactual."""
    front = scores[0, :17]
    top, fg = front.max(0)
    pred = scores[0].argmax(0)
    empty = scores[0, 17]
    sp = frame.support if support is None else support
    mass = front.sum(0)
    amp = mass / sp.clamp_min(1e-30)
    share = top / mass.clamp_min(1e-30)
    cohorts = dict(frame.cohorts, occupied=frame.y < 17)
    out = dict(counts=torch.bincount(frame.y*18+pred, minlength=324).reshape(18, 18).cpu().tolist(),
               empty_fn=int(((frame.y < 17) & (pred == 17)).sum()),
               empty_fp=int(((frame.y == 17) & (pred < 17)).sum()),
               correct_to_empty=int((frame.correct & (pred == 17)).sum()),
               rescued=int((frame.recoverable & (pred == frame.y)).sum()),
               foreground_correct=int(((frame.y < 17) & (fg == frame.y)).sum()),
               cohorts={})
    for name, mask in cohorts.items():
        supported = mask & (sp > 1e-12)
        out["cohorts"][name] = dict(n=int(mask.sum()), supported=int(supported.sum()),
            amplitude_sum=float(amp[supported].double().sum()),
            support_sum=float(sp[mask].double().sum()),
            share_sum=float(share[mask].double().sum()),
            margin_sum=float((top-empty)[mask].double().sum()),
            empty_sum=float(empty[mask].double().sum()),
            top_sum=float(top[mask].double().sum()))
    if semantic is not None:
        a = semantic.sum(-1)
        out["gaussian_amplitude_mean"] = float(a.double().mean())
    return out


def spatial_profile(frame, g):
    """Native backward supplies exact summed spatial weights without a dense query x G matrix."""
    payload = torch.zeros_like(frame.args[2], requires_grad=True)
    response = frame.aggregate(payload)[0]
    total = torch.autograd.grad(response.sum(), payload, retain_graph=True)[0][0, :-1, 0]
    empty = torch.autograd.grad(response[frame.y == 17].sum(), payload)[0][0, :-1, 0]
    valid = total > 1e-12
    h = empty / total.clamp_min(1e-30)
    assert h[valid].min() >= -1e-6 and h[valid].max() <= 1.00001
    target = torch.full_like(total, -1, dtype=torch.long)
    target[frame.point_valid] = frame.assigned
    a = frame.sem.sum(-1)[0]
    size = frame.args[3][0, :-1].prod(-1).pow(1/3)
    arr = dict(h=h.detach().cpu().numpy(), support=total.detach().cpu().numpy(),
               amplitude=a.detach().cpu().numpy(), size=size.detach().cpu().numpy(),
               target=target.cpu().numpy(), valid=valid.detach().cpu().numpy())
    d = frame.u.sigmoid()
    for name in SOURCE_NAMES:
        arr["amplitude_derivative_" + name] = (-(d*g[name]).sum(-1)[0]).cpu().numpy()
    arr["valid"] &= arr["target"] >= 0
    return arr


def local(epoch, limit=0, pilot=False):
    p = initialize(); seed_all(928)
    network, mods = build_network(epoch)
    dest = RUN / ("pilot" if pilot else "local") / f"epoch{epoch}"
    dest.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for split in ["train", "val"]:
        selected = p["scene_samples"][split]
        if limit:
            selected = selected[:limit]
        _, loader, _ = build_loader(split, [m["index"] for m in selected])
        for n, (meta, batch) in enumerate(zip(selected, loader)):
            path = dest / f'{split}_{meta["index"]:04d}.json'
            if path.exists():
                continue
            batch = to_cuda(batch)
            with torch.no_grad():
                output = network_output(network, batch)
            frame = Frame(network, output, batch, mods)
            raw = frame.u.clone().requires_grad_(True)
            gradients, values, audit = source_gradients(frame, raw)
            dirs, projection_audit = directions(frame.u, gradients)
            spatial = spatial_profile(frame, gradients)
            np.savez_compressed(path.with_suffix(".npz"), **spatial)
            scale = float(frame.u[0, frame.point_valid].double().norm())
            joint_norm = float(dirs["combo_111"].double().norm())
            baseline = measure(frame, frame.base, frame.sem)
            rows = []
            with torch.no_grad():
                for radius in RADII:
                    eta = radius * scale / max(joint_norm, 1e-30)
                    for name, gradient in dirs.items():
                        updated = frame.u - eta*gradient
                        semantic = F.softplus(updated)
                        scores = frame.render(semantic)
                        torch.testing.assert_close(scores[:, 17], frame.base[:, 17], atol=0, rtol=0)
                        rows.append(dict(condition=name, radius=radius, eta=eta,
                            raw_step_relative=float((updated-frame.u).double().norm())/scale,
                            gradient_norm=float(gradient.double().norm()),
                            outcome=measure(frame, scores, semantic)))
            write_json(path, dict(status="complete", epoch=epoch, split=split, frame=meta,
                baseline=baseline, interventions=rows, loss_values=values, audit=audit,
                projection_audit=projection_audit, renderer_error=frame.parity_error,
                protocol_sha256=file_sha256(RUN/"protocol.json"),
                source_sha256=file_sha256(__file__), elapsed_seconds=time.monotonic()-start))
            del raw, output, frame, gradients, dirs, batch
            print(f"stage2 local epoch={epoch} {split} {n+1}/{len(selected)} "
                  f"elapsed={time.monotonic()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["initialize", "local"])
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.stage == "initialize":
        initialize()
    else:
        local(args.epoch, args.limit, args.pilot)


if __name__ == "__main__":
    main()
