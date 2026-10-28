"""Evaluate A0 perception with C1 forecasting, without updating any weights.

The shared ego encoder, Gaussian attribute decoder, renderer calibration and
query timestamp assignment belong to the perception donor: all are required to
preserve its current prediction. Only the six exclusive forecasting modules
come from C1. The model itself is built from the complete C1 architecture config.
"""

import argparse
from collections import Counter
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from tools.pmg_world.evaluate import evaluate
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


FORECAST_MODULES = frozenset(
    (
        "ego_cross_attn",
        "position_encoder",
        "reg_branch",
        "vel_branch",
        "cls_branch",
        "traj_head",
    )
)
PERCEPTION_MODULES = frozenset(
    (
        "img_backbone",
        "img_neck",
        "pts_bbox_head",
        "semantic_loss",
        "plan_head",
        "points_scale_branch",
        "gaussian_head",
    )
)


def splice_states(perception, forecast):
    """Fail closed on new/unclassified modules, missing keys or shape changes."""
    first, second = perception["state_dict"], forecast["state_dict"]
    if first.keys() != second.keys():
        raise ValueError("donor checkpoint key sets differ")
    state, ownership = {}, {}
    for key, value in first.items():
        module = key.split(".", 1)[0]
        if module not in PERCEPTION_MODULES | FORECAST_MODULES:
            raise ValueError(f"unclassified checkpoint tensor: {key}")
        other = second[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"donor tensor shape/dtype mismatch: {key}")
        owner = "C1_forecast" if module in FORECAST_MODULES else "A0_perception"
        state[key] = other if owner == "C1_forecast" else value
        ownership[key] = owner
        if state[key].is_floating_point() and not torch.isfinite(state[key]).all():
            raise ValueError(f"nonfinite donor tensor: {key}")
    return state, ownership


def audit_sources(saved, label):
    changed = [
        path
        for path, expected in saved["identity"]["source_sha256"].items()
        if not Path(path).is_file() or runtime.digest(path) != expected
    ]
    if changed:
        raise ValueError(f"{label} frozen source files changed: {changed}")


@torch.no_grad()
def check_current_parity(model, cfg, perception, dataset, output):
    """Compare native A0 and the C1-architecture splice on fixed validation items."""
    native_cfg = Config.fromfile(cfg.checkpoint_splice.perception_config)
    native = build_segmentor(native_cfg.model).cuda()
    runtime.load_evaluation_checkpoint(native, perception)
    native.eval()
    model.eval()
    records = []
    for index in (0, 341, 682, 1023):
        batch = custom_collate_fn_temporal([dataset[index]])
        images, metas = runtime.move_batch(batch, "cuda")
        for network in (native, model):
            network.memory.clear()
            network.memory_queue.clear()
        base = native(imgs=images, metas=metas)
        mixed = model(imgs=images, metas=metas)
        checks = {
            "current_grid_equal": torch.equal(
                base["final_occ_grid"], mixed["final_occ_grid"]
            ),
            "query_features_equal": torch.equal(
                base["query_feat"], mixed["query_feat"]
            ),
            "perception_points_equal": torch.equal(
                base["all_refine_pts"][-1], mixed["all_refine_pts"][-1]
            ),
            "query_stamps_equal": torch.equal(
                base["query_stamps"], mixed["query_stamps"]
            ),
            "finite_forecast_points": all(
                torch.isfinite(p).all().item() for p in mixed["forecast_points_list"]
            ),
        }
        records.append(dict(index=index, **checks))
        if not all(checks.values()):
            runtime.atomic_json(
                output / "current_parity.json", dict(passed=False, records=records)
            )
            raise ValueError(f"A0 current-output parity failed: {records[-1]}")
        print("CURRENT_PARITY", index, "passed", flush=True)
        del batch, images, metas, base, mixed
    del native
    model.memory.clear()
    model.memory_queue.clear()
    gc.collect()
    torch.cuda.empty_cache()
    runtime.atomic_json(
        output / "current_parity.json", dict(passed=True, records=records)
    )


def scene_statistics(paths, output):
    """Paired scene bootstrap, conditional on the previously selected donors."""
    generator = np.random.default_rng(20261022)
    names, weights, scores = None, None, {}
    for name, path in paths.items():
        with np.load(path) as arrays:
            if names is None:
                names = arrays["scenes"]
                weights = generator.multinomial(
                    len(names), np.full(len(names), 1 / len(names)), size=2000
                )
            if not np.array_equal(names, arrays["scenes"]):
                raise ValueError(
                    "scene ordering differs between splice and donor evaluations"
                )
            scene = arrays["scene_confusion"]
            sampled = (weights @ scene.reshape(len(names), -1)).reshape(2000, 7, 18, 18)
            diagonal = np.diagonal(sampled, axis1=-2, axis2=-1)
            union = sampled.sum(-1) + sampled.sum(-2) - diagonal
            class_iou = (
                np.divide(
                    diagonal,
                    union,
                    out=np.zeros_like(diagonal, dtype=float),
                    where=union > 0,
                )
                * 100
            )
            scores[name] = class_iou[..., :17].mean(-1)
    comparisons = {}
    for donor in ("A0", "C1"):
        delta = scores["splice"] - scores[donor]
        comparisons[f"splice_minus_{donor}"] = dict(
            per_horizon_95ci_pp=np.quantile(delta, [0.025, 0.975], axis=0).T.tolist(),
            primary_95ci_pp=np.quantile(
                delta[:, [0, 2, 4, 6]].mean(-1), [0.025, 0.975]
            ).tolist(),
            future_123_95ci_pp=np.quantile(
                delta[:, [2, 4, 6]].mean(-1), [0.025, 0.975]
            ).tolist(),
        )
    result = dict(
        scene_count=len(names),
        iterations=2000,
        seed=20261022,
        comparisons=comparisons,
        caveat="Conditional on fixed donor checkpoints; no correction for training seeds, checkpoint selection, repeated tuning or multiple comparisons.",
    )
    runtime.atomic_json(output / "scene_statistics.json", result)
    return result


def write_comparison(cfg, hybrid, audit, output):
    settings = cfg.checkpoint_splice
    conditions = {}
    count_paths = {}
    for name, path_key, epoch_key in (
        ("A0", "perception_metrics", "perception_epoch"),
        ("C1", "forecast_metrics", "forecast_epoch"),
    ):
        path = Path(settings[path_key])
        conditions[name] = json.loads(path.read_text())
        if (
            conditions[name]["epoch"] != audit[epoch_key]
            or conditions[name]["num_samples"] != 1024
        ):
            raise ValueError(
                "donor metric file does not match selected checkpoint epoch/sample count"
            )
        count_paths[name] = path.with_name(
            path.name.replace("_metrics.json", "_counts.npz")
        )
    conditions["splice"] = hybrid
    count_paths["splice"] = output / f"epoch_{audit['forecast_epoch']:02d}_counts.npz"
    with np.load(count_paths["A0"]) as base, np.load(count_paths["splice"]) as mixed:
        current_equal = np.array_equal(base["confusion"][0], mixed["confusion"][0])
        current_scene_equal = np.array_equal(
            base["scene_confusion"][:, 0], mixed["scene_confusion"][:, 0]
        )
    if not current_equal or not current_scene_equal:
        raise ValueError("full-validation current confusion differs from native A0")
    statistics = scene_statistics(count_paths, output)
    runtime.atomic_json(
        output / "comparison.json",
        dict(
            status="completed",
            samples=1024,
            training_updates=0,
            conditions=conditions,
            full_current_confusion_equal_to_A0=current_equal,
            full_current_scene_confusion_equal_to_A0=current_scene_equal,
            donor_audit=audit,
            statistics=statistics,
        ),
    )
    rows = [
        dict(
            condition=name,
            seconds=h["seconds"],
            class_index=i,
            class_name=CLASS_NAMES[i],
            iou=h["class_iou"][i],
            miou=h["miou"],
            occupied_iou=h["occupied_iou"],
        )
        for name, record in conditions.items()
        for h in record["horizons"]
        for i in range(18)
    ]
    with (output / "per_horizon_class_iou.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# A0感知器＋C1预测器权重拼接评测",
        "",
        "日期：2026-10-22。没有训练或微调；在固定1024个验证样本上评测。",
        "",
        "本次使用刚完成全时刻监督实验的T-A0和T-C1：A0感知来源为第10轮最佳权重，C1预测来源为第9轮最佳权重。它们均按前10轮的当前/1/2/3秒平均mIoU选中。本报告中的A0/C1均指这两组T模型，不混用原时间监督的旧C1。",
        "",
        "## 拼接边界",
        "",
        "模型按C1架构构建：A4的prox几何更新加Gaussian内容持久化。A0提供图像骨干/FPN、全部感知编码器与初始query、Gaussian属性解码、query时间分配、自车状态编码和初始点尺度，以及共享渲染器的空类标定。C1提供ego_cross_attn、position_encoder、reg_branch、vel_branch、cls_branch、traj_head六个独立预测模块。",
        "",
        "plan_head会通过points_scale_branch改变感知初始点，因此必须随A0一起保留，才能保证同一个感知输出。Gaussian属性解码器也被未来出生query复用，本架构没有完全独立的未来属性头。全部参数和buffer均显式指定来源并严格加载，没有平均、随机补齐或训练。",
        "",
        f"两个来源的1040个query中，时间标签不同的有{audit['different_query_stamps']}个，其中当前/未来身份不同的有{audit['different_current_membership']}个。拼接模型使用A0的分配与注意力mask；没有强行套用C1的分配。",
        "",
        "四个固定样本的当前体素预测、query特征、感知点和时间标签逐元素完全一致；完整1024样本的当前帧全局与逐场景混淆矩阵也与A0完全一致。因此下面的当前感知确实恢复到A0，而非仅替换了一个名称相同的模块。",
        "",
        "## 全局结果",
        "",
        "单位为%。mIoU平均17个占据语义类；IoU合并所有占据类，不计具体语义。全空间评测，不使用可见性mask。",
        "",
        "| 时刻 | A0 mIoU | C1 mIoU | 拼接 mIoU | 拼接−A0 pp | 拼接−C1 pp | A0 IoU | C1 IoU | 拼接 IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i in range(7):
        a, c, m = [conditions[name]["horizons"][i] for name in ("A0", "C1", "splice")]
        lines.append(
            f"| {i*.5:g}s | {a['miou']:.4f} | {c['miou']:.4f} | {m['miou']:.4f} | {m['miou']-a['miou']:+.4f} | {m['miou']-c['miou']:+.4f} | {a['occupied_iou']:.4f} | {c['occupied_iou']:.4f} | {m['occupied_iou']:.4f} |"
        )
    lines += ["", "| 汇总指标 | A0 | C1 | 拼接 |", "|---|---:|---:|---:|"]
    for key, label in (
        ("selection_mean_miou", "当前/1/2/3秒平均mIoU"),
        ("future_123_mean_miou", "未来1/2/3秒平均mIoU"),
        ("all_horizons_mean_miou", "七时刻平均mIoU"),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(f"{conditions[n][key]:.4f}" for n in ("A0", "C1", "splice"))
            + " |"
        )
    lines += [
        "",
        "## 目标区域结果",
        "",
        "快区准确率与F-only/P-only保持率以各自固定GT占据体素为分母；快区邻域IoU还计入邻域空空间误报。区域使用此前冻结的同一套GT速度和旧D的预测/感知优势标签。",
        "",
        "| 指标 | 秒 | A0 | C1 | 拼接 | 拼接−A0 pp | 拼接−C1 pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for region, metric, label in (
        ("fast", "occupied_semantic_accuracy", "快区准确率"),
        ("fast_neighborhood", "occupied_iou", "快区邻域IoU"),
        ("F_only", "occupied_semantic_accuracy", "F-only保持率"),
        ("P_only", "occupied_semantic_accuracy", "P-only恢复率"),
    ):
        for i in range(3):
            a, c, m = [
                conditions[n]["regions"][i]["metrics"][region][metric]
                for n in ("A0", "C1", "splice")
            ]
            lines.append(
                f"| {label} | {i+1} | {a:.4f} | {c:.4f} | {m:.4f} | {m-a:+.4f} | {m-c:+.4f} |"
            )
    lines += [
        "",
        "## 统计与解释边界",
        "",
        "150个场景、2000次配对bootstrap，固定已选中的来源权重。未校正训练种子、选轮、多重比较和验证集反复调参。",
        "",
        "| 差值 | 主指标95%区间（pp） | 未来1/2/3秒均值95%区间（pp） |",
        "|---|---|---|",
    ]
    for name, stats in statistics["comparisons"].items():
        lines.append(
            f"| {name} | {stats['primary_95ci_pp']} | {stats['future_123_95ci_pp']} |"
        )
    lines += [
        "",
        "这项实验检验现成权重能否直接组合，不等于使用相同冻结感知器重新训练两种预测架构。C1预测层原先与自己的query特征、Gaussian内容、自车特征和出生query联合训练；替换输入分布后，提升或下降都混合了初始感知质量与接口适配的影响。未对齐潜在特征基底，也未按结果搜索拼接方案。",
        "",
        "## 文件与复现",
        "",
        f"- 完整独立配置：[配置](../../../{settings['eval_config']})。",
        f"- 拼接权重：`{settings['checkpoint_output']}`。",
        "- [权重逐键来源及哈希](splice_audit.json)、[四样本当前输出一致性](current_parity.json)。",
        "- [完整对照JSON](comparison.json)、[逐时刻逐类IoU CSV](per_horizon_class_iou.csv)、[配对统计](scene_statistics.json)。",
        f"- 运行：`PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u -m tools.pmg_world_checkpoint_splice --config {settings['eval_config']}`。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if "_base_" in Path(args.config).read_text():
        raise ValueError("complete standalone configuration required")
    cfg = Config.fromfile(args.config)
    settings = cfg.checkpoint_splice
    output = Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(settings.checkpoint_output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed = output / "completed.json"
    if completed.exists():
        raise ValueError(
            "evaluation already completed; use a new output directory to rerun"
        )
    torch.set_num_threads(4)
    first = torch.load(
        settings.perception_checkpoint, map_location="cpu", weights_only=False
    )
    second = torch.load(
        settings.forecast_checkpoint, map_location="cpu", weights_only=False
    )
    audit_sources(first, "A0")
    audit_sources(second, "C1")
    state, ownership = splice_states(first, second)
    manifests = runtime.audit_manifests(cfg)
    if (
        manifests != first["identity"]["manifests"]
        or manifests != second["identity"]["manifests"]
    ):
        raise ValueError(
            "donor checkpoints and evaluation use different data manifests"
        )
    audit = dict(
        perception_checkpoint=settings.perception_checkpoint,
        perception_sha256=runtime.digest(settings.perception_checkpoint),
        perception_epoch=first["epoch_index"] + 1,
        forecast_checkpoint=settings.forecast_checkpoint,
        forecast_sha256=runtime.digest(settings.forecast_checkpoint),
        forecast_epoch=second["epoch_index"] + 1,
        config=args.config,
        config_sha256=runtime.digest(args.config),
        runner_sha256=runtime.digest(__file__),
        manifests=manifests,
        forecast_modules=sorted(FORECAST_MODULES),
        perception_and_shared_modules=sorted(PERCEPTION_MODULES),
        tensor_counts_by_donor=dict(Counter(ownership.values())),
        tensor_sources=ownership,
        different_query_stamps=int(
            (first["query_stamps"] != second["query_stamps"]).sum()
        ),
        different_current_membership=int(
            ((first["query_stamps"] == 0) != (second["query_stamps"] == 0)).sum()
        ),
        query_stamp_counts=torch.bincount(first["query_stamps"]).tolist(),
        query_stamps_source="A0",
        training_updates=0,
    )
    hybrid = dict(
        state_dict=state,
        epoch_index=second["epoch_index"],
        query_stamps=first["query_stamps"].clone(),
        splice_audit=audit,
    )
    runtime.atomic_save(checkpoint_path, hybrid)
    audit["hybrid_checkpoint"] = str(checkpoint_path)
    audit["hybrid_checkpoint_sha256"] = runtime.digest(checkpoint_path)
    runtime.atomic_json(output / "splice_audit.json", audit)
    model = build_segmentor(cfg.model)
    model.load_state_dict(state, strict=True)
    if model.geometry_mode != "prox" or not model.persistent_content:
        raise ValueError("the hybrid must use the C1 architecture")
    print(
        "SPLICE_PREPARED",
        audit["tensor_counts_by_donor"],
        "donor epochs",
        audit["perception_epoch"],
        audit["forecast_epoch"],
        flush=True,
    )
    if args.prepare_only:
        return
    runtime.configure_runtime(cfg)
    model.cuda()
    epoch = runtime.load_evaluation_checkpoint(model, hybrid)
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if len(dataset) != 1024:
        raise ValueError(
            "splice evaluation requires the complete fixed 1024-sample validation set"
        )
    runtime.atomic_json(
        output / "status.json", dict(status="current_parity", training_updates=0)
    )
    check_current_parity(model, cfg, first, dataset, output)
    del first, second, state, hybrid
    gc.collect()
    runtime.atomic_json(
        output / "status.json",
        dict(status="evaluating", samples=1024, training_updates=0),
    )
    started = time.monotonic()
    metrics = evaluate(model, dataset, cfg, output, epoch)
    if metrics["num_samples"] != 1024 or metrics["partial"]:
        raise ValueError("evaluation did not cover the complete validation set")
    write_comparison(cfg, metrics, audit, output)
    for path_key, hash_key in (
        ("perception_checkpoint", "perception_sha256"),
        ("forecast_checkpoint", "forecast_sha256"),
    ):
        if runtime.digest(audit[path_key]) != audit[hash_key]:
            raise ValueError("a source checkpoint changed during evaluation")
    result = dict(
        status="completed",
        samples=1024,
        training_updates=0,
        evaluation_seconds=time.monotonic() - started,
        checkpoint_sha256=audit["hybrid_checkpoint_sha256"],
        report=str(output / "report.md"),
    )
    runtime.atomic_json(completed, result)
    runtime.atomic_json(output / "status.json", result)
    print("SPLICE_COMPLETED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
