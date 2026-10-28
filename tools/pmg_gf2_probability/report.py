"""Live comparison with the preserved A0 run, using matched epoch budgets."""

import json
from pathlib import Path

from tools.pmg_world.evaluate import refresh as refresh_arm
from tools.sparse_opus_gausisan_world import runtime


def refresh(cfg):
    refresh_arm(cfg)
    root = Path(cfg.report_directory).parent
    baseline = [json.loads(p.read_text()) for p in sorted(Path("analysis/pmg_world_stage1/A0").glob("epoch_*_metrics.json"))]
    candidate = [json.loads(p.read_text()) for p in sorted(Path(cfg.report_directory).glob("epoch_*_metrics.json"))]
    if not candidate:
        return
    common = min(max(r["epoch"] for r in baseline), max(r["epoch"] for r in candidate))
    best_baseline = max((r for r in baseline if r["epoch"] <= common), key=lambda r: r["selection_mean_miou"])
    best_matched = max((r for r in candidate if r["epoch"] <= common), key=lambda r: r["selection_mean_miou"])
    best_all = max(candidate, key=lambda r: r["selection_mean_miou"])
    comparison = dict(completed_epochs=len(candidate), planned_epochs=15,
                      matched_epoch_budget=common, baseline_best=best_baseline,
                      candidate_matched_best=best_matched, candidate_all_best=best_all,
                      matched_selection_delta_pp=best_matched["selection_mean_miou"] - best_baseline["selection_mean_miou"],
                      matched_future_delta_pp=best_matched["future_123_mean_miou"] - best_baseline["future_123_mean_miou"])
    runtime.atomic_json(root / "comparison.json", comparison)
    lines = ["# PMG A0：GF2 概率分离实验", "",
             "固定 4096 训练／1024 验证，seed=17；从原图像权重初始化，正式训练15轮，每轮完整验证七个时刻。",
             "上游 3b34b50 的概率 CUDA 核保持原样；适配 raw logits、batch=2 及越界点过滤。",
             "联合概率为 p(c)=o·q(c)、p(empty)=1−o；使用联合 argmax，无阈值调参。",
             "原 point/world/trajectory 项及体素权重保留；体素 CE 改为概率 NLL，Lovasz 直接接收概率，条件项接收独立 q。", "",
             f"已完成 {len(candidate)}/15 轮；原 A0 有 {len(baseline)} 轮完整验证。", "",
             "| 比较范围 | 模型 | 最佳轮次 | 0/1/2/3s mIoU | future 1/2/3s mIoU | 0s | 1s | 2s | 3s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope, name, record in [(f"前{common}轮", "原 A0", best_baseline),
                                 (f"前{common}轮", "A0 GF2", best_matched),
                                 (f"全部{len(candidate)}轮", "A0 GF2", best_all)]:
        scores = " | ".join(f"{record['horizons'][h]['miou']:.4f}" for h in (0, 2, 4, 6))
        lines.append(f"| {scope} | {name} | {record['epoch']} | {record['selection_mean_miou']:.4f} | {record['future_123_mean_miou']:.4f} | {scores} |")
    lines += ["", f"同预算主指标变化：{comparison['matched_selection_delta_pp']:+.4f} pp；未来均值变化：{comparison['matched_future_delta_pp']:+.4f} pp。",
              "当前及未来结果均来自同一个被选 checkpoint。原 A0 已按用户要求停止，不继续旧队列；完整15轮GF2成绩与12轮A0须区别预算。", "",
              "逐时刻、逐类和固定速度区域见 A0_GF2/report.md 及每轮 JSON；速度显存对照见 validation.json。",
              "以上为单种子、重复使用子集的探索结果。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
