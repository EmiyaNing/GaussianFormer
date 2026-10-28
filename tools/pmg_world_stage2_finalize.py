"""Watch the authorized two-arm queue and write a conservative final interpretation.

Kept outside the frozen training package: this process only reads final
metrics and writes Markdown/JSON reports. It never modifies configurations,
weights or training source and never launches additional training arms.
"""

import json
from pathlib import Path
import time


ROOT = Path("analysis/pmg_world_stage2")


def write_summary():
    comparison = json.loads((ROOT / "time_supervision_comparison.json").read_text())
    stats = json.loads((ROOT / "final_scene_statistics.json").read_text())
    selected = comparison["selected"]
    lines = [
        "# PMG 第二阶段：时间监督对照的机制判断",
        "",
        "状态：T-A0/T-C1均完成10轮，并已独立重载选中权重复评。下列解释只针对固定总权重的时间监督分配；场景区间以验证集所选checkpoint为条件，不包含训练种子波动。",
        "",
        "## 1. 同预算结果",
        "",
        "| 条件 | 选中轮次 | 原主指标 | 七时刻均值 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in selected.items():
        lines.append(
            f"| {name} | {row['epoch']} | {row['selection_mean_miou']:.4f} | {row['all_horizons_mean_miou']:.4f} |"
        )
    lines += [
        "",
        "## 2. C1中间时刻优势是否缩小",
        "",
        "差值变化=(C1_dense−A0_dense)−(C1_sparse−A0_sparse)。负值表示C1相对优势缩小；不能单独区分A0变好和C1变差，需结合下一列。",
        "",
        "| 秒 | 原C1−A0 | 新C1−A0 | 差值变化95%区间 | A0_dense−A0_sparse的95%区间 |",
        "|---|---:|---:|---|---|",
    ]
    a0 = stats["comparisons"]["A0_dense_minus_A0_sparse"]["per_horizon_95ci_pp"]
    did = stats["difference_in_differences_95ci_pp"]
    for h in (1, 3, 5):
        row = comparison["per_horizon"][h]
        lines.append(
            f"| {h*.5:g} | {row['sparse_c1_minus_a0']:+.4f} | {row['dense_c1_minus_a0']:+.4f} | {did[h]} | {a0[h]} |"
        )
    improved = [h * 0.5 for h in (1, 3, 5) if a0[h][0] > 0]
    narrowed = [h * 0.5 for h in (1, 3, 5) if did[h][1] < 0]
    if improved and narrowed:
        lines += [
            "",
            f"A0在{improved}秒的改善区间高于0，C1相对差值在{narrowed}秒缩小，支持时间监督／读出适配对部分中间时刻收益有贡献。它不证明这是唯一机制。",
        ]
    else:
        lines += [
            "",
            "本次对照尚未同时提供A0中间时刻可靠改善和C1相对优势可靠缩小的证据。应逐时刻阅读区间；不能把未排除0解释为已证明没有作用。",
        ]
    lines += [
        "",
        "## 3. 快速目标是否改善",
        "",
        "以下比较同为全时刻监督的C1与A0。单位pp。",
        "",
        "| 秒 | 快速正确率差值95%区间 | 邻域IoU差值95%区间 | F-only保持率差值95%区间 |",
        "|---|---|---|---|",
    ]
    regions = stats["comparisons"]["C1_dense_minus_A0_dense"]["regions"]
    for i, seconds in enumerate((1, 2, 3)):
        lines.append(
            f"| {seconds} | {regions['fast_accuracy'][i]} | {regions['fast_neighborhood_iou'][i]} | {regions['F_only'][i]} |"
        )
    harmed = [
        seconds
        for i, seconds in enumerate((1, 2, 3))
        if regions["fast_accuracy"][i][1] < 0
    ]
    if harmed:
        lines += [
            "",
            f"C1在{harmed}秒仍有快速正确率下降证据。充分时间监督没有在这些时刻消除其快速退化；下一项应分离A4几何与内容约束，而非继续把中间时刻收益解释为运动提升。",
        ]
    else:
        lines += [
            "",
            "快速差异需结合三个时刻的准确率、邻域IoU和F-only共同判断。没有可靠下降证据不等于通过非劣；原F-only容限为−0.5 pp，全域及静态mIoU容限为−0.2 pp。",
        ]
    lines += [
        "",
        "## 4. 与P0发现组合后的解释",
        "",
        "固定中心的语义干预已经表明，部分半秒时刻收益可由读出变化产生；速度开关全部打开未得到一致收益；两份样本的属性梯度未表现为负向冲突。时间监督对照是在此基础上的训练干预，不能由其中一项结果推出单一根因。",
        "",
        "本实验在固定总监督权重下给更多时刻分配信号，同时降低原未来直接监督项的单项权重。若结果不理想，不能据此认定新增中间时刻监督在所有权重设计下无效。",
        "",
        "下一步按证据选择原传播＋单属性保留、可靠实例定位或运动修正；未自动启动新的训练组。该子集已反复使用，正式论文结论仍需多种子、全量及独立场景验证。",
        "",
        "复核：[训练报告](training_report.md)、[完整配对统计](final_scene_statistics.md)、[P0初步发现](preliminary_findings.md)、[实施记录](../../plans/pmg_world_stage2_execution_20261020.md)。",
    ]
    (ROOT / "mechanism_summary.md").write_text("\n".join(lines) + "\n")


def main():
    deadline = time.monotonic() + 72 * 3600
    status_path = ROOT / "finalizer_status.json"
    while time.monotonic() < deadline:
        queue = json.loads((ROOT / "queue_status.json").read_text())
        if queue["status"] == "completed":
            write_summary()
            status_path.write_text(
                json.dumps(
                    dict(status="completed", report="mechanism_summary.md"), indent=2
                )
                + "\n"
            )
            return
        if queue["status"] in ("failed", "stopped_by_request"):
            status_path.write_text(
                json.dumps(
                    dict(status="queue_ended_before_completion", queue=queue), indent=2
                )
                + "\n"
            )
            return
        status_path.write_text(
            json.dumps(
                dict(
                    status="waiting_for_10_epochs",
                    queue_arm=queue.get("arm"),
                    queue_epoch=queue.get("epoch"),
                ),
                indent=2,
            )
            + "\n"
        )
        time.sleep(30)
    status_path.write_text(
        json.dumps(dict(status="timed_out_after_72_hours"), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
