"""Render the mechanism report from editable prose and archived numeric results.

The template owns explanations and notation. Only numeric tables are generated.
No experiments, summaries or historical execution manifests are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "work_dirs/empty_mechanism_stage1"
TEMPLATE = ROOT / "tools/report_templates/empty_mechanism_stage1_results.md"
OUTPUT = ROOT / "analysis/empty_mechanism_stage1_results.md"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def condition(summary, mode, epoch, group, *, branch="set_cls",
              path="full", radius=None):
    if radius is None:
        radius = .001 if mode == "output" else .0001
    matches = [row for row in summary[mode][str(epoch)][group]["conditions"]
               if (row["branch"], row["pathway"], row["radius"])
               == (branch, path, radius)]
    if len(matches) != 1:
        raise ValueError(f"Expected one condition: {mode}, {epoch}, {group}, {branch}, {path}")
    return matches[0]


def delta(row, metric="fnr"):
    return f"{row['effect'][metric]['delta']:+.4f}"


def interval(row, metric="fnr"):
    effect = row["effect"][metric]
    low, high = effect["ci95"]
    return f"{effect['delta']:+.4f} [{low:+.4f}, {high:+.4f}]"


def make_tables(summary):
    audit = summary["audit"]
    assert audit["status"] == "passed"
    census = summary["census"]
    totals, metrics = census["totals"], census["metrics"]
    tables = {}
    tables["census"] = table(["统计量", "结果"], [
        ["原生 mIoU / 二值占有 IoU", f"{metrics['mIoU']:.4f}% / {metrics['IoU']:.4f}%"],
        ["empty 漏检率", f"{metrics['occupied_FNR']:.4f}%"],
        ["empty 误占用率", f"{metrics['empty_FPR']:.4f}%"],
        ["可恢复漏检数", f"{totals['recoverable']:,} 个"],
        ["占所有真实前景 / 占所有 empty 漏检",
         f"{100*totals['recoverable']/totals['occupied']:.2f}% / "
         f"{100*totals['recoverable']/(totals['occupied']*metrics['occupied_FNR']/100):.2f}%"],
        ["出现可恢复漏检的帧", f"{census['frames_with_recoverable']} / {census['frames']}"],
        ["可匹配正确前景对照的漏检",
         f"{totals['matched_population']:,} 个；覆盖 {100*totals['matched_population']/totals['recoverable']:.2f}%"],
        ["可恢复漏检中的零／近零支持体素",
         f"{totals['recoverable_zero_support']:,} 个；占该组 {100*totals['recoverable_zero_support']/totals['recoverable']:.4f}%"],
    ])
    rows = []
    for key, label in [("log_support", "空间支持 S"),
                       ("log_amplitude", "体素平均语义幅度 A"),
                       ("log_share", "体素前景集中度")]:
        item = census["matched_differences"][key]
        low, high = item["ratio_ci95"]
        rows.append([label, f"{item['geometric_ratio']:.3f}", f"[{low:.3f}, {high:.3f}]"])
    tables["matched"] = table(["比较的体素量", "漏检 / 正确对照", "95% 区间"], rows)
    for key, mode, group in [("output_paths", "output", "val"),
                             ("parameter_paths", "parameters", "heldout"),
                             ("train_paths", "output", "train")]:
        rows = [[epoch] + [delta(condition(summary, mode, epoch, group, path=path))
                          for path in ["amplitude", "composition", "full"]]
                for epoch in [1, 3, 5]]
        tables[key] = table(["Epoch", "仅幅度更新", "仅组成：恢复幅度", "完整语义更新"], rows)
    rows, prevalence = [], []
    for epoch in [1, 3, 5]:
        item = next(row for row in summary["channel_trace"]["conditions"][str(epoch)]["val"]
                    if row["radius"] == .001)["cohorts"]
        cohort = item["recoverable"]
        rows.append([epoch] + [f"{cohort[key]['mean']:+.6f}"
                               for key in ["positive_sum", "negative_sum", "total_sum"]])
        prevalence.append([epoch, f"{cohort['amplitude_declined']['mean']:.2f}%",
                           f"{item['gaussian']['amplitude_declined']['mean']:.2f}%"])
    tables["channel_delta"] = table(
        ["Epoch", "目标类贡献", "其余 16 类的合计贡献", "平均幅度总变化"], rows)
    tables["channel_prevalence"] = table(
        ["Epoch", "可恢复漏检体素中，平均幅度下降的比例", "有效 Gaussian 中，总幅度下降的比例"],
        prevalence)
    rows = []
    for branch, label in [("set_cls", "Set 分类"), ("voxel_ce", "体素交叉熵"),
                          ("voxel_lovasz", "Lovász"), ("voxel_total", "场损失"),
                          ("joint", "联合目标")]:
        row = condition(summary, "output", 5, "val", branch=branch)
        rows.append([label] + [delta(row, metric) for metric in ["fnr", "fpr", "miou", "field"]])
    tables["tradeoff"] = table(
        ["更新方向", "漏检率 Δ", "误占用率 Δ", "mIoU Δ", "场损失 Δ"], rows)
    rows = []
    for mode, group, label in [("output", "val", "输出干预，32 个验证场景"),
                                ("parameters", "heldout", "参数干预，8 个独立场景")]:
        rows.append([label] + [
            condition(summary, mode, 5, group, path=path)["correct_to_empty"]
            for path in ["amplitude", "composition", "full"]])
    tables["transitions"] = table(
        ["原正确前景转 empty 的数量（个）", "仅幅度", "仅组成", "完整更新"], rows)
    tables["execution"] = table(["实验", "实际完成量", "统计含义"], [
        ["输出干预", f"{audit['output_interventions']} 次",
         "3 checkpoint × 48 场景帧 × 5 方向 × 3 步长 × 3 条件"],
        ["参数干预", f"{audit['parameter_interventions']} 条结果",
         "3 checkpoint × 8 对场景 × 3 方向 × 2 步长 × 2 评价场景 × 3 条件"],
        ["基线普查", f"{audit['census_frames']} 帧、{census['scenes']} 场景", "epoch 5 原预测"],
        ["追加通道归因", f"{audit['channel_trace_interventions']} 次",
         "144 场景帧 × 3 输出步长"],
    ])
    rows = []
    for epoch in [1, 3, 5]:
        rows.append([epoch, interval(condition(summary, "output", epoch, "val")),
                     interval(condition(summary, "parameters", epoch, "heldout")),
                     delta(condition(summary, "parameters", epoch, "source"))])
    tables["key_intervals"] = table(
        ["Epoch", "输出干预：完整更新 [95% 区间]", "参数独立场景：完整更新 [95% 区间]",
         "参数源场景：完整更新"], rows)
    controls = audit["controls"]
    tables["controls"] = table(["检查", "原实验结果"], [
        ["有效域重渲染与原生输出对齐", f"最大绝对差 {census['renderer_max_error']:.1e}"],
        ["幅度分解后重组", "6 组均为零分数差、零预测变化"],
        ["重合副本各分一半 opacity",
         f"最大分数差 {max(row['split']['max_score_error'] for row in controls):.2e}；预测不变"],
        ["缓存 Set 分类与原接口",
         f"最大损失差 {max(row['loss_parity']['classification_value_error'] for row in controls):.2e}；梯度差为 0"],
        ["完整场损失与预筛选有效域的输入",
         f"追加 144 帧最大差 {summary['channel_trace']['max_original_field_loss_error']:.2e}"],
        ["参数更新后的几何与 opacity", "24 对场景全部通过，最大差为 0"],
        ["empty 分数与 checkpoint", "干预中 empty 分数固定，三个权重 SHA-256 保持一致"],
        ["测试", "原实验 5 项通过，包括解析梯度与有限差分"],
        ["查询网格", "640000 个体素对应 640000 个唯一网格单元"],
    ])
    return tables


def render():
    summary = json.loads((RUN / "summary.json").read_text())
    tables = make_tables(summary)
    source = TEMPLATE.read_text()
    used = []

    def substitute(match):
        key = match.group(1)
        used.append(key)
        return tables[key]

    result = re.sub(r"\{\{ table:([a-z_]+) \}\}", substitute, source)
    if set(used) != set(tables) or len(used) != len(set(used)):
        raise ValueError("Every generated table must appear exactly once")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Check that the Markdown agrees with its template and archived tables")
    args = parser.parse_args()
    result = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != result:
            raise SystemExit("Report differs from its template/data; regenerate it.")
        print("Report matches its template and archived numeric summary.")
    else:
        OUTPUT.write_text(result)
        print(OUTPUT)


if __name__ == "__main__":
    main()
