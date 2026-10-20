"""Readable stage-two report. Tables are generated from the audited experiment summary."""
from pathlib import Path
import argparse
import re
import csv
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage2 import RUN,read,write_json

OUT=ROOT/"analysis/empty_mechanism_stage2_results.md"
ASSETS=ROOT/"analysis/empty_mechanism_stage2"
TEMPLATE=ROOT/"tools/report_templates/empty_mechanism_stage2_results.md"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def find(rows,name,radius=None):
    values=[r for r in rows if r["condition"]==name and (radius is None or r.get("radius")==radius)]
    assert len(values)==1,(name,radius,len(values))
    return values[0]


def d(row,key):
    return f'{row["effect"][key]["delta"]:+.4f}'


def ci(row,key):
    v=row["effect"][key]
    return f'{v["delta"]:+.4f} [{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]'


def figures(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,"font.size":11})
    names=["Set 正类","Set 负类","CE 前景","CE empty","Lovász","联合"]
    keys=["set_positive","set_negative","ce_occupied","ce_empty","combo_001","combo_111"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),sharey=True)
    for ax,e in zip(axes,["1","3","5"]):
        rows=[find(s["local"][e]["val"]["conditions"],k,.001) for k in keys]
        y=np.array([r["effect"]["fnr"]["delta"] for r in rows])
        bounds=np.array([r["effect"]["fnr"]["ci95"] for r in rows]).T
        ax.errorbar(range(len(keys)),y,yerr=np.maximum(np.vstack([y-bounds[0],bounds[1]-y]),0),
                    fmt="o",capsize=3,color="#24689a")
        ax.axhline(0,color="#aaa",lw=1);ax.set_xticks(range(len(keys)),names,rotation=25)
        ax.set_title(f"Epoch {e} · 32 验证场景");ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("漏检率变化（百分点）")
    fig.suptitle("共同更新系数：来源不同，局部作用方向不同")
    fig.tight_layout();fig.savefig(ASSETS/"source_effects.png",dpi=180,bbox_inches="tight");plt.close(fig)
    keys=["B0","B0_reference","B1","B2","B3","half_set","project_set","norm_set","no_all_set"]
    names=["原目标","原接口参照","删末阶段分类","删 CE empty","同时删","分类减半","幅度投影","范数匹配","删全部分类"]
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
    for ax,e in zip(axes,["1","3"]):
        rows=[find(s["short_contrasts"][e],k) for k in keys]
        y=np.array([r["effect"]["miou"]["delta"] for r in rows])
        bounds=np.array([r["effect"]["miou"]["ci95"] for r in rows]).T
        ax.errorbar(range(len(keys)),y,yerr=np.maximum(np.vstack([y-bounds[0],bounds[1]-y]),0),
                    fmt="o",capsize=3,color="#237c57")
        ax.axhline(0,color="#aaa",lw=1)
        ax.set_xticks(range(len(keys)),names,rotation=40,ha="right")
        ax.set_title(f"从 epoch {e} 分叉 · 第 64 步");ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("相对 B0 的 mIoU 变化（百分点）")
    fig.suptitle("真实 AdamW 续训：路径干预与强度对照")
    fig.tight_layout();fig.savefig(ASSETS/"short64_comparison.png",dpi=180,bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,e in zip(axes,["1","3"]):
        rows=[s["equal_bias"]["contrasts"][e][k] for k in ["native","calibrated"]]
        y=np.array([r["effect"]["miou"]["delta"] for r in rows])
        bounds=np.array([r["effect"]["miou"]["ci95"] for r in rows]).T
        ax.errorbar([0,1],y,yerr=np.maximum(np.vstack([y-bounds[0],bounds[1]-y]),0),
                    fmt="o",capsize=4,color="#8a507f")
        ax.axhline(0,color="#aaa",lw=1);ax.set_xticks([0,1],["原生","同等类别校准后"])
        ax.set_xlim(-.5,1.5);ax.set_title(f"起点 epoch {e} · 完整 1024 验证帧");ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("投影模型相对 B0 的 mIoU（百分点）")
    fig.tight_layout();fig.savefig(ASSETS/"equal_bias_comparison.png",dpi=180,bbox_inches="tight");plt.close(fig)


def export_csv(s):
    rows=[]
    for e,groups in s["local"].items():
        for split,group in groups.items():
            for row in group["conditions"]:
                value=dict(experiment="output",epoch=e,group=split,condition=row["condition"],radius=row["radius"])
                for k,v in row["effect"].items():
                    value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
                rows.append(value)
    for key,group in s["adamw_contrasts"].items():
        for row in group:
            value=dict(experiment="adamw_vs_B0",epoch=key.split("_")[0],group=key.split("_")[1],condition=row["condition"],radius="")
            for k,v in row["effect"].items():
                value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
            rows.append(value)
    for e,group in s["short_contrasts"].items():
        for row in group:
            value=dict(experiment="short64_vs_B0",epoch=e,group="val32",condition=row["condition"],radius="")
            for k,v in row["effect"].items():
                value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
            rows.append(value)
    for e,modes in s["adamw"].items():
        for mode,groups in modes.items():
            for role,group in groups.items():
                for row in group:
                    value=dict(experiment="adamw_vs_checkpoint",epoch=e,group=mode+"_"+role,
                               condition=row["condition"],radius="")
                    for k,v in row["effect"].items():
                        value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
                    rows.append(value)
    for e,group in s["short"].items():
        for row in group:
            value=dict(experiment="short_vs_checkpoint",epoch=e,group="val32",
                       condition=row["condition"],radius="",step=row["step"])
            for k,v in row["effect"].items():
                value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
            rows.append(value)
    for e,group in s["equal_bias"]["contrasts"].items():
        for state,row in group.items():
            value=dict(experiment="equal_bias_full1024",epoch=e,group=state,
                       condition="project_set_vs_B0",radius="")
            for k,v in row["effect"].items():
                value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
            rows.append(value)
    for e,group in s["short_strength_contrasts"].items():
        for row in group:
            value=dict(experiment="short64_vs_strength",epoch=e,group=row["reference"],
                       condition="project_set",radius="")
            for k,v in row["effect"].items():
                value.update({k+"_delta":v["delta"],k+"_ci_low":v["ci95"][0],k+"_ci_high":v["ci95"][1]})
            rows.append(value)
    with (ASSETS/"all_conditions.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--document-only",action="store_true",
                      help="Rewrite Markdown only; preserve experiment records, CSV and figures.")
    args=parser.parse_args()
    s=read(RUN/"summary.json")
    audit=read(RUN/"completion_audit.json")
    assert audit["status"]=="passed"
    ASSETS.mkdir(exist_ok=True)
    names={"B0":"原训练目标（B0）","B0_reference":"原损失接口参照",
           "B1":"删除末阶段 Set 分类","B2":"删除 CE empty","B3":"同时删除两项",
           "half_set":"Set 分类减半","project_set":"Set 幅度投影",
           "norm_set":"匹配投影梯度长度","no_all_set":"删除全部阶段 Set 分类",
           "zero_current":"当前梯度置零，保留动量"}
    source_rows=[]
    for key,label in [("set_positive","Set 正类"),("set_negative","Set 负类"),
                      ("ce_occupied","CE 前景"),("ce_empty","CE empty"),
                      ("combo_001","Lovász"),("combo_010","完整 CE"),
                      ("combo_011","CE + Lovász"),("combo_111","完整联合")]:
        r=find(s["local"]["5"]["val"]["conditions"],key,.001)
        source_rows.append([label,d(r,"fnr"),d(r,"fpr"),d(r,"miou")])
    source_table=table(["更新来源","漏检率变化 ↓","误占用率变化 ↓","mIoU 变化 ↑"],source_rows)
    broad_rows=[]
    for e in ["1","3","5"]:
        a=find(s["local"][e]["val"]["conditions"],"set_negative",.001)
        b=find(s["local"][e]["val"]["conditions"],"ce_empty",.001)
        broad_rows.append([e,ci(a,"fnr"),ci(b,"fnr")])
    broad_table=table(["Checkpoint","Set 负类：漏检变化 [95% 区间]","CE empty：漏检变化 [95% 区间]"],broad_rows)
    spatial_rows=[]
    for e in ["1","3","5"]:
        v=s["spatial"][e]["val"]["matched_high_minus_low_relative_ce_empty_derivative"]
        spatial_rows.append([e,f'{v["value"]*1e5:+.3f}',
            f'[{v["ci95"][0]*1e5:+.3f}, {v["ci95"][1]*1e5:+.3f}]',
            f'{v["scenes_with_matches"]}/32'])
    spatial_table=table(["Epoch","高覆盖 − 低覆盖","95% 区间","具有有效分层的场景"],spatial_rows)
    mixed_text="、".join(f'{s["spatial"][e]["val"]["shared_coverage"]["mixed_fraction"]:.1%}' for e in ["1","3","5"])
    interaction_rows=[]
    for e in ["1","3","5"]:
        v=next(r for r in s["local"][e]["val"]["interactions"] if r["radius"]==.001)["statistics"]["fnr"]
        interaction_rows.append([e,f'{v["value"]:+.5f}',f'[{v["ci95"][0]:+.5f}, {v["ci95"][1]:+.5f}]'])
    interaction_table=table(["Epoch","漏检率交互量","95% 区间"],interaction_rows)
    interaction_zero=all(
        next(r for r in s["local"][e]["val"]["interactions"] if r["radius"]==.001)["statistics"]["fnr"]["ci95"][0]<=0<=
        next(r for r in s["local"][e]["val"]["interactions"] if r["radius"]==.001)["statistics"]["fnr"]["ci95"][1]
        for e in ["1","3","5"])
    interaction_text=("三个阶段的交互区间均包含零，没有观察到稳定的额外放大。"
                      if interaction_zero else "部分阶段的交互区间不包含零，需结合步长复核其范围。")
    ratio_text="、".join(f'{find(s["local"][e]["val"]["conditions"],"combo_100",.001)["actual_radius_mean"]/.001:.1%}' for e in ["1","3","5"])
    projection_rows=[]
    for e in ["1","3","5"]:
        group=s["local"][e]["val"]["conditions"]
        b=find(group,"combo_111",.001)
        for key,label in [("C_half_set","分类减半"),("C_project_set","幅度投影"),("C_norm_set","范数匹配")]:
            r=find(group,key,.001)
            projection_rows.append([e,label]+[f'{r["effect"][k]["delta"]-b["effect"][k]["delta"]:+.4f}' for k in ["fnr","fpr","miou"]])
    projection_table=table(["Epoch","相对原联合更新","漏检率 Δ","误占用率 Δ","mIoU Δ"],projection_rows)
    adam_rows=[]
    for e in ["1","3","5"]:
        group=s["adamw"][e]["full"]["heldout"]
        for key in ["B0","zero_current","project_set"]:
            r=find(group,key)
            adam_rows.append([e,names[key],d(r,"fnr"),d(r,"fpr"),d(r,"miou")])
    adam_table=table(["Epoch","真实全网络一步","漏检率 Δ","误占用率 Δ","mIoU Δ"],adam_rows)
    short_text=[]
    parameter_rows=[]
    ranking_rows=[]
    candidate_results=[]
    for e in ["1","3"]:
        group=s["short_contrasts"][e]
        rows=[]
        for key in ["B0","B0_reference","B1","B2","B3","half_set","project_set","norm_set","no_all_set"]:
            r=find(group,key)
            rows.append([names[key],f'{r["effect"]["miou"]["after"]:.4f}',ci(r,"miou"),d(r,"fnr"),d(r,"fpr")])
        short_text.append(f"### 起点 epoch {e}：64 步后\n\n"+table(
            ["训练条件","mIoU（%）","mIoU 差值 [95% 区间]","漏检率差值","误占用率差值"],rows))
        candidate_results.append(find(group,"project_set"))
        for key in ["B0","project_set","norm_set"]:
            r=next(v for v in s["short"][e] if v["condition"]==key and v["step"]==64)
            parameter_rows.append([e,names[key],f'{r["effect"]["gaussian_amplitude"]["after"]:.4f}',
                                   f'{r["empty_bias_after"]:.4f}',f'{r["empty_scalar_after"]:.4f}'])
            ranking=find(group,key)
            ranking_rows.append([e,names[key],f'{ranking["effect"]["fg_rank_correct_rate"]["after"]:.4f}',
                                 ci(ranking,"fg_rank_correct_rate")])
    parameter_table=table(["起点 Epoch","第 64 步条件","Gaussian 平均总幅度","全局 empty bias","empty scalar"],parameter_rows)
    ranking_table=table(["起点 Epoch","第 64 步条件","前景首选正确率（%）","相对 B0 [95% 区间]"],ranking_rows)
    strength_rows=[]
    for e,group in s["short_strength_contrasts"].items():
        for r in group:
            strength_rows.append([e,"投影 − "+names[r["reference"]],ci(r,"miou")])
    strength_table=table(["起点 Epoch","配对比较","mIoU 变化 [95% 区间]"],strength_rows)
    # A rise in FP is not automatically failure: mIoU already accounts for
    # class-dependent unions. Require a path-specific gain over strength controls.
    gate_evidence=[]
    for e,r in zip(["1","3"],candidate_results):
        rows=s["short_contrasts"][e]
        native_gain=r["effect"]["miou"]["delta"]
        norm_gain=find(rows,"norm_set")["effect"]["miou"]["delta"]
        half_gain=find(rows,"half_set")["effect"]["miou"]["delta"]
        gate_evidence.append(dict(epoch=e,versus_B0=native_gain,versus_norm=native_gain-norm_gain,
                                  versus_half=native_gain-half_gain))
    native_gate=all(min(v["versus_B0"],v["versus_norm"],v["versus_half"])>0 for v in gate_evidence)
    equal=s["equal_bias"]
    calibration_gate=all(equal["contrasts"][e]["calibrated"]["effect"]["miou"]["delta"]>0 for e in ["1","3"])
    gate=native_gate and calibration_gate
    calibration_sections=[];calibration_comparison=[];calibration_overview_rows=[]
    for e in ["1","3"]:
        rows=[]
        for key in ["B0","project_set"]:
            record=equal["models"][e][key]
            calibration_overview_rows.append([
                e,names[key],f'{record["native"]["mIoU"]:.4f}',
                f'{record["calibrated"]["mIoU"]:.4f}'])
            for state,label in [("native","原生"),("calibrated","类别校准")]:
                v=record[state]
                rows.append([names[key]+" · "+label,f'{v["mIoU"]:.4f}',
                             f'{v["IoU"]:.4f}',f'{v["occupied_FNR"]:.4f}',f'{v["empty_FPR"]:.4f}'])
        calibration_sections.append(f"### 起点 epoch {e}：完整验证集\n\n"+table(
            ["模型与推理方式","mIoU（%）","二值 IoU（%）","漏检率（%）","误占用率（%）"],rows))
        raw=equal["contrasts"][e]["native"];cal=equal["contrasts"][e]["calibrated"]
        calibration_comparison.append([e,ci(raw,"miou"),ci(cal,"miou")])
    calibration_table=table(["起点","原生差值 [95% 区间]","校准后差值 [95% 区间]"],calibration_comparison)
    calibration_overview=table(["起点","模型","原生 mIoU","类别校准后 mIoU"],calibration_overview_rows)
    gate_record = dict(
        candidate="project_set",endpoint=64,epochs=[1,3],passed=gate,
        native_screen_passed=native_gate,equal_bias_screen_passed=calibration_gate,
        rule="both forks improve native mIoU versus B0, half_set and norm_set AND calibrated mIoU versus B0; FP rise alone is not failure",
        evidence=gate_evidence,calibrated_evidence=equal["contrasts"])
    if gate:
        gate_text="幅度投影通过原生对照与同等类别 bias 校准筛选；完整重训练与多 seed 复核仍需继续，本报告目前不将其写为已完成的方法收益。"
    else:
        if native_gate:
            gate_text="幅度投影通过了原生指标筛选，但未通过同等类别 bias 校准筛选。因此，本轮完成机制诊断与校准比较后，没有启动该方案的完整重训练和多 seed 方法实验。"
        else:
            gate_text="幅度投影没有同时通过跨起点的原目标／强度对照比较和同等校准筛选，因此本轮暂不启动该投影方案的完整重训练和多 seed 方法实验。"
        gate_text+=" 这里的停止依据是校准后的 mIoU 表现，误占用率单项上升不作为失败条件。"
    short_summary_text="；".join(
        f'从 epoch {e} 出发，投影相对原训练目标的 mIoU 变化为 {r["effect"]["miou"]["delta"]:+.4f} 个百分点'
        for e,r in zip([1,3],candidate_results))
    calibration_summary_text="完整 1024 帧的同等类别校准后，投影相对原训练目标（B0）的 mIoU 变化为 "+ "、".join(
        f'epoch {e} 起点 {equal["contrasts"][e]["calibrated"]["effect"]["miou"]["delta"]:+.4f} 个百分点'
        for e in ["1","3"])
    context = {
        'short_summary_text': str(short_summary_text),
        'calibration_summary_text': calibration_summary_text,
        'source_table': str(source_table),
        'broad_table': str(broad_table),
        'ratio_text': str(ratio_text),
        'mixed_text': str(mixed_text),
        'spatial_table': str(spatial_table),
        'interaction_table': str(interaction_table),
        'interaction_text': str(interaction_text),
        'projection_table': str(projection_table),
        'adam_table': str(adam_table),
        'short_sections': "\n\n".join(short_text),
        'parameter_table': str(parameter_table),
        'ranking_table': str(ranking_table),
        'strength_table': strength_table,
        'calibration_sections': "\n\n".join(calibration_sections),
        'calibration_overview': calibration_overview,
        'calibration_table': str(calibration_table),
        'gate_text': str(gate_text),
        'audit_local_frames': str(audit['local_frames']),
        'audit_output_interventions': str(audit['output_interventions']),
        'audit_adamw_pairs': str(audit['adamw_pairs']),
        'audit_adamw_optimizer_updates': str(audit['adamw_optimizer_updates']),
        'audit_short_branches': str(audit['short_branches']),
        'audit_short_optimizer_updates': str(audit['short_optimizer_updates']),
        'audit_short_evaluation_scene_frames': str(audit['short_evaluation_scene_frames']),
        'audit_calibration_models': str(audit['calibration_models']),
        'audit_max_ce_value_error': format(audit['max_ce_value_error'], '.2e'),
        'audit_max_set_value_error': format(audit['max_set_value_error'], '.2e'),
        'audit_max_decomposition_joint_gradient_error': format(audit['max_decomposition_joint_gradient_error'], '.2e'),
        'audit_parity_full_cached_vs_original_relative_gradient_error': format(audit['parity']['full']['cached_vs_original_relative_gradient_error'], '.3%'),
        'audit_parity_full_original_replay_relative_gradient_noise': format(audit['parity']['full']['original_replay_relative_gradient_noise'], '.3%'),
    }
    used = []
    def substitute(match):
        key = match.group(1)
        used.append(key)
        return context[key]
    text = re.sub(r"\{\{ value:([a-zA-Z0-9_]+) \}\}", substitute, TEMPLATE.read_text())
    if set(used) != set(context):
        raise ValueError("Report template does not use every numeric/explanation block")
    if args.check:
        if not OUT.exists() or OUT.read_text()!=text:
            raise SystemExit("Report differs from its template or audited summary")
        print("Report matches its template and audited numeric summary.")
        return
    OUT.write_text(text)
    if args.document_only:
        print(OUT)
        return
    write_json(RUN/"method_gate.json",gate_record)
    export_csv(s);figures(s)
    print(OUT)


if __name__=="__main__":main()
