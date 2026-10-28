"""Write the experiment status, paired comparisons, class tables and figures."""

import csv
import json
from pathlib import Path

import numpy as np

from tools.pmg_world.regions import REGIONS
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import CLASS_NAMES


ROOT = Path("analysis/pmg_world_stage3_20261023")


def regional_sufficient(counts):
    fast = counts[..., REGIONS.index("fast"), :, :]
    near = counts[..., REGIONS.index("fast_neighborhood"), :, :]
    fonly = counts[..., REGIONS.index("F_only"), :, :]
    return np.stack((np.trace(fast[..., :17,:17],axis1=-2,axis2=-1),fast[..., :17,:].sum((-1,-2)),
        near[..., :17,:17].sum((-1,-2)),near[..., :17,:].sum((-1,-2)),near[..., :,:17].sum((-1,-2)),
        np.trace(fonly[..., :17,:17],axis1=-2,axis2=-1),fonly[..., :17,:].sum((-1,-2))),-1)


def regional_values(parts):
    return dict(fast_accuracy=100*parts[...,0]/np.maximum(parts[...,1],1),
        fast_neighborhood_iou=100*parts[...,2]/np.maximum(parts[...,3]+parts[...,4]-parts[...,2],1),
        F_only=100*parts[...,5]/np.maximum(parts[...,6],1))


def paired_intervals(scene_parts, left, right, weights):
    sampled=(weights @ scene_parts.reshape(len(weights[0]),-1)).reshape(len(weights),*scene_parts.shape[1:])
    values=regional_values(sampled)
    return {name:dict(per_horizon_delta_95ci_pp=np.quantile(v[:,left]-v[:,right],[.025,.975],axis=0).T.tolist(),
        mean_delta_95ci_pp=np.quantile((v[:,left]-v[:,right]).mean(-1),[.025,.975]).tolist()) for name,v in values.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows=[]
    lines=["# PMG 新阶段实验结果汇总", "",
           "本报告区分GT辅助输出诊断、冻结基线短程学习和全量从头训练。GT辅助恢复量不作为可部署方法成绩。", "",
           "主指标是当前、1、2、3秒mIoU的平均。表内指标为百分数，差值单位为百分点；快速GT正确率不是IoU。", ""]
    validation=Path("out/pmg_world_stage3_20261023/baseline_validation/validation.json")
    if validation.exists():
        v=json.loads(validation.read_text())
        lines += ["## R0：训练稳定性", "",
            f"microbatch=1、累积4完成{v['updates']}次真实更新，包含验证后继续训练与完整状态重载后继续训练。峰值显存{v['peak_cuda_mib']/1024:.3f} GiB。",
            "连续验证期间，优化完成后的已分配显存约0.90 GiB，没有逐步增长；这不代表任意样本或并发任务下都不会OOM。", "",
            "随后与诊断并发的正式训练及首次短程训练发生CUDA OOM。诊断进程的GPU占用曾约17.1 GiB，而PyTorch日志仅记录约2.3 GiB峰值已分配张量；不能仅依据该峰值安排并发。已改为诊断、两组短程训练、全量训练顺序执行。失败日志保留，不能计作完成的训练预算。", ""]
    diagnosis=ROOT/"diagnosis_1024"
    folder=diagnosis if (diagnosis/"completed.json").exists() else ROOT/"diagnosis_64"
    if not (folder/"metrics.json").exists():
        (ROOT/"results_summary.md").write_text("\n".join(lines+["诊断尚未完成。"])+"\n")
        return
    protocol=json.loads((folder/"protocol.json").read_text())
    data=json.loads((folder/"metrics.json").read_text())
    variants=protocol["variants"]
    metrics=data["variants"]
    class_comparisons={name:metrics[name]["horizons"] for name in ("D00","D10_r20")}
    if "D11_r20_rank_swap" in metrics:
        class_comparisons["D11_r20_rank_swap"]=metrics["D11_r20_rank_swap"]["horizons"]
    with np.load(folder/"scene_counts.npz") as a:
        scenes=a["scenes"].copy()
        globals=a["scene_global_confusion"].copy()
        regions=a["scene_region_confusion"].copy()
    sufficient=regional_sufficient(regions)
    weights=np.random.default_rng(20261023).multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=2000)
    comparisons={}
    pairs=[("D10_r20","position_random_direction"),("D11_r20","D10_r20"),("D11_r20","D01")]
    if "D01_rank_swap" in variants:
        pairs += [("D01_rank_swap","D00"),("D11_r20_rank_swap","D10_r20"),("D11_r20_rank_swap","D01_rank_swap")]
    for left,right in pairs:
        comparisons[left+"_minus_"+right]=paired_intervals(sufficient,variants.index(left),variants.index(right),weights)
    runtime.atomic_json(ROOT/"additional_paired_comparisons.json",comparisons)
    if data["num_samples"]==1024:
        with np.load("analysis/pmg_world_stage2/T_A0_dense/epoch_10_counts.npz") as old:
            parity=dict(global_confusion_exact=bool(np.array_equal(globals.sum(0)[0],old["confusion"])),
                        region_confusion_exact=bool(np.array_equal(regions.sum(0)[0],old["region_confusion"])))
        runtime.atomic_json(ROOT/"native_parity.json",parity)
        lines += [f"## R1：完整1024样本诊断（{len(scenes)}场景）", "",
            f"D00与原T-A0第10轮记录：全域计数完全一致={parity['global_confusion_exact']}；区域计数完全一致={parity['region_confusion_exact']}。", ""]
    else:
        lines += [f"## R1：已完成{data['num_samples']}样本；1024复核尚在运行", ""]
    lines += ["| 干预 | 0/1/2/3秒平均mIoU | 1s快区正确率 | 2s | 3s | 1s邻域IoU | 2s | 3s | 3s F-only |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in variants:
        m=metrics[name]
        values=[m["primary_miou"],*[r["fast"]["occupied_semantic_accuracy"] for r in m["regions"]],
                *[r["fast_neighborhood"]["occupied_iou"] for r in m["regions"]],m["regions"][2]["F_only"]["occupied_semantic_accuracy"]]
        lines.append("| "+name+" | "+" | ".join(f"{v:.4f}" for v in values)+" |")
        for h,value in enumerate(m["horizons"]):
            for c,cls in enumerate(CLASS_NAMES):
                rows.append(dict(experiment_scope="GT_assisted_diagnostic",condition=name,seconds=h*.5,class_index=c,class_name=cls,
                                 iou_percent=value["class_iou"][c],miou_percent=value["miou"]))
    lines += ["", "### 匹配范围与解释边界", "",
        "关联只限有效继承Gaussian与2m内最近fast GT体素，不是物理实例追踪；出生query未修改。以下覆盖率是原始关联到的不同GT体素占比，不能等同于渲染支持率。", "",
        "| 秒 | 匹配Gaussian/有效继承Gaussian | 关联到的不同fast GT体素/全部fast GT体素 |", "|---|---:|---:|"]
    for h in (2,4,6):
        coverage=[r for r in data["coverage"] if r["horizon"]==h]
        matched=sum(r["matched_gaussians"] for r in coverage)
        active=sum(r["active_inherited_gaussians"] for r in coverage)
        touched=sum(r["matched_distinct_fast_voxels"] for r in coverage)
        fast=sum(r["fast_voxels"] for r in coverage)
        lines.append(f"| {h*.5:g} | {100*matched/max(active,1):.3f}% | {100*touched/max(fast,1):.3f}% |")
    lines += ["", "### 应当怎样解释", "",
        "位置纠正和类别组成纠正会分别改变空间贡献和类别竞争。单类别GT注入还会提高类别集中度，因此1024样本额外加入分数排序交换对照，保持每个Gaussian的总和、峰值和熵。组合恢复大于单项并不能单独证明内部动力学耦合；IoU非线性和覆盖改变也会产生非加性。",
        "正确方向与随机方向的配对对照、排序保持控制见additional_paired_comparisons.json。所有结果来自重复使用的开发验证集和一个检查点。", ""]
    def mean_region(name, region, field):
        return float(np.mean([metrics[name]["regions"][h][region][field] for h in range(3)]))
    position_gain=mean_region("D10_r20","fast_neighborhood","occupied_iou")-mean_region("D00","fast_neighborhood","occupied_iou")
    random_gain=mean_region("position_random_direction","fast_neighborhood","occupied_iou")-mean_region("D00","fast_neighborhood","occupied_iou")
    lines += [f"1. **位置是当前Gaussian模型中可恢复的故障环节。** 2m位置纠正的平均快速邻域IoU提高{position_gain:.4f} pp；等长度随机方向提高{random_gain:.4f} pp。正确方向相对随机方向的配对区间见JSON，不能将全部恢复量都归于准确运动方向。该诊断没有验证速度估计的上游原因。", ""]
    if "D01_rank_swap" in variants:
        semantic_gain=mean_region("D01_rank_swap","fast_neighborhood","occupied_iou")-mean_region("D00","fast_neighborhood","occupied_iou")
        joint_gain=mean_region("D11_r20_rank_swap","fast_neighborhood","occupied_iou")-mean_region("D00","fast_neighborhood","occupied_iou")
        fonly_delta=metrics["D11_r20_rank_swap"]["regions"][2]["F_only"]["occupied_semantic_accuracy"]-metrics["D00"]["regions"][2]["F_only"]["occupied_semantic_accuracy"]
        lines += [f"2. **正确类别不能弥补错误的空间贡献。** 保持每Gaussian分数峰值与熵的类别交换，平均邻域IoU变化{semantic_gain:+.4f} pp；配合位置纠正后提高{joint_gain:.4f} pp。类别与位置都存在可恢复量，但这种输出交互不能直接证明内部动力学存在某种唯一耦合。", "",
            f"3. **即使GT辅助修正，也没有自动保住原优势。** 排序保持的组合纠正在3秒F-only变化{fonly_delta:+.4f} pp。局部目标附近的Gaussian同时服务其他空间位置，修改它们会改变周边预测。下一步需要学习何处、如何修正，而非对所有候选统一执行oracle操作。", ""]
    labels=[name for name in variants if name not in ("D00","amplitude_half","semantic_wrong_control")]
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharey=True)
    base=metrics["D00"]
    for ax,region,field,title in zip(axes,["fast","fast_neighborhood","F_only"],["occupied_semantic_accuracy","occupied_iou","occupied_semantic_accuracy"],["Fast semantic accuracy","Fast-neighborhood IoU","F-only retention"]):
        delta=[np.mean([metrics[n]["regions"][h][region][field]-base["regions"][h][region][field] for h in range(3)]) for n in labels]
        ax.barh(np.arange(len(labels)),delta,color=["#247ba0" if d>=0 else "#d66a5e" for d in delta])
        ax.axvline(0,color="black",linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("Mean change at 1/2/3 s (pp)")
        ax.grid(axis="x",alpha=.2)
    axes[0].set_yticks(np.arange(len(labels)),labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(ROOT/"diagnostic_overview.png",dpi=180)
    fig.savefig(ROOT/"diagnostic_overview.svg")
    plt.close(fig)
    lines += ["![诊断结果对比](diagnostic_overview.png)", "", "## R2：冻结基线短程训练", "",
        "两组从同一T-A0出发，只训练同容量的位置修正头，每组512次更新、2048个训练样本。它们不是从头15轮或全量收敛结果。", "",
        "| 条件 | 状态 | 主指标 | 1s快区正确率 | 2s | 3s | 1s邻域IoU | 2s | 3s |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    adaptation_parts=[]
    adaptation_globals=[]
    adaptation_names=[]
    invariants={}
    for arm in ("current","history"):
        report=ROOT/f"frozen_M_{arm}"
        completed=Path(f"out/pmg_world_stage3_20261023/frozen_M_{arm}/completed.json")
        if not completed.exists():
            lines.append(f"| M-{arm} | 尚未完成完整训练和验证 | — | — | — | — | — | — | — |")
            continue
        m=json.loads((report/"epoch_01_metrics.json").read_text())
        class_comparisons[f"M_{arm}"]=m["horizons"]
        runtime.atomic_json(report/"final_512updates_metrics.json",dict(m,
            actual_training_updates=512,training_samples_consumed=2048,evaluation_round=1,
            budget_note="Frozen checkpoint adaptation; epoch_01 is an evaluation index, not a complete scratch-training epoch."))
        (report/"README.md").write_text(
            f"# M-{arm} 冻结基线短程对照\n\n"
            "完成512次优化更新，使用2048个训练样本，完整验证1024样本。原生T-A0参数冻结，只训练位置修正头。\n\n"
            "epoch_01文件名指本次唯一一次最终评估，不代表完成一个4096样本epoch，更不代表15轮从头训练。\n\n"
            "[明确预算的最终指标](final_512updates_metrics.json)；完整对比、逐类CSV和区间见上一级results_summary.md。\n")
        import torch
        source=torch.load("out/pmg_world_stage2/T_A0_dense/best.pth",map_location="cpu",mmap=True,weights_only=False)
        learned=torch.load(Path(f"out/pmg_world_stage3_20261023/frozen_M_{arm}/latest.pth"),map_location="cpu",mmap=True,weights_only=False)
        changed=[key for key,value in source["state_dict"].items() if not torch.equal(value,learned["state_dict"][key])]
        invariants[arm]=dict(native_state_changed_keys=changed,native_state_exact=not changed,
            query_assignment_exact=bool(torch.equal(source["query_stamps"],learned["query_stamps"])),
            correction_parameter_count=sum(v.numel() for k,v in learned["state_dict"].items() if k.startswith("motion_correction.")))
        del source,learned
        values=[m["selection_mean_miou"],*[r["metrics"]["fast"]["occupied_semantic_accuracy"] for r in m["regions"]],
                *[r["metrics"]["fast_neighborhood"]["occupied_iou"] for r in m["regions"]]]
        lines.append(f"| M-{arm} | 完成512更新和1024验证 | "+" | ".join(f"{v:.4f}" for v in values)+" |")
        with np.load(report/"epoch_01_counts.npz") as a:
            assert np.array_equal(a["scenes"],scenes)
            adaptation_parts.append(regional_sufficient(a["scene_region_confusion"]))
            adaptation_globals.append(a["scene_confusion"].copy())
            adaptation_names.append(arm)
        for h,value in enumerate(m["horizons"]):
            for c,cls in enumerate(CLASS_NAMES):
                rows.append(dict(experiment_scope="frozen_short_adaptation",condition=f"M_{arm}",seconds=h*.5,class_index=c,class_name=cls,
                                 iou_percent=value["class_iou"][c],miou_percent=value["miou"]))
    if adaptation_parts:
        runtime.atomic_json(ROOT/"frozen_state_invariants.json",invariants)
        combined=np.stack([sufficient[:,0],*adaptation_parts],axis=1)
        comparisons={name+"_minus_frozen_T_A0":paired_intervals(combined,i+1,0,weights) for i,name in enumerate(adaptation_names)}
        if len(adaptation_parts)==2:
            comparisons["history_minus_current"]=paired_intervals(combined,2,1,weights)
        global_combined=np.stack([globals[:,0],*adaptation_globals],axis=1)
        global_boot=(weights.astype(float) @ global_combined.reshape(len(scenes),-1).astype(float)).reshape(2000,len(adaptation_globals)+1,7,18,18)
        diagonal=np.diagonal(global_boot,axis1=-2,axis2=-1)
        union=global_boot.sum(-1)+global_boot.sum(-2)-diagonal
        global_iou=100*np.divide(diagonal,union,out=np.zeros_like(diagonal,dtype=float),where=union>0)
        global_miou=global_iou[...,:17].mean(-1)
        pairs=[(i+1,0,name+"_minus_frozen_T_A0") for i,name in enumerate(adaptation_names)]
        if len(adaptation_names)==2:
            pairs.append((2,1,"history_minus_current"))
        for left,right,name in pairs:
            delta=global_miou[:,left]-global_miou[:,right]
            comparisons[name]["global_miou"]=dict(
                per_horizon_delta_95ci_pp=np.quantile(delta,[.025,.975],axis=0).T.tolist(),
                primary_delta_95ci_pp=np.quantile(delta[:,[0,2,4,6]].mean(-1),[.025,.975]).tolist(),
                future_123_delta_95ci_pp=np.quantile(delta[:,[2,4,6]].mean(-1),[.025,.975]).tolist())
        runtime.atomic_json(ROOT/"adaptation_scene_statistics.json",comparisons)
        lines += ["", "| 对照 | 平均快速邻域IoU差值95%区间pp | 3s F-only差值95%区间pp |",
                  "|---|---|---|"]
        for name,value in comparisons.items():
            a=value["fast_neighborhood_iou"]["mean_delta_95ci_pp"]
            b=value["F_only"]["per_horizon_delta_95ci_pp"][2]
            lines.append(f"| {name} | [{a[0]:+.4f}, {a[1]:+.4f}] | [{b[0]:+.4f}, {b[1]:+.4f}] |")
        lines += ["", "原生状态和query分配的逐元素核对见frozen_state_invariants.json。冻结对照中的变化只能归于新增输出修正及其渲染后果，不能归于原生参数被重新训练。"]
        if len(adaptation_names)==2:
            lines += ["", "### 本轮方法筛选结论", "",
                "两组在平均快速邻域IoU上的提升均不足0.1 pp，未达到预设1.0 pp门槛；F-only三个时刻均有明显损失，未通过−0.5 pp非劣要求。因此本轮不将这两个原型扩展为正式15轮架构比较。",
                "这是512更新的冻结基线对照，不能据此否定更久训练、端到端适配或其他历史运动读取。普通历史投影平均并不保留逐帧对应，也没有显式物体运动对齐。", ""]
    audit=ROOT/"learned_head_audit.json"
    if audit.exists():
        audited=json.loads(audit.read_text())
        lines += ["## 修正头审计", "",
            "固定4个样本上，归零修正后七时刻原生计数完全复现；实际修正前后Gaussian语义、尺度、方向和出生状态保持一致。只改变继承中心及由ROI有效性引起的opacity遮罩。", "",
            "| 组别 | 1s平均位移m | 2s | 3s | 3s有效继承Gaussian中位移>1cm的比例 |", "|---|---:|---:|---:|---:|"]
        for arm,value in audited.items():
            magnitudes=[np.mean([r['correction_l2_m']['mean'] for r in value['records'] if r['seconds']==h]) for h in (1.,2.,3.)]
            fraction=np.mean([r['active_correction']['fraction_gt_1cm'] for r in value['records'] if r['seconds']==3.])
            lines.append(f"| {arm} | {magnitudes[0]:.4f} | {magnitudes[1]:.4f} | {magnitudes[2]:.4f} | {100*fraction:.3f}% |")
        lines += ["", "这些是4样本的描述性审计，不是真实实例速度误差。更新并未集中在少数Gaussian；原生状态冻结且不反馈递推仍不足以保证F-only保持。位移选择性是下一项直接干预候选，但尚未证明它单独解释所有损失。完整输入尺度、类别分组、门值和ROI进出见learned_head_audit.json。", ""]
    lines += ["", "R2的最终判断须结合adaptation_scene_statistics.json、F-only保持率和匹配预算，不能由GT辅助大幅恢复直接推出修正头能学到同样收益。", "",
        "## 完整数据baseline", "",
        "完整计划为15轮，短程运行上限3轮或5小时。并发失败后安排在R1和R2进程退出后独占GPU重启；实际运行状态与断点见out/sparse_opus_gaussian_world_A0_full_dense_mb1_15e，不把短程稳定性验证写成已完成epoch。", "",
        "## 文件", "",
        "- [实验设计](experiment_design.md)",
        "- [具体机制：语义分配如何改变二值占据](mechanism_analysis.md)",
        f"- [本次诊断完整表]({folder.name}/report.md)",
        f"- [诊断逐时刻逐类别IoU]({folder.name}/per_horizon_class_iou.csv)",
        "- [全部实验逐时刻逐类别汇总](all_experiment_class_iou.csv)",
        "- [主要条件的逐类可读对比](per_class_comparison.md)",
        "- [额外配对区间](additional_paired_comparisons.json)", "",
        "- [本轮决策与下一步](post_run_decision.md)", "",
        "待执行：从头训练的候选架构、完整15轮、多种子、反馈/不反馈与显式运动对齐对照、独立数据范围确认。"]
    runtime_folder=Path("out/sparse_opus_gaussian_world_A0_full_dense_mb1_15e")
    if (runtime_folder/"resume_audit.json").exists():
        audit=json.loads((runtime_folder/"resume_audit.json").read_text())
        snapshot=json.loads((runtime_folder/"progress.json").read_text())
        lines += ["", "### 全量运行的本次状态快照", "",
            f"正式断点在{audit['saved_updates']}次优化更新、{audit['saved_samples']}个样本处保存，并已恢复继续训练；检查时优化器包含{audit['optimizer_state_entries']}个参数状态。",
            f"本次汇总读取到第{snapshot['epoch']}轮、{snapshot['updates']}次累计更新、该轮{snapshot.get('samples',0)}/19730样本。尚未完成一个全量epoch的验证，不提供全量性能结论。实时状态请查看训练目录progress.json与train.log。",
            "[全量baseline说明](../sparse_opus_gaussian_world_A0_full_dense_mb1_15e/README.md)。"]
    with (ROOT/"all_experiment_class_iou.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    class_lines=["# 每个时刻、每个类别的IoU对比", "",
        "单位为百分数；后续表为相对冻结T-A0基线的百分点变化。mIoU只平均17个非空类，free单独展示。GT诊断与学习模型分开命名；这些逐类差值是描述统计，未对多重比较作显著性校正。", "",
        "## 冻结T-A0原始IoU", "", "| 类别 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c,name in enumerate(CLASS_NAMES):
        class_lines.append("| "+name+" | "+" | ".join(f"{h['class_iou'][c]:.4f}" for h in class_comparisons['D00'])+" |")
    for condition,horizons in class_comparisons.items():
        if condition=='D00':
            continue
        class_lines += ["",f"## {condition}：相对基线的变化", "", "| 类别 | 0s | 0.5s | 1s | 1.5s | 2s | 2.5s | 3s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c,name in enumerate(CLASS_NAMES):
            class_lines.append("| "+name+" | "+" | ".join(f"{horizons[h]['class_iou'][c]-class_comparisons['D00'][h]['class_iou'][c]:+.4f}" for h in range(7))+" |")
    (ROOT/"per_class_comparison.md").write_text("\n".join(class_lines)+"\n")
    (ROOT/"results_summary.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
