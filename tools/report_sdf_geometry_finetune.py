"""Write paired full-validation performance and Gaussian geometry results."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.run_sdf_geometry_finetune import RUN, REPORT, SDF_NAMES
import tools.report_gaussian_geometry_causal as base


def geometry(a):
    x = dict(zip(SDF_NAMES, a['sdf_geometry']))
    n=max(x['gaussians'],1); cv=max(x['center_valid'],1); sv=max(x['shape_valid'],1)
    return dict(SDF_center_valid_percent=100*x['center_valid']/n,
        SDF_shape_valid_percent=100*x['shape_valid']/n,
        SDF_outside_percent=100*x['outside_count']/cv,
        SDF_positive_distance_m=x['sum_positive_sdf']/cv,
        SDF_signed_distance_m=x['sum_signed_sdf']/cv,
        SDF_normal_scale_m=x['sum_normal_scale']/sv,
        SDF_shape_violation_percent=100*x['shape_violation_count']/sv,
        GS_camera_foreground_percent=100*x['center_camera_foreground']/n,
        GS_camera_empty_percent=100*x['center_camera_empty']/n,
        GS_camera_unknown_percent=100*x['center_unknown']/n,
        GS_center_class_accuracy_percent=100*x['center_foreground_same_class']/max(x['center_camera_foreground'],1))


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report():
    base.RUN=RUN
    (RUN/'aggregate').mkdir(exist_ok=True)
    arrays={}; scenes={}; summary={}; scene_rows=[]; class_rows=[]
    for name in ('before','after'):
        a,s,rows=base.aggregate(name,'val',dict(frames=6019,scenes=150))
        arrays[name]=a;scenes[name]=s
        summary[name]={**base.metrics(a),**geometry(a)}
        for row in rows:
            with np.load(RUN/'eval'/name/'val'/(row['scene']+'.npz')) as data:
                row.update(geometry(data))
            scene_rows.append(row)
        class_rows.extend(base.class_rows(name,'val',a['confusion']))
    assert np.array_equal(arrays['before']['confusion'].sum(1),arrays['after']['confusion'].sum(1))
    assert int(arrays['before']['loss'][2])==6019 and int(arrays['after']['loss'][2])==6019
    interval=base.bootstrap(scenes['before'],scenes['after'])
    delta={k:summary['after'][k]-summary['before'][k] for k in summary['before']}
    result=dict(status='complete',validation_frames=6019,validation_scenes=150,
                models=summary,delta=delta,scene_bootstrap_95=interval)
    base.write(RUN/'summary.json',result)
    csv_write(RUN/'scene_metrics.csv',scene_rows);csv_write(RUN/'class_metrics.csv',class_rows)
    performance=[('mIoU','mIoU（%）'),('IoU','二值 IoU（%）'),('FN_rate','漏检率：FN／GT前景（%）'),
        ('FPR','误占用率：FP／GT empty（%）'),('FP_over_GTfg','FP／GT前景（%）'),
        ('FP_over_predfg','FP／预测前景（%）'),('semantic_error','GT前景错类率（%）')]
    geometric=[('GS_camera_foreground_percent','中心位于可见前景／全部GS（%）'),
        ('GS_camera_empty_percent','中心位于可见empty／全部GS（%）'),
        ('GS_camera_unknown_percent','中心位于camera mask外／全部GS（%）'),
        ('SDF_center_valid_percent','SDF中心监督有效GS／全部GS（%）'),
        ('SDF_shape_valid_percent','SDF形状监督有效GS／全部GS（%）'),
        ('SDF_outside_percent','中心位于前景外／SDF有效GS（%）'),
        ('SDF_positive_distance_m','正距离均值：内部记0（m）'),
        ('SDF_normal_scale_m','法线方向尺度／形状有效GS（m）'),
        ('SDF_shape_violation_percent','违反形状约束／形状有效GS（%）'),
        ('n_1sigma','GT前景平均1σ覆盖数'),('n_2sigma','GT前景平均2σ覆盖数'),
        ('no_1sigma','GT前景没有1σ覆盖（%）'),('no_2sigma','GT前景没有2σ覆盖（%）'),
        ('GS_sum_A','总语义幅度均值'),('GS_sum_alpha','不透明度均值'),
        ('GS_sum_sigma','三轴尺度均值（m）'),('GS_maxscale_ge_03999','最大尺度饱和GS（%）')]
    def table(rows):
        text='| 指标 | 原始权重 | 微调1 epoch | 差值：微调−原始 |\n| --- | ---: | ---: | ---: |\n'
        for key,label in rows:
            b=summary['before'][key];a=summary['after'][key]
            text+=f'| {label} | {b:.4f} | {a:.4f} | {a-b:+.4f} |\n'
        return text
    text='# SDF 替换 Set 几何监督：完整数据微调结果\n\n'
    text+='**本轮已完成：训练700场景／28,130帧恰好一整轮，微调前后分别评估完整150场景／6,019帧。**\n\n'
    text+=f"微调后 mIoU 为 **{summary['after']['mIoU']:.4f}%**，相对原权重变化 **{delta['mIoU']:+.4f} 个百分点**；二值 IoU 变化 **{delta['IoU']:+.4f} 个百分点**。\n\n"
    text+='## 1. 性能变化\n\n'+table(performance)
    text+='\n指标由完整camera mask范围内的混淆矩阵汇总；未额外调节empty偏移、逐类别bias或温度。模型训练所得empty参数保留。\n\n'
    text+='## 2. 几何分布与覆盖\n\n'+table(geometric)
    text+='\n中心／形状SDF指标的有效集合会随预测位置变化，所以同时给出有效比例。判断几何改善时，以固定GT前景上的覆盖指标和全部GS中心分布共同解释；不能只看被筛选后的距离均值。这里的中心类别准确率只在中心位于camera-valid前景的GS中统计。\n\n'
    text+='## 3. 配对场景不确定性\n\n| 指标差值 | 场景bootstrap 95%区间 |\n| --- | --- |\n'
    for key,values in interval.items():text+=f'| {key} | [{values[0]:+.4f}, {values[1]:+.4f}] |\n'
    text+='\n该区间描述固定这两个权重在不同场景上的差异。本轮只有一次训练，且没有保留原几何损失的同条件继续训练对照，因此不单独估计替换损失的净因果收益。\n\n'
    text+='## 4. 本轮结论\n\n'
    if delta['no_1sigma']<0 and delta['GS_camera_empty_percent']<0:
        text+='全前景的零核心覆盖比例和可见empty中心比例同时下降，支持本次微调改善了这两项几何分布指标。仍需结合误占用与漏检，判断改进是否转化为更好的预测。\n\n'
    else:
        text+='几何指标没有同时朝预期方向变化。分别检查中心分布、SDF有效比例、前景覆盖与尺度饱和，不能据某一距离或尺度指标单独宣布几何优化成功。\n\n'
    text+='本轮移除了所有阶段的双向Set几何项，SDF只作用于最后一层。因而结果还反映覆盖监督的变化；后续可与保留GT→GS项的方案区分，但本报告不混入未运行对照。\n\n'
    text+='## 5. 复核资料\n\n'
    text+='- [实验设计](sdf_geometry_replacement_full_design.md)\n'
    text+='- [配置](../config/sdf_geometry_replacement_full/opusv1_gaussian_softplus_tiny_8f_sdf_1ep.py)\n'
    text+='- [汇总计数与指标](../work_dirs/sdf_geometry_replacement_full/summary.json)\n'
    text+='- [逐场景指标](../work_dirs/sdf_geometry_replacement_full/scene_metrics.csv)\n'
    text+='- [逐类别指标](../work_dirs/sdf_geometry_replacement_full/class_metrics.csv)\n'
    text+='- [训练完成记录](../work_dirs/sdf_geometry_replacement_full/train/complete.json)\n'
    text+='- [微调权重](../work_dirs/sdf_geometry_replacement_full/train/final.pth)\n'
    REPORT.write_text(text)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':report()
