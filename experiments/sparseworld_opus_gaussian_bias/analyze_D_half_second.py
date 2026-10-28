"""Read-only epoch-15 diagnosis of D's half-second prediction gap."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy
from zoneinfo import ZoneInfo

import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'out/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024'
WORK = ROOT/'D_balanced_conditional_ce'
ANALYSIS = REPO/'analysis/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024'
CONFIG = REPO/'config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_continue_4096_1024_15e.py'
PROBE = ANALYSIS/'half_second_probe'
NAMES = ['others','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian',
         'traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','free']


def read(path):
    return json.loads(path.read_text())


def collect():
    import torch
    from mmengine.config import Config
    from mmengine.runner import set_random_seed
    from mmseg.models import build_segmentor
    from dataset import OPENOCC_DATASET
    from . import train as runner
    from .conditional_first_epoch.diagnostics import conditional_confusion
    from .fullspace_diagnostics import partition_histograms, region_metrics

    if (PROBE/'probe.json').exists():
        raise ValueError('read-only probe already exists; preserve it')
    PROBE.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(str(CONFIG))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    set_random_seed(cfg.seed)
    checkpoint=WORK/'best_mIoU.pth'
    if not checkpoint.exists():
        checkpoint=WORK/'latest.pth'
    before=runner.digest(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    epoch_index=saved.get('epoch_index',saved.get('epoch',0)-1)
    if epoch_index!=14:
        raise ValueError('expected D epoch fifteen')
    model=build_segmentor(cfg.model)
    model.init_weights()
    model.load_state_dict(saved['state_dict'],strict=True)
    model.cuda()
    model.set_epoch(epoch_index)
    model.pts_bbox_head.ind_stamps_all=saved['query_stamps'].cuda()
    model.pts_bbox_head.reset_mask()
    model.eval()
    model.memory.clear()
    model.memory_queue.clear()
    del saved
    dataset=OPENOCC_DATASET.build(cfg.val_dataset_config)
    indices=np.linspace(0,len(dataset)-1,16,dtype=int).tolist()
    formal=np.zeros((7,18,18),dtype=np.int64)
    conditional=np.zeros((7,17,17),dtype=np.int64)
    deltas=[0.,1.,2.,3.]
    alternatives=np.zeros((len(deltas),7,18,18),dtype=np.int64)
    margins=[]
    original=model.gaussian_head.render_state
    captured=[]

    def capture(gaussian,xyz):
        logits=original(gaussian,xyz)
        captured.append(logits)
        return logits

    model.gaussian_head.render_state=capture
    try:
        with torch.no_grad():
            for sample,data in enumerate(runner.loader(dataset,cfg.val_loader,order=indices)):
                captured.clear()
                images,metas=runner.move_batch(data,'cuda')
                output=model(imgs=images,metas=metas)
                if len(captured)!=7:
                    raise ValueError('expected all seven rendered horizons')
                for h,logits in enumerate(captured):
                    target=metas['occ_label'] if h==0 else metas['future_occ_labels'][:,h-1]
                    camera=metas['occ_cam_mask'] if h==0 else metas['future_occ_cam_masks'][:,h-1]
                    labels=target.flatten(1).long()
                    foreground,classes=logits[:,:17].max(1)
                    margin=foreground-logits[:,17]
                    occupied=(labels>=0)&(labels<17)
                    formal[h]+=partition_histograms(logits.argmax(1).view_as(target),target,camera)['full']
                    conditional[h]+=conditional_confusion(logits,target,camera)['full']
                    for i,delta in enumerate(deltas):
                        prediction=torch.where(margin+delta>0,classes,torch.full_like(classes,17))
                        alternatives[i,h]+=partition_histograms(prediction.view_as(target),target,camera)['full']
                    conditional_correct=occupied&(classes==labels)
                    margins.append(dict(index=indices[sample],horizon=h,
                        GT_occupied_margin_quantiles=np.quantile(margin[occupied].cpu().numpy(),[.1,.5,.9]).tolist(),
                        GT_occupied_correct_conditional_voxels=int(conditional_correct.sum()),
                        correct_conditional_but_formally_free=int((conditional_correct&(margin<=0)).sum())))
                print(f'HALF_SECOND_PROBE samples={sample+1}/16',flush=True)
                del output,images,metas
    finally:
        model.gaussian_head.render_state=original
        captured.clear()
    if runner.digest(checkpoint)!=before:
        raise ValueError('read-only probe changed checkpoint')
    if not np.array_equal(conditional.sum(2),formal.sum(2)[:,:17]):
        raise ValueError('conditional and formal GT counts differ')
    original_probe=read(ANALYSIS/'epoch_diagnostics/epoch_15_diagnostics.json')
    matches=[]
    for h in range(7):
        expected=original_probe['render_probes'][f'h{h}/default/full']
        actual=region_metrics(formal[h])
        matches.append(bool(np.allclose(actual['class_iou'],expected['class_iou'],atol=1e-10,rtol=1e-10)))
    np.savez_compressed(PROBE/'confusion.npz',formal=formal,conditional=conditional,alternatives=alternatives)
    runner.atomic_json(PROBE/'probe.json',dict(status='passed',samples=16,sample_indices=indices,
        checkpoint=str(checkpoint),checkpoint_sha256=before,checkpoint_unchanged=True,training_advanced=False,
        inference_only=True,output_bias_deltas=deltas,default_matches_saved_sixteen_sample_probes=matches,
        margin_records=margins,conditional_metrics=[dict(horizon=h,
            micro_accuracy_percent=100*float(np.trace(hist))/int(hist.sum()),
            macro_recall_percent=100*float(np.divide(hist.diagonal(),hist.sum(1),out=np.zeros(17),where=hist.sum(1)>0).mean()),
            class_recall_percent=(100*np.divide(hist.diagonal(),hist.sum(1),out=np.zeros(17),where=hist.sum(1)>0)).tolist())
            for h,hist in enumerate(conditional)],
        alternative_metrics=[[region_metrics(hist) for hist in arm] for arm in alternatives],
        scope='Fixed sixteen validation samples only; bias sweeps diagnose calibration and do not define a selected deployment threshold.'))


def report():
    cfg=runpy.run_path(str(CONFIG))
    best={label:read(ROOT/name/'best.json') for label,name in [('Strict','sparseworld_strict'),('D','D_balanced_conditional_ce')]}
    hist={label:np.load(ROOT/name/f"epoch_{best[label]['epoch']:02d}_confusion.npz",allow_pickle=False)['confusion']
          for label,name in [('Strict','sparseworld_strict'),('D','D_balanced_conditional_ce')]}
    diag=read(ANALYSIS/'epoch_diagnostics/epoch_15_diagnostics.json')
    probe=read(PROBE/'probe.json')
    c,d=best['Strict']['horizons'][1],best['D']['horizons'][1]
    if c['class_gt_count']!=d['class_gt_count']:
        raise ValueError('D/Strict GT differs')
    class_rows=[]
    for i,name in enumerate(NAMES[:17]):
        row=dict(class_name=name,Strict_iou=c['class_iou'][i],D_iou=d['class_iou'][i],delta_iou_pp=d['class_iou'][i]-c['class_iou'][i])
        for label in ('Strict','D'):
            h=hist[label][1]
            gt,hit=int(h[i].sum()),int(h[i,:17].sum())
            row[label]=dict(gt_voxels=gt,tp=int(h[i,i]),fp=int(h[:,i].sum()-h[i,i]),
                GT_predicted_free_percent=100*int(h[i,17])/gt,
                correct_given_occupied_hit_percent=100*int(h[i,i])/max(hit,1),
                GT_predicted_road_percent=100*int(h[i,11])/gt)
        class_rows.append(row)
    attrs=[]
    for h in range(7):
        rows=[r for r in diag['attributes'] if r['horizon']==h]
        attrs.append(dict(horizon=h,queries=float(np.mean([r['queries'] for r in rows])),
            feature_median_mean=float(np.mean([r['feature_norm_quantiles'][2] for r in rows])),
            semantic_max_median_mean=float(np.mean([r['softplus_semantic_max_quantiles'][2] for r in rows])),
            legal_centers_percent=float(np.mean([r['in_range_percent'] for r in rows]))))
    epochs=[read(WORK/f'epoch_{e:02d}_metrics.json') for e in range(1,16)]
    best_half=max(epochs,key=lambda x:x['horizons'][1]['miou'])
    result=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        formal_samples=1024,best_epochs={k:v['epoch'] for k,v in best.items()},
        gaussian_loss_horizons=cfg['model']['gaussian_loss_horizons'],
        current_half_second_miou_delta_pp=d['miou']-c['miou'],half_second_classes=class_rows,
        attributes_16_samples=attrs,read_only_probe=probe,
        best_half_second_epoch=best_half['epoch'],best_half_second_miou=best_half['horizons'][1]['miou'],
        best_half_second_gain_over_selected_pp=best_half['horizons'][1]['miou']-d['miou'],
        first_forecast_loss_curriculum=dict(set_loss_epochs=10,direct_gaussian_epochs=0),
        training_advanced=False)
    (ANALYSIS/'D_half_second_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# D的0.5秒预测为何偏低', '',
        'D完整15轮训练已完成并通过最优checkpoint独立复评。主比较为D第15轮与Strict第14轮；二者共用1024样本、全空间评测和相同GT。这里分析语义mIoU缺口，不把单seed差异称为已验证的训练统计显著性。', '',
        '## 事实：0.5秒的语义mIoU低，占用IoU几乎持平', '',
        '| 指标 | Strict % | D % | Δ pp |','|---|---:|---:|---:|']
    for key,label in [('miou','0.5秒mIoU'),('occupied_iou','0.5秒二元占用IoU'),('occupied_precision','占用precision'),('occupied_recall','占用recall')]:
        lines.append(f'| {label} | {c[key]:.6f} | {d[key]:.6f} | {d[key]-c[key]:+.6f} |')
    lines+=['','D的0.5秒TP比Strict少2,548,483，FP少12,120,708，FN多2,548,483。表现为更保守、更多漏检。所有命中非空体素的语义正确率为D 80.5010%、Strict 79.4645%，但这一按体素加权的指标掩盖了个别类别的严重混淆。','',
        '## 1. 直接监督时刻与输出时刻不一致','',
        '配置gaussian_loss_horizons=[0,2,4,6]，对应0、1、2、3秒；0.5、1.5、2.5秒不加入Gaussian体素损失。模型对这些时刻的Gaussian构建和渲染使用torch.no_grad，输出也不进入gaussian_logits_by_horizon；因此原18类体素CE、Lovasz和D的平衡17类辅助CE都不直接监督0.5秒输出。','',
        '0.5秒仍受SparseWorld原有未来点集合分类/位置损失监督，不能说完全没有训练。未来集合监督从第6轮开始，该时刻有10轮集合监督；1秒从第7轮开始有9轮直接Gaussian体素监督。这个时间差说明0.5秒的弱势不能简单解释为训练轮数更少。','',
        '集合损失主要约束点位置与点语义；正式评测却使用受协方差、opacity、多个高斯重叠和空类偏置共同影响的稠密体素输出。D在直接监督时刻学到的渲染校准，没有保证迁移到0.5秒。共享参数仍带来间接训练影响，但不能替代该时刻的直接损失。','',
        '## 2. 为什么同样没有直接监督的1.5、2.5秒没有落后','',
        '遗漏直接监督是已确认的机制，不能单独解释全部时刻差异。滚动预测的第一步由感知特征首次叠加ego和position信息，并拼入0.5秒query；后续递归在已更新的world特征上继续滚动。1.5、2.5秒分别夹在1/2秒、2/3秒这两对直接监督world状态之间；0.5秒处于感知到world的首次转换。共享属性分支对这些分布的泛化可能不同，这是结合代码和属性统计的推断，需要补监督消融确认。','',
        '| 时刻 | 直接Gaussian监督 | 特征范数中位值的样本均值 | softplus语义最大值中位值的样本均值 | 合法中心 % |','|---|---|---:|---:|---:|']
    for a in attrs:
        lines.append(f"| {a['horizon']*.5:g}s | {'有' if a['horizon'] in [0,2,4,6] else '无'} | {a['feature_median_mean']:.4f} | {a['semantic_max_median_mean']:.4f} | {a['legal_centers_percent']:.4f} |")
    lines+=['','0.5秒的语义幅值约0.7274，低于1秒的0.9126；但合法中心比例更高，不能把低分简单归于高斯中心越界。每步共享empty bias=2.882466，实际非空证据幅值与空间重叠却不同，因而同一偏置产生不同precision/recall。属性统计来自固定16样本，不是完整1024样本的分布估计。','',
        '## 3. 缺口集中在other_flat、vegetation和车辆','',
        '| 类别 | Strict IoU % | D IoU % | Δ pp | Strict GT预测为空 % | D GT预测为空 % | Strict命中后语义正确 % | D命中后语义正确 % |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for i in [12,16,4,10,1,3,2,6]:
        r=class_rows[i]
        lines.append(f"| {r['class_name']} | {r['Strict_iou']:.4f} | {r['D_iou']:.4f} | {r['delta_iou_pp']:+.4f} | {r['Strict']['GT_predicted_free_percent']:.4f} | {r['D']['GT_predicted_free_percent']:.4f} | {r['Strict']['correct_given_occupied_hit_percent']:.4f} | {r['D']['correct_given_occupied_hit_percent']:.4f} |")
    lines+=['','other_flat单类差距对17类平均贡献−0.4533 pp，已占净mIoU缺口约55.66%。该类被误判道路的比例28.7988%→49.7259%，而预测为空的比例36.8021%→34.5746%，所以这一主要缺口是类别混淆，降低空类分数不能直接修复。vegetation预测为空74.2132%→86.9732%，则主要体现漏检；car、truck同时存在漏检和语义混淆。','',
        '## 4. 已有与新增只读诊断：校准问题存在，但阈值不能包治','',
        '原训练固定16样本上，0.5秒默认mIoU=8.4641%，empty bias额外+1/+2时为9.1575%/9.5511%；同一批样本1秒默认10.6808%，加偏置反而下降。0.5秒点栅格化8.8650%也高于Gaussian默认，而其他时刻Gaussian均高于对应点栅格化。该现象支持0.5秒没有充分获得Gaussian渲染收益。探针数值不应混同1024样本正式成绩。','',
        '新增探针只读取D第15轮最佳checkpoint，在同样16个预定样本上补充所有时刻条件17类统计及偏置诊断，不推进训练、不更改checkpoint。','',
        '| 时刻 | 条件17类micro % | 条件17类macro召回 % | 默认mIoU % | bias+1 mIoU % | bias+2 mIoU % |','|---|---:|---:|---:|---:|---:|']
    for h,cond in enumerate(probe['conditional_metrics']):
        alt=probe['alternative_metrics']
        lines.append(f"| {h*.5:g}s | {cond['micro_accuracy_percent']:.4f} | {cond['macro_recall_percent']:.4f} | {alt[0][h]['miou']:.4f} | {alt[1][h]['miou']:.4f} | {alt[2][h]['miou']:.4f} |")
    lines+=['',f"新增探针与原训练16样本逐时刻类别IoU一致性：{probe['default_matches_saved_sixteen_sample_probes']}。偏置扫描仅用于定位，不将验证集上选择的偏置当作正式改进成绩。",'',
        '## 5. 不是选择了错误的checkpoint','',
        f"D在0.5秒单独最优为第{best_half['epoch']}轮，mIoU={best_half['horizons'][1]['miou']:.6f}%；比共同选择的第15轮仅高{result['best_half_second_gain_over_selected_pp']:.6f} pp，仍低于Strict。因此主选择指标没有包含0.5秒，虽然可能降低关注度，但不是当前缺口的主要原因。第11～15轮该时刻持续偏弱，也不像一次评测偶然波动。",'',
        '## 建议的最小验证顺序','',
        '1. 优先补上0.5秒完整Gaussian体素监督，包含原18类CE、Lovasz及D的平衡17类辅助CE；其余训练流程不变，并设置未来总损失权重保持一致的对照，区分类别/时刻覆盖收益与总梯度增强。','2. 若第一步有效，再对所有半秒时刻覆盖直接体素监督，检查时刻间precision/recall振荡是否收敛；不先改几何架构。','3. 保持同一checkpoint做仅校准诊断，可以量化可恢复的漏检；正式偏置应在训练侧校准数据上确定，other_flat→道路的混淆仍需语义监督修复。','',
        '本次只完成原因分析与只读探针，没有修改训练配置或启动新训练。','',
        '- [完整分析数据](D_half_second_analysis.json)','- [新增16样本只读探针](half_second_probe/probe.json)','- [15轮正式结果](report.md)','- [时刻性能与占用召回图](half_second_horizon_diagnosis.png)','']
    (ANALYSIS/'D_half_second_analysis.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    times=np.arange(7)*.5
    for label,color in [('Strict','#596b87'),('D','#208578')]:
        axes[0].plot(times,[h['miou'] for h in best[label]['horizons']],marker='o',label=label,color=color)
        axes[1].plot(times,[h['occupied_recall'] for h in best[label]['horizons']],marker='o',label=label,color=color)
    for ax in axes:
        ax.axvspan(.43,.57,color='#d67a2e',alpha=.15)
        ax.set_xlabel('Prediction horizon (s)')
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_ylabel('17-class mIoU (%)')
    axes[1].set_ylabel('Binary occupancy recall (%)')
    fig.suptitle('1024 validation samples; selected D epoch 15 / Strict epoch 14; one seed')
    fig.tight_layout()
    fig.savefig(ANALYSIS/'half_second_horizon_diagnosis.png',dpi=180)
    fig.savefig(ANALYSIS/'half_second_horizon_diagnosis.pdf')
    plt.close(fig)
    print(f'Half-second analysis: {ANALYSIS / "D_half_second_analysis.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    collect() if args.probe else report()
