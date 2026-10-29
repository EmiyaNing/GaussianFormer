"""All-scene mature-checkpoint continuation summary, with original census reference."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.gaussian_geometry_causal import RUN,write
from tools.report_gaussian_geometry_causal import aggregate,metrics,cm_metrics,bootstrap,calibrated,class_rows
from tools.gaussian_geometry_mature_control import NAMES


def report():
    assert json.loads((RUN/'mature_complete.json').read_text())['status']=='complete'
    protocol=json.loads((RUN/'protocol.json').read_text())
    mature_protocol=json.loads((RUN/'mature_protocol.json').read_text())
    reference_summary=json.loads((ROOT/'work_dirs/gaussian_scale_supervision_full/summary.json').read_text())
    assert mature_protocol['checkpoint_sha256']==reference_summary['checkpoint_sha256']
    (RUN/'aggregate').mkdir(exist_ok=True)
    allarrays={};allscenes={};summary={};rows=[]
    for name in NAMES:
        summary[name]={}
        for split in ('train','val'):
            a,s,r=aggregate(name,split,protocol['expected'][split])
            allarrays[name,split]=a;allscenes[name,split]=s;summary[name][split]=metrics(a);rows.extend(r)
    reference={};class_output=[]
    # Borrow only exact native confusion counts from the already audited full
    # census of the identical original checkpoint. Do not borrow operating-point
    # sweeps or CE values with different averaging conventions.
    for split in ('train','val'):
        cm=np.zeros((18,18),dtype=np.int64);frames=0
        paths=sorted((ROOT/'work_dirs/gaussian_scale_supervision_full/scenes'/split).glob('*.json'))
        assert len(paths)==protocol['expected'][split]['scenes']
        for path in paths:
            data=json.loads(path.read_text())
            for r in data['records']:
                cm+=np.asarray(r['confusion'][0],dtype=np.int64);frames+=1
        assert frames==protocol['expected'][split]['frames']
        for name in NAMES:assert np.array_equal(cm.sum(1),allarrays[name,split]['confusion'].sum(1))
        reference[split]={k:float(v) for k,v in cm_metrics(cm).items()}
        class_output.extend(class_rows('original_mature_reference',split,cm,'existing_full_census'))
        for name in NAMES:class_output.extend(class_rows(name,split,allarrays[name,split]['confusion']))
    intervals={split:bootstrap(allscenes[NAMES[0],split],allscenes[NAMES[1],split]) for split in ('train','val')}
    calibration={name:calibrated(allarrays[name,'train'],allarrays[name,'val'],summary[NAMES[0]]['train']['FP']) for name in NAMES}
    trace=json.loads((RUN/'train'/f'{NAMES[1]}_s932/trace.jsonl').read_text().splitlines()[0])
    traces=[json.loads(x) for x in (RUN/'train'/f'{NAMES[1]}_s932/trace.jsonl').read_text().splitlines()]
    dose=dict(mean_camera_empty_fraction=float(np.mean([r['losses']['selective_camera_empty_fraction'] for r in traces])),
        mean_eligible_fraction=float(np.mean([r['losses']['selective_eligible_fraction'] for r in traces])),
        mean_blocked_fraction=float(np.mean([r['losses']['selective_blocked_fraction'] for r in traces])),
        max_forward_error=max(r['losses']['selective_forward_error'] for r in traces))
    write(RUN/'mature_summary.json',dict(reference=reference,trained=summary,intervals=intervals,
        calibration=calibration,dose=dose))
    with (RUN/'mature_scene_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (RUN/'mature_class_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(class_output[0]));w.writeheader();w.writerows(class_output)
    lines=['\n## 9. 对实际 30 epoch 权重的补充检验\n',
        '前面使用重新建立的短程共同起点；这里改用此前完成全量现象统计的 **opus_gaussian_8f_tiny.pth**，从同一模型和 AdamW 状态比较标准继续训练与针对性控制。每组 1,024 个 microbatch、128 次更新，种子 932，empty 参数共同固定。共同起点完整验证 mIoU 仅 4.37%，因此在任何成熟组训练开始前，延长了成熟配对的预算；这项决定未参考成熟组的干预结果。\n',
        '保存检查点的实际终止学习率约为 2 × 10⁻⁷。恢复审计后、任何成熟模型实验开始前，将两组统一设为保存值的 10 倍，约 2 × 10⁻⁶；这是声明过的小步诊断，不是声称原 30 epoch 调度自然继续。初始登记及修订均已保留。\n',
        '两组分别重新统计全部 850 个场景。原成熟权重的参考性能直接来自此前相同检查点、相同预处理的完整统计，只复用原生混淆计数，不混用其他口径的 CE 或阈值曲线。\n',
        '| 模型 | 集合 | mIoU | 占有 IoU | 漏检率 | 假阳性率 |\n|---|---|---:|---:|---:|---:|']
    labels={NAMES[0]:'标准继续训练',NAMES[1]:'针对性控制'}
    for name,values in [('reference',reference),*summary.items()]:
        for split in ('train','val'):
            m=values[split]
            lines.append('| '+' | '.join([('原成熟权重' if name=='reference' else labels[name]),
                '训练' if split=='train' else '验证',*[f'{m[k]:.4f}%' for k in ('mIoU','IoU','FN_rate','FPR')]])+' |')
    lines.extend(['\n针对性控制相对标准继续训练，验证集 mIoU 差值的场景 bootstrap 95% 区间为 '
        f'[{intervals["val"]["mIoU"][0]:+.4f}, {intervals["val"]["mIoU"][1]:+.4f}] 个百分点。这不是多种子训练区间。\n',
        '| 模型 | 验证 2σ 覆盖数 | 验证无 1σ 覆盖 | 训练加权 CE | 验证二值 AP |\n|---|---:|---:|---:|---:|'])
    for name in NAMES:
        m=summary[name]['val'];tr=summary[name]['train']
        lines.append(f'| {labels[name]} | {m["n_2sigma"]:.5f} | {m["no_1sigma"]:.4f}% | {tr["CE_weighted"]:.6f} | {m["binary_AP_hist"]:.4f}% |')
    lines.extend([f'\n这组干预平均每帧命中的 GS 占全部预测 GS 的 **{dose["mean_blocked_fraction"]*100:.5f}%**；满足可见 empty 中心且核心无前景的 GS 占 **{dose["mean_eligible_fraction"]*100:.3f}%**。因此必须结合实际命中强度解释结果，不能把稀少或很小的干预误写成强否定。\n',
        '这项补充可以检验成熟状态对当前训练路径的短程敏感性，不能回溯原始训练中每条路径的累计因果贡献，也不能仅凭 128 次更新判断新方法最终收敛性能。\n',
        '原始结果见 `mature_summary.json`、`mature_scene_metrics.csv`、`mature_pairing_audit.json` 和 `mature_protocol.json`。本轮新增完整评估合计 **8 个检查点、273,192 次 keyframe 评估**；原成熟参考的既有统计另行复用，不计入新增次数。\n'])
    p=ROOT/'analysis/gaussian_geometry_causal_results.md'
    text=p.read_text().split('\n## 9. 对实际 30 epoch 权重的补充检验')[0]
    p.write_text(text+'\n'.join(lines))
    print(json.dumps(dict(reference=reference,summary=summary,dose=dose,intervals=intervals),ensure_ascii=False,indent=2))


if __name__=='__main__':report()
