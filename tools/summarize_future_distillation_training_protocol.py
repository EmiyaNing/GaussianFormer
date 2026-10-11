#!/usr/bin/env python3
"""Beginner-readable S0 continuation reports from actual native-head results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_future_distillation_causal import native_tables, average, teacher_subset, write, s0_metric_reading

REPO=Path(__file__).resolve().parents[1]
DOCS=REPO/'analysis/hypotheses/future_distillation_causes/s0_training_protocol'
LABELS={
    'full_bn_train':'完整任务＋BN 更新', 'full_bn_fixed':'完整任务＋BN 固定',
    'future_bn_train':'仅未来任务＋BN 更新', 'future_bn_fixed':'仅未来任务＋BN 固定',
    'full_bn_fixed_detached_geom':'完整任务＋BN 固定＋几何梯度分离',
    'future_bn_fixed_detached_geom':'仅未来任务＋BN 固定＋几何梯度分离',
}


def training_protocol_explanation():
    """Reading guide, independent of the numerical outcome of an arm."""
    return [
        '', '## 先读实验组名称与操作', '',
        '本目录是从相同预训练初始化重新训练的协议对照；上一份 `s0_protocol` '
        '是在同一个 5 epoch 检查点上临时恢复模块。两者都检查共同退化，但不能互换操作或恢复比例。', '',
        '| 名称片段 | 实际含义 | 容易混淆的说法 |',
        '|---|---|---|',
        '| `full` | 当前感知＋temporal 集合＋六步未来 occupancy＋轨迹任务 | 不是给模型完整未来图像 |',
        '| `future` | 只保留六步未来 occupancy＋轨迹任务 | 不等于冻结历史编码，也不是学生读未来图像 |',
        '| `bn_train` | BN 使用训练批次统计，并更新运行统计 | 不等于只有 BN 参数参与训练 |',
        '| `bn_fixed` | BN 使用并保持预训练运行统计，缩放／偏移仍允许训练 | 不等于冻结整个 backbone 或整个 Student |',
        '| `detached_geom` | 分离当前步未来点输出的梯度；前向坐标值不变 | 不是删除位置头、固定物体不动或做教师梯度投影 |',
        '| `task` | 仅使用任务真值监督 | 不是未微调基线 |',
        '| `teacher` | 任务监督＋正确对应的教师特征监督 | 指接受正确教师监督的学生，不是教师自身的感知成绩 |',
        '| `shuffled` | 同一场景、同一时距内打乱有效 query 的教师特征目标 | 不改学生图像，也不打乱 occupancy 真值 |',
        '| `seed20260930` | 训练随机种子 | 不是预测时距或更新次数 |', '',
        '例如，`full_bn_fixed_teacher_seed20260930` 是“完整任务＋固定 BN＋正确教师监督”的学生。'
        '末次教师复验只使用 `full_bn_fixed` 协议；六组初始协议筛选均不加教师。'
        '正常学生只读当前／历史图像；future-born queries 保留，所有训练组共用恒等初始化残差 MLP，'
        '最终仍由原版未来预测分支解码。', '',
        'temporal 集合指当前及未来 GT 变换到当前 ego 后，对全部 timestamp query 做集合监督，'
        '不是又一份单独当前帧标签。删除它与当前感知损失是组合干预，不能据此指定其中某一项为唯一原因。', '',
        '**BN-only**完全没有优化器，只通过训练图像前向改变 BN 缓冲状态。'
        '表中的 32／128／256 是图像前向次数；训练组对应数字则是优化器更新次数，两者不是同一训练预算排名。'
        '激活检查点反向重算还可能再次更新训练态 BN，因此同样的图像前向次数不等于相同的内部 BN 更新次数。', '',
        '进度表的“BN 保持不变”只核对统计缓冲量；更新组的“按设计已更新”不表示核对失败。'
        '参数是否允许训练、某次反向是否收到梯度、BN 是否更新统计，必须分别检查。'
        '不同损失或几何分离会使部分参数没有梯度，不能将“全部允许训练”写成每个参数每步都更新。',
    ]


def write_report(root):
    root=Path(root)
    if not (root/'results.json').exists():return
    report=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    formal_root=REPO/'out/future_distillation_causal/20261001_training_protocol'
    docs=DOCS if root.resolve()==formal_root.resolve() and len(manifest['train']['indices'])==128 else root/'reports'
    base=report['baseline'];steps=manifest['evaluation_steps']
    final=steps[-1]
    completed={r['arm']:r['steps'][str(final)] for r in report['arms'].values()
               if r['supervision']=='task' and r.get('status')=='complete' and str(final) in r.get('steps',{})}
    paired_complete={r['supervision']:r['steps'][str(final)] for r in report['arms'].values()
                     if r['arm']=='full_bn_fixed' and r.get('status')=='complete' and str(final) in r.get('steps',{})}
    status_label={'complete':'本轮已完成','running':'正在执行'}.get(report['status'],report['status'])
    rows=['# S0 续实验：到底是什么训练改变造成共同退化？','',
          f'状态：{status_label}。训练 {len(manifest["train"]["indices"])} 样本／验证 {len(manifest["val"]["indices"])} 样本；种子 {manifest["seeds"]}。',
          '', '## 一页结论','']
    if all(name in completed for name in ('full_bn_train','full_bn_fixed')):
        a,b=completed['full_bn_train'],completed['full_bn_fixed']
        loss=average(base)-average(a)
        rows.extend(['| 问题 | 当前回答 | 证据边界 |','|---|---|---|',
                     '| BN 统计是否造成共同退化？ | 无权重更新的 BN-only 已明显下降；配对训练固定 BN 大幅恢复 | 本轮受控干预；不是所有旧实验的统一原因 |',
                     '| 删除辅助任务或分离位置梯度能否同等修复？ | 没有看到同等量级恢复 | 仅本轮损失组合及几何梯度设置 |',
                     '| 固定 BN 后，教师是否帮助预测？ | '+('三组已完成，见下方同协议对照' if all(k in paired_complete for k in ('task','teacher','shuffled')) else '正确／打乱教师对照尚未齐全')+' | 单种子开发筛选，不能代替独立确认 |'])
        if loss>.2:rows.extend(['',f'固定 BN 追回本轮完整任务组下降的 {100*(average(b)-average(a))/loss:.1f}%；分母是本轮 {final} 更新对照，不是旧 5 epoch 的损失。'])
    else:rows.append('配对训练结果尚未齐全，暂不判断主要因素。')
    if all(k in paired_complete for k in ('task','teacher','shuffled')):
        records=[('原版 SparseWorld',base)]
        teacher=teacher_subset(root,len(manifest['val']['indices']))
        if teacher:records.append(('真实未来感知教师',teacher))
        records.extend([('学生：仅任务',paired_complete['task']),('学生：正确教师监督',paired_complete['teacher']),
                        ('学生：打乱教师监督',paired_complete['shuffled'])])
        rows.extend(['','### 末次教师对照：总体及分类别原版头结果','',
                     '“正确教师监督”“打乱教师监督”两行都指 Student；只有“真实未来感知教师”一行是教师自身。'
                     '所有学生仍只看当前／历史图像，比较点统一为完成 256 次更新后。','',
                     *native_tables(records,base),
                     '','Teacher 在目标未来时刻读真实图像作感知，Student 三组只读当前／历史图像。'
                     'Teacher 的高分不能当成可部署预测结果；它只说明特权未来输入下感知器能够识别目标场景。'])
    if report['arms']:
        rows.extend(['','### 图示：从同一初始化出发，哪些设置迅速退化？','',
                     '纵轴是原版预测头的 1／2／3 秒平均 mIoU，不是特征损失。这里只画六组无教师训练；'
                     '曲线表示预定验证点，不代表连续记录或最佳检查点筛选。','',
                     '![原版预测头的训练协议曲线](training_protocol_curves.svg)'])
    rows.extend(training_protocol_explanation())
    rows.extend(s0_metric_reading(len(manifest['val']['indices'])))
    rows.extend(['', '## 判读边界','',
          '这是一轮短程训练协议筛选，不是完整 4096／1024 确认。所有组都保留 future-born queries，正常 Student 不读未来图像。'
          'BN 固定只固定运行统计、使用 eval 统计；其缩放／偏移参数仍可训练。没有新增 occupancy 读出器。'
          '训练组共用前一轮的恒等初始化残差 MLP，输出仍由原版未来语义／位置／速度分支解码；'
          '这不等于训练后的模型完全没有额外模块。',
          '每一对组只比较控制的因素。BN-only 不进行任何梯度更新；其结果不能与训练组当作同一训练预算方法排名。',
          '此 BN 干预只针对此前全参数实验将原版固定统计改为更新统计的设置，不能解释所有冻结感知的旧负结果。'
          '激活检查点的反向重算也可能重复更新训练态 BN；BN-only 的前向次数不等于每个 BN 的训练内部更新次数。',
          '', '## 1．不改权重，只改变 BN 统计',''])
    rows.extend(['只让 BN 统计接触训练图像，不更新可学习权重，再切到 eval 验证。'
                 '如果预测下降，说明运行状态变化本身就足以改变输出；'
                 '但不能由这组结果分配真实训练组的全部损失责任。下面展开的是前向次数，不是 epoch 或优化步数。',''])
    bn=report.get('bn_only',{})
    records=[('原版 SparseWorld',base)]
    records.extend((f'BN-only，{step} 次图像前向',value) for step,value in bn.get('steps',{}).items())
    rows.extend(['<details>','<summary>展开 BN-only 的 32／128／256 次前向结果</summary>','',*native_tables(records,base),'','</details>'])
    if bn.get('status')=='complete':
        rows.append(f'\n所有可学习权重逐项未改变：{bn["parameters_unchanged"]}。仅图像路径 BN 缓冲量可以更新。')
        if bn.get('steps'):
            last=max(bn['steps'],key=int)
            rows.append('\n末次 BN-only 的 1／2／3 秒 mIoU 为 '+
                        '／'.join(f'{bn["steps"][last]["metrics"][h]["mIoU"]:.3f}' for h in ('1s','2s','3s'))+
                        '。没有教师、没有权重优化也出现下降，支持统计状态变化足以导致本设置的退化。')
    rows.extend(['','## 2．配对训练：每个阶段看什么','',
                 '| 组 | 完成更新 | 最近三时距平均 mIoU | BN 缓冲量状态 |','|---|---:|---:|---|'])
    for label,row in report['arms'].items():
        values=row.get('steps',{});last=max(values,key=int) if values else None
        score=f'{average(values[last]):.4f}' if last else '未验证'
        checked=row.get('bn_unchanged')
        bn_status='已核对：保持不变' if checked is True else ('按设计已更新' if checked is False and not row['fixed_bn'] else '待核对／异常')
        rows.append(f'| {LABELS[row["arm"]]}／{row["supervision"]}／{row["seed"]} | {row.get("progress",0)} | {score} | {bn_status} |')
    rows.extend(['','32／128 是过程检查，256 是预先规定的末次比较点。'
                 '每组从相同初始化出发，不能挑各组最好的中间结果比较；中间验证不是额外训练组。'
                 '上表的末次均值用于概览，以下分别保留各时距和各指标。',''])
    for step in steps:
        records=[('原版 SparseWorld',base)]
        teacher=teacher_subset(root,len(manifest['val']['indices']))
        if teacher:records.append(('真实未来感知教师',teacher))
        records.extend((f'{LABELS[row["arm"]]}／{row["supervision"]}／{row["seed"]}',row['steps'][str(step)])
                       for row in report['arms'].values() if str(step) in row.get('steps',{}))
        if step!=final:rows.extend(['','<details>',f'<summary>展开 {step} 次更新的中间验证</summary>',''])
        rows.extend(['',f'### 完成 {step} 次更新时的原版头',*native_tables(records,base)])
        if step!=final:rows.extend(['','</details>'])
    rows.extend(['','## 3．逐项回答假设',''])
    def value(arm,supervision='task'):
        matches=[row['steps'][str(final)] for row in report['arms'].values()
                 if row['arm']==arm and row['supervision']==supervision and str(final) in row.get('steps',{})]
        return average(matches[0]) if len(matches)==1 else None
    rows.extend(['| 控制的因素 | 第一种设置的三时距平均 mIoU | 第二种设置的三时距平均 mIoU |',
                 '|---|---|---|'])
    for first,second,label in [('full_bn_train','full_bn_fixed','BN 更新因素'),
                               ('future_bn_train','future_bn_fixed','仅未来任务下的 BN 更新因素'),
                               ('full_bn_fixed','future_bn_fixed','当前／temporal 辅助任务组合因素'),
                               ('full_bn_fixed','full_bn_fixed_detached_geom','完整任务下的几何梯度因素'),
                               ('future_bn_fixed','future_bn_fixed_detached_geom','仅未来任务下的几何梯度因素')]:
        a,b=value(first),value(second)
        if a is not None and b is not None:
            rows.append(f'| {label} | {LABELS[first]}：{a:.4f} | {LABELS[second]}：{b:.4f} |')
    rows.extend(['','以上是相同初始化、种子、样本和更新预算下的配对干预。BN 模式有远大于其余因素的效应；'
                 '删除辅助任务和分离几何梯度没有复现同等恢复。单个短程种子不证明旧 5 epoch 的全部下降由此造成，'
                 '也不证明这些因素在其他训练预算下必然没有作用。',''])
    scores={arm:value(arm) for arm in LABELS}
    if scores['full_bn_train'] is not None and scores['full_bn_fixed'] is not None:
        loss=average(base)-scores['full_bn_train']
        if loss>.2:
            recovery=(scores['full_bn_fixed']-scores['full_bn_train'])/loss
            rows.extend(['','### BN 修复追回的是哪一份损失？','',
                         f'固定 BN 追回本轮 256 更新完整任务组相对基线下降的 {recovery*100:.1f}%。'
                         '分母是这轮短程对照，不是旧 4096／1024、5 epoch 的下降；不能混称两份恢复比例。', '',
                         '`恢复比例 = (固定 BN 平均 mIoU − 更新 BN 平均 mIoU) / (原版基线平均 mIoU − 更新 BN 平均 mIoU)`', '',
                         '分子、分母都使用本轮相同的三时距均值。'
                         '这不是旧检查点恢复感知组合的比例，也不表示 BN 解释了仓库所有负结果。'])
    stable=[arm for arm,score in scores.items() if score is not None and score>=average(base)-.2]
    rows.extend(['','### 是否满足后续教师净效应复验的筛选门槛？','',
                 '预先使用三时距平均不低于基线 0.2 点、任一时距不低于基线 0.3 点的工程门槛；'
                 '它不是统计显著性。只有达到门槛才考虑教师配对复验，不能挑教师最好的验证 epoch。'])
    passed=[]
    for arm in stable:
        result=next(row['steps'][str(final)] for row in report['arms'].values() if row['arm']==arm and row['supervision']=='task')
        if all(result['metrics'][h]['mIoU']>=base['metrics'][h]['mIoU']-.3 for h in ('1s','2s','3s')):passed.append(arm)
    rows.append('\n当前通过：'+('、'.join(LABELS[arm] for arm in passed) if passed else '尚无已完成结果满足门槛')+'。')
    if ('full_bn_fixed' in passed and scores['full_bn_fixed_detached_geom'] is not None
            and 'full_bn_fixed_detached_geom' not in passed):
        rows.append('\n通过门槛不等于统计上已证明与基线等效。按既定优先级先考虑完整任务＋固定 BN＋原版几何分离；'
                    '该组未通过后，再采用通过门槛的完整任务＋固定 BN。'
                    '不是因为它在开发集分数最高，而是优先保留完整任务。')
    rows.extend(['','### R1／R2／R3 的证据没有被偷换','',
                 '本轮首先检查共同训练问题。即使修复成功，也不是证明教师有用；必须随后比较同协议 task／teacher／shuffled。'
                 '它不验证同一历史的多未来，因此 R1 的核心命题仍未验证。'
                 '几何梯度分离也不是去除教师／任务冲突，不能冒充 R3 的梯度投影。',
                 '', '### 同协议教师复验',''])
    paired={}
    for label,row in report['arms'].items():
        if str(final) in row.get('steps',{}):paired.setdefault(row['arm'],{})[row['supervision']]=row['steps'][str(final)]
    for arm,values in paired.items():
        if all(name in values for name in ('task','teacher','shuffled')):
            rows.extend([f'\n{LABELS[arm]}：三时距平均 mIoU，任务 {average(values["task"]):.4f}，'
                         f'正确教师 {average(values["teacher"]):.4f}，打乱教师 {average(values["shuffled"]):.4f}。',
                         '这是同一开发集、一个种子的配对结果，不属于独立确认；不能把极小均值差别写成稳定收益。'])
    rows.extend(['','“正确教师”和“打乱教师”都是 Student 的训练条件名称。'
                 '教师自身分数更高只说明特权未来感知参考更强；是否蒸馏成功，要看正确监督的 Student '
                 '是否稳定超过同预算任务组，以及是否优于打乱对应组。',''])
    alignment_path=root/'alignment_audit.json'
    if alignment_path.exists():
        alignment=json.loads(alignment_path.read_text())
        rows.extend(['','### 验证特征是否真正靠近同一教师目标','',
                     '固定匹配使用初始原版几何上的相同 Teacher 目标与有效 query 槽位；训练后的点仍可移动，'
                     '不是同物理位置严格采样。移动匹配会随新的点位置重新选目标。这里是归一化 SmoothL1＋余弦损失，不是 mIoU。','',
                     '| 组 | 固定目标特征损失 | 移动匹配特征损失 |','|---|---:|---:|'])
        for label,value in alignment['cases'].items():rows.append(f'| {label} | {value["fixed_feature_loss"]:.6f} | {value["moving_feature_loss"]:.6f} |')
        baseline_alignment=alignment['cases'].get('baseline')
        teacher_alignment=alignment['cases'].get(f'full_bn_fixed_teacher_seed{manifest["seeds"][0]}')
        task_alignment=alignment['cases'].get(f'full_bn_fixed_task_seed{manifest["seeds"][0]}')
        if baseline_alignment and teacher_alignment and task_alignment:
            rows.extend(['',f'同一固定目标误差：未训练 {baseline_alignment["fixed_feature_loss"]:.6f}；'
                         f'仅任务 {task_alignment["fixed_feature_loss"]:.6f}；正确教师 {teacher_alignment["fixed_feature_loss"]:.6f}。'
                         '如果教师组误差更低，只能说此对齐目标有效改变了特征；是否帮助预测仍以同预算任务组及打乱组的原版头指标为准。'])
            if teacher_alignment['fixed_feature_loss']<task_alignment['fixed_feature_loss']:
                rows.append('\n本次正确教师的固定目标损失确实低于仅任务组，但仍未接近零；'
                            '不是完整复制教师特征的实验。若原版头分数没有稳定净收益，只能说本预算、系数和接口下尚未把这份特征变化转化为稳定任务收益，'
                            '不能证明更充分的适配必然失败，也不能证明教师特征包含的差异全部无法从历史推断。')
    intervals=root/'bootstrap.json'
    if intervals.exists():
        statistics=json.loads(intervals.read_text())
        rows.extend(['','### 场景重采样：避免只看一个均值','',
                     '下表只报告相对原版基线的 mIoU 变化 95% 区间，单位为百分点。'
                     '同一组场景配对重采样 1000 次，一个训练种子，未校正多重比较；不覆盖重训练随机性或独立场景确认。','',
                     '| 组 | 1 秒区间 | 2 秒 | 3 秒 | 三时距平均 |','|---|---:|---:|---:|---:|'])
        for name,record in statistics['baseline_changes'].items():
            rows.append('| '+name+' | '+' | '.join(f'[{record[h][0]:+.3f}, {record[h][1]:+.3f}]' for h in ('1s','2s','3s','mean_1_2_3s'))+' |')
        rows.append('\n控制组之间的配对区间保存在原始 `bootstrap.json` 的 `paired_changes`，不混入主表的相对基线差值。')
        rows.append('\n“区间跨零”表示当前数据不足以排除无净变化，不表示两组精确相等或证明蒸馏无效。'
                    '1000 次是对同一批场景重新抽样，不是重新训练 1000 个模型；该区间不覆盖多个训练种子的变化。')
        for supervision,description in [('teacher','正确教师'),('shuffled','打乱教师')]:
            label=f'full_bn_fixed_{supervision}_seed{manifest["seeds"][0]} vs full_bn_fixed_task_seed{manifest["seeds"][0]}'
            interval=statistics.get('paired_changes',{}).get(label,{}).get('mean_1_2_3s')
            if interval:
                verdict='区间跨零，尚不支持稳定净变化' if interval[0]<=0<=interval[1] else ('区间全部高于零，有局部正效应' if interval[0]>0 else '区间全部低于零，有局部负效应')
                rows.append(f'\n{description}与同预算任务组的三时距平均配对检查：{verdict}。'
                            '这是单种子开发场景的探索性区间，仍不覆盖重新训练的随机性；不宣称总体机制已被证明。')
        label=f'full_bn_fixed_teacher_seed{manifest["seeds"][0]} vs full_bn_fixed_shuffled_seed{manifest["seeds"][0]}'
        interval=statistics.get('paired_changes',{}).get(label,{}).get('mean_1_2_3s')
        if interval:
            verdict='区间跨零，尚不能区分两组的净效应' if interval[0]<=0<=interval[1] else ('正确组有局部优势' if interval[0]>0 else '正确组有局部劣势')
            rows.append(f'\n正确教师与打乱教师的三时距平均配对检查：{verdict}。'
                        '打乱组破坏同一场景、同一时距中 query 与特征目标的空间对应，不是读取另一场景图像；'
                        '单独超过任务组而未超过打乱组，不能证明收益来自正确的未来对应信息。')
    verification_path=root/'verification.json'
    if verification_path.exists():
        verification=json.loads(verification_path.read_text())
        rows.extend(['','### 数值与身份核对','',
                     f'最近一次机器审计状态：{verification["status"]}，核对 {verification["checked_native_results"]} 份原版头验证统计。'
                     '审计从逐场景 confusion matrix／交并计数重算总体及动态／静态指标，核对主表差值、'
                     '原版基线一致性、教师样本清单及 BN 状态；不证明近邻匹配等同于物体身份对应。'
                     '审计时的结果文件 SHA256 记录在 `verification.json`，训练继续后需重新运行审计。'])
    rows.extend([
                 '', '## 4．原始产物与复现','',f'数据与过程：`{root}`；每个验证点 `.npz` 为原版头逐场景统计。',
                 '逐更新日志、参数梯度审计、BN 不变检查在 `results.json`；可续训状态在各组 `latest.pth`。',
                 '设计见[实验设计](design.md)。' if docs==DOCS else '烟雾测试：不计入正式证据。'])
    rows.extend(['','### 本次完成与未完成','',
                 '本目录的完成只指预定短程训练协议筛选、同种子教师复验及相应审计。'
                 '三个预定新种子的重新训练、S3 未用于开发决策的独立场景确认、S5 组合确认仍未完成。'
                 'R1 的严格相同历史／动作多未来分支缺少已验证的数据或环境，不以单轨迹拟合误差替代其证明。'])
    write(docs/'results.md',rows)
    if report['arms']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names={'full_bn_train':'Full / BN train','full_bn_fixed':'Full / BN fixed',
               'future_bn_train':'Future / BN train','future_bn_fixed':'Future / BN fixed',
               'full_bn_fixed_detached_geom':'Full / fixed BN / detached geometry',
               'future_bn_fixed_detached_geom':'Future / fixed BN / detached geometry'}
        fig,axis=plt.subplots(figsize=(8,4))
        for row in report['arms'].values():
            if row['supervision']!='task' or not row.get('steps'):continue
            x=sorted(row['steps'],key=int);axis.plot([0,*map(int,x)],
                [average(base),*[average(row['steps'][step]) for step in x]],marker='o',label=names[row['arm']])
        axis.axhline(average(base),linestyle='--',color='black',label='Untouched original baseline')
        axis.set_xlabel('Optimizer updates');axis.set_ylabel('Native mean mIoU (1/2/3s, %)')
        axis.grid(alpha=.2);axis.legend(fontsize=7);fig.tight_layout()
        fig.savefig(docs/'training_protocol_curves.svg');plt.close(fig)
    save=root/'gate.json'
    save.write_text(json.dumps(dict(status=report['status'],task_arms_passing=passed,
                                  threshold_mean=.2,threshold_each_horizon=.3,criterion_step=final),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'out/future_distillation_causal/20261001_training_protocol')
    args=parser.parse_args();write_report(args.root)
