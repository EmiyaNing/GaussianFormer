"""Keep a readable on-disk progress page while the durable queue runs."""
from pathlib import Path
from datetime import datetime
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.gf2_probability_experiment import RUN,read


def refresh():
    path=RUN/'queue_status.json';state=read(path) if path.exists() else dict(status='initializing')
    rows=['# 概率渲染实验：实时进度','',f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        '',f"队列状态：`{state['status']}`。本页约每 45 秒刷新；完成判据见结果报告。",'',
        '| 阶段 | 实际完成量 |','| --- | --- |']
    for label,folder in [('B0 原生基线',ROOT/'work_dirs/empty_mechanism_history/train/B0'),('P 概率渲染',RUN/'train/probability')]:
        trace=folder/'microbatch_trace.jsonl';n=0
        if trace.exists():n=len(trace.read_text().splitlines())
        complete=(folder/'complete.json').exists()
        rows.append(f"| {label}训练 | {n}/20480 帧次，{n//4}/5120 次更新；完整性确认：{'已通过' if complete else '待完成'} |")
    rows.append(f"| P 完整模型预检 | {'通过' if (RUN/'native_pilot/pilot_success.json').exists() else '待完成'} |")
    for name in ['baseline','probability']:
        dest=RUN/'evaluation'/name;n=len(list((dest/'frames').glob('*.npz'))) if (dest/'frames').exists() else 0
        rows.append(f"| {name} 验证提取 | {n}/1024 帧 |")
        rows.append(f"| {name} 三项诊断统计 | {'完成' if (dest/'statistics_complete.json').exists() else '待完成'} |")
    if 'command' in state:rows+=['',f"当前命令：`{' '.join(state['command'])}`。"]
    if 'error' in state:rows+=['',f"失败信息：`{state['error']}`。保留现场后再修复，不能将中断训练当作完整轨迹。"]
    rows+=['','[实验设计](gf2_probability_decomposition_design.md) · [结果报告](gf2_probability_decomposition_results.md)。','']
    path=ROOT/'analysis/probability_rendering/gf2_probability_decomposition_progress.md';temp=path.with_suffix('.tmp');temp.write_text('\n'.join(rows));temp.replace(path)
    return state['status']


if __name__=='__main__':
    while True:
        if refresh() in ['complete','failed']:break
        time.sleep(45)
