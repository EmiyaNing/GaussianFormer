"""Compact, read-only experiment progress plus a human-readable status page."""
from pathlib import Path
from datetime import datetime
import json
import os
import re

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def last_trace(path):
    if not path.exists():return {}
    try:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-16384));lines=f.read().decode().splitlines()
        for line in reversed(lines):
            try:return json.loads(line)
            except json.JSONDecodeError:pass
    except OSError:pass
    return {}


def main():
    late=ROOT/'work_dirs/empty_mechanism_stage3';history=ROOT/'work_dirs/empty_mechanism_history'
    late_train=read(late/'queue_status.json');late_queue=read(late/'completion_queue_status.json');hist_queue=read(history/'queue_status.json')
    failures=[]
    for name,status in [('末期训练',late_train),('末期评价',late_queue),('全程实验',hist_queue)]:
        if status.get('status') in ['failed','training_failed']:failures.append(dict(phase=name,detail=status))
    late_progress={};history_progress={}
    for name in ['B0','S','E','SE']:
        folder=late/'train'/name;complete=read(folder/'complete.json');progress=read(folder/'progress.json')
        late_progress[name]=dict(complete=complete.get('status')=='complete',steps=complete.get('steps',progress.get('step',0)))
        logfile=late/f'train_{name}.log'
        if not late_progress[name]['complete'] and logfile.exists():
            with logfile.open('rb') as f:
                f.seek(max(0,logfile.stat().st_size-4096));tail=f.read().decode(errors='replace')
            reported=re.findall(r'step=(\d+)/1024',tail)
            if reported:late_progress[name]['steps']=max(late_progress[name]['steps'],int(reported[-1]))
        folder=history/'train'/name;complete=read(folder/'complete.json');trace=last_trace(folder/'microbatch_trace.jsonl')
        history_progress[name]=dict(complete=complete.get('status')=='complete',
            frames=complete.get('frames',trace.get('global_iter',-1)+1))
    cancelled=hist_queue.get('status')=='cancelled_by_user'
    completed=late_queue.get('status')=='complete' and hist_queue.get('status')=='complete'
    active=late_train.get('condition','待开始')
    message=f"末期训练：{sum(v['steps'] for v in late_progress.values())}/4096 次更新"
    if late_train.get('status')=='running':message+=f"，当前 {active}"
    labels={'waiting_for_training':'等待联合干预训练完成','evaluating':'提取完整验证预测',
            'statistics':'汇总体素统计','endpoint_factor_swaps':'检查输出因素交换',
            'reporting':'生成配对统计与报告','auditing':'完成完整性审计','complete':'已完成',
            'waiting_for_final_epoch_training':'等待末期训练结束后自动开始'}
    phase=late_queue.get('status','待开始')
    message+=f"；末期评价：{labels.get(phase,phase)}"
    total=sum(v['frames'] for v in history_progress.values())
    message+=f"；全程训练：{total}/81920 帧次"
    command=hist_queue.get('command',[])
    if command:
        name=command[-1];kind=command[1] if len(command)>1 else ''
        if kind=='train' and name in history_progress:
            n=history_progress[name]['frames'];epoch=min(5,max(1,(max(1,n)-1)//4096+1));frame=(n-1)%4096+1 if n else 0
            message+=f"，{name} 第 {epoch}/5 个 epoch，{frame}/4096 帧"
        else:message+=f"，{kind} {name}"
    else:
        phase=hist_queue.get('status','待开始');message+=f"，{labels.get(phase,phase)}"
    if cancelled:message+='；其余全程实验已按用户要求取消，已有结果保留，切换概率渲染实验'
    if failures:message='实验队列报告错误，需要检查：'+str(failures)
    out=dict(complete=completed,cancelled=cancelled,failed=bool(failures),message=message,
             late_training=late_progress,history_training=history_progress,failures=failures)
    date=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    lines=['# Empty 机制验证：实时进度','',f'更新：{date}。','',message+'。','',
           '## 完整末期训练','', '| 条件 | 已完成更新 / 1024 | 状态 |','| --- | ---: | --- |']
    for name,r in late_progress.items():lines.append(f'| {name} | {r["steps"]} | {"完成" if r["complete"] else "执行中或待执行"} |')
    lines+=['','[末期实验设计](stage3/empty_mechanism_stage3_design.md) · [末期结果报告](stage3/empty_mechanism_stage3_results.md)','',
            '## 从训练起点开始的五个 epoch 对照','', '| 条件 | 已完成训练帧次 / 20480 | 状态 |','| --- | ---: | --- |']
    for name,r in history_progress.items():lines.append(f'| {name} | {r["frames"]} | {"完成" if r["complete"] else "已取消，保留部分记录" if cancelled and r["frames"] else "已取消" if cancelled else "执行中或待执行"} |')
    lines+=['','[全程实验设计](history/empty_mechanism_history_design.md) · [全程结果报告](history/empty_mechanism_history_results.md)','',
            '结果文件在正式训练、完整验证和配对统计完成后自动更新；执行中的阶段不作正式成因结论。','']
    path=ROOT/'analysis/empty_mechanism/empty_mechanism_progress.md';temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text('\n'.join(lines));temp.replace(path)
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
