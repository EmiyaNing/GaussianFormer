#!/usr/bin/env python3
"""Wait for the full census, then produce reviewable reports in dependency order."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sparseworld_full_regions_common import ROOT
from tools.sparseworld_region_utils import write_json


def main():
    stages={'status.json':'推理','statistics_status.json':'区域统计','motion_status.json':'实例诊断','query_status.json':'query 诊断'}
    while True:
        status={k:json.loads((ROOT/k).read_text()) for k in stages}
        assert not any(d['status']=='failed' for d in status.values()),'Upstream failed; inspect producer logs'
        message='；'.join(f"{stages[k]} {d.get('completed',d.get('sources',d.get('samples',0)))}/29049" for k,d in status.items())
        print('[全量进度] '+message,flush=True)
        if all(d['status']=='complete' for d in status.values()):break
        time.sleep(50)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLCONFIGDIR='/tmp/matplotlib-cache')
    steps=[('summary','tools/summarize_sparseworld_full_regions.py',[]),
           ('motion_summary','tools/summarize_sparseworld_full_motion.py',[]),
           ('explanation','tools/fit_sparseworld_region_explanations.py',['--root',str(ROOT)]),
           ('report','tools/write_sparseworld_full_region_report.py',[]),
           ('cases','tools/plot_sparseworld_full_region_cases.py',[])]
    for name,script,args in steps:
        write_json(ROOT/'finish_status.json',dict(status='summarizing',stage=name,pid=os.getpid()))
        print('[全量汇总] '+name+' 开始',flush=True)
        with Path('/tmp',f'sparseworld_full_regions_{name}.log').open('w') as f:
            subprocess.run([sys.executable,script,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('[全量汇总] '+name+' 完成',flush=True)
    write_json(ROOT/'finish_status.json',dict(status='ready_for_review',reports_written=True))
    print('统计、机制、解释模型、报告与图例已生成，等待内容检查及全量验收。',flush=True)


if __name__=='__main__':main()
