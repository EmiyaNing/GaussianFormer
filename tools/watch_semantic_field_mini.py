"""Refresh only the pending report; never overwrite the completed analysis."""
from pathlib import Path
import json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.report_semantic_field_mini import progress,NAMES
from tools.semantic_field_experiment_common import RUN
if __name__=='__main__':
    while True:
        q=json.loads((RUN/'queue_progress.json').read_text())
        if q.get('status')=='complete':break
        if not all((RUN/'eval'/n/'complete.json').exists() for n in NAMES):progress()
        if q.get('status')=='failed':break
        try:os.kill(q['pid'],0)
        except ProcessLookupError:break
        time.sleep(55)
