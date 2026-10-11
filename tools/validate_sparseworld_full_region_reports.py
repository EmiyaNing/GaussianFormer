#!/usr/bin/env python3
"""Refresh report hashes after review, preserving the completed census audit."""
import json,re,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sparseworld_full_regions_common import ROOT,REPORT
from tools.sparseworld_region_utils import sha256,write_json


def main():
    path=ROOT/'final_validation.json';prior=json.loads(path.read_text())
    assert prior['status']=='passed' and prior['sources']==29049 and prior['old_exact_replays']==512
    allowed={'tools/write_sparseworld_full_region_report.py','tools/update_sparseworld_full_region_research.py'}
    changed=[p for p,h in prior['code_sha256'].items() if sha256(p)!=h]
    assert set(changed)<=allowed,('Scientific code changed; rerun full census validation',changed)
    update=json.loads((ROOT/'research_update_audit.json').read_text())
    assert update['status']=='complete' and update['history_preserved'] and update['full_reports_promoted']
    for p,h in update['report_sha256'].items():assert sha256(p)==h,('Research document changed since promotion',p)
    base=REPORT.parent;analysis=base.parent.parent
    docs=sorted(set(base.rglob('*.md'))|set((base.parent/'fast_motion_distillation').rglob('*.md'))|
                {analysis/n for n in ('README.md','experimental_data_index.md',
                 'sparseworld_prediction_perception_regions_plan.md','sparseworld_prediction_perception_regions_protocol.md')})
    for p in docs:
        text=p.read_text();assert text.count('```')%2==0,p
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http:','https:','#')):assert (p.parent/link.split('#')[0]).resolve().exists(),(p,link)
    # Confirm that the promoted text differs only in relative link locations.
    from tools.update_sparseworld_full_region_research import rewrite_links
    for name in ('results.md','statistics.md','reproduction.md'):
        assert (base/name).read_text()==rewrite_links((REPORT/name).read_text(),REPORT,base)
    assert sha256(base/'attribute_statistics.csv')==sha256(REPORT/'region_statistics.csv')
    for name in ('status.json','statistics_status.json','motion_status.json','query_status.json','finish_status.json'):
        assert json.loads((ROOT/name).read_text())['status']=='complete'
    original_hash=sha256(path);snapshot=ROOT/('final_validation_before_report_review_'+original_hash[:12]+'.json')
    if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
    artifacts=sorted(set(docs)|set(REPORT.glob('*.csv'))|set((REPORT/'figures').glob('*')))
    final=dict(prior)
    final['code_sha256']={p:sha256(p) for p in prior['code_sha256']}
    final['code_sha256'][str(Path(__file__)) if not Path(__file__).is_absolute() else 'tools/'+Path(__file__).name]=sha256(__file__)
    final['report_sha256']={str(p):sha256(p) for p in artifacts}
    final['checks']=prior['checks']+['Final report review, promoted tables, historical archive and all research links verified; census code unchanged']
    final['count']=len(final['checks'])
    final['report_review']=dict(status='passed',prior_audit=str(snapshot),prior_audit_sha256=original_hash,
        changed_report_code=changed,scientific_code_unchanged=True,markdown_files=len(docs),
        scope='Report content/hash refresh only; previous exhaustive source/cache validation preserved')
    write_json(path,final)
    print(f'Final report review passed: {len(docs)} Markdown files; full census audit preserved.',flush=True)


if __name__=='__main__':main()
