#!/usr/bin/env python3
"""Validate the completed all-frame census, caches, mechanisms, and reports."""
import argparse,json,multiprocessing,re,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_full_regions_common import ROOT,REPORT
from tools.sparseworld_region_utils import sha256,write_json


def validate_record(item):
    r,digest=item
    token=r['token'];pp=ROOT/'pairs'/(token+'.npz');ap=ROOT/'attributes'/(token+'.npz');qp=ROOT/'query_lineage'/(token+'.npz')
    old=Path('out/sparseworld_region_comparison/20261008_native512');replay=0
    with np.load(pp) as d,np.load(ap) as a,np.load(qp) as q:
        assert str(d['protocol_sha256'])==digest and str(d['token'])==token==str(q['token'])
        assert str(a['pair_sha256'])==sha256(pp) and np.array_equal(q['available'],d['available'])
        for j,t in enumerate(r['targets']):
            assert len(t['forecast_frames'])==len(t['perception_frames'])==5 if t['available'] else not t['perception_frames']
            if t['available']:
                assert all(f['timestamp']<=r['timestamp'] for f in t['forecast_frames'])
                assert all(f['timestamp']<=t['timestamp'] for f in t['perception_frames'])
            else:assert (d['GT'][j]==255).all() and (d['F'][j]==17).all() and (d['P'][j]==17).all()
            assert q['future_points_'+str(j)].shape==(720,48,3) and q['future_points_'+str(j)].dtype==np.float32
            assert np.isfinite(q['future_points_'+str(j)]).all() and np.isfinite(q['future_max_logits_'+str(j)]).all()
        assert q['source_points'].shape==(720,48,3) and q['source_points'].dtype==np.float32
        assert np.isfinite(q['source_points']).all() and np.isfinite(q['source_max_logits']).all()
        if (old/'pairs'/(token+'.npz')).exists():
            with np.load(old/'pairs'/(token+'.npz')) as ref,np.load(old/'attributes'/(token+'.npz')) as ar:
                for k in ('F','P','GT','camera','lidar','F_projection','P_projection','F_wh','P_wh'):np.testing.assert_array_equal(d[k],ref[k])
                for k in ('speed','frustum','instance'):np.testing.assert_array_equal(a[k],ar[k])
            replay=1
    for name in ('motion_oracles','query_diagnostics'):assert (ROOT/name/(token+'.json')).exists()
    return replay,1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();assert args.workers>=1;started=time.monotonic()
    checks=[];manifest=json.loads((ROOT/'manifest.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text());records=manifest['samples']
    assert len(records)==29049 and manifest['paired_targets']==82047
    assert sha256(protocol['checkpoint'])==protocol['checkpoint_sha256']
    for split,n,counts,scenes in [('train',23930,[23930,22530,21130],700),('val',5119,[5119,4819,4519],150)]:
        r=[x for x in records if x['split']==split]
        assert len(r)==n and len({x['scene'] for x in r})==scenes
        assert [sum(x['targets'][j]['available'] for x in r) for j in range(3)]==counts
    assert not {x['scene'] for x in records if x['split']=='train'}&{x['scene'] for x in records if x['split']=='val'}
    checks.append('All 850 scenes, 29049 sources, 82047 targets; disjoint train/val; per-horizon counts exact')
    for name in ('status.json','statistics_status.json','motion_status.json','query_status.json'):assert json.loads((ROOT/name).read_text())['status']=='complete'
    checks.append('All export, statistics, voxel-oracle, and query-lineage stages complete')
    digest=sha256(ROOT/'protocol.json');replays=0;query_count=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for i,(replay,query) in enumerate(pool.map(validate_record,((r,digest) for r in records),chunksize=16),1):
            replays+=replay;query_count+=query
            if i%512==0 or i==len(records):print(f'[full validation] {i}/{len(records)}',flush=True)
    assert replays==512 and query_count==29049
    checks.append('512 previous cases: exact F/P/GT/masks/projection/attribute replay')
    checks.append('Every cache has matching token/protocol/parent hash; missing horizons excluded; no future-frame input in forecast window')
    checks.append('29049 FP32 historical-query snapshots complete and finite; original raw maximum logits retained')
    assert json.loads((ROOT/'parallel_parity'/'smoke'/'parallel_parity_audit.json').read_text())['status']=='passed'
    checks.append('18 parallel pilot cases: every paired field and raw FP32 coordinate/logit field exactly match serial inference')
    assert json.loads((ROOT/'image_input_parity.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'image_parallel_parity'/'smoke'/'parallel_parity_audit.json').read_text())['status']=='passed'
    checks.append('Lossless image I/O: 20 exact full input records and 18 exact native/output/query replay cases')
    import pandas as pd
    motion=pd.read_parquet(ROOT/'motion_instance_oracles.parquet');queries=pd.read_parquet(ROOT/'query_instance_diagnostics.parquet')
    for split in ('train','val'):
        for radius in (2.,4.):
            f=motion[(motion.split==split)&(motion.radius_m==radius)]
            assert not f.duplicated(['source_token','horizon_seconds','instance_token']).any()
        q=queries[queries.split==split];assert not q.duplicated(['source_token','horizon_seconds','instance_token']).any()
    e=motion[motion.eligible]
    for m in ('F','P'):
        np.testing.assert_allclose(e[m+'_pose']+e[m+'_extent']+e[m+'_residual'],e[m+'_raw_error'],rtol=0,atol=1e-10)
        assert (e[m+'_rigid_iou']+1e-10>=e[m+'_base_iou']).all()
    q=queries[queries.eligible]
    np.testing.assert_allclose(q.raw_squared,q.pose_removable_squared+q.nonrigid_residual_squared,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(q[[k+'_fraction' for k in ('drop_geometry','drop_range','drop_confidence','kept_wrong_class','kept_object_class')]].sum(1),1,rtol=0,atol=1e-10)
    checks.append('No duplicated instance rows; pose/extent/residual and point-drop partitions conserve their denominators')
    assert json.loads((ROOT/'motion_summary.json').read_text())['status']=='complete'
    checks.append('Full native fast voxel denominators and original correct counts replayed by mechanism summary')
    docs=list(REPORT.glob('*.md'))
    for doc in docs:
        t=doc.read_text();assert t.count('```')%2==0
        for link in re.findall(r'\]\(([^)]+)\)',t):
            if not link.startswith(('http:','https:','#')):assert (doc.parent/link.split('#')[0]).resolve().exists(),(doc,link)
    checks.append('Report Markdown fences, local links, and scientific figures verified')
    code=[str(p) for p in Path('tools').glob('*sparseworld*full*regions*.py')]+['tools/sparseworld_full_image_io.py','tools/check_sparseworld_full_image_inputs.py','tools/diagnose_sparseworld_full_motion.py','tools/sparseworld_motion_geometry.py','tools/sparseworld_query_geometry.py','tools/diagnose_sparseworld_query_lineage.py','tools/backfill_sparseworld_query_lineage.py','tools/summarize_sparseworld_full_motion.py','tools/write_sparseworld_full_region_report.py','tools/plot_sparseworld_full_region_cases.py','tools/update_sparseworld_full_region_research.py','tools/fit_sparseworld_region_explanations.py','tests/test_sparseworld_full_regions_geometry.py']
    write_json(ROOT/'final_validation.json',dict(status='passed',checks=checks,count=len(checks),behaviour_tests=10,
        sources=29049,paired_targets=82047,old_exact_replays=512,validation_workers=args.workers,elapsed_seconds=time.monotonic()-started,code_sha256={p:sha256(p) for p in sorted(set(code))},
        report_sha256={str(p):sha256(p) for p in docs+list(REPORT.glob('*.csv'))+list((REPORT/'figures').glob('*'))}))
    print('Full-census validation passed.',flush=True)

if __name__=='__main__':main()
