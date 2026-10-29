"""Check saved full-census identities, partitions, strata and attribute bounds."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.stat_query_budget import RUN,FIELDS,Q,CHILDREN
from tools.stat_boundary_gaussians import read,write,sha


def audit(partial=False):
    protocol=read(RUN/'protocol.json')
    identities=set();tokens=set();counts={};totals={}
    for split in ('train','val'):
        scenes=frames=0
        total=np.zeros(len(FIELDS),dtype=np.float64)
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            meta=read(path)
            assert meta['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert meta['indices']==[r['index'] for r in meta['records']]
            for r in meta['records']:
                key=(split,r['sample_token'])
                assert key not in identities,key
                identities.add(key)
                assert r['sample_token'] not in tokens, 'Repeated sample token across split/scene'
                tokens.add(r['sample_token'])
            with np.load(path.with_suffix('.npz')) as z:
                a,classes,strata=z['metrics'],z['class_counts'],z['strata']
            assert a.shape==(meta['frames'],Q,len(FIELDS))
            assert classes.shape==(meta['frames'],Q,17)
            assert np.isfinite(a).all() and (a>=0).all()
            assert (classes>=0).all() and (strata>=0).all()
            f=lambda key:a[...,FIELDS.index(key)]
            np.testing.assert_array_equal(f('gt_foreground'),classes.sum(2))
            np.testing.assert_array_equal(f('gt_foreground'),f('gt_own_core')+f('gt_neighbor_only_core')+f('gt_no_core'))
            np.testing.assert_array_equal(f('gt_foreground'),f('gt_correct')+f('gt_wrong_class')+f('gt_miss'))
            np.testing.assert_array_equal(f('gt_classes'),(classes>0).sum(2))
            np.testing.assert_array_equal(f('core_coverage_all_fg'),f('core_exclusive')+f('core_shared_other_query'))
            for key in ('gt_no_core_correct','gt_no_core_detected','gt_semantic_boundary','gt_empty_boundary','gt_components','gt_classes'):
                assert (f(key)<=f('gt_foreground')).all(),key
            assert (f('gt_no_core_correct')<=f('gt_no_core_detected')).all()
            assert (f('gt_no_core_detected')<=f('gt_no_core')).all()
            assert (f('gt_components')[f('gt_foreground')>0]>=1).all()
            assert (f('core_gs_touches_fg')<=f('native_gs_touches_fg')).all()
            assert (f('core_gs_touches_fg')<=f('gs_camera_core')).all()
            assert (f('gs_camera_core')<=f('gs_camera_native')).all()
            for key in ('core_gs_touches_fg','native_gs_touches_fg','gs_camera_native','gs_camera_core'):
                assert (f(key)<=CHILDREN).all()
            assert (f('opacity_mean')<=1.000001).all()
            assert (f('effective_amplitude_mean')<=f('amplitude_mean')+1e-5).all()
            assert (f('scale_mean')>=.079999).all() and (f('scale_max_mean')<=.400001).all()
            assert (f('dom_no_core_correct_from_same_query')<=f('no_core_correct_dom_n')).all()
            for j,key in enumerate(('gt_foreground','gt_no_core','gt_miss','gt_no_core_correct','gt_own_core')):
                assert int(strata[...,j].sum())==int(f(key).sum(dtype=np.float64)),(path,key)
            assert int(f('core_coverage_all_fg').sum(dtype=np.float64))==sum(r['distinct_query_coverage_sum'] for r in meta['records'])
            assert int(f('dominant_fp_voxels').sum(dtype=np.float64))==sum(r['false_positive'] for r in meta['records'])
            total+=a.sum((0,1),dtype=np.float64)
            scenes+=1;frames+=meta['frames']
        counts[split]=dict(scenes=scenes,frames=frames)
        totals[split]={k:float(total[j]) for j,k in enumerate(FIELDS)}
    complete=counts==protocol['expected']
    if not partial:
        assert complete,counts
        assert len(identities)==34149
    result=dict(status='passed' if complete else 'partial_passed',scope=counts,
        frame_identities=len(identities),query_frame_records=len(identities)*Q,
        unique_sample_tokens=len(tokens),
        checkpoint_sha256=protocol['checkpoint_sha256'],totals=totals,
        checks=['identities','GT ownership partition','outcome partition','class histogram',
            'query core union partition','GS contact bounds','attribute bounds','voxel strata totals','FP attribution totals'],
        postprocessing_source_sha256={p:sha(ROOT/p) for p in (
            'tools/report_query_budget.py','tools/audit_query_budget_stats.py',
            'tests/test_query_budget_statistics.py','tests/test_query_budget_reporting.py')})
    write(RUN/'audit.json',result)
    print({k:result[k] for k in ('status','scope','frame_identities','query_frame_records')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    audit(args.partial)
