"""Independent saved-count and identity audit for the small-checkpoint census."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.diagnose_small_gaussians import RUN,Q,CHILDREN,GROUPS,ATTRS,FLAGS
from tools.stat_query_budget import FIELDS
from tools.stat_boundary_gaussians import read,write,sha


def audit(partial=False):
    protocol=read(RUN/'protocol.json')
    for source,digest in {**protocol['inherited_source_sha256'],**protocol['source_sha256']}.items():
        assert sha(ROOT/source)==digest,source
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    totals={};tokens=set();identities=set()
    for split in ('train','val'):
        frames=scenes=0
        for path in sorted((RUN/'scenes'/split).glob('*.json')):
            meta=read(path)
            assert meta['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert meta['indices']==[r['index'] for r in meta['records']]
            old=read(ROOT/'work_dirs/query_budget_full/scenes'/split/path.name)
            assert meta['indices']==old['indices']
            assert [r['sample_token'] for r in meta['records']]==[r['sample_token'] for r in old['records']]
            assert [r['foreground'] for r in meta['records']]==[r['foreground'] for r in old['records']]
            for r in meta['records']:
                assert r['sample_token'] not in tokens
                tokens.add(r['sample_token'])
                identity=(split,r['index'])
                assert identity not in identities
                identities.add(identity)
                assert r['false_positive']>=0 and r['unattributed_fp']==0
            with np.load(path.with_suffix('.npz')) as z:
                a,classes,strata=z['metrics'],z['class_counts'],z['strata']
                n,sums,flags=z['group_counts'],z['attribute_sums'],z['flag_counts']
                cm,errors,bq=z['confusion'],z['boundary_errors'],z['boundary_queries']
            assert a.shape==(meta['frames'],Q,len(FIELDS))
            assert classes.shape==(meta['frames'],Q,17)
            assert np.isfinite(a).all() and (a>=0).all()
            f=lambda name:a[...,FIELDS.index(name)]
            np.testing.assert_array_equal(f('gt_foreground'),classes.sum(-1))
            np.testing.assert_array_equal(f('gt_foreground'),f('gt_own_core')+f('gt_neighbor_only_core')+f('gt_no_core'))
            np.testing.assert_array_equal(f('gt_foreground'),f('gt_correct')+f('gt_wrong_class')+f('gt_miss'))
            np.testing.assert_array_equal(f('core_coverage_all_fg'),f('core_exclusive')+f('core_shared_other_query'))
            for k in ('core_gs_touches_fg','native_gs_touches_fg','gs_camera_native','gs_camera_core'):
                assert (f(k)<=CHILDREN).all()
            for j,k in enumerate(('gt_foreground','gt_no_core','gt_miss','gt_no_core_correct','gt_own_core')):
                assert int(strata[...,j].sum())==int(f(k).sum(dtype=np.float64))
            assert int(f('dominant_fp_voxels').sum(dtype=np.float64))==int(cm[17,:17].sum())
            assert int(f('gt_foreground').sum(dtype=np.float64))==int(cm[:17].sum())
            assert int(f('gt_no_core').sum(dtype=np.float64))==sum(r['no_core'] for r in meta['records'])
            assert n.shape==(len(GROUPS),) and sums.shape==(len(GROUPS),len(ATTRS)) and flags.shape==(len(GROUPS),len(FLAGS))
            assert n[0]==meta['frames']*76800 and n[1:4].sum()==n[0]
            assert (flags>=0).all() and (flags<=n[:,None]).all()
            assert np.isfinite(sums).all()
            assert np.all(sums[:,ATTRS.index('opacity')]<=n+1e-6)
            assert sums[0,-1]==cm[17,:17].sum()
            np.testing.assert_array_equal(errors[:,0],errors[:,4]+errors[:,7]+errors[:,8])
            correct=np.trace(cm[:17,:17]);wrong=cm[:17,:17].sum()-correct
            np.testing.assert_array_equal(errors[1,0],[correct,wrong,cm[:17,17].sum(),cm[17,:17].sum(),cm[17,17]])
            assert bq.shape==(meta['frames'],Q,2,3) and (bq>=0).all() and (bq<=CHILDREN).all()
            for s,scope in enumerate((1,3)):
                np.testing.assert_array_equal(bq[:,:,s].sum((0,1)),[meta['gaussian_counts'][scope]['boundary_'+k] for k in ('any','foreground','fg_empty')])
            assert n[GROUPS.index('mixed_foreground_camera_3sigma')]==meta['gaussian_counts'][3]['boundary_foreground']
            assert n[GROUPS.index('pure_foreground_camera_3sigma')]+n[GROUPS.index('pure_empty_camera_3sigma')]+meta['gaussian_counts'][3]['boundary_any']==meta['gaussian_counts'][3]['supported']
            scenes+=1;frames+=meta['frames']
        totals[split]=dict(scenes=scenes,frames=frames)
    complete=totals==protocol['expected']
    if not partial:
        assert complete,totals
        assert len(tokens)==len(identities)==34149
    result=dict(status='passed' if complete else 'partial_passed',scope=totals,unique_sample_tokens=len(tokens),
        query_frame_records=len(tokens)*Q,checkpoint_sha256=protocol['checkpoint_sha256'],
        checks=['source/checkpoint hashes','tiny/small frame identity and GT denominators','query partitions',
            'query coverage/contact bounds','voxel strata','confidence flag denominators',
            'boundary GS counts by query','confusion matrix and boundary error agreement','FP ownership'],
        audit_source_sha256=sha(Path(__file__)))
    write(RUN/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    audit(args.partial)
