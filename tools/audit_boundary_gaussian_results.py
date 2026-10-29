"""Independent metadata/count/source audit of the complete boundary census."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/boundary_gaussian_full'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    protocol=read(RUN/'protocol.json')
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    for path,digest in protocol['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
        assert sha(RUN/'sources'/path)==digest,path
    pooled={};seen_scenes=set();manifest={}
    for split in ('train','val'):
        annotation=ROOT/f'data/nuscenes_cam/nuscenes_infos_{split}_sweeps_occ.pkl'
        assert sha(annotation)==protocol['annotation_sha256'][split]
        data=pickle.loads(annotation.read_bytes())
        keys=sorted(data['metadata'],key=lambda x:x[0]+'{:0>3}'.format(str(x[1])))
        expected={}
        for i,(scene,frame) in enumerate(keys):expected.setdefault(scene,[]).append((i,int(frame)))
        paths=sorted((RUN/'scenes'/split).glob('*.json'))
        assert {p.stem for p in paths}==set(expected)
        assert not (seen_scenes&set(expected))
        seen_scenes.update(expected)
        global_counts=np.zeros((4,9,5),np.int64)
        global_g=None;frames=0
        for path in paths:
            row=read(path);scene=path.stem
            assert row['status']=='complete' and row['split']==split and row['scene_token']==scene
            assert row['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert [(f['index'],f['scene_frame']) for f in row['records']]==expected[scene]
            assert row['indices']==[i for i,_ in expected[scene]]
            assert row['frames']==len(row['records'])
            for f in row['records']:
                info=data['infos'][scene][f['scene_frame']]
                assert f['sample_token']==info['token']
                assert row['scene_name']==Path(info['occ_path']).parts[-3]
                c=np.array(f['error_counts'],np.int64)
                assert c.shape==(4,9,5) and (c>=0).all()
                np.testing.assert_array_equal(c[0,0],c[2,0])
                np.testing.assert_array_equal(c[1,0],c[3,0])
                np.testing.assert_array_equal(c[:,0],c[:,4]+c[:,7]+c[:,8])
                assert (c[:,1:4]>=c[:,4:7]).all()
                assert (c[:,1:4]<=c[:,0:1]).all()
                assert (c[1,0]<=c[0,0]).all()
                assert (c[3,0]<=c[2,0]).all()
                for g in f['gaussian_counts']:
                    assert g['predicted']==76800
                    h=g['class_count_histogram']
                    assert sum(h)==g['predicted']
                    assert g['supported']==g['predicted']-h[0]
                    assert g['boundary_any']==sum(h[2:])
                    assert max(g['boundary_foreground'],g['boundary_fg_empty'])<=g['boundary_any']
                    assert g['boundary_foreground']+g['boundary_fg_empty']>=g['boundary_any']
                for field in ('supported','boundary_any','boundary_foreground','boundary_fg_empty'):
                    gs=f['gaussian_counts']
                    assert gs[1][field]<=gs[0][field] and gs[2][field]<=gs[0][field]
                    assert gs[3][field]<=gs[1][field] and gs[3][field]<=gs[2][field]
            expected_c=np.array([f['error_counts'] for f in row['records']],np.int64).sum(0)
            np.testing.assert_array_equal(expected_c,row['error_counts'])
            for s in range(4):
                for field,value in row['gaussian_counts'][s].items():
                    expected_g=np.array([f['gaussian_counts'][s][field] for f in row['records']]).sum(0)
                    np.testing.assert_array_equal(expected_g,value)
            if global_g is None:global_g=[dict(g) for g in row['gaussian_counts']]
            else:
                for s in range(4):
                    for k,v in row['gaussian_counts'][s].items():
                        global_g[s][k]=(np.asarray(global_g[s][k])+v).tolist()
            global_counts+=expected_c;frames+=row['frames']
            manifest[str(path.relative_to(RUN))]=sha(path)
        pooled[split]=dict(counts=global_counts,gaussians=global_g,frames=frames,scenes=len(paths))
        print(split,frames,len(paths),'metadata and counts passed',flush=True)
        del data
    summary=read(RUN/'summary.json')
    assert summary['status']=='complete'
    for split,value in pooled.items():
        actual=summary['statistics'][split]
        assert actual['frames']==value['frames'] and actual['scenes']==value['scenes']
        np.testing.assert_array_equal(actual['error_counts'],value['counts'])
        for s in range(4):
            for k,v in value['gaussians'][s].items():np.testing.assert_array_equal(v,actual['gaussian_counts'][s][k])
    np.testing.assert_array_equal(summary['statistics']['total']['error_counts'],
                                  pooled['train']['counts']+pooled['val']['counts'])
    with (RUN/'scene_statistics.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==850 and {r['scene_token'] for r in rows}==seen_scenes
    for r in rows:
        raw=read(RUN/'scenes'/r['split']/(r['scene_token']+'.json'))
        for s,prefix in ((0,'all'),(1,'camera')):
            g=raw['gaussian_counts'][s]
            assert float(r[prefix+'_boundary_any_fraction'])==g['boundary_any']/g['predicted']
            assert int(r[prefix+'_union_voxels'])==sum(raw['error_counts'][s][1])
            for i,name in ((1,'wrong_class'),(2,'miss'),(3,'false_positive')):
                assert int(r[prefix+'_union_'+name])==raw['error_counts'][s][1][i]
    out=dict(status='passed',frames=34149,scenes=850,
        checkpoint_sha256=protocol['checkpoint_sha256'],checks=[
            'current and archived source hashes','checkpoint and annotation hashes',
            'exact metadata scene/frame/sample-token correspondence',
            'disjoint train/validation scenes','every-frame cohort partition and coverage bounds',
            'Gaussian class-histogram and boundary-definition identities',
            'camera/core support monotonicity','frame-to-scene-to-global raw count sums',
            '850-row CSV keys, denominators and error counts'],
        scene_file_sha256=manifest,audit_source_sha256=sha(__file__))
    (RUN/'integrity_audit.json').write_text(json.dumps(out,indent=2))
    print('all integrity checks passed',flush=True)


if __name__=='__main__':main()
