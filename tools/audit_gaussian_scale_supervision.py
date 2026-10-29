"""Check complete frame identities and frozen sources after the full census."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/gaussian_scale_supervision_full'


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    from mmengine import Config
    protocol=read(RUN/'protocol.json');original=read(ROOT/'work_dirs/boundary_gaussian_full/protocol.json')
    assert protocol['version']=='exact_raw_attribute_center_support_v3'
    assert read(RUN/'execution_complete.json')['status']=='complete'
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    cfg=Config.fromfile(str(ROOT/protocol['config']))
    config_digest=hashlib.sha256(cfg.pretty_text.encode()).hexdigest()
    assert config_digest==original['resolved_config_sha256']
    cfg.dump(RUN/'resolved_config.py')
    for split,digest in original['annotation_sha256'].items():
        assert sha(Path(cfg[split+'_dataset_config'].imageset))==digest,split
    for source,digest in {**original['source_sha256'],**protocol['source_sha256']}.items():
        assert sha(ROOT/source)==digest,source
    result={'status':'passed','checkpoint_sha256_verified':True,'frozen_sources_unchanged':True,
        'resolved_config_sha256':config_digest,'annotation_sha256':original['annotation_sha256'],'splits':{}}
    all_tokens=[]
    for split,expected in protocol['expected'].items():
        files=sorted((RUN/'scenes'/split).glob('*.json'))
        old=ROOT/'work_dirs/boundary_gaussian_full/scenes'/split
        assert {p.name for p in files}=={p.name for p in old.glob('*.json')}
        indices=[];tokens=[]
        for p in files:
            row=read(p);reference=read(old/p.name)
            assert row['status']=='complete' and row['checkpoint_sha256']==protocol['checkpoint_sha256']
            assert row['indices']==reference['indices']
            assert row['frames']==len(row['records'])==len(row['indices'])
            assert [(r['index'],r['sample_token']) for r in row['records']]==[
                (r['index'],r['sample_token']) for r in reference['records']]
            assert p.with_suffix('.npz').is_file()
            indices.extend(row['indices']);tokens.extend(r['sample_token'] for r in row['records'])
        assert len(files)==expected['scenes']
        assert len(indices)==len(set(indices))==len(set(tokens))==expected['frames']
        assert sorted(indices)==list(range(expected['frames']))
        result['splits'][split]={'scenes':len(files),'frames':len(indices),'unique_frames':len(set(indices)),
            'gaussian_instances':len(indices)*76800,'every_expected_index_present_once':True,
            'all_sample_tokens_match_original_census':True}
        all_tokens.extend(tokens)
    assert len(all_tokens)==len(set(all_tokens))==34149
    (RUN/'frame_source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
