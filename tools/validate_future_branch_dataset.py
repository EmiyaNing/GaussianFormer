#!/usr/bin/env python3
"""Mechanical U2-A checks only; never turns test fixtures into scene evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO=Path(__file__).resolve().parents[1]


def fingerprint(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def validate(source):
    if source is None:
        return dict(status='blocked_no_source',root_states=0,branches=0,
                    actual_u2_experiment=False,reason='No branch manifest supplied.')
    source=Path(source).resolve();record=json.loads(source.read_text());base=source.parent
    if record.get('schema')!=1 or record.get('horizons')!=[1,2,3]:
        raise ValueError('Expected schema 1 and horizons [1,2,3] seconds')
    for key in ('environment_provenance','teacher_checkpoint_sha256','sensor_configuration'):
        if not record.get(key):raise ValueError(f'Missing provenance: {key}')
    roots=record.get('roots',[])
    if not roots:raise ValueError('No roots')
    scene_splits={};root_ids=set();rows=[];source_hashes={str(source):fingerprint(source)}

    def file(path):
        value=(base/path).resolve()
        if not value.is_file():raise ValueError(f'Missing artifact: {value}')
        source_hashes[str(value)]=fingerprint(value)
        return value

    def files(paths):
        if not paths:raise ValueError('Empty input file list')
        return [source_hashes[str(file(p))] for p in paths]

    for root in roots:
        if root['id'] in root_ids:raise ValueError('Duplicate root id')
        root_ids.add(root['id'])
        split=root['split'];scene=root['scene_family']
        if split not in ('train','development','confirmation'):raise ValueError('Unknown root split')
        if scene in scene_splits and scene_splits[scene]!=split:
            raise ValueError('Scene family crosses data splits')
        scene_splits[scene]=split
        history=files(root['student_history_files'])
        action=fingerprint(file(root['ego_action_file']))
        state=fingerprint(file(root['fork_state_file']))
        anchors=np.load(file(root['world_anchors_file']),allow_pickle=False)
        if anchors.ndim!=2 or anchors.shape[1]!=3 or not len(anchors) or not np.isfinite(anchors).all():
            raise ValueError('Invalid world anchors')
        branches=root['branches']
        if len(branches)<2:raise ValueError('Need at least two futures per root')
        targets=[];branch_ids=set()
        for branch in branches:
            if branch['id'] in branch_ids:raise ValueError('Duplicate branch id')
            branch_ids.add(branch['id'])
            if files(branch['student_history_files'])!=history:raise ValueError('Different student history')
            if fingerprint(file(branch['ego_action_file']))!=action:raise ValueError('Different ego actions')
            if fingerprint(file(branch['fork_state_file']))!=state:raise ValueError('Different fork state')
            if branch.get('teacher_checkpoint_sha256')!=record['teacher_checkpoint_sha256']:
                raise ValueError('Different teacher identity')
            for h in ('1','2','3'):
                files(branch['future_image_files'][h])
            with np.load(file(branch['anchor_targets_file']),allow_pickle=False) as data:
                labels=data['gt_labels'].copy();features=data['teacher_features'].copy()
                valid=data['valid'].copy();positions=data['world_anchors'].copy()
                repeat=data['teacher_repeats'].copy()
                ego=data['future_ego_poses'].copy()
            n=len(anchors)
            if labels.shape!=(3,n) or valid.shape!=(3,n) or features.shape!=(3,n,256):
                raise ValueError('Target shape mismatch')
            if valid.dtype!=np.bool_ or not np.issubdtype(labels.dtype,np.integer) or ((labels<0)|(labels>17)).any():
                raise ValueError('Invalid labels or mask')
            if positions.shape!=anchors.shape or not np.array_equal(positions,anchors):
                raise ValueError('Different physical world anchors')
            if repeat.ndim!=4 or repeat.shape[0]<5 or repeat.shape[1:]!=(3,n,256):
                raise ValueError('Need five same-input teacher repeat measurements')
            if ego.shape!=(3,4,4) or not all(np.isfinite(a).all() for a in (features,repeat,ego)):
                raise ValueError('Nonfinite teacher features or invalid ego poses')
            targets.append(dict(labels=labels,features=features,valid=valid,repeats=repeat,ego=ego))
        # This initial protocol fixes actual ego poses, not only action commands.
        if any(not np.array_equal(t['ego'],targets[0]['ego']) for t in targets[1:]):
            raise ValueError('Initial replay-ego protocol requires identical future ego poses')
        common=np.logical_and.reduce([t['valid'] for t in targets])
        if not common.any():raise ValueError('No common valid anchor')
        different=np.any(np.stack([t['labels']!=targets[0]['labels'] for t in targets[1:]]),axis=0)&common
        rows.append(dict(root=root['id'],split=split,branches=len(branches),
                         common_anchors=int(common.sum()),gt_disagreement_anchors=int(different.sum())))
    return dict(status='mechanical_contract_passed',root_states=len(roots),
                branches=sum(r['branches'] for r in rows),roots=rows,
                actual_u2_experiment=False,source_hashes=source_hashes,
                limitation='File equality and metadata checks do not establish physically valid branching, simulator restoration, unbiased probabilities, or a causal training result. Provenance must be reviewed before U2-B/C.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,default=REPO/'out/future_distillation_causal/20261008_u2_entry/validation.json')
    args=parser.parse_args();result=validate(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temporary.replace(args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))
