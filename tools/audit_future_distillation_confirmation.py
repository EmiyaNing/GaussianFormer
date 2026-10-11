#!/usr/bin/env python3
"""CPU-only S3/S5/U2 entry audit. Does not train or claim causal confirmation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = Path('out/future_distillation_causal/20261001_training_protocol')
DEVELOPMENT_ROOTS = (
    Path('out/future2current'), Path('out/future_distillation_causal'),
    Path('out/future_distillation_response_based'),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scene_audit(available, used):
    available, used = set(available), set(used)
    return dict(available_scenes=len(available), development_scenes=len(used),
                used_available_scenes=len(available & used),
                unused_scenes=sorted(available - used),
                unused_count=len(available - used))


def confirmation_gate(unused_count, seeds, epochs):
    reasons = []
    if unused_count == 0:
        reasons.append('No untouched validation scene in the audited local inventory.')
    if len(set(seeds)) < 3:
        reasons.append('Fewer than three paired training seeds.')
    if epochs != 5:
        reasons.append('The preregistered final epoch is five.')
    return dict(eligible=not reasons, reasons=reasons)


def combination_gate(accepted):
    # Only independently confirmed effects enter S5; a screening point gain
    # or BN protocol repair does not count as a confirmed teacher intervention.
    names = sorted({row['name'] for row in accepted
                    if row.get('independent_confirmation') is True
                    and row.get('paired_seeds', 0) >= 3
                    and row.get('useful') is True})
    return dict(eligible=len(names) >= 2, confirmed_interventions=names,
                required_interventions=2)


def inventory(path):
    # The existing repository's trusted annotation pickle is read-only.
    with path.open('rb') as stream:
        value = pickle.load(stream)
    rows = value['infos']
    if any(not row.get('scene_token') for row in rows):
        raise ValueError(f'Missing scene identity: {path}')
    return dict(path=str(path.relative_to(REPO)), sha256=sha256(path),
                rows=len(rows), scenes=sorted({row['scene_token'] for row in rows}))


def audit(output):
    manifests, used = [], {'train': set(), 'val': set()}
    for relative in DEVELOPMENT_ROOTS:
        for path in sorted((REPO / relative).rglob('manifest.json')):
            if 'smoke' in str(path.relative_to(REPO)):
                continue
            value = json.loads(path.read_text())
            record = dict(path=str(path.relative_to(REPO)), sha256=sha256(path))
            for split in ('train', 'val'):
                part = value.get(split)
                if not isinstance(part, dict) or not part.get('scenes'):
                    continue
                if len(part['scenes']) != len(part['indices']):
                    raise ValueError(f'Sample/scene count mismatch: {path}')
                used[split].update(part['scenes'])
                record[split] = dict(samples=len(part['indices']),
                                     scenes=len(set(part['scenes'])))
            manifests.append(record)
    inventories = {split: inventory(REPO / f'data/nuscenes_infos_{split}_w_3occ.pkl')
                   for split in ('train', 'val')}
    coverage = {split: scene_audit(item['scenes'], used[split])
                for split, item in inventories.items()}
    if not manifests:
        raise ValueError('No development evidence found; absence is not independence.')
    root = REPO / PROTOCOL
    report = json.loads((root / 'results.json').read_text())
    intervals = json.loads((root / 'bootstrap.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    final = str(manifest['evaluation_steps'][-1])
    evidence = {}
    for name in ('task', 'teacher', 'shuffled'):
        label = f'full_bn_fixed_{name}_seed20260930'
        row = report['arms'][label]
        if row['status'] != 'complete':
            raise ValueError(f'Incomplete source arm: {label}')
        evidence[name] = row['steps'][final]['metrics']
    teacher_task = intervals['paired_changes'][
        'full_bn_fixed_teacher_seed20260930 vs full_bn_fixed_task_seed20260930']
    teacher_shuffled = intervals['paired_changes'][
        'full_bn_fixed_teacher_seed20260930 vs full_bn_fixed_shuffled_seed20260930']
    branch_candidates = sorted(str(path.relative_to(REPO))
                               for path in (REPO / 'data').rglob('*branch*') if path.is_file())
    value = dict(
        status='prerequisite_audit_complete_not_training_complete',
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        manifests=manifests, inventories=inventories, scene_coverage=coverage,
        S3=dict(status='blocked_independent_confirmation',
                gate=confirmation_gate(coverage['val']['unused_count'],
                                       [20260930, 20261001, 20261002], 5),
                proposed_budget=dict(train=4096, val=1024, epochs=5,
                                     seeds=[20260930, 20261001, 20261002]),
                new_training_runs=0),
        S5=dict(status='not_entered', gate=combination_gate([]),
                accepted_ledger=[], local_seed_count=len(manifest['seeds']),
                previous_native_metrics=evidence,
                previous_teacher_vs_task_ci=teacher_task,
                previous_teacher_vs_shuffled_ci=teacher_shuffled,
                new_training_runs=0),
        U2=dict(status='blocked_no_verified_branch_source',
                carla_python_available=importlib.util.find_spec('carla') is not None,
                servers_on_path={name: shutil.which(name)
                                 for name in ('CarlaUE4.sh', 'CarlaUnreal.sh')},
                branch_named_candidate_files=branch_candidates,
                verified_branch_datasets=[], new_branch_rollouts=0,
                scope='Checks local data filenames and Python/PATH only; not proof no remote simulator exists.'),
        source_hashes={str(path.relative_to(REPO)): sha256(path)
                       for path in (root/'results.json', root/'bootstrap.json', root/'manifest.json')},
        limitations=[
            'Inventory rows are not all necessarily forecasting-eligible; scene exhaustion already holds at the raw-scene superset level.',
            'Previous results above are cited evidence, not new S3 or S5 runs.',
            'No new data acquisition, simulator installation, split relabeling, or training was performed.',
        ])
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'entry_audit.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(destination)
    print(json.dumps(dict(artifact=str(destination), stages={stage: value[stage]['status']
          for stage in ('S3', 'S5', 'U2')}, scene_coverage=coverage), ensure_ascii=False, indent=2))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=REPO/'out/future_distillation_causal/20261007_confirmation_entry')
    audit(parser.parse_args().output_dir)
