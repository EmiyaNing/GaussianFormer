"""Verify all causal-trial artifacts after execution; no inference or training."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work_dirs/gaussian_geometry_causal'
NAMES = ('common', 'standard', 'no_amplitude', 'no_geometry', 'no_both',
         'no_empty_center_boost', 'mature_standard', 'mature_no_empty_center_boost')
EXPECTED = {'train': (700, 28130), 'val': (150, 6019)}
GS_PER_FRAME = 76800


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def main():
    assert (RUN / 'all_execution_complete.json').exists()
    protocol = read(RUN / 'protocol.json')
    audit = {'status': 'passed', 'models': {}, 'source_checks': {}, 'pairing': {}}
    # Frozen registrations retain their original hashes, including the actual
    # common launch snapshot. A later prose revision must not rewrite history.
    for group in ('source_sha256', 'inherited_source_sha256'):
        for rel, digest in protocol[group].items():
            path = ROOT / rel
            if rel.startswith('analysis/'):
                path = RUN / 'sources/paired' / rel
            assert sha(path) == digest, str(path)
            audit['source_checks'][str(path.relative_to(ROOT))] = digest
    common_source = RUN / 'sources/common/gaussian_geometry_causal.py'
    assert sha(common_source) == protocol['common_launch_source_sha256']
    for filename, key in (('selective_protocol.json', 'source_sha256'),
                          ('mature_protocol.json', 'source_sha256')):
        registration = read(RUN / filename)
        source = ROOT / ('tools/gaussian_geometry_selective_control.py'
                         if filename.startswith('selective')
                         else 'tools/gaussian_geometry_mature_control.py')
        assert sha(source) == registration[key]
    reference_gt = {}
    empty_parameters = {}
    for name in NAMES:
        complete = read(RUN / 'eval' / name / 'complete.json')
        assert complete['status'] == 'complete'
        checkpoint = Path(complete['checkpoint'])
        assert sha(checkpoint) == complete['checkpoint_sha256']
        assert complete['source_sha256'] == protocol['source_sha256'][
            'tools/eval_gaussian_geometry_causal.py']
        audit['models'][name] = {'checkpoint_sha256': complete['checkpoint_sha256'],
                                 'splits': {}}
        for split, (scene_count, frame_count) in EXPECTED.items():
            folder = RUN / 'eval' / name / split
            paths = sorted(folder.glob('*.json'))
            assert len(paths) == scene_count
            assert len(list(folder.glob('*.npz'))) == scene_count
            indices = []
            cm = np.zeros((18, 18), dtype=np.int64)
            gs_count = raw_count = coverage_targets = 0
            for path in paths:
                meta = read(path)
                assert meta['status'] == 'complete'
                assert meta['checkpoint_sha256'] == complete['checkpoint_sha256']
                assert len(meta['indices']) == meta['frames'] == len(meta['records'])
                assert [r['index'] for r in meta['records']] == meta['indices']
                indices.extend(meta['indices'])
                with np.load(path.with_suffix('.npz')) as a:
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                    matrix = a['confusion']
                    assert int(a['gaussian_total'][0]) == meta['frames'] * GS_PER_FRAME
                    assert int(a['raw_attribute'][0]) == meta['frames'] * GS_PER_FRAME
                    assert int(a['voxels'].sum()) == int(matrix[:17].sum())
                    assert int(a['margin_hist'].sum()) == int(matrix.sum())
                    assert np.array_equal(a['margin_hist'].sum((1, 2)), matrix.sum(1))
                    assert int(a['loss'][2]) == meta['frames']
                    cm += matrix
                    gs_count += int(a['gaussian_total'][0])
                    raw_count += int(a['raw_attribute'][0])
                    coverage_targets += int(a['voxels'].sum())
            assert sorted(indices) == list(range(frame_count)), (name, split)
            assert gs_count == raw_count == frame_count * GS_PER_FRAME
            gt = cm.sum(1)
            if split in reference_gt:
                assert np.array_equal(gt, reference_gt[split])
            else:
                reference_gt[split] = gt
            audit['models'][name]['splits'][split] = {
                'scenes': scene_count, 'frames': frame_count, 'GS_observations': gs_count,
                'GT_foreground_voxels': coverage_targets,
                'GT_empty_voxels': int(gt[17]), 'all_frame_indices_exact': True}
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        empty_parameters[name] = {
            k: float(state['state_dict']['head.' + k])
            for k in ('empty_bias', 'empty_scalar')}
        del state
    for name in NAMES[1:6]:
        assert empty_parameters[name] == empty_parameters['common']
    original = torch.load(ROOT / 'ckpts/opus_gaussian_8f_tiny.pth',
                          map_location='cpu', weights_only=False)
    original_empty = {k: float(original['state_dict']['head.' + k])
                      for k in ('empty_bias', 'empty_scalar')}
    del original
    for name in NAMES[6:]:
        assert empty_parameters[name] == original_empty
    audit['empty_parameters'] = empty_parameters
    for filename in ('train/pairing_audit.json', 'selective_pairing_audit.json',
                     'mature_pairing_audit.json'):
        value = read(RUN / filename)
        assert value['status'] == 'passed' and value['all_data_fingerprints_equal']
        assert value['microbatches'] == 1024
        audit['pairing'][filename] = value
    audit['optimizer_diagnostics'] = {}
    for name in NAMES[1:]:
        seed = 932 if name.startswith('mature_') else 928
        folder = RUN / 'train' / f'{name}_s{seed}'
        value = read(folder / 'complete.json')
        assert value['microbatches'] == 1024 and value['optimizer_steps'] == 128
        assert value['source_sha256'] == protocol['source_sha256'][
            'tools/gaussian_geometry_causal.py']
        assert value['initial_sha256'] == sha(value['initial'])
        traces = [json.loads(s) for s in (folder / 'trace.jsonl').read_text().splitlines()]
        assert len(traces) == 1024
        assert [r['microbatch'] for r in traces] == list(range(1024))
        assert np.isfinite([r['total'] for r in traces]).all()
        norms = np.asarray([r['grad_norm'] for r in traces if 'grad_norm' in r])
        assert len(norms) == 128 and np.isfinite(norms).all()
        audit['optimizer_diagnostics'][name] = {
            'updates': 128, 'median_preclip_norm': float(np.median(norms)),
            'maximum_preclip_norm': float(norms.max()),
            'clipped_fraction': float((norms > 35).mean())}
    common = read(RUN / 'common/standard_s928/complete.json')
    assert common['microbatches'] == 2048 and common['optimizer_steps'] == 256
    audit['evaluation_frame_visits'] = len(NAMES) * sum(x[1] for x in EXPECTED.values())
    audit['formal_microbatches'] = 9216
    audit['formal_optimizer_updates'] = 1152
    audit['passed_at'] = datetime.now().astimezone().isoformat()
    write(RUN / 'final_integrity_audit.json', audit)

    import mmengine
    import mmcv
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                   'numpy': np.__version__, 'mmengine': mmengine.__version__,
                   'mmcv': mmcv.__version__}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True)
    environment['gpu'] = gpu.stdout.strip()
    write(RUN / 'execution_environment.json', environment)

    # Preserve the final reporting sources and human-readable deliverables
    # separately from the pre-training snapshots.
    sources = sorted((ROOT / 'tools').glob('*gaussian_geometry*.py'))
    sources += [ROOT / 'analysis/gaussian_geometry_causal_design.md',
                ROOT / 'analysis/gaussian_geometry_causal_results.md',
                ROOT / 'loss/opus_set_loss.py', ROOT / 'loss/occupancy_loss.py',
                ROOT / 'loss/multi_loss.py',
                ROOT / 'model/head/gaussian_opus_head.py',
                ROOT / 'model/head/opus_v1_gaussian_head.py',
                ROOT / 'model/head/opus_v1_gaussian_trainable_bias_head.py',
                ROOT / 'model/encoder/opus_encoder.py',
                ROOT / 'model/encoder/opus_v1_gaussian_encoder.py',
                ROOT / 'model/encoder/opus_v1_gaussian_softplus_encoder.py',
                ROOT / 'model/encoder/opus_v1_gaussian_temporal_encoder.py']
    native = ROOT / 'model/head/localagg'
    sources += [p for p in native.rglob('*') if p.is_file()
                and 'build' not in p.parts and '__pycache__' not in p.parts
                and p.suffix in ('.cu', '.h', '.cpp', '.py', '.so')]
    manifest = {}
    for source in sorted(set(sources)):
        rel = source.relative_to(ROOT)
        dest = RUN / 'sources/final' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest[str(rel)] = sha(source)
    artifacts = sorted(RUN.glob('*.json')) + sorted(RUN.glob('*.csv'))
    artifacts += sorted((ROOT / 'analysis/figures/gaussian_geometry_causal').glob('*'))
    write(RUN / 'final_artifact_manifest.json', {
        'written_at': datetime.now().astimezone().isoformat(),
        'final_source_sha256': manifest,
        'artifacts_sha256': {str(p.relative_to(ROOT)): sha(p) for p in artifacts
                            if p.name != 'final_artifact_manifest.json'}})
    print(json.dumps({'status': 'passed', 'models': len(NAMES),
                      'evaluation_frame_visits': audit['evaluation_frame_visits'],
                      'empty_parameters_unchanged_within_pairs': True}, indent=2))


if __name__ == '__main__':
    main()
