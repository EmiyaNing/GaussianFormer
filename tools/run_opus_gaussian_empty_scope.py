"""Frozen OPUS Gaussian: compare camera, camera-mask complement, and full space."""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.center_support_common import write_json
from tools.diagnose_opus_v1_gaussian import choose_empty_shift, file_sha256

RUN = ROOT / 'work_dirs/opus_gaussian_empty_scope'
REPORT = ROOT / 'plans/sparse_render_offical_9_28/opus_gaussian_empty_scope_results.md'
MODELS = {
    'current': dict(label='当前版：softplus＋camera mask＋learned bias',
        config='config/center_support_contribution1/a0_original.py',
        checkpoint='work_dirs/center_support_contribution1/train/a0_original_seed928/epoch_5.pth',
        raw='work_dirs/center_support_contribution1/train/a0_original_seed928/evaluation_epoch5.json',
        calibrated='work_dirs/no_set_loss_ablation_mini/with_set_loss_official_calibrated.json'),
    'original': dict(label='原始版：raw logits＋全域 voxel 监督',
        config='config/mini_occ3d_4096_1024/opusv1_gaussian_sparse_render.py',
        checkpoint='work_dirs/mini_occ3d_4096_1024/opusv1_gaussian_sparse_render/epoch_5.pth'),
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def job(name, command, requirements, checkpoint_hash):
    output = RUN / (name + '.json')
    result = read(output)
    if result and result.get('status') == 'complete':
        assert all(result['protocol'].get(k) == v for k, v in requirements.items()), name
        assert result['checkpoint_sha256'] == checkpoint_hash, name
        return result
    write_json(RUN / 'status.json', dict(status='running', task=name, command=command,
        updated_at=datetime.now().astimezone().isoformat()))
    print('Starting', name, flush=True)
    with (RUN / (name + '.log')).open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = read(output)
    if not result or result.get('status') != 'complete':
        raise RuntimeError(f'{name} exited without a complete result')
    return job(name, command, requirements, checkpoint_hash)


def sweep(name, settings, split, shifts, checkpoint_hash):
    limit = 256 if split == 'train' else 1024
    task = f'{name}_{split}_' + ('initial' if len(shifts) == 17 else 'wide')
    cmd = [sys.executable, 'tools/diagnose_gaussianformer_empty.py', '--py-config', settings['config'],
        '--checkpoint', settings['checkpoint'], '--split', split, '--limit', str(limit),
        '--include-camera-invisible', '--shifts', *map(str, shifts), '--out',
        str((RUN / (task + '.json')).relative_to(ROOT))]
    if split == 'train':
        cmd += ['--selection-manifest', str((RUN / 'training_calibration_manifest.json').relative_to(ROOT))]
    result = job(task, cmd, dict(py_config=settings['config'], checkpoint=settings['checkpoint'],
        split=split, limit=limit, shifts=shifts, include_camera_invisible=True), checkpoint_hash)
    assert result['samples'] == limit and result['native_zero_offset_parity_passed']
    assert result['diagnostic_source_sha256'] == file_sha256(ROOT / 'tools/diagnose_gaussianformer_empty.py')
    for b in shifts:
        counts = [np.asarray(result['masks'][mask]['empty_logit_shift_metrics'][str(b)]['confusion'])
                  for mask in ['all_valid', 'camera', 'camera_invisible']]
        assert np.array_equal(counts[0], counts[1]+counts[2]), 'Mask partition is inconsistent'
    return result


def select(training):
    return choose_empty_shift(dict(split=training['split'],
        empty_logit_shift_metrics=training['masks']['camera']['empty_logit_shift_metrics']))


def official(name, settings, shift, calibrated, checkpoint_hash):
    mode = 'calibrated' if calibrated else 'raw'
    cached = settings.get(mode)
    if cached:
        e = read(ROOT / cached)
        if e and e.get('status') == 'complete' and e['protocol'].get('empty_logit_shift', 0.) == shift:
            assert e['protocol']['py_config'] == settings['config']
            assert e['protocol']['checkpoint'] == settings['checkpoint']
            if e.get('checkpoint_sha256'):
                assert e['checkpoint_sha256'] == checkpoint_hash
            else:
                # The legacy raw evaluation predates checkpoint hashes. Verify
                # against its contemporaneous, hashed train diagnostic as well.
                reference = read(ROOT / 'work_dirs/no_set_loss_ablation_mini/with_set_loss_train_initial.json')
                assert reference['checkpoint_sha256'] == checkpoint_hash
                assert reference['protocol']['checkpoint'] == settings['checkpoint']
            return dict(e, reused_from=cached, reused_artifact_sha256=file_sha256(ROOT / cached))
    task = f'{name}_official_{mode}'
    cmd = [sys.executable, 'tools/eval_center_support.py', '--py-config', settings['config'],
        '--checkpoint', settings['checkpoint'], '--empty-logit-shift', str(shift),
        '--out', str((RUN / (task + '.json')).relative_to(ROOT))]
    return job(task, cmd, dict(py_config=settings['config'], checkpoint=settings['checkpoint'],
        empty_logit_shift=shift), checkpoint_hash)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'work_dirs/center_support_contribution1/execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    models = {}
    provenance = read(RUN / 'initial_provenance.json')
    for path, expected in provenance['sha256'].items():
        assert file_sha256(ROOT / path) == expected, path
    for name, settings in MODELS.items():
        checkpoint_hash = file_sha256(ROOT / settings['checkpoint'])
        shifts = list(map(float, range(-8, 9)))
        training = sweep(name, settings, 'train', shifts, checkpoint_hash)
        if abs(select(training)) == 8:
            shifts = list(map(float, range(-16, 17)))
            training = sweep(name, settings, 'train', shifts, checkpoint_hash)
        b = select(training)
        validation = sweep(name, settings, 'val', shifts, checkpoint_hash)
        assert training['imageset_sha256'] != validation['imageset_sha256']
        assert training['checkpoint_sha256'] == validation['checkpoint_sha256']
        raw = official(name, settings, 0., False, checkpoint_hash)
        calibrated = official(name, settings, b, True, checkpoint_hash)
        for shift, e in [(0., raw), (b, calibrated)]:
            assert e['samples'] == 1024 and e['streaming_official_parity_passed']
            m = validation['masks']['camera']['empty_logit_shift_metrics'][str(shift)]
            delta = {k:e['camera_metrics'][k]-m[k] for k in ['mIoU', 'IoU', 'precision', 'recall']}
            assert all(abs(v) < 1e-8 for v in delta.values()), delta
        models[name] = dict(label=settings['label'], selected_shift=b, training=training,
            validation=validation, raw_official=raw, calibrated_official=calibrated)
        write_json(RUN / 'partial_summary.json', dict(status='running', models=models))
    gf = read(ROOT / 'work_dirs/gaussianformer_empty_diagnostic/summary.json')
    assert gf['validation']['imageset_sha256'] == models['current']['validation']['imageset_sha256']
    summary = dict(status='complete', finished_at=datetime.now().astimezone().isoformat(), models=models,
        gaussianformer_reference=dict(selected_shift=gf['selected_shift'], validation=gf['validation'],
            raw_official=gf['raw_official'], calibrated_official=gf['calibrated_official']),
        source_sha256={p:file_sha256(ROOT / p) for p in ['tools/run_opus_gaussian_empty_scope.py',
            'tools/diagnose_gaussianformer_empty.py', 'tools/eval_center_support.py']})
    from tools.report_opus_gaussian_empty_scope import write_report, plot
    plot(summary)
    write_report(summary)
    write_json(RUN / 'summary.json', summary)
    write_json(RUN / 'completion_audit.json', dict(status='complete', samples_train_per_model=256,
        samples_validation_per_model=1024, checkpoint_and_input_hashes_verified=True,
        zero_shift_native_parity=True, camera_plus_complement_equals_full_confusion=True,
        camera_sweep_official_exact_parity=True, official_ray_count_parity=True))
    write_json(RUN / 'status.json', dict(status='complete', finished_at=summary['finished_at']))
    print('Completed', {name: m['selected_shift'] for name, m in models.items()}, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(RUN / 'status.json', dict(status='failed', error=repr(error)))
        raise
