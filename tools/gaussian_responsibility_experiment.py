"""Register, preflight and run A1 -> A2 -> A3 using the native five-epoch runner."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'work_dirs/gaussian_responsibility'
CONFIGS = dict(A0='config/gf2_probability_decomposition/probability.py',
    A1='config/gaussian_responsibility/a1_softmax.py',
    A2='config/gaussian_responsibility/a2_render.py',
    A3='config/gaussian_responsibility/a3_transport.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from tools.center_support_common import write_json
    write_json(path, value)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(name):
    if name == 'A0':
        return ROOT / 'work_dirs/gf2_probability_decomposition/train/probability/epoch_5.pth'
    return RUN / 'train' / name / 'epoch_5.pth'


def initialize():
    from mmengine import Config
    import mmengine
    path = RUN / 'protocol.json'
    if path.exists():
        return read(path)
    cfg = Config.fromfile(CONFIGS['A0'])
    prior = read(ROOT / 'work_dirs/gf2_probability_decomposition/protocol.json')
    old = read(ROOT / 'work_dirs/gf2_probability_decomposition/train/probability/complete.json')
    assert old['checkpoint_sha256'] == sha(checkpoint('A0'))
    scenes = {}
    for split in ('train', 'val'):
        data = mmengine.load(cfg[split + '_dataset_config'].imageset)
        first = {}
        for index, (scene, frame) in enumerate(data['metadata']):
            first.setdefault(scene, dict(index=index, scene=scene, scene_frame=int(frame)))
        keys = sorted(first)
        import numpy as np
        ids = np.linspace(0, len(keys) - 1, 32, dtype=int)
        scenes[split] = [first[keys[i]] for i in ids]
    sources = ['loss/gaussian_responsibility_loss.py',
        'loss/gaussian_responsibility_experiment_loss.py',
        'tools/gaussian_responsibility_experiment.py',
        'train.py', 'model/head/opus_gf2_probability_head.py',
        'model/head/localagg_prob/src/forward.cu', 'tests/test_gaussian_responsibility.py',
        *CONFIGS.values()]
    p = dict(registered_at=datetime.now().astimezone().isoformat(), train_frames=4096,
        validation_frames=1024, epochs=5, optimizer_updates=5120, train_frame_visits=20480,
        seed=928, accumulation=4, baseline_reused=True, baseline_sha256=sha(checkpoint('A0')),
        configs=CONFIGS, affected_stage='final Gaussian classification only; other five Set stages unchanged',
        gt_domain='All valid foreground for Set, unchanged from A0; default camera-mask occupancy loss/evaluation',
        class_objective=dict(A0='original nearest-GT sigmoid focal', A1='nearest-GT softmax CE',
            A2='renderer-responsibility soft-target CE', A3='sparse unbalanced-transport soft-target CE'),
        gradient_control='Per microbatch match final-logit L2 classification gradient norm to native focal at current output; detached multiplier; original nearest-GT class/distance weights in all arms',
        assignment=dict(stop_gradient=True, entropy_epsilon=1., tau_row=1., tau_voxel=10., iterations=24,
            capacity='a_i proportional to sqrt(A2 responsibility mass); sum a = number of GT foreground voxels',
            kernel='column-normalized native density including covariance determinant and opacity; no semantic cost',
            candidates='exact native integer boxes, foreground GT only; same nearest-Gaussian guard for unsupported GT in A2 and A3',
            empty_gaussian_rows='keep original nearest-GT label, preserving class normalization and supervised Gaussian count'),
        probe_samples=scenes, probe_rms_steps=[.01, .03, .1],
        probe_definition='Output-logit steps with fixed geometry on each probe frame, not fitted model generalization; o must be unchanged',
        fit_scenes=prior['fit_scenes'], tune_scenes=prior['tune_scenes'],
        evaluation='Every 1024 validation frame, native joint and hierarchical decisions plus equal train-only global/class calibration',
        uncertainty='1000 paired validation-scene bootstrap samples, seed928; single training seed',
        primary_comparisons=['A1-A0 classification form', 'A2-A1 responsibility definition', 'A3-A2 constrained allocation'],
        pretrain=dict(path=cfg.load_from, sha256=sha(ROOT / cfg.load_from)),
        annotation_sha256={s: sha(cfg[s+'_dataset_config'].imageset) for s in ('train', 'val')},
        source_sha256={s: sha(ROOT / s) for s in sources})
    write(path, p)
    for name in sources:
        target = RUN / 'sources_registered' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return p


class PilotComplete(Exception):
    pass


def pilot(name):
    import numpy as np
    import torch
    from mmengine import Config
    initialize()
    folder = RUN / 'pilot' / name
    if (folder / 'success.json').exists():
        return
    if (folder / 'microbatch_trace.jsonl').exists():
        raise RuntimeError('Retain and inspect partial pilot before an explicit retry')
    folder.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(CONFIGS[name]); cfg.loss.trace_dir = str(folder)
    cfg.dump(folder / 'pilot.py')
    original = torch.optim.AdamW.step
    def step(self, *args, **kwargs):
        assert all(torch.isfinite(p.grad).all() for group in self.param_groups
                   for p in group['params'] if p.grad is not None)
        original(self, *args, **kwargs)
        raise PilotComplete()
    torch.optim.AdamW.step = step
    sys.argv = ['tools/train_center_support.py', '--py-config', str(folder / 'pilot.py'),
        '--work-dir', str(folder), '--seed', '928', '--gradient-accumulation', '4']
    try:
        runpy.run_path(str(ROOT / 'tools/train_center_support.py'), run_name='__main__')
    except PilotComplete:
        pass
    finally:
        torch.optim.AdamW.step = original
    rows = [json.loads(line) for line in (folder / 'microbatch_trace.jsonl').read_text().splitlines()]
    reference = ROOT / 'work_dirs/gf2_probability_decomposition/train/probability'
    old = [json.loads(line) for line in (reference / 'microbatch_trace.jsonl').read_text().splitlines()[:4]]
    assert len(rows) == 4
    assert [r['data_fingerprint'] for r in rows] == [r['data_fingerprint'] for r in old]
    a, b = np.load(folder / 'initial_forward.npz'), np.load(reference / 'initial_forward.npz')
    np.testing.assert_allclose(a['raw'], b['raw'], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(a['labels'], b['labels'])
    for r in rows:
        assert np.isfinite(r['loss'])
        np.testing.assert_allclose(r['matched_grad_norm'], r['reference_grad_norm'], rtol=2e-6)
    write(folder / 'success.json', dict(status='passed', frames=4, optimizer_updates=1,
        initial_raw_error=float(abs(a['raw'] - b['raw']).max()), paired_inputs=True,
        finite_gradients=True, rows=rows))


def train(name):
    import numpy as np
    protocol = initialize()
    folder = RUN / 'train' / name
    if (folder / 'complete.json').exists():
        return
    if (folder / 'microbatch_trace.jsonl').exists():
        raise RuntimeError('Partial run exists; do not silently resume without RNG parity')
    folder.mkdir(parents=True, exist_ok=True)
    sources = protocol['source_sha256']
    assert all(sha(ROOT / s) == digest for s, digest in sources.items())
    with (folder / 'console.log').open('a') as stream:
        code = subprocess.call([sys.executable, 'tools/train_center_support.py',
            '--py-config', CONFIGS[name], '--work-dir', str(folder),
            '--seed', '928', '--gradient-accumulation', '4'], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f'{name} training failed, code {code}; inspect {folder}/console.log')
    rows = [json.loads(line) for line in (folder / 'microbatch_trace.jsonl').read_text().splitlines()]
    old = [json.loads(line) for line in (ROOT / 'work_dirs/gf2_probability_decomposition/train/probability/microbatch_trace.jsonl').read_text().splitlines()]
    assert len(rows) == 20480 and [r['global_iter'] for r in rows] == list(range(20480))
    assert [r['data_fingerprint'] for r in rows] == [r['data_fingerprint'] for r in old]
    assert np.isfinite([r['loss'] for r in rows]).all()
    assert all(sha(ROOT / s) == digest for s, digest in sources.items())
    assert all((folder / f'epoch_{i}.pth').exists() for i in range(1, 6))
    relative = [abs(r['matched_grad_norm'] - r['reference_grad_norm']) / max(r['reference_grad_norm'], 1e-20) for r in rows]
    assert max(relative) < 2e-6
    write(folder / 'complete.json', dict(status='complete', epochs=5, frames=20480,
        optimizer_updates=5120, paired_inputs=True, finite_losses=True,
        max_gradient_norm_relative_error=max(relative), checkpoint_sha256=sha(checkpoint(name))))


def queue():
    initialize()
    for name in ('A1', 'A2', 'A3'):
        train(name)
        subprocess.run([sys.executable, 'tools/eval_gaussian_responsibility.py', '--model', name],
                       cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'tools/eval_gaussian_responsibility.py', '--model', 'A0'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'tools/probe_gaussian_responsibility.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'tools/report_gaussian_responsibility.py'], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('initialize', 'pilot', 'train', 'queue'))
    p.add_argument('--model', choices=('A1', 'A2', 'A3'))
    args = p.parse_args()
    if args.stage == 'initialize':
        initialize()
    elif args.stage == 'queue':
        queue()
    else:
        if args.model is None:
            p.error('--model is required')
        (pilot if args.stage == 'pilot' else train)(args.model)


if __name__ == '__main__':
    main()
