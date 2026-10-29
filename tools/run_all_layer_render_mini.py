"""Register and run the paired 4,090-frame Small experiment and full validation."""
from datetime import datetime
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.all_layer_render_experiment import *


def initialize():
    selection = prepare_annotation()
    baseline, treatment = config('baseline'), config('all_layers')
    assert baseline.expected_train_frames == treatment.expected_train_frames == 4090
    assert baseline.expected_val_frames == treatment.expected_val_frames == 6019
    assert treatment.model.encoder.attribute_stages == list(range(6))
    assert treatment.model.head.apply_loss_type == 'all'
    assert baseline.loss.loss_cfgs[0] == treatment.loss.loss_cfgs[0]
    occ = dict(treatment.loss.loss_cfgs[1])
    assert occ.pop('stage_weights') == [1.] * 6
    occ['type'] = 'OccupancyLoss'
    assert occ == baseline.loss.loss_cfgs[1]
    sources = source_files()
    protocol = dict(version=1, model='small', initial=str(INITIAL.relative_to(ROOT)),
        initial_sha256=sha(INITIAL), train_frames=4090, train_scenes=698, validation_frames=6019,
        validation_scenes=150, epochs=1, accumulation=2, optimizer_updates=2045, seed=928,
        arms=dict(baseline='Original learned final GS attributes, six Set losses, final occupancy only',
            all_layers='Six learned GS attribute heads, six Set losses, six full-weight occupancy losses'),
        initialization='Load every pretrained parameter exactly; only new attribute branches 0..4 use original branch constructor initialization. Fresh matched AdamW for each arm.',
        training_selection=selection,
        loss='sum_l [original OPUSSetLoss_l + original occupancy_l]; Set stage weights=1, occupancy stage weights=1; CE=10, Lovasz=1; no division by six',
        gt_domain='Original unmodified Set domain; camera-mask occupancy supervision and evaluation',
        optimizer=dict(type='AdamW', lr=2e-5, weight_decay=.01, accumulation=2,
            warmup_updates=50, cosine_min_ratio=.1, gradient_clip=35,
            multipliers=dict(backbone=.1, sampling_offset=.1, empty_bias=25)),
        pairing='Same selected 4090 frames, order and per-index augmentation seeds; exact transformed-image/projection/GT SHA-256 check for every training frame',
        evaluation='Every 6019 full-validation frame, native camera domain; all six renders, per-layer distance/GS core correctness, and fixed initial final-load quintiles',
        annotations={s: sha(baseline[s+'_dataset_config'].imageset) for s in ('train', 'val')},
        resolved_config_sha256={arm: hashlib.sha256(config(arm).pretty_text.encode()).hexdigest() for arm in ARMS},
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources})
    path = RUN / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen inputs changed; preserve this run and use a new directory'
    else:
        write(path, protocol)
        for arm in ARMS:
            config(arm).dump(RUN / f'{arm}_resolved_config.py')
        for p in sources:
            destination = RUN / 'sources' / p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, destination)
    return protocol


def call(script, arm, phase):
    assert_protocol()
    folder = RUN / phase / arm
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'console.log').open('a', buffering=1) as log:
        result = subprocess.run([sys.executable, '-u', str(ROOT / 'tools' / script), arm],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{phase}/{arm} failed: inspect {folder}/console.log')


def run():
    initialize()
    pilot_record = RUN / 'pilot.json'
    if not pilot_record.exists():
        with (RUN / 'pilot_console.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'tools/pilot_all_layer_render_mini.py')],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert json.loads(pilot_record.read_text())['status'] == 'passed'
    for arm in ARMS:
        call('train_all_layer_render_mini.py', arm, 'train')
        call('eval_all_layer_render_mini.py', arm, 'eval')
    call('eval_all_layer_render_mini.py', 'initial', 'eval')
    from tools.report_all_layer_render_mini import main as report
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('register', 'run', 'report'), default='run')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        with (RUN / 'experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'register':
                initialize()
            elif args.action == 'report':
                from tools.report_all_layer_render_mini import main as report
                report()
            else:
                run()
    except Exception as error:
        write(RUN / 'error.json', dict(status='failed', type=type(error).__name__, message=str(error),
            updated_at=datetime.now().astimezone().isoformat()))
        write(RUN / 'status.json', dict(status='failed', message=str(error)))
        raise
