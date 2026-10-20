"""From-start factorial verification of the empty-amplitude mechanism."""
from pathlib import Path
from datetime import datetime
import argparse
import copy
import hashlib
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage3 import read,write_json,file_sha256

RUN=ROOT/'work_dirs/empty_mechanism_history'
MODELS=['B0','S','E','SE']


def initialize():
    RUN.mkdir(exist_ok=True)
    path=RUN/'protocol.json'
    if path.exists():return read(path)
    from mmengine import Config
    cfg=Config.fromfile('config/center_support_contribution1/a0_original.py')
    pretrain=ROOT/cfg.get('train_load_from',cfg.load_from)
    p=dict(at=datetime.now().astimezone().isoformat(),registered_before_from_start_runs=True,
        motivation='A final-epoch intervention cannot identify causes acquired during earlier epochs; repeat the same 2x2 paths from the common training initialization.',
        config_base='config/center_support_contribution1/a0_original.py',original_training_seed=928,
        checkpoint=dict(path=str(pretrain.relative_to(ROOT)),sha256=file_sha256(pretrain)),
        conditions=MODELS,train_frames=4096,validation_frames=1024,max_epochs=5,
        training_epoch_count=5,total_optimizer_steps=5120,total_frame_visits_per_model=20480,
        stage_scope='Only occupancy-rendered shared semantic logits for Set-negative and CE-empty; currently final stage5. Earlier Set classification gradients remain unchanged.',
        intervention='Identical source-gradient operation to stage3; native train.py initialization, data loader, AdamW, scheduling, clipping, original weights and AMP=false.',
        primary='Net correct rescue (gross correct rescue minus newly empty originally-correct B0 foreground), divided by the fixed endpoint B0 recoverable-miss cohort; native and matched validation false-positive count',
        dominance_rule='Lower95 net rescue >50%; native and matched-FPR quantity decisions separately; class ranking and calibrated mIoU reported separately as method utility, not causal quantity gates',
        evaluation='All1024 validation frames; same698 training scene representatives for class calibration; source-stage3 evaluator reused',
        preexisting_results_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in [
            ROOT/'work_dirs/empty_mechanism_stage2/summary.json',ROOT/'work_dirs/empty_mechanism_metric_effects/summary.json']},
        source_sha256=file_sha256(__file__))
    p['scene_samples']=read(ROOT/'work_dirs/empty_mechanism_stage3/protocol.json')['scene_samples']
    write_json(path,p)
    configdir=ROOT/'config/empty_mechanism_history';configdir.mkdir(exist_ok=True)
    for name in MODELS:
        trace=f'work_dirs/empty_mechanism_history/train/{name}'
        config=f'''"""Five-epoch from-start source-path intervention: {name}."""
_base_ = ['../center_support_contribution1/a0_original.py']
custom_imports = dict(imports=['loss.empty_history_path_loss'], allow_failed_imports=False)
loss = dict(_delete_=True, type='EmptyHistoryPathLoss',condition='{name}',trace_dir='{trace}')
loss_input_convertion = dict(gaussians='gaussians',supervised_stage_indices='supervised_stage_indices')
'''
        (configdir/f'{name}.py').write_text(config)
    return p


def train(name):
    p=initialize();folder=RUN/'train'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    if (folder/'microbatch_trace.jsonl').exists():
        raise RuntimeError('Partial native trajectory exists; preserve and inspect it before an explicit restart. Native checkpoints do not contain full RNG state.')
    source_paths=['loss/empty_history_path_loss.py','tools/empty_mechanism_stage3.py',
                  'tools/empty_mechanism_history.py','train.py']
    sources={path:file_sha256(ROOT/path) for path in source_paths}
    write_json(folder/'training_sources.json',sources)
    # Native runner handles initialization and full epoch validation. No reuse of
    # the historical B0 is claimed as a paired from-start baseline.
    with (folder/'console.log').open('a') as log:
        result=subprocess.run([sys.executable,'tools/train_center_support.py',
            '--py-config',f'config/empty_mechanism_history/{name}.py',
            '--work-dir',str(folder.relative_to(ROOT)),'--seed','928','--gradient-accumulation','4'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} native full training failed: {result.returncode}')
    rows=[json.loads(line) for line in (folder/'microbatch_trace.jsonl').read_text().splitlines()]
    assert len(rows)==20480 and [r['global_iter'] for r in rows]==list(range(20480))
    assert all(file_sha256(ROOT/path)==sha for path,sha in sources.items())
    assert all(row['rendered_stages']==[5] for row in rows)
    assert all(len(row['data_fingerprint'])==64 for row in rows)
    trace=[dict(step=i+1,loss=sum(r['loss'] for r in rows[i*4:i*4+4])/4,sources={}) for i in range(5120)]
    for i,t in enumerate(trace):
        for source in ['S','E']:
            r=[a['sources'][source] for a in rows[i*4:i*4+4] if source in a['sources']]
            if r:t['sources'][source]={k:sum(a[k] for a in r)/len(r) for k in r[0]}
    write_json(folder/'complete.json',dict(status='complete',condition=name,steps=5120,frames=20480,
        epochs=5,trace=trace,initial_raw_sha256=rows[0]['initial_raw_sha256'],
        checkpoint_sha256=file_sha256(folder/'epoch_5.pth'),
        protocol_sha256=file_sha256(RUN/'protocol.json'),source_sha256=file_sha256(__file__),
        elapsed_seconds=read(folder/'training_runtime.json')['training_and_validation_seconds']))


def prepare_evaluation():
    p=initialize()
    path=RUN/'evaluation_protocol.json'
    if path.exists():return read(path)
    from tools.eval_empty_mechanism_stage3 import evaluation_protocol
    prior=copy.deepcopy(evaluation_protocol())
    prior.update(models=MODELS,endpoint_kind='full_history',at=datetime.now().astimezone().isoformat())
    prior['origin']='Five complete epochs from common pretraining and seed928 initialization'
    if not path.exists():write_json(path,prior)
    return read(path)


def setup_evaluator():
    import tools.eval_empty_mechanism_stage3 as e
    e.RUN=RUN;e.MODELS=MODELS;e.initialize=initialize;e.evaluation_protocol=prepare_evaluation
    e.checkpoint_path=lambda name:RUN/'train'/name/'epoch_5.pth'
    return e


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['initialize','train','extract','statistics'])
    parser.add_argument('--condition',choices=MODELS,default='B0');a=parser.parse_args()
    if a.stage=='initialize':initialize();prepare_evaluation();print(RUN/'protocol.json')
    elif a.stage=='train':train(a.condition)
    elif a.stage=='extract':setup_evaluator().extract(a.condition)
    else:setup_evaluator().make_statistics(a.condition)


if __name__=='__main__':main()
