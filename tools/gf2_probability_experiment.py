"""Native five-epoch GF2-rendering replacement on the existing mini split."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'work_dirs/gf2_probability_decomposition'
CONFIG='config/gf2_probability_decomposition/probability.py'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def initialize():
    RUN.mkdir(exist_ok=True)
    p=RUN/'protocol.json'
    if p.exists():return read(p)
    from mmengine import Config
    c=Config.fromfile('config/center_support_contribution1/a0_original.py')
    prior=read(ROOT/'work_dirs/empty_decision_optimization/protocol.json')
    protocol=dict(at=datetime.now().astimezone().isoformat(),registered_before_formal_probability_training=True,
        train_frames=4096,validation_frames=1024,epochs=5,optimizer_updates=5120,train_frame_visits=20480,
        seed=928,accumulation=4,baseline='work_dirs/empty_mechanism_history/train/B0/epoch_5.pth',
        baseline_complete='work_dirs/empty_mechanism_history/train/B0/complete.json',
        probability_config=CONFIG,pretrain=dict(path=c.load_from,sha256=sha(ROOT/c.load_from)),
        annotation_sha256={s:sha(c[s+'_dataset_config'].imageset) for s in ['train','val']},
        fit_scenes=prior['fit_scenes'],tune_scenes=prior['tune_scenes'],
        fit_protocol='Same523fit/175tune train scenes, one deterministic frame per scene; no validation-label parameter selection',
        changes=['Unmodified official GF2 probability kernel: noisy-OR geometry without opacity, density/opacity-weighted normalized semantic mixture',
            'Raw OPUS logits through softmax for Gaussian semantics; no global empty Gaussian or learned global empty-logit bias',
            'Occupancy CE uses probability NLL and Lovasz consumes probabilities; Set objective and original weights unchanged'],
        baseline_caveat='Baseline has its original learned global empty bias; report native differences and equal train-only global/class calibration opportunities separately',
        decision=dict(ratios=[0,.25,.5,.75,1],temperatures=[.5,.75,1,1.5,2],
            joint_ratio=1,hierarchical_ratio=0,semantic_risk='C_FP=C_FN=1; C_class=ratio; class intercepts fit on training only',
            fitting_threshold_grid=[-256,256,1/32],matched_fpr='Descriptive validation-budget matching; primary paired scene bootstrap reselects threshold'),
        evaluation='All1024validation frames, default camera mask; five fixed-occupancy temperatures per model, natural decision-gap counts, risk-family fit/tune/evaluation',
        uncertainty='1000 paired bootstrap resamples of validation scenes; one training seed, no training-seed confidence claim',
        method_claim='Controlled OPUS rendering replacement, not a reproduction of the whole GF2 architecture or pretrained checkpoints',
        scheduling='User cancelled remaining old full-history interventions; run probability experiment immediately using completed B0 as baseline',
        old_results_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'work_dirs/empty_mechanism_stage2/summary.json',ROOT/'work_dirs/empty_mechanism_stage3/summary.json']})
    write(p,protocol)
    return protocol


def train():
    initialize();folder=RUN/'train/probability';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    if (folder/'microbatch_trace.jsonl').exists():
        raise RuntimeError('Partial native trajectory exists; retain it and inspect before restarting; full RNG is not checkpointed')
    sources={name:sha(ROOT/name) for name in ['model/head/opus_gf2_probability_head.py',
        'model/head/gf2_probability_backend.py','loss/gf2_probability_experiment_loss.py',
        'tools/gf2_probability_experiment.py',CONFIG,'train.py']}
    write(folder/'training_sources.json',sources)
    with (folder/'console.log').open('a') as log:
        code=subprocess.call([sys.executable,'tools/train_center_support.py','--py-config',CONFIG,
            '--work-dir',str(folder.relative_to(ROOT)),'--seed','928','--gradient-accumulation','4'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'Native probability training failed: {code}')
    rows=[json.loads(x) for x in (folder/'microbatch_trace.jsonl').read_text().splitlines()]
    assert len(rows)==20480 and [x['global_iter'] for x in rows]==list(range(20480))
    assert all(sha(ROOT/name)==value for name,value in sources.items())
    baseline=ROOT/'work_dirs/empty_mechanism_history/train/B0/microbatch_trace.jsonl'
    reference=[json.loads(x) for x in baseline.read_text().splitlines()]
    assert [x['data_fingerprint'] for x in rows]==[x['data_fingerprint'] for x in reference]
    import numpy as np
    assert np.isfinite([x['loss'] for x in rows]).all()
    assert read(folder/'training_runtime.json')['status']=='complete'
    assert all((folder/f'epoch_{epoch}.pth').exists() for epoch in range(1,6))
    a=np.load(folder/'initial_forward.npz');b=np.load(baseline.parent/'initial_forward.npz')
    np.testing.assert_array_equal(a['labels'],b['labels'])
    np.testing.assert_allclose(a['raw'],b['raw'],rtol=1e-5,atol=1e-6)
    write(folder/'complete.json',dict(status='complete',epochs=5,updates=5120,frames=20480,
        paired_inputs=True,finite_losses=True,native_nonfinite_gradient_guard=True,
        initial_forward_max_error=float(abs(a['raw']-b['raw']).max()),
        checkpoint_sha256=sha(folder/'epoch_5.pth'),sources=sources))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['initialize','build','train'])
    a=p.parse_args()
    if a.stage=='initialize':print(initialize())
    elif a.stage=='build':
        from model.head.gf2_probability_backend import probability_backend
        print(probability_backend())
    else:train()


if __name__=='__main__':main()
