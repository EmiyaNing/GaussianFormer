"""Integrity checks for the third-stage training, endpoints and attribution."""
from pathlib import Path
import sys
from datetime import datetime
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage3 import RUN,read,write_json,file_sha256
from tools.eval_empty_mechanism_stage3 import MODELS


def main():
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    p=read(RUN/'protocol.json');s=read(RUN/'summary.json')
    check('summary_complete',s['status']=='complete')
    baseline=None;sequence=None
    for name in ['B0','S','E','SE']:
        folder=RUN/'train'/name;r=read(folder/'complete.json')
        check(name+'_full_final_epoch',r['steps']==1024 and r['frames']==4096 and len(r['trace'])==1024)
        check(name+'_all_training_frames_once',sorted(r['sequence'])==list(range(4096)))
        if sequence is None:sequence=r['sequence']
        check(name+'_paired_sequence',sequence==r['sequence'])
        check(name+'_finite_gradient_and_loss',all(np.isfinite(t['loss']) and np.isfinite(t['grad_norm']) for t in r['trace']))
        counts=np.asarray([v['counts'] for v in read(folder/'baseline.json')['records']])
        if baseline is None:baseline=counts
        check(name+'_same_origin_predictions',np.array_equal(counts,baseline))
        check(name+'_registered_source',r['source_sha256']==read(RUN/'execution_addendum.json')['formal_code_sha256'])
        for source in ['S','E']:
            if source in name:
                audit=s['training_audit'][name]['sources'][source]
                check(name+'_'+source+'_path_suppression',abs(audit['removed_derivative_fraction']-1)<1e-8 and audit['steps']==1024)
    meta=read(RUN/'evaluation/B0/validation_frames.json')
    for name in MODELS:
        folder=RUN/'evaluation'/name;extract=read(folder/'extraction_complete.json');r=read(folder/'statistics.json')
        check(name+'_all_validation_frames',extract['validation_frames']==1024 and len(r['records'])==1024)
        check(name+'_same_frame_order',read(folder/'validation_frames.json')==meta)
        check(name+'_calibration_boundary_clear',not any(read(folder/'selection.json')['boundary']))
        check(name+'_matched_fp_budget',r['achieved_fp']<=r['target_fp'])
        for row in r['cohort_records']:
            for c in row['modes'].values():
                check_sum=c['rescued']+c['remaining_empty']+c['wrong_foreground']
                if check_sum!=row['cohort']:raise AssertionError((name,row['index'],'cohort partition'))
        check(name+'_cohort_partition',True)
    for path,sha in p['preexisting_results_sha256'].items():check('immutable_'+path,file_sha256(ROOT/path)==sha)
    check('origin_checkpoint_unchanged',file_sha256(ROOT/p['checkpoint']['path'])==p['checkpoint']['sha256'])
    med=read(RUN/'mediation/summary.json')
    check('factor_swaps_complete',med['status']=='complete' and med['frames']==32 and med['factor_combinations']==16)
    check('factor_renderer_parity',max(x['max_margin_error'] for row in med['parity'] for x in row)<=5e-4)
    original=np.asarray(s['models']['original5']['modes']['native']['counts'])
    historical=np.asarray(read(ROOT/'work_dirs/center_support_contribution1/train/a0_original_seed928/evaluation_epoch5.json')['camera_confusion'])
    check('archived_original_evaluation_parity',np.array_equal(original,historical))
    write_json(RUN/'completion_audit.json',dict(status='passed',checks=checks,check_count=len(checks),
        train_conditions=4,training_frame_visits=16384,optimizer_updates=4096,
        evaluated_models=len(MODELS),validation_frame_visits=1024*len(MODELS),
        primary_scene_bootstrap=1000,exploratory_factor_swaps=32*16,
        source_sha256=file_sha256(__file__),summary_sha256=file_sha256(RUN/'summary.json')))
    design=ROOT/'analysis/empty_mechanism_stage3_design.md'
    text=design.read_text().replace('状态：实验执行中。','状态：**完整末期训练、验证和统计已完成，完整性审计通过。**')
    design.write_text(text)
    for name in ['design','results']:
        with (RUN/f'document_{name}.log').open('w') as log:
            result=subprocess.run(['node','work_dirs/empty_mechanism_stage2/document_review/validate.js',
                f'analysis/empty_mechanism_stage3_{name}.md'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Document validation failed: {name}')
    write_json(RUN/'closeout.json',dict(status='complete',at=datetime.now().astimezone().isoformat(),
        required_work_remaining=[],scope='Final-epoch intervention; full-history study recorded separately',
        audit_sha256=file_sha256(RUN/'completion_audit.json'),summary_sha256=file_sha256(RUN/'summary.json'),
        documents=['analysis/empty_mechanism_stage3_design.md','analysis/empty_mechanism_stage3_results.md']))
    print(f'{len(checks)} integrity checks passed.')


if __name__=='__main__':main()
