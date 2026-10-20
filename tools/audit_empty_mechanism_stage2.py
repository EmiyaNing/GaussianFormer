"""Verify every registered condition, fixed cohort, source and original artifact."""
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage2 import RUN,read,write_json,file_sha256,CONDITIONS


def main():
    p=read(RUN/"protocol.json")
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
    check("queue_complete",read(RUN/"queue_status.json")["status"]=="complete")
    check("registered_plan_snapshot_matches_protocol",
          file_sha256(RUN/"sources_executed"/"registered_plan.md")==p["plan_sha256"])
    protocol_hash=file_sha256(RUN/"protocol.json")
    local=[]
    for e in [1,3,5]:
        for split,count in [("train",16),("val",32)]:
            files=sorted((RUN/"local"/f"epoch{e}").glob(split+"_*.json"))
            records=[read(f) for f in files];local.extend(records)
            check(f"local_e{e}_{split}_count",len(records)==count)
            for r in records:
                assert r["source_sha256"]==file_sha256(ROOT/"tools/empty_mechanism_stage2.py")
                assert r["protocol_sha256"]==protocol_hash
                assert len(r["interventions"])==34 and r["renderer_error"]<=2e-4
                wanted={(c,radius) for c in CONDITIONS for radius in [1e-4,1e-3]}
                actual={(v["condition"],v["radius"]) for v in r["interventions"]}
                assert wanted==actual
                for radius in [1e-4,1e-3]:
                    rows=[v for v in r["interventions"] if v["radius"]==radius]
                    assert len({v["eta"] for v in rows})==1
                    zero=next(v for v in rows if v["condition"]=="combo_000")
                    assert zero["outcome"]["counts"]==r["baseline"]["counts"]
                for row in r["interventions"]:
                    a=np.asarray(row["outcome"]["counts"]);b=np.asarray(r["baseline"]["counts"])
                    assert np.array_equal(a.sum(1),b.sum(1))
                    assert a[:17,17].sum()==row["outcome"]["empty_fn"]
                    assert a[17,:17].sum()==row["outcome"]["empty_fp"]
                    for cohort in r["baseline"]["cohorts"]:
                        assert row["outcome"]["cohorts"][cohort]["empty_sum"]==r["baseline"]["cohorts"][cohort]["empty_sum"]
                assert r["audit"]["lovasz_empty_voxel_gradient"]==0
                assert r["projection_audit"]["projection_orthogonality_error"]<1e-12
    check("all_local_conditions_invariants_and_provenance",True)
    adamw=[]
    for e in [1,3,5]:
        for mode in ["classifier","full"]:
            files=sorted((RUN/"adamw"/f"epoch{e}"/mode).glob("pair_*.json"))
            records=[read(f) for f in files];adamw.extend(records)
            check(f"adamw_e{e}_{mode}_pairs",len(records)==8)
            for r in records:
                assert r["source_sha256"]==file_sha256(ROOT/"tools/train_empty_mechanism_stage2.py")
                assert r["protocol_sha256"]==protocol_hash
                assert len(r["interventions"])==9 and r["optimizer_state_restored"]
                assert r["pair"]["source"]["scene"]!=r["pair"]["heldout"]["scene"]
                for row in r["interventions"]:
                    for role in ["source","heldout"]:
                        o=row["outcomes"][role];b=r["baselines"][role]
                        assert np.array_equal(np.asarray(o["counts"]).sum(1),np.asarray(b["counts"]).sum(1))
                        for cohort in b["cohorts"]:
                            assert o["cohorts"][cohort]["n"]==b["cohorts"][cohort]["n"]
                        if mode=="classifier":
                            assert max(row["frozen_errors"][role].values())==0
    check("adamw_source_and_fixed_geometry_controls",True)
    addendum=read(RUN/"execution_addendum.json")
    registered=datetime.fromisoformat(addendum["method_screening_clarification"]["at"]).timestamp()
    check("method_screening_registered_before_candidate_endpoints",
          all((RUN/"short"/f"epoch{e}"/"project_set"/"step_64.json").stat().st_mtime>registered
              for e in [1,3]))
    shorts=[]
    for e in [1,3]:
        sequences=[]
        for condition in addendum["short_conditions"]:
            folder=RUN/"short"/f"epoch{e}"/condition
            done=read(folder/"complete.json")
            assert done["status"]=="complete" and done["endpoints"]==[1,16,64]
            assert done["source_sha256"]==file_sha256(ROOT/"tools/train_empty_mechanism_stage2.py")
            sequences.append(done["sequence"])
            before=read(folder/"baseline.json")["records"]
            assert len(before)==32
            for step in [1,16,64]:
                after=read(folder/f"step_{step:02d}.json")
                assert after["source_sha256"]==done["source_sha256"]
                assert len(after["records"])==32 and len(after["trace"])==step
                for b,a in zip(before,after["records"]):
                    assert b["frame"]==a["frame"]
                    for cohort in b["outcome"]["cohorts"]:
                        assert b["outcome"]["cohorts"][cohort]["n"]==a["outcome"]["cohorts"][cohort]["n"]
            shorts.append(done)
        assert all(x==sequences[0] for x in sequences)
    check("all_18_short_branches_complete_and_paired",len(shorts)==18)
    pilots={}
    for mode in ["classifier","full"]:
        pilot=read(RUN/"adamw_pilot"/"epoch5"/mode/"pair_00.json")["parity"]
        assert pilot["cached_vs_original_relative_gradient_error"]<max(
            5*pilot["original_replay_relative_gradient_noise"],5e-5)
        pilots[mode]=pilot
    check("native_multiloss_backward_parity",True)
    check("necessary_numerical_tests_passed","3 passed" in (RUN/"tests.log").read_text())
    check("candidate_loss_gradient_isolation_tests_passed","2 passed" in (RUN/"candidate_tests.log").read_text())
    check("equal_bias_counting_and_optimizer_tests_passed","2 passed" in (RUN/"calibration_tests.log").read_text())
    check("equal_bias_queue_complete",read(RUN/"equal_bias"/"queue_status.json")["status"]=="complete")
    calibrations=[]
    for e in [1,3]:
        for condition in ["B0","project_set"]:
            folder=RUN/"equal_bias"/f"epoch{e}_{condition}"
            done=read(folder/"complete.json")
            assert done["status"]=="complete" and done["train_frames"]==4096 and done["validation_frames"]==1024
            assert done["source_sha256"]==file_sha256(ROOT/"tools/calibrate_empty_mechanism_stage2.py")
            assert done["checkpoint_sha256"]==file_sha256(RUN/"short"/f"epoch{e}"/condition/"final.pth")
            selection=read(folder/"selection.json")
            assert len(selection["thresholds"])==17 and not any(selection["boundary"])
            assert len(read(folder/"validation_frames.json"))==1024
            calibrations.append(done)
    check("equal_bias_all_four_models_and_selection_provenance",len(calibrations)==4)
    for e,item in p["checkpoint"].items():
        check("original_checkpoint_epoch"+e+"_immutable",file_sha256(ROOT/item["path"])==item["sha256"])
    check("original_stage1_summary_immutable",file_sha256(ROOT/"work_dirs/empty_mechanism_stage1/summary.json")==p["original_summary_sha256"])
    record=dict(status="passed" if all(v["passed"] for v in checks) else "failed",
        checks=checks,local_frames=len(local),output_interventions=sum(len(r["interventions"]) for r in local),
        adamw_pairs=len(adamw),adamw_optimizer_updates=sum(len(r["interventions"]) for r in adamw),
        short_branches=len(shorts),short_optimizer_updates=sum(max(r["endpoints"]) for r in shorts),
        short_evaluation_scene_frames=len(shorts)*3*32,parity=pilots,
        calibration_models=len(calibrations),
        calibration_training_frame_forwards=sum(r["train_frames"] for r in calibrations),
        calibration_validation_frame_forwards=sum(r["validation_frames"] for r in calibrations),
        max_decomposition_joint_gradient_error=max(r["audit"]["joint_gradient_error"] for r in local),
        max_ce_value_error=max(r["audit"]["ce"]["value_error"] for r in local),
        max_set_value_error=max(r["audit"]["set"]["value_error"] for r in local))
    write_json(RUN/"completion_audit.json",record)
    assert record["status"]=="passed"
    print(json.dumps(record,indent=2))


if __name__=="__main__":main()
