"""Run the authorized PMG experiment sequence, one GPU process at a time.

The frozen reference can be launched separately. This coordinator waits for its
completed marker, validates CUDA contracts, then runs all five arms round-robin
through 15 epochs. Each child retains the full 15-epoch optimizer schedule.
"""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.sparse_opus_gausisan_world.runtime import atomic_json
from .summarize import ARMS, ROOT, refresh


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, PYTHONPATH=".", MPLCONFIGDIR="/tmp/matplotlib-cache",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    refresh()
    while not (ROOT / "reference" / "completed.json").exists():
        reference_log = ROOT / "reference.log"
        if reference_log.exists() and "Traceback (most recent call last)" in reference_log.read_text()[-6000:]:
            atomic_json(ROOT / "queue_status.json", dict(status="reference_failed", log=str(reference_log)))
            raise SystemExit("frozen reference diagnosis failed; inspect reference.log")
        atomic_json(ROOT / "queue_status.json", dict(status="waiting_for_frozen_diagnosis",
                     arms=list(ARMS), planned_epochs=15, pid=os.getpid()))
        time.sleep(15)
    if json.loads((ROOT / "reference" / "completed.json").read_text())["samples"] != 1024:
        raise ValueError("the reference must cover all 1024 validation samples")

    def run(command, log, status):
        atomic_json(ROOT / "queue_status.json", dict(status, pid=os.getpid(), command=command))
        with Path(log).open("a") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode:
            atomic_json(ROOT / "queue_status.json", dict(status="failed", returncode=completed.returncode,
                         command=command, log=str(log), pid=os.getpid()))
            raise SystemExit(completed.returncode)

    audit_path = ROOT / "region_denominator_audit.json"
    if not audit_path.exists() or not json.loads(audit_path.read_text())["full_coverage"]:
        run([sys.executable, "-m", "tools.pmg_world.audit_regions"], ROOT / "region_audit.log",
            dict(status="auditing_region_denominators"))
    if not json.loads(audit_path.read_text()).get("phenomenon_confirmed", False):
        raise SystemExit("frozen Gaussian diagnosis has not confirmed the fast-region effect")
    if not (ROOT / "reference_geometry.json").exists():
        run([sys.executable, "-m", "tools.pmg_world.diagnose_geometry"], ROOT / "geometry.log",
            dict(status="diagnosing_gaussian_structure"))
    validation = ROOT / "validation.json"
    if not validation.exists() or json.loads(validation.read_text())["status"] != "passed":
        run([sys.executable, "-m", "tools.pmg_world.validate"], ROOT / "validation.log",
            dict(status="validating_actual_gpu_contracts"))
    if json.loads(validation.read_text())["status"] != "passed":
        raise ValueError("GPU validation failed")
    for epoch in range(1, 16):
        for arm in ARMS:
            marker = ROOT / arm / f"epoch_{epoch:02d}_metrics.json"
            completed_path = Path(f"out/pmg_world_stage1/{arm}/completed.json")
            if marker.exists() and (epoch < 15 or completed_path.exists()):
                continue
            command = [sys.executable, "-m", "tools.pmg_world.train", "--config",
                       f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py",
                       "--work-dir", f"out/pmg_world_stage1/{arm}",
                       "--stop-after-epoch", str(epoch)]
            run(command, ROOT / f"{arm}_train.log", dict(status="training", arm=arm, epoch=epoch,
                                                       planned_epochs=15))
            if not marker.exists():
                raise ValueError("child exited without completing formal validation")
            refresh()
    refresh()
    atomic_json(ROOT / "queue_status.json", dict(status="completed", arms=list(ARMS), epochs=15))


if __name__ == "__main__":
    main()
