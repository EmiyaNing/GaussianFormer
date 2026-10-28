"""Run the two predeclared frozen-state adaptation arms sequentially."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path("analysis/pmg_world_stage3_20261023")
    gate = json.loads(Path("out/pmg_world_stage3_20261023/frozen_history_validation/validation.json").read_text())
    if gate["status"] != "passed":
        raise ValueError("historical-image correction forward/backward validation failed")
    # A completed process releases CUDA caches; a completion file alone can
    # appear just before process exit. Keep a short additional release window.
    deadline = time.monotonic() + 7200
    while not (root / "diagnosis_1024" / "completed.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("1024-sample diagnosis did not complete")
        (root / "adaptation_queue_status.json").write_text(json.dumps(dict(status="waiting_for_diagnosis"),indent=2)+"\n")
        time.sleep(5)
    time.sleep(5)
    environment = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for arm in ("current", "history"):
        work = Path(f"out/pmg_world_stage3_20261023/frozen_M_{arm}")
        command = [sys.executable, "-u", "-m", "tools.pmg_world_stage3.train_frozen",
                   "--config", f"config/sparse_opus_gaussian_world/pmg_stage3_frozen_M_{arm}_512updates.py",
                   "--work-dir", str(work)]
        status = dict(status="training", arm=arm, command=command)
        (root / "adaptation_queue_status.json").write_text(json.dumps(status, indent=2)+"\n")
        with (root / f"frozen_M_{arm}.log").open("a") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
        if code:
            (root / "adaptation_queue_status.json").write_text(json.dumps(dict(status="failed",arm=arm,returncode=code),indent=2)+"\n")
            raise SystemExit(code)
        subprocess.run([sys.executable, "-m", "tools.pmg_world_stage3.summarize_experiments"], env=environment, check=True)
    config = "config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_A0_full_dense_mb1_15e.py"
    directory = Path("out/sparse_opus_gaussian_world_A0_full_dense_mb1_15e")
    command = [sys.executable, "-u", "-m", "tools.pmg_world_full.train", "--config", config,
               "--work-dir", str(directory), "--stop-after-epoch", "3", "--max-hours", "5"]
    old = directory / "launch.json"
    if old.exists():
        (directory / "launch_before_serial_restart.json").write_text(old.read_text())
    with (directory / "train.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment, start_new_session=True)
    old.write_text(json.dumps(dict(pid=process.pid, command=command, mode="isolated_after_R1_R2"),indent=2)+"\n")
    (root / "adaptation_queue_status.json").write_text(json.dumps(dict(status="completed",arms=["current","history"],baseline_pid=process.pid),indent=2)+"\n")
    subprocess.run([sys.executable, "-m", "tools.pmg_world_stage3.summarize_experiments"], env=environment, check=True)


if __name__ == "__main__":
    main()
