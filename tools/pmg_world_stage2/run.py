"""Persistent round-robin queue, using only explicitly selected stage2 arms."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .report import ARMS, ROOT, diagnosis_report, overview
from tools.sparse_opus_gausisan_world.runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    args = parser.parse_args()
    lock = (ROOT / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validation = json.loads((ROOT / "validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError("actual batch-two validation has not passed")
    decision = json.loads((ROOT / "execution_decision.json").read_text())
    if decision["training_arms"] != args.arms:
        raise RuntimeError("selected arms differ from recorded decision")
    stop = False
    child = None

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    environment = dict(
        os.environ,
        PYTHONPATH=".",
        MPLCONFIGDIR="/tmp/matplotlib-cache",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    diagnosis_report()
    overview()
    for epoch in range(1, 11):
        for arm in args.arms:
            if stop:
                break
            marker = ROOT / arm / f"epoch_{epoch:02d}_metrics.json"
            completion = Path(f"out/pmg_world_stage2/{arm}/completed.json")
            if marker.exists() and (epoch < 10 or completion.exists()):
                continue
            command = [
                sys.executable,
                "-u",
                "-m",
                "tools.pmg_world_stage2.train",
                "--config",
                f"config/sparse_opus_gaussian_world/pmg_stage2_{arm}_4096_1024_10e.py",
                "--work-dir",
                f"out/pmg_world_stage2/{arm}",
                "--stop-after-epoch",
                str(epoch),
            ]
            log = ROOT / f"{arm}_train.log"
            atomic_json(
                ROOT / "queue_status.json",
                dict(
                    status="training",
                    arm=arm,
                    epoch=epoch,
                    planned_epochs=10,
                    schedule_epochs=15,
                    pid=os.getpid(),
                    command=command,
                    log=str(log),
                ),
            )
            with log.open("a") as handle:
                child = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=environment,
                )
                code = child.wait()
            if stop:
                break
            if code or not marker.exists():
                atomic_json(
                    ROOT / "queue_status.json",
                    dict(
                        status="failed",
                        arm=arm,
                        epoch=epoch,
                        returncode=code,
                        log=str(log),
                        pid=os.getpid(),
                    ),
                )
                raise SystemExit(code or 1)
            overview()
        if stop:
            break
    if stop:
        atomic_json(
            ROOT / "queue_status.json",
            dict(status="stopped_by_request", pid=None, arms=args.arms),
        )
    else:
        from .statistics import final_statistics

        final_statistics()
        atomic_json(
            ROOT / "queue_status.json",
            dict(status="completed", pid=None, arms=args.arms, epochs=10),
        )


if __name__ == "__main__":
    main()
