"""Launch only the selected and validated stage2 experiment arms."""

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.sparse_opus_gausisan_world.runtime import atomic_json


def main():
    root = Path("analysis/pmg_world_stage2")
    validation = json.loads((root / "validation.json").read_text())
    if validation["status"] != "passed":
        raise RuntimeError("real batch-two validation must pass before launch")
    decision = json.loads((root / "execution_decision.json").read_text())
    if len(decision["training_arms"]) != 2:
        raise ValueError("the first batch must contain exactly the two selected arms")
    marker = root / "launched.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("the previous stage2 queue process is still alive")
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.pmg_world_stage2.run",
        "--arms",
        *decision["training_arms"],
    ]
    environment = dict(
        os.environ,
        PYTHONPATH=".",
        MPLCONFIGDIR="/tmp/matplotlib-cache",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    with (root / "queue.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
    result = dict(
        status="launched",
        pid=process.pid,
        process_group=process.pid,
        command=command,
        arms=decision["training_arms"],
        planned_epochs=10,
        schedule_epochs=15,
        log=str(root / "queue.log"),
    )
    atomic_json(marker, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
