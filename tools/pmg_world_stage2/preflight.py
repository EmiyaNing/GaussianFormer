"""Run finite GPU gate diagnostics then actual batch-two implementation checks."""

import os
from pathlib import Path
import subprocess
import sys

from tools.sparse_opus_gausisan_world.runtime import atomic_json


def main():
    root = Path("analysis/pmg_world_stage2")
    tasks = [
        (
            "gate_interventions",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.pmg_world_stage2.gates",
                "--samples",
                "64",
            ],
        ),
        (
            "batch_two_validation",
            [sys.executable, "-u", "-m", "tools.pmg_world_stage2.validate"],
        ),
    ]
    for name, command in tasks:
        atomic_json(
            root / "preflight_status.json",
            dict(status="running", stage=name, pid=os.getpid(), command=command),
        )
        with (root / (name + ".log")).open("a") as log:
            code = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if code:
            atomic_json(
                root / "preflight_status.json",
                dict(status="failed", stage=name, returncode=code),
            )
            raise SystemExit(code)
    atomic_json(root / "preflight_status.json", dict(status="passed", pid=None))


if __name__ == "__main__":
    main()
