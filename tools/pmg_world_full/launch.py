"""Launch the validated full-split baseline as a resumable background process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_A0_full_dense_15e.py")
    parser.add_argument("--work-dir", default="out/sparse_opus_gaussian_world_A0_full_dense_15e")
    parser.add_argument("--smoke-dir", default="out/sparse_opus_gaussian_world_A0_full_dense_15e_smoke")
    parser.add_argument("--stop-after-epoch", type=int, default=3)
    parser.add_argument("--max-hours", type=float, default=5)
    args = parser.parse_args()
    validation = json.loads((Path(args.smoke_dir) / "smoke.json").read_text())
    if not validation["passed"] or validation["active_horizons"] != list(range(7)):
        raise RuntimeError("seven-horizon GPU smoke validation must pass first")
    if validation.get("config_sha256") != hashlib.sha256(Path(args.config).read_bytes()).hexdigest():
        raise RuntimeError("configuration changed since GPU validation")
    directory = Path(args.work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "launch.json"
    if marker.exists():
        previous_pid = json.loads(marker.read_text())["pid"]
        try:
            os.kill(previous_pid, 0)
        except ProcessLookupError:
            pass
        else:
            # A reaped child may briefly remain a zombie; it holds no run lock.
            status = Path(f"/proc/{previous_pid}/stat")
            if not status.exists() or status.read_text().split()[2] != "Z":
                raise RuntimeError(f"the previous training process is still alive: {previous_pid}")
    command = [sys.executable, "-u", "-m", "tools.pmg_world_full.train",
               "--config", args.config, "--work-dir", args.work_dir,
               "--stop-after-epoch", str(args.stop_after_epoch),
               "--max-hours", str(args.max_hours)]
    environment = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       PYTHONUNBUFFERED="1")
    with (directory / "train.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment,
                                   start_new_session=True)
    payload = dict(pid=process.pid, command=command, log=str(directory / "train.log"))
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
