"""Launch the validated, isolated 15-epoch job in a persistent process session."""

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.sparse_opus_gausisan_world.runtime import atomic_json


def main():
    root = Path("analysis/pmg_gf2_probability_mini15e")
    validation = json.loads((root / "validation.json").read_text())
    if validation["status"] != "passed":
        raise ValueError("real batch-2 validation must pass before training")
    marker = root / "launched.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("the previous training process is still running")
    command = [sys.executable, "-u", "-m", "tools.pmg_gf2_probability.train", "--config",
               "config/sparse_opus_gaussian_world/pmg_A0_gf2_probability_4096_1024_direct15e.py",
               "--work-dir", "out/pmg_gf2_probability_mini15e/A0_GF2"]
    environment = dict(os.environ, MPLCONFIGDIR="/tmp/matplotlib-cache",
        TORCH_EXTENSIONS_DIR="/tmp/pmg_gf2_torch_extensions", OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    with (root / "train.log").open("a") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment, start_new_session=True)
    result = dict(pid=process.pid, process_group=process.pid, command=command,
                  planned_epochs=15, status="launched", log=str(root / "train.log"))
    atomic_json(marker, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
