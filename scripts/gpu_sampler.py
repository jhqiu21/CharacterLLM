"""
Legacy (v1 throughput table, results/throughput.md): log the GPU compute processes every few seconds, with timestamps,
to runs/gpu_usage.jsonl.

    nohup python scripts/gpu_sampler.py &      # or inside tmux

results/aggregate.py --sync uses this log to decide whether a run had the GPU to itself during
its throughput-timing window (iterations 1000-2000), which the launcher's per-run flag cannot tell.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "runs" / "gpu_usage.jsonl"
INTERVAL_S = 5


def sample():
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    procs = []
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        cmd = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout.strip()
        procs.append({"gpu_uuid": uuid, "pid": int(pid), "mem_mib": int(mem), "cmd": cmd[:200]})
    return {"t": datetime.now().isoformat(timespec="seconds"), "procs": procs}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    while True:
        with open(OUT, "a") as f:
            f.write(json.dumps(sample()) + "\n")
        time.sleep(INTERVAL_S)
