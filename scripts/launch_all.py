"""
Legacy (v1/v2): queue the Table V / ablation runs of results/REPORT.md, one run per free GPU (CUDA_VISIBLE_DEVICES=i).
The GPU helpers (gpu_state, is_free, descendants) are also used by scripts/launch_v3*.py and scripts/launch_v4.py.

    nohup python scripts/launch_all.py > runs/launch_all.log 2>&1 &     # or inside tmux
    python scripts/launch_all.py --smoke     # 300-iteration end-to-end check of every config
    python scripts/launch_all.py --dry-run   # print the queue

By default a GPU counts as free only if it has no compute processes. With --allow-shared a run may
start on a GPU used by other jobs if at least --min-free-mib MiB are free (throughput is then only
indicative). A run whose runs/<run_name>/eval.json already holds the full-context result is skipped, so the
launcher can simply be restarted after a crash. Failed runs are recorded, not retried.
Status lines go to runs/launch_status.jsonl (runs/smoke_status.jsonl for --smoke), including any
foreign GPU processes seen while a run was going (throughput numbers are only valid without them).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [4212, 4213, 4214]
POLL_S = 20


def build_queue():
    """(run_name, config, extra args). TR2o-legacy first: it feeds the sanity gate."""
    queue = [(f"TR2o-legacy_s{s}", "configs/TR2o-nomask.yaml", ["--legacy-dropout-rng"]) for s in SEEDS]
    for s in SEEDS:
        queue += [
            (f"TR2o_s{s}", "configs/TR2o.yaml", []),
            (f"TR2o-rngfix_s{s}", "configs/TR2o-nomask.yaml", []),
            (f"TR8o_s{s}", "configs/TR8o.yaml", []),
            (f"LR-1_s{s}", "configs/LR-1.yaml", []),
            (f"LR-2_s{s}", "configs/LR-2.yaml", []),
        ]
    return [(name, cfg, ["--seed", name.rsplit("_s", 1)[1]] + extra) for name, cfg, extra in queue]


def gpu_state():
    """{gpu index: {"mem_used": MiB, "pids": [...]}} from nvidia-smi."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    state, uuid_to_idx = {}, {}
    for line in q.strip().splitlines():
        idx, uuid, mem, total = [x.strip() for x in line.split(",")]
        state[int(idx)] = {"mem_used": int(mem), "mem_total": int(total), "pids": []}
        uuid_to_idx[uuid] = int(idx)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    for line in apps.strip().splitlines():
        if line.strip():
            uuid, pid = [x.strip() for x in line.split(",")]
            if uuid in uuid_to_idx:
                state[uuid_to_idx[uuid]]["pids"].append(int(pid))
    return state


def is_free(s, allow_shared=False, min_free_mib=0):
    if allow_shared:
        return s["mem_total"] - s["mem_used"] >= min_free_mib
    return not s["pids"] and s["mem_used"] < 1000


def is_done(run_name):
    p = ROOT / "runs" / run_name / "eval.json"
    if not p.exists():
        return False
    try:
        return "full_context" in json.load(open(p))["checkpoints"]["best_loss_all_"]
    except Exception:
        return False


def descendants(pid):
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout.split()
    kids = [int(p) for p in out]
    return kids + [d for k in kids for d in descendants(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU indices (default: every GPU free at start)")
    ap.add_argument("--only", default=None, help="comma-separated run names to include")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-shared", action="store_true", help="start runs on GPUs used by other jobs if memory allows")
    ap.add_argument("--min-free-mib", type=int, default=30000)
    args = ap.parse_args()

    queue = build_queue()
    if args.smoke:
        seen, smoke = set(), []
        for name, cfg, extra in queue:  # one smoke run per (config, legacy flag), seed 4212
            key = (cfg, "--legacy-dropout-rng" in extra)
            if key not in seen and name.endswith("_s4212"):
                seen.add(key)
                smoke.append((f"smoke_{name}", cfg, extra + ["--epochs", "300", "--val-interval", "100", "--fc-stride", "64"]))
        queue = smoke
    if args.only:
        keep = set(args.only.split(","))
        queue = [q for q in queue if q[0] in keep]
    queue = [q for q in queue if not is_done(q[0])]

    status_path = ROOT / "runs" / ("smoke_status.jsonl" if args.smoke else "launch_status.jsonl")
    status_path.parent.mkdir(exist_ok=True)
    state = gpu_state()
    free = lambda s: is_free(s, args.allow_shared, args.min_free_mib)  # noqa: E731
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else [i for i, s in state.items() if free(s)]
    print(f"[{datetime.now():%F %T}] GPUs: {gpus}; state: {state}")
    print("Queue:\n  " + "\n  ".join(f"{n}: {c} {' '.join(e)}" for n, c, e in queue), flush=True)
    if args.dry_run or not queue:
        return
    if not gpus:
        sys.exit("no free GPU")

    running = {}  # gpu -> dict
    while queue or running:
        state = gpu_state()
        for gpu, r in list(running.items()):
            ours = {r["proc"].pid, *descendants(r["proc"].pid)}
            foreign = [p for p in state[gpu]["pids"] if p not in ours]
            r["foreign_pids"].update(foreign)
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["log"].close()
            rec = {"run": r["name"], "gpu": gpu, "cmd": r["cmd"], "start": r["start"], "end": f"{datetime.now():%F %T}",
                   "returncode": rc, "complete": is_done(r["name"]) if not args.smoke else rc == 0,
                   "foreign_gpu_pids_seen": sorted(r["foreign_pids"]), "gpu_state_at_start": r["gpu_state_at_start"]}
            with open(status_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['end']}] finished {r['name']} on GPU {gpu}: rc={rc}, foreign pids {rec['foreign_gpu_pids_seen']}",
                  flush=True)
            del running[gpu]
        for gpu in gpus:
            if not queue or gpu in running:
                continue
            if not free(state[gpu]):
                continue  # busy with someone else's job: wait
            name, cfg, extra = queue.pop(0)
            run_dir = ROOT / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "train.py", "--config", cfg, "--run-name", name] + extra
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false")
            log = open(run_dir / "launcher.log", "w")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = {"name": name, "cmd": " ".join(cmd), "proc": proc, "log": log, "start": f"{datetime.now():%F %T}",
                            "foreign_pids": set(), "gpu_state_at_start": state[gpu]}
            print(f"[{datetime.now():%F %T}] started {name} on GPU {gpu}: {' '.join(cmd)}", flush=True)
        time.sleep(POLL_S)
    print(f"[{datetime.now():%F %T}] queue empty", flush=True)


if __name__ == "__main__":
    main()
