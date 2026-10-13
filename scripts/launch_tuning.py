"""
Run the v3 hyperparameter search; the grid and the decision rules live in results/v3_tuning/aggregate_tuning.py.

    nohup setsid python scripts/launch_tuning.py > runs/launch_tuning.log 2>&1 &
    python scripts/launch_tuning.py --dry-run     # print the pending runs and the GPU state

Loop: copy finished runs into results/v3_tuning/raw/, ask the rules which runs are pending (per family: the grid; once
the grid is complete, the edge extension if the rule calls for it; once that is complete, the eps = 0 run), and start
them on one GPU, at most --max-jobs at a time and only while at least --min-free-mib MiB are free on it. Runs that
depend on others (edge, eps = 0) go first, and LSTM before Transformer (LSTM runs are ~3x longer). A finished run is
recorded in runs/tuning_status.jsonl as done, diverged (non-finite loss) or failed; failed runs are not retried in the
same session. Throughput is not measured here: two runs share the GPU.
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
sys.path.insert(0, str(ROOT / "results" / "v3_tuning"))
import aggregate_tuning as agg  # noqa: E402

STATUS = ROOT / "runs" / "tuning_status.jsonl"
POLL_S = 20
MIN_GAP_S = 60  # between two starts, so the memory reading includes the previous run
PHASE_PRIORITY = {"ls": 0, "edge": 1, "grid": 2}
FAMILY_PRIORITY = {"LSTM": 0, "TR": 1}


def now():
    return f"{datetime.now():%F %T}"


def gpu_state(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    uuid, used, total = [x.strip() for x in q.strip().split(",")]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    pids = [int(p) for u, p in (line.split(",") for line in apps.strip().splitlines() if line.strip()) if u.strip() == uuid]
    return {"mem_used": int(used), "mem_total": int(total), "free": int(total) - int(used), "pids": pids}


def descendants(pid):
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout.split()
    kids = [int(p) for p in out]
    return kids + [d for k in kids for d in descendants(k)]


def pending(rows, skip):
    todo = [s for s in agg.plan(rows) if not agg.finished(rows, s) and s["name"] not in skip]
    return sorted(todo, key=lambda s: (PHASE_PRIORITY[s["phase"]], FAMILY_PRIORITY[s["family"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-jobs", type=int, default=2)
    ap.add_argument("--min-free-mib", type=int, default=30 * 1024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    agg.sync()
    rows = agg.load_rows()
    print(f"[{now()}] {len(rows)} runs finished; GPU {args.gpu}: {gpu_state(args.gpu)}")
    print("Pending now:\n  " + "\n  ".join(f"{s['name']} ({s['phase']})" for s in pending(rows, set())), flush=True)
    if args.dry_run:
        return

    running, failed, last_start = {}, set(), 0.0
    while True:
        agg.sync()
        rows = agg.load_rows()
        state = gpu_state(args.gpu)
        for name, r in list(running.items()):
            rc = r["proc"].poll()
            if rc is None:
                ours = {r["proc"].pid, *descendants(r["proc"].pid)}
                theirs = {p for other in running.values() for p in (other["proc"].pid, *descendants(other["proc"].pid))}
                r["foreign_pids"].update(p for p in state["pids"] if p not in ours | theirs)
                continue
            r["log"].close()
            outcome = rows[name]["status"] if name in rows else "failed"
            if outcome == "failed":
                failed.add(name)
            rec = {"run": name, "phase": r["phase"], "cmd": r["cmd"], "start": r["start"], "end": now(), "returncode": rc,
                   "outcome": outcome, "gpu_at_start": r["gpu_at_start"], "foreign_gpu_pids_seen": sorted(r["foreign_pids"])}
            with open(STATUS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['end']}] finished {name}: rc={rc}, {outcome}"
                  + (f", val BPC {rows[name]['bpc']:.4f}" if outcome == "done" else ""), flush=True)
            del running[name]

        todo = pending(rows, set(running) | failed)
        if not todo and not running:
            break
        if todo and len(running) < args.max_jobs and time.time() - last_start >= MIN_GAP_S and state["free"] >= args.min_free_mib:
            s = todo[0]
            run_dir = ROOT / "runs" / s["name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "train.py", "--run-name", s["name"]] + s["args"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false")
            log = open(run_dir / "launcher.log", "w")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[s["name"]] = {"phase": s["phase"], "cmd": " ".join(cmd), "proc": proc, "log": log, "start": now(),
                                  "gpu_at_start": state, "foreign_pids": set()}
            last_start = time.time()
            print(f"[{now()}] started {s['name']} ({s['phase']}; {len(running)} running; GPU free {state['free']} MiB): "
                  f"{' '.join(cmd)}", flush=True)
        time.sleep(POLL_S)

    print(f"[{now()}] nothing pending" + (f"; FAILED (not retried): {sorted(failed)}" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
