"""
Queue the v3 evaluations (prompts/03_final_and_extensions.md, Part 2) on fixed GPUs, one job per GPU, after training.

    nohup setsid python scripts/launch_v3_evals.py --gpus 2,3 > runs/launch_v3_evals.txt 2>&1 &

- length extrapolation (scripts/eval_extrapolation.py --sanity, W = 128/256/512/1024) for every seed of the six
  positional encodings (blocks 1-2) and of the LSTMs LR-1 and LR-2;
- attention analysis (scripts/analyze_attention.py) for seed 42 of the six encodings at W = 128, and of
  rope / alibi / sinusoidal / relative at W = 512;
- PE tables (scripts/export_pe_tables.py, a few seconds on a GPU) for seed 42 of every encoding.
A job is skipped when its output exists or its run has not finished. GPU rules as in launch_v3.py
(a GPU is used only without anyone else's process on it). Records: runs/v3_evals_status.jsonl.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_all import gpu_state, is_free  # noqa: E402
from launch_v3 import SEEDS, gpu_env, is_done  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PES = ["rope", "learned", "sinusoidal", "hybrid", "alibi", "relative"]
EXTRAPOLATING = ["rope", "alibi", "sinusoidal", "relative"]
POLL_S = 15


def tr_run(pe, seed):
    return f"TR-final_s{seed}" if pe == "rope" else f"TR-final-{pe}_s{seed}"


def build_jobs():
    """[(job name, run, output file, argv, needs GPU)] in priority order."""
    py = sys.executable
    jobs = []
    for s in SEEDS:
        for run in [tr_run(pe, s) for pe in PES] + [f"LSTM-final_s{s}", f"LR-1-final_s{s}"]:
            jobs.append((f"extrap:{run}", run, f"runs/{run}/extrapolation.json",
                         [py, "scripts/eval_extrapolation.py", "--run", run, "--lengths", "128", "256", "512", "1024", "--sanity"], True))
    for pe in PES:
        run = tr_run(pe, 42)
        jobs.append((f"attn128:{run}", run, f"runs/{run}/attention_W128/stats.json",
                     [py, "scripts/analyze_attention.py", "--run", run, "--length", "128"], True))
        jobs.append((f"pe:{run}", run, f"runs/{run}/pe_tables.json", [py, "scripts/export_pe_tables.py", "--run", run], True))
    for pe in EXTRAPOLATING:
        run = tr_run(pe, 42)
        jobs.append((f"attn512:{run}", run, f"runs/{run}/attention_W512/stats.json",
                     [py, "scripts/analyze_attention.py", "--run", run, "--length", "512"], True))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", default="", help="comma-separated job names to rerun even if their output exists")
    args = ap.parse_args()
    force = set(filter(None, args.force.split(",")))
    jobs = [j for j in build_jobs() if j[0] in force or not (ROOT / j[2]).exists()]
    missing = [j[0] for j in jobs if not is_done(j[1])]
    jobs = [j for j in jobs if is_done(j[1])]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[{datetime.now():%F %T}] GPUs {gpus}; runs not finished (skipped): {missing}")
    print("Jobs:\n  " + "\n  ".join(j[0] for j in jobs), flush=True)
    if args.dry_run:
        return

    status = ROOT / "runs" / "v3_evals_status.jsonl"
    log_dir = ROOT / "runs" / "eval_logs"
    log_dir.mkdir(exist_ok=True)

    def record(name, rc, t0, slot):
        rec = {"job": name, "slot": slot, "returncode": rc, "seconds": round(time.time() - t0, 1), "end": f"{datetime.now():%F %T}"}
        with open(status, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{rec['end']}] finished {name} ({slot}): rc={rc}, {rec['seconds']:.0f}s", flush=True)

    for name, _, _, argv, gpu_needed in [j for j in jobs if not j[4]]:  # CPU jobs first, one by one
        t0 = time.time()
        with open(log_dir / f"{name.replace(':', '_')}.txt", "w") as log:
            rc = subprocess.run(argv, cwd=ROOT, env=dict(gpu_env(0), JAX_PLATFORMS="cpu"), stdout=log, stderr=subprocess.STDOUT).returncode
        record(name, rc, t0, "cpu")
    queue = [j for j in jobs if j[4]]
    running = {}
    while queue or running:
        state = gpu_state()
        for gpu, (name, proc, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            record(name, proc.returncode, t0, f"gpu{gpu}")
            del running[gpu]
        for gpu in gpus:
            if queue and gpu not in running and is_free(state[gpu]):
                name, _, _, argv, _ = queue.pop(0)
                log = open(log_dir / f"{name.replace(':', '_')}.txt", "w")
                proc = subprocess.Popen(argv, cwd=ROOT, env=gpu_env(gpu), stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = (name, proc, log, time.time())
                print(f"[{datetime.now():%F %T}] started {name} on GPU {gpu}", flush=True)
        time.sleep(POLL_S)
    print(f"[{datetime.now():%F %T}] all jobs done", flush=True)


if __name__ == "__main__":
    main()
