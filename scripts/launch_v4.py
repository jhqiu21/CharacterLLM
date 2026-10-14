"""
Queue the v4 runs and evaluations on fixed GPUs, one job per GPU (results/v4/REPORT.md).

    nohup setsid python scripts/launch_v4.py --gpus 2,3 > runs/launch_v4.txt 2>&1 &
    python scripts/launch_v4.py --dry-run

Jobs, in priority order (a job starts only once the runs it needs have finished):
- NoPE final runs TR-final-nope_s{42,43,44}: configs/final/TR.yaml with model.pos_encoding_type=none, 50,000 iterations;
- position interpolation (eval_extrapolation.py --interpolate, W = 256/512/1024) for every seed of the v3 RoPE and
  sinusoidal runs, and the attention analysis of RoPE + PI at W = 512 (seed 42); no training needed;
- lr bracket check (validation only, the v3 tuning protocol: --tuning, seed 4212, 10,000 iterations, warmup 2,000,
  validation every 500, scored by full-context BPC on the validation split with stride 8) at the final configs (dropout 0,
  label smoothing 0): two new learning rates per family plus a rerun of the family's lowest-BPC lr on this hardware;
- NoPE evaluations: extrapolation (W = 128/256/512/1024, --sanity) for every seed, attention at W = 128 and 512 (seed 42).
Every training run uses --set evaluation.full_context_batch_size=2048 (8192 runs out of memory on a 24 GB card).
A GPU is used only when nobody else has a process on it. Finished jobs are skipped, so the launcher can be restarted;
failed jobs are recorded, not retried. Logs: runs/<run>/launcher.txt (training), runs/eval_logs/<job>.txt (evaluations).
Records: runs/v4_status.jsonl (one line per finished job).
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_all import descendants, gpu_state, is_free  # noqa: E402
from launch_v3 import FC_BATCH, SEEDS, gpu_env, is_done  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLL_S = 20
TUNING = ["--seed", "4212", "--tuning", "--set", "training.epochs=10000", "--set", "scheduler.warmup_iters=2000",
          "--set", "training.validation_interval=500"]
# family: (config, lowest-BPC lr of the v3 search (rerun here), two new lrs: further steps of factor 1.5)
BRACKET = {"TR": ("configs/final/TR.yaml", "2.7e-3", ["4.05e-3", "6.075e-3"]),
           "LSTM": ("configs/final/LSTM.yaml", "7.2e-3", ["1.08e-2", "1.62e-2"])}
PI_PES = ["rope", "sinusoidal"]


def tr_run(pe, seed):
    """Run name of the final Transformer with encoding `pe` (RoPE is the base run; NoPE runs are named 'nope')."""
    return f"TR-final_s{seed}" if pe == "rope" else f"TR-final-{'nope' if pe == 'none' else pe}_s{seed}"


def bracket_run(family, lr):
    return f"bracket_{family}_lr{lr}"


def build_jobs():
    """[(name, kind, runs it needs, done-check path or None for training, argv)] in priority order."""
    py = sys.executable
    jobs = []

    def train_job(run, cfg, extra):
        jobs.append((run, "train", [], None, [py, "train.py", "--config", cfg, "--run-name", run] + extra + FC_BATCH))

    def eval_job(name, run, out, argv):
        jobs.append((name, "eval", [run], f"runs/{run}/{out}", [py] + argv))

    for s in SEEDS:
        train_job(tr_run("none", s), "configs/final/TR.yaml", ["--seed", str(s), "--set", "model.pos_encoding_type=none"])
    for pe in PI_PES:
        for s in SEEDS:
            run = tr_run(pe, s)
            eval_job(f"pi:{run}", run, "extrapolation_pi.json",
                     ["scripts/eval_extrapolation.py", "--run", run, "--lengths", "128", "256", "512", "1024", "--interpolate"])
    eval_job(f"attn512pi:{tr_run('rope', 42)}", tr_run("rope", 42), "attention_W512_pi/stats.json",
             ["scripts/analyze_attention.py", "--run", tr_run("rope", 42), "--length", "512", "--interpolate"])
    for family, (cfg, best, new) in BRACKET.items():
        for lr in [best] + new:
            train_job(bracket_run(family, lr), cfg, TUNING + ["--set", f"training.learning_rate={lr}"])
    for s in SEEDS:
        run = tr_run("none", s)
        eval_job(f"extrap:{run}", run, "extrapolation.json",
                 ["scripts/eval_extrapolation.py", "--run", run, "--lengths", "128", "256", "512", "1024", "--sanity"])
    for W in (128, 512):
        run = tr_run("none", 42)
        eval_job(f"attn{W}:{run}", run, f"attention_W{W}/stats.json", ["scripts/analyze_attention.py", "--run", run, "--length", str(W)])
    return jobs


def finished(job):
    name, kind, _, out, _ = job
    return is_done(name) if kind == "train" else (ROOT / out).exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--only", default=None, help="comma-separated job names to include")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only.split(",")]
    skipped = [j[0] for j in jobs if finished(j)]
    queue = [j for j in jobs if j[0] not in skipped]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[{datetime.now():%F %T}] GPUs {gpus}; skipped (finished): {skipped}")
    print("Queue:\n  " + "\n  ".join(f"{j[0]}: {' '.join(j[4][1:])}" for j in queue), flush=True)
    if args.dry_run or not queue:
        return

    status = ROOT / "runs" / "v4_status.jsonl"
    (ROOT / "runs" / "eval_logs").mkdir(parents=True, exist_ok=True)
    running, failed = {}, set()
    while queue or running:
        state = gpu_state()
        for gpu, r in list(running.items()):
            ours = {r["proc"].pid, *descendants(r["proc"].pid)}
            r["foreign"].update(p for p in state[gpu]["pids"] if p not in ours)
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["log"].close()
            ok = rc == 0 and finished(r["job"])
            if not ok:
                failed.add(r["job"][0])
            rec = {"job": r["job"][0], "kind": r["job"][1], "gpu": gpu, "cmd": " ".join(r["job"][4]), "start": r["start"],
                   "end": f"{datetime.now():%F %T}", "seconds": round(time.time() - r["t0"], 1), "returncode": rc,
                   "complete": ok, "foreign_gpu_pids_seen": sorted(r["foreign"])}
            with open(status, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['end']}] finished {rec['job']} on GPU {gpu}: rc={rc}, complete={ok}, {rec['seconds'] / 60:.1f} min", flush=True)
            del running[gpu]
        # a job whose dependency failed can never start
        for j in [j for j in queue if any(d in failed for d in j[2])]:
            queue.remove(j)
            failed.add(j[0])
            print(f"[{datetime.now():%F %T}] dropped {j[0]}: dependency failed", flush=True)
        for gpu in gpus:
            if gpu in running or not is_free(state[gpu]):
                continue
            ready = [j for j in queue if all(is_done(d) for d in j[2])]
            if not ready:
                break
            job = ready[0]
            queue.remove(job)
            name, kind, _, _, argv = job
            if kind == "train":
                (ROOT / "runs" / name).mkdir(parents=True, exist_ok=True)
                log_path = ROOT / "runs" / name / "launcher.txt"
            else:
                log_path = ROOT / "runs" / "eval_logs" / f"{name.replace(':', '_')}.txt"
            log = open(log_path, "w")
            proc = subprocess.Popen(argv, cwd=ROOT, env=gpu_env(gpu), stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = {"job": job, "proc": proc, "log": log, "start": f"{datetime.now():%F %T}", "t0": time.time(), "foreign": set()}
            print(f"[{datetime.now():%F %T}] started {name} on GPU {gpu}: {' '.join(argv)}", flush=True)
        time.sleep(POLL_S)
    print(f"[{datetime.now():%F %T}] queue empty; failed: {sorted(failed)}", flush=True)


if __name__ == "__main__":
    main()
