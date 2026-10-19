"""
Queue the v6 capacity experiments on fixed GPUs, one job per GPU (results/v6/REPORT.md).

    nohup setsid bash -c 'for p in v5topk search formal seeds evals; do python scripts/launch_v6.py --gpus 3 --phase $p || exit 1; done' \
        > runs/launch_v6.txt 2>&1 &
    python scripts/launch_v6.py --phase formal --dry-run

Every phase builds its queue when it starts, from the finished results of the earlier phases, so run the phases as
separate, consecutive invocations (as above).

Models (results/v6/configs/): AL-3M (4 layers, d_model 256, 4 heads, 3,169,280 parameters) and AL-14M (8 layers, 384,
6 heads, 14,204,928), both ALiBi, trained from random initialisation on 512-character windows, batch 32 (16,384
training characters per update), everything else as configs/final/TR.yaml. Checkpoints are selected on the fixed
validation set by accuracy, ties broken by the lower loss (training.checkpoint_selection = acc).

Phases:
- v5topk: the best v5 models (ALiBi + 4,000 iterations of 512-window fine-tuning; RoPE + the same fine-tuning without
  PI) and the 128-trained ALiBi runs, all seeds, re-evaluated on the test set (W = 128..1024, stride 128) to add top-k
  accuracy; written to extrapolation_v6.json, the v5 / v3 files are not touched.
- search: learning rate 0.5x / 1x / 2x of TR.yaml's 1.8e-3 for each model, SEARCH_ITERS iterations (warmup 2,000,
  validation every 500, cosine to 0), seed SEARCH_SEED, validation only (--tuning). Same budget for both models.
- formal: seed 42, FORMAL_ITERS = 50,000 iterations (819,200,000 training characters) at each model's chosen lr: the
  search lr with the highest validation accuracy of the selected checkpoint on the whole validation split (W = 512,
  stride 128; val_eval.json), ties broken by the lower validation BPC.
- seeds: seeds 43 and 44 of both models, only if the seed-42 runs show a clear capacity gain (GAIN_RULE_PP, fixed
  before the formal runs): AL-14M's validation accuracy is at least GAIN_RULE_PP percentage points above AL-3M's and
  its validation BPC is lower. Otherwise the phase records the decision and queues nothing.
- evals: every finished formal / seed run: validation split (W = 128, 256, 512, 1024, stride 128), then the test split
  (same W), then the test-set error breakdown by position in the word (scripts/analyze_errors.py, W = 512) of these runs
  and of the two best v5 conditions. The test set is opened only in this phase and in v5topk; nothing is selected on it.
- bracket (added after the search, whose best lr was the upper edge 3.6e-3 for both models): BRACKET_LR = 4x, same
  protocol as the search, both models. A post-hoc check of whether the optimum is bracketed; it is not used to choose
  the lr of the formal runs (those had started).
A GPU is used only when nobody else has a process on it. Finished jobs are skipped, so the launcher can be restarted;
a training job whose process is still running (e.g. left by a stopped launcher) is waited for, never started twice;
failed jobs are recorded, not retried. Each invocation snapshots the code (HEAD and the uncommitted diff, untracked
source files included) into results/v6/code/. Records: runs/v6_status.jsonl (one line per finished job), logs:
runs/<run>/launcher.txt (training), runs/eval_logs/<job>.txt (evaluations).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_all import descendants, gpu_state, is_free  # noqa: E402
from launch_v3 import gpu_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLL_S = 10
MODELS = {"AL-3M": "results/v6/configs/AL-3M.yaml", "AL-14M": "results/v6/configs/AL-14M.yaml"}
BASE_LR = 1.8e-3  # configs/final/TR.yaml
LR_GRID = ["9e-4", "1.8e-3", "3.6e-3"]  # 0.5x, 1x, 2x
BRACKET_LR = "7.2e-3"  # 4x, post-hoc check (bracket phase)
SEARCH_SEED = 4212  # the tuning seed of v3 / v4, distinct from the final seeds
SEARCH_ITERS = 10000
FORMAL_ITERS = 50000
FORMAL_SEED = 42
EXTRA_SEEDS = [43, 44]
GAIN_RULE_PP = 0.3
EVAL_W = ["128", "256", "512", "1024"]
V5_RUNS = [f"ft4k-alibi512_lr3e-5_s{s}" for s in (42, 43, 44)] + [f"ft4k-long512_lr3e-4_s{s}" for s in (42, 43, 44)] \
    + [f"TR-final-alibi_s{s}" for s in (42, 43, 44)]
SOURCE_DIRS = ("train.py", "models/", "utils/", "scripts/", "tests/", "configs/", "results/v6/configs/", "results/v6/aggregate_v6.py")


def search_run(model, lr):
    return f"v6search-{model}_lr{lr}"


def formal_run(model, seed):
    return f"v6-{model}_s{seed}"


def val_eval(run):
    """val_eval.json of a finished --tuning run, or None."""
    try:
        e = json.load(open(ROOT / "runs" / run / "val_eval.json"))
        return e if "full_context" in e["checkpoints"][e["selected_checkpoint"]] else None
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def selected_val(run):
    """(validation accuracy, validation BPC) of the selected checkpoint on the whole validation split."""
    e = val_eval(run)
    fc = e["checkpoints"][e["selected_checkpoint"]]["full_context"]
    return fc["acc"], fc["bpc"]


def chosen_lr(model):
    """Search lr with the highest validation accuracy, ties broken by the lower BPC; None while the search runs."""
    if any(val_eval(search_run(model, lr)) is None for lr in LR_GRID):
        return None
    return max(LR_GRID, key=lambda lr: (selected_val(search_run(model, lr))[0], -selected_val(search_run(model, lr))[1]))


def gain_decision():
    """(clear gain?, text) from the seed-42 formal runs (GAIN_RULE_PP)."""
    small, big = formal_run("AL-3M", FORMAL_SEED), formal_run("AL-14M", FORMAL_SEED)
    if val_eval(small) is None or val_eval(big) is None:
        return None, "seed-42 formal runs not finished"
    (a3, b3), (a14, b14) = selected_val(small), selected_val(big)
    gain = 100 * (a14 - a3)
    ok = gain >= GAIN_RULE_PP and b14 < b3
    return ok, (f"validation acc AL-14M {100 * a14:.3f}% vs AL-3M {100 * a3:.3f}% (gain {gain:+.3f} pp, rule >= {GAIN_RULE_PP}), "
                f"BPC {b14:.4f} vs {b3:.4f} -> {'clear gain: run seeds 43, 44' if ok else 'no clear gain: no further seeds'}")


def snapshot_code(tag):
    """Write HEAD, `git status` and the uncommitted diff (tracked changes + untracked source files) to results/v6/code/."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout
    head = git("rev-parse", "HEAD").strip()
    patch = git("diff", "HEAD", "--binary")
    untracked = [f for f in git("ls-files", "--others", "--exclude-standard").split()
                 if f.startswith(SOURCE_DIRS) and "__pycache__" not in f]
    for f in untracked:
        patch += git("diff", "--no-index", "--binary", "/dev/null", f)
    sha = hashlib.sha256(patch.encode()).hexdigest()[:12]
    out = ROOT / "results" / "v6" / "code"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{tag}_{head[:7]}_{sha}"
    (out / f"{name}.patch").write_text(patch)
    (out / f"{name}.txt").write_text(f"HEAD {head}\npatch sha256[:12] {sha}\nuntracked source files: {untracked}\n\n"
                                      f"git status --porcelain:\n{git('status', '--porcelain')}")
    return name


def build_jobs(phase):
    """[(name, kind, runs it needs, done-check path or None for training, argv)]."""
    py = sys.executable
    jobs = []

    def train_job(run, model, lr, seed, iters, warmup=2000):
        jobs.append((run, "train", [], None, [py, "train.py", "--config", MODELS[model], "--seed", str(seed), "--run-name", run,
                                              "--tuning", "--set", f"training.learning_rate={lr}", "--set", f"training.epochs={iters}",
                                              "--set", f"scheduler.warmup_iters={warmup}",
                                              "--set", f"training.stage_checkpoint_interval={iters}"]))

    def eval_job(name, run, out, argv):
        jobs.append((name, "eval", [run], f"runs/{run}/{out}", [py] + argv))

    if phase == "v5topk":
        for run in V5_RUNS:
            eval_job(f"topk:{run}", run, "extrapolation_v6.json",
                     ["scripts/eval_extrapolation.py", "--run", run, "--lengths", *EVAL_W, "--tag", "v6"])
    elif phase in ("search", "bracket"):
        for model in MODELS:
            for lr in (LR_GRID if phase == "search" else [BRACKET_LR]):
                train_job(search_run(model, lr), model, lr, SEARCH_SEED, SEARCH_ITERS)
    elif phase in ("formal", "seeds"):
        if phase == "seeds":
            ok, text = gain_decision()
            print(f"phase seeds: {text}", flush=True)
            if not ok:
                return jobs
        for model in MODELS:
            lr = chosen_lr(model)
            if lr is None:
                raise SystemExit(f"phase {phase}: the lr search of {model} has not finished")
            print(f"phase {phase}: {model} lr {lr} "
                  f"(search: {[(x, *[round(v, 6) for v in selected_val(search_run(model, x))]) for x in LR_GRID]})", flush=True)
            for seed in ([FORMAL_SEED] if phase == "formal" else EXTRA_SEEDS):
                train_job(formal_run(model, seed), model, lr, seed, FORMAL_ITERS)
        if phase == "seeds":  # largest model first, so that on two GPUs the two long runs go in parallel
            jobs.sort(key=lambda j: -list(MODELS).index(j[0].split("_s")[0].removeprefix("v6-")))
    elif phase == "evals":
        runs = [formal_run(m, s) for s in [FORMAL_SEED] + EXTRA_SEEDS for m in MODELS if val_eval(formal_run(m, s))]
        for split in ("val", "test"):
            for run in runs:
                eval_job(f"{split}:{run}", run, "extrapolation_val.json" if split == "val" else "extrapolation.json",
                         ["scripts/eval_extrapolation.py", "--run", run, "--split", split, "--lengths", *EVAL_W])
        for run in runs + [r for r in V5_RUNS if r.startswith("ft4k-")]:
            eval_job(f"errors:{run}", run, "errors_W512.json", ["scripts/analyze_errors.py", "--run", run, "--length", "512"])
    else:
        raise SystemExit(f"unknown phase {phase!r}")
    return jobs


def finished(job):
    name, kind, _, out, _ = job
    return val_eval(name) is not None if kind == "train" else (ROOT / out).exists()


def in_progress(job):
    """A training job whose train.py process is running already (started by an earlier, stopped launcher)."""
    if job[1] != "train":
        return False
    pattern = f"train[.]py .*--run-name {re.escape(job[0])}( |$)"
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="3")
    ap.add_argument("--phase", required=True, choices=["v5topk", "search", "formal", "seeds", "evals", "bracket"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.phase)
    skipped = [j[0] for j in jobs if finished(j)]
    queue = [j for j in jobs if j[0] not in skipped]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[{datetime.now():%F %T}] GPUs {gpus}; phase {args.phase}; skipped (finished): {skipped}")
    print("Queue:\n  " + "\n  ".join(f"{j[0]}: {' '.join(j[4][1:])}" for j in queue), flush=True)
    if args.dry_run or not queue:
        return
    snapshot = snapshot_code(args.phase)
    print(f"code snapshot: results/v6/code/{snapshot}.patch", flush=True)

    status = ROOT / "runs" / "v6_status.jsonl"
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
            rec = {"job": r["job"][0], "kind": r["job"][1], "phase": args.phase, "gpu": gpu, "cmd": " ".join(r["job"][4]),
                   "start": r["start"], "end": f"{datetime.now():%F %T}", "seconds": round(time.time() - r["t0"], 1),
                   "returncode": rc, "complete": ok, "foreign_gpu_pids_seen": sorted(r["foreign"]), "code_snapshot": snapshot}
            with open(status, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['end']}] finished {rec['job']} on GPU {gpu}: rc={rc}, complete={ok}, {rec['seconds'] / 60:.1f} min", flush=True)
            del running[gpu]
        for j in [j for j in queue if any(d in failed for d in j[2])]:
            queue.remove(j)
            failed.add(j[0])
            print(f"[{datetime.now():%F %T}] dropped {j[0]}: dependency failed", flush=True)
        for gpu in gpus:
            if gpu in running or not is_free(state[gpu]):
                continue
            ready = [j for j in queue if not in_progress(j)]
            if not ready:
                break
            job = ready[0]
            queue.remove(job)
            if finished(job):
                print(f"[{datetime.now():%F %T}] skipped {job[0]}: finished meanwhile", flush=True)
                continue
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
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
