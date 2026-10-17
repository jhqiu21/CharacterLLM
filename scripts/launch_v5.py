"""
Queue the v5 long-context experiments on fixed GPUs, one job per GPU (results/v5/REPORT.md).

    nohup setsid python scripts/launch_v5.py --gpus 3 --phase grid infer > runs/launch_v5.txt 2>&1 &
    python scripts/launch_v5.py --phase ft --dry-run

(The ft phase needs the finished grid when its queue is built, so run it as a separate, later invocation.)

Phases, run in the order given by --phase (a job starts only once the runs it needs have finished):
- grid: fine-tuning learning-rate grid on seed 42, validation only (--tuning). Every condition starts from the
  best-loss checkpoint (iteration 49,999) of the 50k run with the same seed (--init-from), with a fresh AdamW state, and
  trains FT_ITERS iterations (warmup 50, cosine to 0) at each lr of LR_GRID. Every condition sees the same
  16,384 characters per step (32 x 512 or 128 x 128), so the same number of extra training characters.
- infer: evaluation-only variants of the RoPE runs (every seed, no fine-tuning) with scripts/eval_extrapolation.py,
  first on the validation split (used to choose among them), then on the test split: local attention (32/64/128),
  SelfExtend 'floor' and 'linear' (near window 16/32/64); plain RoPE and RoPE + PI on the validation split (their
  test results are in results/v3 and results/v4).
- ft: the seeds 43 and 44 of every condition at the lr chosen in the grid (lowest best fixed-validation loss; the
  seed-42 grid run at that lr is the seed-42 run; a condition whose grid never improved on its starting validation loss
  is not continued, since its selected checkpoint is the source model), then the test evaluations of every fine-tuned run (W = 128..1024,
  stride 128, plus stride 1 at W = 128), PI applied once more at W = 1024 for the PI condition, and the attention
  analysis at W = 512 (seed 42) of the two fine-tuned RoPE conditions.
- budget: every continued condition again with BUDGET_ITERS = 4,000 iterations (warmup and validation interval 5% of it)
  at the lr chosen for 1,000, all seeds, with the same test evaluations: does the ranking hold with 4x the adaptation?
A GPU is used only when nobody else has a process on it. Finished jobs are skipped, so the launcher can be restarted;
failed jobs are recorded, not retried. Logs: runs/<run>/launcher.txt (training), runs/eval_logs/<job>.txt (evaluations).
Records: runs/v5_status.jsonl (one line per finished job).
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
from launch_v3 import SEEDS, gpu_env, is_done  # noqa: E402
from launch_v4 import tr_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POLL_S = 10
GRID_SEED = 42
FT_ITERS = 1000
# 3e-3 was added after the first four: the PI condition's best lr was 1e-3, the upper edge (rule: extend by one step)
LR_GRID = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]
# validation split scored at W = max_len, stride 128; 256 windows per batch (2048 x 512 runs out of memory)
BUDGET_ITERS = 4000  # budget phase: 4x the fine-tuning budget, warmup and validation interval scaled with it


def ft_common(iters):
    return ["--tuning", "--set", f"training.epochs={iters}", "--set", f"scheduler.warmup_iters={iters // 20}",
            "--set", f"training.validation_interval={iters // 20}", "--set", "evaluation.tuning_val_stride=128",
            "--set", "evaluation.full_context_batch_size=256"]


FT_COMMON = ft_common(FT_ITERS)
LONG = ["--set", "model.max_len=512", "--set", "training.sequence_length=512", "--set", "training.batch_size=32"]
# condition: (encoding of the source run, overrides); 32 x 512 and 128 x 128 are both 16,384 characters per step
CONDITIONS = {
    "pi512": ("rope", LONG + ["--set", "model.pos_scale=0.25"]),  # RoPE + PI + long-window training
    "long512": ("rope", LONG),                                    # RoPE + long-window training
    "short128": ("rope", []),                                     # RoPE + the same extra training at 128
    "alibi512": ("alibi", LONG),                                  # ALiBi + long-window training (comparison)
}
INFER_W = ["256", "512", "1024"]
# SelfExtend near windows; 96 and 112 were added after the first three (the best, 64, was the upper edge)
SE_NEAR = (16, 32, 64, 96, 112)
# variant: (eval_extrapolation.py arguments, output-name part, lengths, splits)
INFER = {
    "plain": ([], "", ["128"] + INFER_W, ["val"]),
    "pi": (["--interpolate"], "_pi", INFER_W, ["val"]),
    **{f"local{n}": (["--local-window", str(n)], f"_local{n}", ["128"] + INFER_W, ["val", "test"]) for n in (32, 64, 128)},
    **{f"se{m}{n}": (["--self-extend", str(n), "--self-extend-mode", m], f"_se{m}{n}", INFER_W, ["val", "test"])
       for m in ("floor", "linear") for n in SE_NEAR},
}


def ft_run(cond, lr, seed, iters=FT_ITERS):
    return f"ft-{cond}_lr{lr}_s{seed}" if iters == FT_ITERS else f"ft{iters // 1000}k-{cond}_lr{lr}_s{seed}"


def grid_val_loss(cond, lr):
    """Best fixed-validation loss (nats) of the seed-42 grid run, or None if it has not finished."""
    try:
        e = json.load(open(ROOT / "runs" / ft_run(cond, lr, GRID_SEED) / "val_eval.json"))
        return float(e["checkpoints"]["best_loss_all_"]["val_loss"])
    except (FileNotFoundError, KeyError):
        return None


def chosen_lr(cond):
    """The grid lr with the lowest best fixed-validation loss; None while the grid is incomplete."""
    losses = {lr: grid_val_loss(cond, lr) for lr in LR_GRID}
    if any(v is None for v in losses.values()):
        return None
    return min(LR_GRID, key=lambda lr: losses[lr])


def grid_best_step(cond, lr):
    e = json.load(open(ROOT / "runs" / ft_run(cond, lr, GRID_SEED) / "val_eval.json"))
    return int(e["checkpoints"]["best_loss_all_"]["step"])


def build_jobs(phases):
    """[(name, kind, runs it needs, done-check path or None for training, argv)] in the order of `phases`."""
    py = sys.executable
    jobs = []

    def train_job(cond, lr, seed, iters=FT_ITERS):
        pe, extra = CONDITIONS[cond]
        src = tr_run(pe, seed)
        src_set = [] if pe == "rope" else ["--set", f"model.pos_encoding_type={pe}"]
        run = ft_run(cond, lr, seed, iters)
        jobs.append((run, "train", [src], None, [py, "train.py", "--config", "configs/final/TR.yaml", "--seed", str(seed),
                                                  "--run-name", run, "--init-from", src, "--set", f"training.learning_rate={lr}"]
                     + src_set + extra + ft_common(iters)))
        return run

    def ft_evals(cond, runs):
        for run in runs:
            eval_job(f"extrap:{run}", run, "extrapolation.json",
                     ["scripts/eval_extrapolation.py", "--run", run, "--lengths", "128", "256", "512", "1024", "--stride1", "128"])
            if cond == "pi512":
                eval_job(f"pi:{run}", run, "extrapolation_pi.json",
                         ["scripts/eval_extrapolation.py", "--run", run, "--lengths", "1024", "--interpolate"])

    def eval_job(name, run, out, argv):
        jobs.append((name, "eval", [run], f"runs/{run}/{out}", [py] + argv))

    for phase in phases:
        if phase == "grid":
            for cond in CONDITIONS:
                for lr in LR_GRID:
                    train_job(cond, lr, GRID_SEED)
        elif phase == "infer":
            for split in ("val", "test"):
                for name, (extra, part, lengths, splits) in INFER.items():
                    if split not in splits:
                        continue
                    for s in SEEDS:
                        run = tr_run("rope", s)
                        suffix = "_val" if split == "val" else ""
                        eval_job(f"{name}-{split}:{run}", run, f"extrapolation{part}{suffix}.json",
                                 ["scripts/eval_extrapolation.py", "--run", run, "--split", split, "--lengths", *lengths, *extra])
        elif phase == "ft":
            for cond in CONDITIONS:
                lr = chosen_lr(cond)
                if lr is None:
                    raise SystemExit(f"phase ft: the lr grid of {cond} has not finished")
                if grid_best_step(cond, lr) == 0:
                    # no lr improved on the starting validation loss: the selected checkpoint is the source model itself
                    print(f"phase ft: {cond} not continued (best checkpoint of the grid is iteration 0 at every lr)")
                    continue
                runs = [ft_run(cond, lr, GRID_SEED)] + [train_job(cond, lr, s) for s in SEEDS if s != GRID_SEED]
                ft_evals(cond, runs)
                if cond in ("pi512", "long512"):
                    eval_job(f"attn512:{runs[0]}", runs[0], "attention_W512/stats.json",
                             ["scripts/analyze_attention.py", "--run", runs[0], "--length", "512"])
        elif phase == "budget":
            for cond in CONDITIONS:
                lr = chosen_lr(cond)
                if lr is None or grid_best_step(cond, lr) == 0:
                    continue
                ft_evals(cond, [train_job(cond, lr, s, BUDGET_ITERS) for s in SEEDS])
        else:
            raise SystemExit(f"unknown phase {phase!r}")
    return jobs


def finished(job):
    name, kind, _, out, _ = job
    return is_done(name) if kind == "train" else (ROOT / out).exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="3")
    ap.add_argument("--phase", nargs="+", default=["grid", "infer"], choices=["grid", "infer", "ft", "budget"])
    ap.add_argument("--only", default=None, help="comma-separated job names to include")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.phase)
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only.split(",")]
    skipped = [j[0] for j in jobs if finished(j)]
    queue = [j for j in jobs if j[0] not in skipped]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[{datetime.now():%F %T}] GPUs {gpus}; phases {args.phase}; skipped (finished): {skipped}")
    if "ft" in args.phase:
        for cond in CONDITIONS:
            print(f"  {cond}: grid best val loss {[(lr, grid_val_loss(cond, lr)) for lr in LR_GRID]} -> lr {chosen_lr(cond)}")
    print("Queue:\n  " + "\n  ".join(f"{j[0]}: {' '.join(j[4][1:])}" for j in queue), flush=True)
    if args.dry_run or not queue:
        return

    status = ROOT / "runs" / "v5_status.jsonl"
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
            if finished(job):  # finished by an earlier launcher after this queue was built
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


if __name__ == "__main__":
    main()
