"""
Queue the v3 final runs (prompts/03_final_and_extensions.md, Part 1) on fixed GPUs, one run per GPU.

    nohup setsid python scripts/launch_v3.py --gpus 2,3 > runs/launch_v3.txt 2>&1 &
    python scripts/launch_v3.py --dry-run

Priority order: solo timing runs, block 1 (Table V, tie-rule lr), 1b (lowest-BPC lr), 2 (positional encodings),
3 (dropout ablation). A GPU is used only when it has no compute process of anyone else (so each run has its GPU
to itself). Finished runs (eval.json / val_eval.json with a full-context result) are skipped, so the launcher can be
restarted. Each run logs to runs/<run>/launcher.txt. Records: runs/v3_status.jsonl (one line per finished run);
runs/v3_status.json (snapshot: running, queued, finished, and an estimate of the remaining time once block 1 is done).
Failed runs are recorded, not retried.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_all import descendants, gpu_state, is_free  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
REUSED = {"TR-final_s42", "LSTM-final_s42", "LR-1-final_s42"}  # finished before this task, reused (see REPORT.md)
FC_BATCH = ["--set", "evaluation.full_context_batch_size=2048"]
BASE = {"TR": "configs/final/TR.yaml", "LSTM": "configs/final/LSTM.yaml", "LR-1": "configs/final/LR-1.yaml"}
BEST_LR = {"TR": "2.7e-3", "LSTM": "7.2e-3", "LR-1": "7.2e-3"}
PE_OTHER = ["learned", "sinusoidal", "hybrid", "alibi", "relative"]
TR_DROPOUT_KEYS = ["emb_dropout", "attn_dropout", "resid_dropout", "mlp_dropout"]
POLL_S = 20


def build_queue():
    """[(block, kind, run_name, config, extra args)]; kind groups runs of equal cost for the time estimate."""
    q = [("timing", f, f"timing_{f}", BASE[f], ["--seed", "42", "--tuning", "--epochs", "2000"]) for f in BASE]
    for s in SEEDS:
        for f in BASE:
            q.append(("1", f, f"{f}-final_s{s}", BASE[f], ["--seed", str(s)]))
    for s in SEEDS:
        for f in BASE:
            q.append(("1b", f, f"{f}-final_s{s}_bestlr", BASE[f], ["--seed", str(s), "--set", f"training.learning_rate={BEST_LR[f]}"]))
    for s in SEEDS:
        for pe in PE_OTHER:
            q.append(("2", "TR", f"TR-final-{pe}_s{s}", BASE["TR"], ["--seed", str(s), "--set", f"model.pos_encoding_type={pe}"]))
    for s in SEEDS:
        sets = [a for k in TR_DROPOUT_KEYS for a in ("--set", f"model.{k}=0.1")]
        q.append(("3", "TR", f"TR-final-d0.1_s{s}", BASE["TR"], ["--seed", str(s)] + sets))
    return [(b, k, n, c, e + FC_BATCH) for b, k, n, c, e in q]


def is_done(run_name):
    run_dir = ROOT / "runs" / run_name
    for fname in ("eval.json", "val_eval.json"):
        try:
            if "full_context" in json.load(open(run_dir / fname))["checkpoints"]["best_loss_all_"]:
                return True
        except Exception:
            pass
    return False


def estimate_remaining(records, queue, running, n_gpus):
    """Hours left: mean duration per kind from finished block-1 runs, applied to queued + running runs."""
    dur = {}
    for r in records:
        if r["block"] == "1" and r["complete"]:
            dur.setdefault(r["kind"], []).append(r["seconds"])
    if not dur:
        return None
    mean = {k: sum(v) / len(v) for k, v in dur.items()}
    fallback = max(mean.values())
    todo = sum(mean.get(k, fallback) for _, k, *_ in queue)
    todo += sum(max(0.0, mean.get(r["kind"], fallback) - (time.time() - r["t0"])) for r in running.values())
    return {"hours": round(todo / n_gpus / 3600, 2), "mean_minutes_per_run": {k: round(v / 60, 1) for k, v in mean.items()},
            "based_on": {k: len(v) for k, v in dur.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="2,3")
    ap.add_argument("--only", default=None, help="comma-separated run names to include")
    ap.add_argument("--blocks", default=None, help="comma-separated blocks to include (timing,1,1b,2,3)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    queue = build_queue()
    if args.blocks:
        queue = [q for q in queue if q[0] in args.blocks.split(",")]
    if args.only:
        queue = [q for q in queue if q[2] in args.only.split(",")]
    skipped = [q[2] for q in queue if q[2] in REUSED or is_done(q[2])]
    queue = [q for q in queue if q[2] not in skipped]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[{datetime.now():%F %T}] GPUs {gpus}; skipped (finished or reused): {skipped}")
    print("Queue:\n  " + "\n  ".join(f"[{b}] {n}: {c} {' '.join(e)}" for b, _, n, c, e in queue), flush=True)
    if args.dry_run or not queue:
        return

    status_jsonl, status_json = ROOT / "runs" / "v3_status.jsonl", ROOT / "runs" / "v3_status.json"
    records = [json.loads(line) for line in open(status_jsonl)] if status_jsonl.exists() else []
    running = {}
    block1 = {q[2] for q in queue if q[0] == "1"}
    while queue or running:
        state = gpu_state()
        for gpu, r in list(running.items()):
            ours = {r["proc"].pid, *descendants(r["proc"].pid)}
            r["foreign"].update(p for p in state[gpu]["pids"] if p not in ours)
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["log"].close()
            rec = {"block": r["block"], "kind": r["kind"], "run": r["name"], "gpu": gpu, "cmd": r["cmd"], "start": r["start"],
                   "end": f"{datetime.now():%F %T}", "seconds": round(time.time() - r["t0"], 1), "returncode": rc,
                   "complete": is_done(r["name"]), "foreign_gpu_pids_seen": sorted(r["foreign"])}
            records.append(rec)
            with open(status_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['end']}] finished {r['name']} on GPU {gpu}: rc={rc}, complete={rec['complete']}, "
                  f"{rec['seconds'] / 60:.1f} min", flush=True)
            del running[gpu]
        for gpu in gpus:
            if not queue or gpu in running or not is_free(state[gpu]):
                continue
            block, kind, name, cfg, extra = queue.pop(0)
            run_dir = ROOT / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "train.py", "--config", cfg, "--run-name", name] + extra
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            log = open(run_dir / "launcher.txt", "w")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = {"block": block, "kind": kind, "name": name, "cmd": " ".join(cmd), "proc": proc, "log": log,
                            "start": f"{datetime.now():%F %T}", "t0": time.time(), "foreign": set()}
            print(f"[{datetime.now():%F %T}] started {name} on GPU {gpu}: {' '.join(cmd)}", flush=True)
        block1_left = any(q[2] in block1 for q in queue) or any(r["name"] in block1 for r in running.values())
        snapshot = {
            "updated": f"{datetime.now():%F %T}",
            "running": {str(g): {"run": r["name"], "start": r["start"]} for g, r in running.items()},
            "queued": [q[2] for q in queue],
            "finished": [{"run": r["run"], "complete": r["complete"], "minutes": round(r["seconds"] / 60, 1)} for r in records],
            "remaining_estimate": None if block1_left else estimate_remaining(records, queue, running, len(gpus)),
        }
        json.dump(snapshot, open(status_json, "w"), indent=2)
        time.sleep(POLL_S)
    print(f"[{datetime.now():%F %T}] queue empty", flush=True)


if __name__ == "__main__":
    main()
