"""
Bounded hyperparameter search (validation split only): the grid, the decision rules, and the tables.

    python results/v3_tuning/aggregate_tuning.py           # tables from results/v3_tuning/raw/ -> tuning.csv, tables.md
    python results/v3_tuning/aggregate_tuning.py --sync    # first copy runs/tune_*/{val_eval.json,config.yaml} (and a
                                                           # divergence record for runs that hit a non-finite loss) to raw/
    python results/v3_tuning/aggregate_tuning.py --next    # print the runs the rules call for next (used by
                                                           # scripts/launch_tuning.py)

Every run: --tuning, seed 4212, 10,000 iterations, warmup 2,000, validation every 500 (the fixed 1024 x 128 set picks the
best-loss checkpoint). Score = full-context BPC of that checkpoint on the validation split (stride 8) from val_eval.json.
A diverged run (non-finite loss) scores +inf: worse than every finished run.

Decision rules (prompts/02_tuning.md Part 2), with the details this script fixes before any grid result exists:
  1. Grid per family: lr x dropout (GRID below), label smoothing eps = 0.05 as in the base configs.
  2. Edge rule: if the lowest-BPC grid run has the smallest or largest grid lr, add two lr values beyond that edge
     (edge / 1.5 and / 2.25, or x 1.5 and x 2.25) at that run's dropout. Once. The trigger is the plain lowest-BPC run
     (the question is whether the optimum lies outside the grid), not the run the tie rule picks.
  3. Tie rule over all eps = 0.05 runs of a family (grid + edge): the tied set is every run whose BPC is less than
     TIE_BPC = 0.02 above the lowest. From it pick the run closest to the current config (lr 4e-4, dropout 0.1), where
     distance = |lr steps| + |dropout steps|, counted as index distance in the family's sorted list of lr values tried
     and in DROPOUTS; equal distance -> lower BPC.
  4. Label smoothing: one run at the chosen (lr, dropout) with loss.ls_eps = 0.0. eps = 0 is chosen only if its BPC is
     at least TIE_BPC below the eps = 0.05 run; otherwise (tie or worse) eps stays 0.05.
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw"

SEED = 4212
EPOCHS, WARMUP, VAL_INTERVAL, VAL_STRIDE = 10_000, 2_000, 500, 8
N_VAL_TARGETS = 4_499_999
TIE_BPC = 0.02
EDGE_FACTORS = (1.5, 2.25)
DROPOUTS = (0.0, 0.1)
REF_LS_EPS = 0.05
FAMILIES = {
    "TR": dict(config="configs/TR2o.yaml", model_type="transformer", num_params=3_169_280,
               lrs=(4e-4, 6e-4, 8e-4, 1.2e-3), ref_lr=4e-4, ref_dropout=0.1,
               dropout_keys=("model.emb_dropout", "model.attn_dropout", "model.resid_dropout", "model.mlp_dropout")),
    "LSTM": dict(config="configs/LR-2.yaml", model_type="lstm", num_params=3_177_052,
                 lrs=(4e-4, 8e-4, 1.6e-3, 3.2e-3), ref_lr=4e-4, ref_dropout=0.1,
                 dropout_keys=("model.dropout",)),
}


# ----------------------------------------------------------------------------------------------
# Run specs
# ----------------------------------------------------------------------------------------------
def lr_tag(lr):
    """4e-4, 1.2e-3, 2.667e-4: at most 4 significant digits, no trailing zeros."""
    mant, exp = f"{lr:.3e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def round_lr(lr):
    return float(f"{lr:.4g}")


def run_name(family, lr, dropout, ls_eps=REF_LS_EPS):
    return f"tune_{family}_lr{lr_tag(lr)}_d{dropout}" + ("" if ls_eps == REF_LS_EPS else f"_ls{ls_eps:g}")


def spec(family, lr, dropout, phase, ls_eps=REF_LS_EPS):
    fam = FAMILIES[family]
    sets = [f"training.epochs={EPOCHS}", f"scheduler.warmup_iters={WARMUP}", f"training.validation_interval={VAL_INTERVAL}",
            f"training.learning_rate={lr:.4g}"] + [f"{k}={dropout}" for k in fam["dropout_keys"]]
    if ls_eps != REF_LS_EPS:
        sets.append(f"loss.ls_eps={ls_eps}")
    args = ["--config", fam["config"], "--seed", str(SEED), "--tuning"]
    for s in sets:
        args += ["--set", s]
    return dict(name=run_name(family, lr, dropout, ls_eps), family=family, phase=phase, lr=lr, dropout=dropout,
                ls_eps=ls_eps, args=args)


def grid_specs(family):
    return [spec(family, lr, d, "grid") for lr in FAMILIES[family]["lrs"] for d in DROPOUTS]


# ----------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------
DIVERGED_RE = re.compile(r"non-finite training loss (\S+) at iteration (\d+)")


def sync():
    """Copy runs/tune_*/{val_eval.json, config.yaml}, validation curves and runs/tuning_status.jsonl into raw/;
    record diverged runs."""
    RAW.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted((ROOT / "runs").glob("tune_*")):
        if not (run_dir / "config.yaml").exists():
            continue
        diverged = None
        for log in ("train.log", "launcher.log"):
            if (run_dir / log).exists():
                m = DIVERGED_RE.search((run_dir / log).read_text(errors="replace"))
                if m:
                    diverged = {"reason": "non-finite training loss", "loss": m.group(1), "iteration": int(m.group(2))}
                    break
        if not (run_dir / "val_eval.json").exists() and diverged is None:
            continue  # still running, or failed for another reason
        dst = RAW / run_dir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(run_dir / "config.yaml", dst / "config.yaml")
        if (run_dir / "val_eval.json").exists():
            shutil.copy2(run_dir / "val_eval.json", dst / "val_eval.json")
        if diverged:
            (dst / "diverged.json").write_text(json.dumps(diverged, indent=2) + "\n")
        if (run_dir / "metrics_history.json").exists():
            hist = json.load(open(run_dir / "metrics_history.json"))
            (dst / "val_history.json").write_text(json.dumps({k: hist[k] for k in ("iteration", "loss_all_val", "bpc")}) + "\n")
    if (ROOT / "runs" / "tuning_status.jsonl").exists():
        shutil.copy2(ROOT / "runs" / "tuning_status.jsonl", RAW / "tuning_status.jsonl")


def get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def family_of(cfg):
    return next(f for f, v in FAMILIES.items() if v["model_type"] == cfg["model"]["type"])


def load_row(run_dir):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    family = family_of(cfg)
    fam = FAMILIES[family]
    dropouts = {get(cfg, k) for k in fam["dropout_keys"]}
    assert len(dropouts) == 1, (run_dir.name, dropouts)
    dropout = float(dropouts.pop())
    lr = float(cfg["training"]["learning_rate"])
    ls_eps = float(cfg["loss"]["ls_eps"]) if cfg["loss"]["LABEL_SMOOTHING"] else 0.0
    # the protocol every run must follow
    checks = {"seed": (cfg["run"]["seed"], SEED), "tuning": (cfg["run"]["tuning"], True),
              "epochs": (cfg["training"]["epochs"], EPOCHS), "warmup": (cfg["scheduler"]["warmup_iters"], WARMUP),
              "validation_interval": (cfg["training"]["validation_interval"], VAL_INTERVAL),
              "tuning_val_stride": (cfg["evaluation"]["tuning_val_stride"], VAL_STRIDE),
              "batch_size": (cfg["training"]["batch_size"], 128), "sequence_length": (cfg["training"]["sequence_length"], 128),
              "config_file": (cfg["run"]["config_file"], fam["config"]), "legacy_dropout_rng": (cfg["run"]["legacy_dropout_rng"], False)}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    assert not bad, (run_dir.name, bad)
    assert run_dir.name == run_name(family, lr, dropout, ls_eps), (run_dir.name, run_name(family, lr, dropout, ls_eps))

    phase = "ls" if ls_eps != REF_LS_EPS else ("grid" if any(math.isclose(lr, g) for g in fam["lrs"]) else "edge")
    row = dict(run=run_dir.name, family=family, phase=phase, lr=lr, dropout=dropout, ls_eps=ls_eps,
               status=None, bpc=None, loss=None, perplexity=None, acc=None, ece=None, acc_rare=None, acc_common=None,
               n_scored=None, best_step=None, fixed_val_loss=None, fixed_val_bpc=None, diverged_at=None,
               train_time_s=None, git_commit=cfg["run"]["git_commit"])
    if (run_dir / "diverged.json").exists():
        row["status"] = "diverged"
        row["diverged_at"] = json.load(open(run_dir / "diverged.json"))["iteration"]
        return row
    ev = json.load(open(run_dir / "val_eval.json"))
    assert ev["split"] == "validation" and ev["num_params"] == fam["num_params"], run_dir.name
    ck = ev["checkpoints"]["best_loss_all_"]
    fc = ck["full_context"]
    assert fc["n_scored"] == N_VAL_TARGETS and fc["stride"] == VAL_STRIDE, (run_dir.name, fc["n_scored"], fc["stride"])
    row.update(status="done" if math.isfinite(fc["bpc"]) else "diverged", bpc=fc["bpc"], loss=fc["loss"],
               perplexity=fc["perplexity"], acc=fc["acc"], ece=fc["ece"], acc_rare=fc["acc_rare"], acc_common=fc["acc_common"],
               n_scored=fc["n_scored"], best_step=ck["step"], fixed_val_loss=ck["val_loss"], fixed_val_bpc=ck["val_bpc"],
               train_time_s=(ev.get("throughput") or {}).get("total_train_time_s"))
    return row


def load_rows():
    if not RAW.exists():
        return {}
    return {d.name: load_row(d) for d in sorted(RAW.iterdir()) if d.is_dir() and (d / "config.yaml").exists()}


# ----------------------------------------------------------------------------------------------
# Decision rules
# ----------------------------------------------------------------------------------------------
def score(row):
    return row["bpc"] if row and row["status"] == "done" else math.inf


def finished(rows, s):
    return s["name"] in rows


def argmin(rows_list):
    return min(rows_list, key=score)


def edge_specs(family, rows):
    """None while the grid is unfinished; [] if the lowest-BPC grid lr is interior; else the two extension runs."""
    grid = grid_specs(family)
    if not all(finished(rows, s) for s in grid):
        return None
    best = argmin([rows[s["name"]] for s in grid])
    lrs = FAMILIES[family]["lrs"]
    if math.isinf(score(best)):
        return []  # every grid run diverged: nothing to extend from
    if math.isclose(best["lr"], max(lrs)):
        new = [round_lr(max(lrs) * f) for f in EDGE_FACTORS]
    elif math.isclose(best["lr"], min(lrs)):
        new = [round_lr(min(lrs) / f) for f in EDGE_FACTORS]
    else:
        return []
    return [spec(family, lr, best["dropout"], "edge") for lr in new]


def choose(family, candidates):
    """Tie rule over finished eps = 0.05 runs. Returns (chosen row, info dict)."""
    fam = FAMILIES[family]
    best = argmin(candidates)
    best_score = score(best)
    tied = [r for r in candidates if score(r) - best_score < TIE_BPC] if math.isfinite(best_score) else [best]
    lrs = sorted({r["lr"] for r in candidates})
    ref_lr = next(lr for lr in lrs if math.isclose(lr, fam["ref_lr"]))

    def distance(r):
        return abs(lrs.index(r["lr"]) - lrs.index(ref_lr)) + abs(DROPOUTS.index(r["dropout"]) - DROPOUTS.index(fam["ref_dropout"]))

    chosen = min(tied, key=lambda r: (distance(r), score(r)))
    others = sorted(score(r) for r in candidates if r is not best)
    return chosen, dict(best=best, tied=sorted(tied, key=score), distance={r["run"]: distance(r) for r in tied},
                        runner_up_gap=(others[0] - best_score) if others else None)


def plan(rows):
    """Every run the rules call for given the results so far, per family in phase order. Each phase is listed only
    once the phase it depends on has finished (edge runs need the grid; the eps = 0 run needs grid + edge)."""
    out = []
    for family in FAMILIES:
        grid = grid_specs(family)
        out += grid
        ext = edge_specs(family, rows)
        if ext is None:
            continue
        out += ext
        if not all(finished(rows, s) for s in ext):
            continue
        chosen, _ = choose(family, [rows[s["name"]] for s in grid + ext])
        out.append(spec(family, chosen["lr"], chosen["dropout"], "ls", ls_eps=0.0))
    return out


def decide(rows):
    """Final decision per family, or None if its search is not finished."""
    out = {}
    for family in FAMILIES:
        grid = grid_specs(family)
        ext = edge_specs(family, rows)
        if ext is None or not all(finished(rows, s) for s in ext):
            out[family] = None
            continue
        main_runs = [rows[s["name"]] for s in grid + ext]
        chosen, info = choose(family, main_runs)
        ls_s = spec(family, chosen["lr"], chosen["dropout"], "ls", ls_eps=0.0)
        ls_row = rows.get(ls_s["name"])
        dec = dict(chosen=chosen, info=info, edge=ext, ls_row=ls_row, ls_eps=None, ls_rule=None)
        if ls_row is not None:
            diff = score(ls_row) - score(chosen)  # eps 0 minus eps 0.05
            if diff <= -TIE_BPC:
                dec["ls_eps"], dec["ls_rule"] = 0.0, f"eps 0 better by {-diff:.4f} >= {TIE_BPC}"
            elif diff < TIE_BPC:
                dec["ls_eps"], dec["ls_rule"] = REF_LS_EPS, f"tie (|diff| {abs(diff):.4f} < {TIE_BPC}): keep current {REF_LS_EPS}"
            else:
                dec["ls_eps"], dec["ls_rule"] = REF_LS_EPS, f"eps 0.05 better by {diff:.4f}"
            dec["ls_diff"] = diff
        out[family] = dec
    return out


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
CSV_COLS = ["run", "family", "phase", "lr", "dropout", "ls_eps", "status", "bpc", "loss", "perplexity", "acc", "ece",
            "acc_rare", "acc_common", "n_scored", "best_step", "fixed_val_loss", "fixed_val_bpc", "diverged_at",
            "train_time_s", "git_commit"]


def fmt_bpc(row):
    if row is None:
        return "—"
    if row["status"] == "diverged":
        return "diverged" + (f" (it {row['diverged_at']:,})" if row["diverged_at"] is not None else "")
    return f"{row['bpc']:.4f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_csv(rows):
    order = {"grid": 0, "edge": 1, "ls": 2}
    with open(HERE / "tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: (list(FAMILIES).index(r["family"]), order[r["phase"]], r["lr"], r["dropout"])):
            w.writerow({k: r[k] for k in CSV_COLS})


def grid_table(family, rows, dec):
    lrs = sorted({r["lr"] for r in rows.values() if r["family"] == family and r["phase"] != "ls"} | set(FAMILIES[family]["lrs"]))
    chosen = dec["chosen"]["run"] if dec else None
    best = dec["info"]["best"]["run"] if dec else None
    tied = {r["run"] for r in dec["info"]["tied"]} if dec else set()
    body = []
    for lr in lrs:
        cells = []
        for d in DROPOUTS:
            r = rows.get(run_name(family, lr, d))
            c = fmt_bpc(r)
            if r is not None and r["run"] in tied and r["run"] != chosen:
                c += " ~"
            if r is not None and r["run"] == best:
                c += " †"
            if r is not None and r["run"] == chosen:
                c = f"**{c}**"
            cells.append(c)
        label = lr_tag(lr)
        if not any(math.isclose(lr, g) for g in FAMILIES[family]["lrs"]):
            label += " (edge ext.)"
        if math.isclose(lr, FAMILIES[family]["ref_lr"]):
            label += " (current)"
        body.append([label] + cells)
    return md_table(["lr", "dropout 0.0", "dropout 0.1 (current)"], body)


def selection_text(family, dec):
    if dec is None:
        return f"{family}: search not finished."
    info, chosen = dec["info"], dec["chosen"]
    best = info["best"]
    lines = [f"- lowest BPC: `{best['run']}` {fmt_bpc(best)}"
             + (f"; next best +{info['runner_up_gap']:.4f}" if info["runner_up_gap"] is not None else "")]
    if dec["edge"]:
        side = "high" if dec["edge"][0]["lr"] > max(FAMILIES[family]["lrs"]) else "low"
        lines.append(f"- edge rule: lowest-BPC grid lr at the {side} edge -> added "
                     + ", ".join(f"`{s['name']}`" for s in dec["edge"]))
    else:
        lines.append("- edge rule: not triggered (lowest-BPC grid lr is interior)")
    lines.append(f"- tied set (< {TIE_BPC} BPC above the lowest): "
                 + ", ".join(f"`{r['run']}` {fmt_bpc(r)} (distance {info['distance'][r['run']]})" for r in info["tied"]))
    if len(info["tied"]) == 1:
        rule = "clear winner"
    elif chosen["run"] == best["run"]:
        rule = "tie rule (the lowest-BPC run is also the closest to the current config)"
    else:
        rule = "tie rule (closest to the current config among the tied set)"
    if dec["edge"] and any(chosen["run"] == s["name"] for s in dec["edge"]):
        rule += "; chosen run comes from the edge extension"
    lines.append(f"- chosen: lr {lr_tag(chosen['lr'])}, dropout {chosen['dropout']} -> `{chosen['run']}` {fmt_bpc(chosen)} [{rule}]")
    if dec["ls_row"] is not None:
        lines.append(f"- label smoothing: eps 0.05 {fmt_bpc(chosen)} vs eps 0 {fmt_bpc(dec['ls_row'])} "
                     f"-> eps {dec['ls_eps']} [{dec['ls_rule']}]")
    else:
        lines.append("- label smoothing: eps 0 run not finished")
    return "\n".join(lines)


def write_tables(rows):
    decs = decide(rows)
    parts = ["# v3 tuning tables (generated by aggregate_tuning.py; do not edit)", "",
             f"Score: full-context BPC on the validation split (stride {VAL_STRIDE}, {N_VAL_TARGETS:,} targets) of the "
             f"best-loss checkpoint. Seed {SEED}, {EPOCHS:,} iterations, warmup {WARMUP:,}. "
             f"**bold** = chosen, † = lowest BPC, ~ = tied with the lowest (< {TIE_BPC} BPC), — = not run.", ""]
    for family in FAMILIES:
        fam = FAMILIES[family]
        parts += [f"## {family} (`{fam['config']}`, label smoothing 0.05)", "", grid_table(family, rows, decs[family]), "",
                  selection_text(family, decs[family]), ""]
    parts += ["## Label smoothing", ""]
    ls_body = []
    for family, dec in decs.items():
        if dec is None or dec["ls_row"] is None:
            ls_body.append([family, "—", "—", "—", "—", "—", "—"])
            continue
        c = dec["chosen"]
        ls_body.append([family, lr_tag(c["lr"]), c["dropout"], fmt_bpc(c), fmt_bpc(dec["ls_row"]),
                        f"{dec['ls_diff']:+.4f}" if math.isfinite(dec["ls_diff"]) else "n/a", dec["ls_eps"]])
    parts += [md_table(["family", "lr", "dropout", "BPC eps 0.05", "BPC eps 0", "eps 0 − eps 0.05", "chosen eps"], ls_body), ""]
    parts += ["## Chosen values", ""]
    ch_body = []
    for family, dec in decs.items():
        if dec is None or dec["ls_eps"] is None:
            ch_body.append([family, "—", "—", "—"])
        else:
            ch_body.append([family, lr_tag(dec["chosen"]["lr"]), dec["chosen"]["dropout"], dec["ls_eps"]])
    parts += [md_table(["family", "learning_rate", "dropout (all)", "ls_eps"], ch_body), ""]
    parts += ["## All runs", ""]
    order = {"grid": 0, "edge": 1, "ls": 2}
    body = []
    for r in sorted(rows.values(), key=lambda r: (list(FAMILIES).index(r["family"]), order[r["phase"]], r["lr"], r["dropout"])):
        done = r["status"] == "done"
        body.append([f"`{r['run']}`", r["phase"], lr_tag(r["lr"]), r["dropout"], r["ls_eps"], fmt_bpc(r),
                     f"{100 * r['acc']:.2f}" if done else "", f"{r['ece']:.4f}" if done else "",
                     f"{r['best_step']:,}" if done else "", f"{r['fixed_val_loss']:.4f}" if done else ""])
    parts += [md_table(["run", "phase", "lr", "dropout", "ls_eps", "val BPC (full ctx)", "val acc (%)", "val ECE",
                        "best step", "fixed-set val loss"], body), ""]
    (HERE / "tables.md").write_text("\n".join(parts))
    return decs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--next", action="store_true", help="print the pending runs the rules call for (JSON lines)")
    args = ap.parse_args()
    if args.sync:
        sync()
    rows = load_rows()
    if args.next:
        for s in plan(rows):
            if not finished(rows, s):
                print(json.dumps(s))
        return
    write_csv(rows)
    decs = write_tables(rows)
    for family, dec in decs.items():
        print(selection_text(family, dec))
    print(f"{len(rows)} runs; wrote {HERE / 'tuning.csv'} and {HERE / 'tables.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
