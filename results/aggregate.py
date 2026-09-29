"""
Rebuild every results table from results/raw/<run_name>/eval.json (copies of runs/*/eval.json).

    python results/aggregate.py            # writes results/*.csv|md|tex
    python results/aggregate.py --sync     # first copy runs/*/{eval,training_results,timing}.json
                                           # and runs/launch_status.jsonl into results/raw/

Aggregates are mean ± sample std (ddof=1) over the seeds found for each run group.
"""

import argparse
import csv
import json
import re
import shutil
import statistics
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"

TABLE_V_MODELS = ["TR2o", "TR8o", "LR-1", "LR-2"]
ABLATION = [("TR2o-legacy", "fixed dropout key (legacy), decay on all params"),
            ("TR2o-rngfix", "dropout key split every step, decay on all params"),
            ("TR2o", "dropout key split every step, decay mask")]
PAPER = {  # Table V of the report (old code, legacy32-style evaluation)
    "TR2o": dict(bpc=1.8468, acc=60.28, acc_last=64.08, acc_rare=39.27, acc_common=72.10, ece=0.0830, bleu=0.0159,
                 params=3_169_280, s_per_step=0.027),
    "TR8o": dict(bpc=1.8993, acc=59.50, acc_last=62.94, acc_rare=37.06, acc_common=71.77, ece=0.1322, bleu=0.0983,
                 params=3_238_640, s_per_step=0.023),
    "LR-1": dict(bpc=1.9198, acc=59.02, acc_last=62.53, acc_rare=37.60, acc_common=71.29, ece=0.0227, bleu=0.1541,
                 params=2_115_072, s_per_step=0.044),
    "LR-2": dict(bpc=1.8896, acc=59.67, acc_last=63.45, acc_rare=38.22, acc_common=71.82, ece=0.0260, bleu=0.1212,
                 params=3_177_052, s_per_step=0.069),
}
SANITY_TARGET, SANITY_TOL = PAPER["TR2o"]["bpc"], 0.05
SETUP_SLACK_S = 30  # upper bound on data loading + init between writing config.yaml and the first training step

# (key, header, getter, scale, decimals)
TABLE_V_COLS = [
    ("bpc", "BPC", lambda e: e["ckpt"]["full_context"]["bpc"], 1, 4),
    ("ppl", "PPL", lambda e: e["ckpt"]["full_context"]["perplexity"], 1, 4),
    ("acc", "Acc_all (%)", lambda e: e["ckpt"]["full_context"]["acc"], 100, 2),
    ("ece", "ECE", lambda e: e["ckpt"]["full_context"]["ece"], 1, 4),
    ("acc_rare", "Acc_rare (%)", lambda e: e["ckpt"]["full_context"]["acc_rare"], 100, 2),
    ("acc_common", "Acc_comm (%)", lambda e: e["ckpt"]["full_context"]["acc_common"], 100, 2),
    ("acc_last", "Acc_last@128 (%)", lambda e: e["ckpt"]["chunked128"]["acc_last"], 100, 2),
    ("bleu", "Self-BLEU-4", lambda e: e["ckpt"]["self_bleu_4"], 1, 4),
    ("params", "#Params", lambda e: e["num_params"], 1, 0),
    # solo runs only (GPU not shared during the timing window); see write_table_v / throughput.md
    ("s_per_step", "s/step (solo)", lambda e: e["throughput"]["s_per_step_median"] if e["solo"] else None, 1, 4),
]
MODE_COLS = {
    "legacy32": [("bpc", "BPC", 1, 4), ("acc", "Acc_all (%)", 100, 2), ("acc_last", "Acc_last@31 (%)", 100, 2),
                 ("acc_rare", "Acc_rare* (%)", 100, 2), ("acc_common", "Acc_comm* (%)", 100, 2)],
    "chunked128": [("bpc", "BPC", 1, 4), ("acc", "Acc_all (%)", 100, 2), ("acc_last", "Acc_last@127 (%)", 100, 2)],
    "full_context": [("bpc", "BPC", 1, 4), ("acc", "Acc_all (%)", 100, 2), ("ece", "ECE", 1, 4),
                     ("acc_rare", "Acc_rare (%)", 100, 2), ("acc_common", "Acc_comm (%)", 100, 2)],
}


def sync():
    RAW.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted((ROOT / "runs").iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("smoke_") or not (run_dir / "eval.json").exists():
            continue
        dst = RAW / run_dir.name
        dst.mkdir(exist_ok=True)
        for name in ("eval.json", "training_results.json", "timing.json", "config.yaml"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, dst / name)
        (dst / "timing_window.json").write_text(json.dumps(timing_window(run_dir), indent=2))
        hist = json.load(open(run_dir / "metrics_history.json"))
        (dst / "val_history.json").write_text(json.dumps({k: hist[k] for k in (
            "iteration", "loss_all_val", "acc", "acc_last", "bpc", "ece")}))
    for name in ("launch_status.jsonl", "smoke_status.jsonl"):
        if (ROOT / "runs" / name).exists():
            shutil.copy2(ROOT / "runs" / name, RAW / name)
    for p in RAW.rglob("*"):  # end files with a newline so pre-commit's end-of-file-fixer (CI) leaves them alone
        if p.is_file() and p.read_bytes()[-1:] != b"\n":
            with open(p, "a") as f:
                f.write("\n")


def timing_window(run_dir):
    """Wall-clock span of iterations [0, 2000] and the GPU processes scripts/gpu_sampler.py saw during it."""
    run = run_dir.name
    t_cfg = datetime.fromtimestamp((run_dir / "config.yaml").stat().st_mtime)
    m = re.search(r"^iteration 2000  time: ([0-9.]+) seconds", (run_dir / "train.log").read_text(), re.M)
    end = t_cfg + timedelta(seconds=SETUP_SLACK_S + float(m.group(1))) if m else None
    samples, foreign = 0, {}
    usage = ROOT / "runs" / "gpu_usage.jsonl"
    if end and usage.exists():
        for line in open(usage):
            rec = json.loads(line)
            t = datetime.fromisoformat(rec["t"])
            if t_cfg <= t <= end:
                samples += 1
                for p in rec["procs"]:
                    if f"--run-name {run} " not in p["cmd"] + " ":
                        foreign[p["pid"]] = p["cmd"][:100]
    return {"start": t_cfg.isoformat(timespec="seconds"), "end_conservative": end and end.isoformat(timespec="seconds"),
            "sampler_samples_in_window": samples, "foreign_processes_in_window": foreign,
            "shared_in_timing_window": (len(foreign) > 0) if samples else None}


def load_runs():
    runs = {}
    for p in sorted(RAW.glob("*/eval.json")):
        e = json.load(open(p))
        e["ckpt"] = e["checkpoints"]["best_loss_all_"]
        runs[p.parent.name] = e
    return runs


def load_status():
    """run name -> {"shared": True/False/None, "evidence": str}: was the GPU shared during the timing window?

    Sampler-based timing_window.json when it covers the window, else the hand-recorded evidence in
    results/raw_manual_gpu_sharing.json; None (unknown) otherwise, which counts as not solo."""
    manual = json.load(open(HERE / "raw_manual_gpu_sharing.json")) if (HERE / "raw_manual_gpu_sharing.json").exists() else {}
    status = {}
    for p in sorted(RAW.glob("*/eval.json")):
        run = p.parent.name
        tw_path = p.parent / "timing_window.json"
        tw = json.load(open(tw_path)) if tw_path.exists() else {}
        if tw.get("shared_in_timing_window") is not None:
            procs = "; ".join(sorted(set(tw["foreign_processes_in_window"].values()))) or "none"
            status[run] = {"shared": tw["shared_in_timing_window"],
                           "evidence": f"gpu_sampler: {tw['sampler_samples_in_window']} samples in window, foreign: {procs}"}
        elif run in manual:
            status[run] = {"shared": manual[run]["shared_in_timing_window"], "evidence": "manual: " + manual[run]["evidence"]}
        else:
            status[run] = {"shared": None, "evidence": "unknown"}
    return status


def group(runs, prefix):
    return {name: e for name, e in runs.items() if name.rsplit("_s", 1)[0] == prefix}


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def fmt(m, s, scale, dec, pm="±"):
    if m is None:
        return "n/a"
    if dec == 0:
        return f"{m * scale:,.0f}" + ("" if not s or s != s or s == 0 else f" {pm} {s * scale:,.0f}")
    return f"{m * scale:.{dec}f} {pm} {s * scale:.{dec}f}" if s == s else f"{m * scale:.{dec}f}"


def stat(entries, getter):
    vals = []
    for e in entries.values():
        try:
            vals.append(getter(e))
        except (KeyError, TypeError):
            vals.append(None)
    return mean_std(vals)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_table_v(runs, status):
    for name, e in runs.items():
        e["solo"] = status.get(name, {}).get("shared") is False
    header = ["ID", "seeds"] + [c[1] for c in TABLE_V_COLS] + ["solo runs"]
    rows_md, rows_csv, rows_tex = [], [], []
    for model in TABLE_V_MODELS:
        g = group(runs, model)
        cells_md, cells_csv, cells_tex = [], {"ID": model, "seeds": len(g)}, []
        for key, head, getter, scale, dec in TABLE_V_COLS:
            m, s, n = stat(g, getter)
            cells_md.append(fmt(m, s, scale, dec))
            cells_tex.append(fmt(m, s, scale, dec, pm=r"$\pm$").replace(",", "{,}"))
            cells_csv[f"{key}_mean"] = None if m is None else m * scale
            cells_csv[f"{key}_std"] = None if s is None else s * scale
        cells_csv["runs"] = " ".join(sorted(g))
        n_solo = sum(e["solo"] for e in g.values())
        cells_csv["n_solo_runs_for_s_per_step"] = n_solo
        rows_md.append([model, len(g)] + cells_md + [n_solo])
        rows_tex.append([model, str(len(g))] + cells_tex)
        rows_csv.append(cells_csv)

    with open(HERE / "table_v.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    fc_note = ""
    any_run = next(iter(runs.values()), None)
    if any_run:
        fc = any_run["ckpt"]["full_context"]
        groups = any_run["frequency_groups"]
        fc_note = (f"full_context: sliding window W={fc['window']}, stride S={fc['stride']} over the 5M-character test set "
                   f"(every target scored once; >= {fc['min_context_after_first_window']} chars of context after the first window). "
                   f"Rare = {{{', '.join(repr(g['char']) for g in groups['rare'])}}}, common = "
                   f"{{{', '.join(repr(g['char']) for g in groups['common'])}}} (bottom / top 20% of the 27 characters by "
                   f"training-split frequency).")
    with open(HERE / "table_v.md", "w") as f:
        f.write("# Table V (new code, re-run)\n\n")
        f.write("Test set, checkpoint with minimum validation loss (fixed 1024x128 validation set). Mean ± sample std over seeds.\n"
                "BPC, PPL, Acc_all, ECE, Acc_rare, Acc_comm: `full_context` evaluation. Acc_last: `chunked128` "
                "(non-overlapping 128-char blocks, position 127). s/step: median over iterations 1000-1999, averaged over "
                "the runs that had the GPU to themselves during that window ('solo runs'; see throughput.md).\n\n")
        f.write(fc_note + "\n\n")
        f.write(md_table(header, rows_md))
        f.write("\n## Paper Table V (old code, legacy32-style evaluation), for reference\n\n")
        f.write(md_table(["ID", "BPC", "Acc_all", "Acc_last", "Acc_rare", "Acc_comm", "ECE", "Self-BLEU-4", "#Params", "s/step (8xA5000)"],
                         [[m, p["bpc"], p["acc"], p["acc_last"], p["acc_rare"], p["acc_common"], p["ece"], p["bleu"],
                           f"{p['params']:,}", p["s_per_step"]] for m, p in PAPER.items()]))

    cols = "l" + "c" * (len(TABLE_V_COLS) + 1)
    tex_head = ["ID", "Seeds", "BPC $\\downarrow$", "PPL $\\downarrow$", "Acc$_{\\text{all}}$ (\\%) $\\uparrow$", "ECE $\\downarrow$",
                "Acc$_{\\text{rare}}$ (\\%) $\\uparrow$", "Acc$_{\\text{comm}}$ (\\%) $\\uparrow$",
                "Acc$_{\\text{last}}$ (\\%) $\\uparrow$", "Self-BLEU-4 $\\downarrow$", "\\#Params", "s/step"]
    with open(HERE / "table_v.tex", "w") as f:
        f.write("% Generated by results/aggregate.py -- do not edit by hand.\n")
        f.write("% BPC/PPL/Acc_all/ECE/Acc_rare/Acc_comm: full-context sliding-window evaluation; Acc_last: chunked128.\n")
        f.write("% Requires \\usepackage{booktabs} (and amsmath for \\text).\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{Test-set performance (mean $\\pm$ std over 3 seeds) at the checkpoint with minimum validation loss. "
                "BPC, perplexity, accuracies and ECE use sliding-window evaluation with a 128-character window and stride "
                f"{any_run['ckpt']['full_context']['stride'] if any_run else 1}; "
                "Acc$_{\\text{last}}$ is measured at position 127 of non-overlapping 128-character blocks.}\n")
        f.write("\\label{tab:table-v-rerun}\n")
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        f.write(" & ".join(tex_head) + " \\\\\n\\midrule\n")
        for r in rows_tex:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def mode_rows(runs, prefixes, mode):
    rows = []
    for prefix, desc in prefixes:
        g = group(runs, prefix)
        row = [prefix, desc, len(g)]
        for key, _, scale, dec in MODE_COLS[mode]:
            m, s, _ = stat(g, lambda e, k=key: e["ckpt"][mode][k])
            row.append(fmt(m, s, scale, dec))
        rows.append(row)
    return rows


def write_ablation(runs):
    with open(HERE / "ablation.md", "w") as f:
        f.write("# Ablation: dropout-RNG fix and weight-decay mask (TR2o config, 3 seeds each)\n\n")
        f.write("All arms: TR2o config (RoPE, lr 4e-4, dropout 0.1), fixed 1024x128 validation set for checkpoint selection, "
                "best-validation-loss checkpoint. Mean ± sample std over seeds.\n\n")
        for mode, desc in [("legacy32", "non-overlapping 32-char blocks (the paper's evaluation; *rare/common by per-batch frequency)"),
                           ("chunked128", "non-overlapping 128-char blocks"),
                           ("full_context", "sliding window W=128")]:
            f.write(f"## {mode}: {desc}\n\n")
            f.write(md_table(["arm", "training", "seeds"] + [c[1] for c in MODE_COLS[mode]], mode_rows(runs, ABLATION, mode)))
            f.write("\n")
        f.write("## All Table V models in all three evaluation modes\n\n")
        for mode in MODE_COLS:
            f.write(f"### {mode}\n\n")
            f.write(md_table(["ID", "", "seeds"] + [c[1] for c in MODE_COLS[mode]],
                             mode_rows(runs, [(m, "") for m in TABLE_V_MODELS], mode)))
            f.write("\n")


def write_sanity(runs):
    g = group(runs, "TR2o-legacy")
    vals = {n: e["ckpt"]["legacy32"]["bpc"] for n, e in sorted(g.items())}
    m, s, n = mean_std(list(vals.values()))
    passed = m is not None and abs(m - SANITY_TARGET) <= SANITY_TOL
    with open(HERE / "sanity_gate.md", "w") as f:
        f.write("# Sanity gate: TR2o-legacy, legacy32 test BPC vs paper TR2o\n\n")
        f.write(md_table(["run", "legacy32 BPC"], [[k, f"{v:.4f}"] for k, v in vals.items()]))
        if m is not None:
            f.write(f"\nMean {m:.4f} ± {s:.4f} (n={n}); paper TR2o {SANITY_TARGET}; |diff| = {abs(m - SANITY_TARGET):.4f}; "
                    f"tolerance {SANITY_TOL} -> **{'PASS' if passed else 'FAIL'}**\n")
    return m, s, n, passed


def write_throughput(runs, status):
    rows = []
    for model in TABLE_V_MODELS + ["TR2o-legacy", "TR2o-rngfix"]:
        g = group(runs, model)
        solo = {k: e for k, e in g.items() if status.get(k, {}).get("shared") is False}
        shared = sorted(k for k in g if k not in solo)
        m_all, s_all, n_all = stat(g, lambda e: e["throughput"]["s_per_step_median"])
        m_solo, s_solo, n_solo = stat(solo, lambda e: e["throughput"]["s_per_step_median"])
        tok_solo, _, _ = stat(solo, lambda e: e["throughput"]["tokens_per_s"])
        tok_all, _, _ = stat(g, lambda e: e["throughput"]["tokens_per_s"])
        dev = next((e["throughput"]["device"] for e in g.values() if e.get("throughput")), "n/a")
        rows.append([model, dev, n_solo, fmt(m_solo, s_solo, 1, 4), "n/a" if tok_solo is None else f"{tok_solo:.3g}",
                     n_all, fmt(m_all, s_all, 1, 4), "n/a" if tok_all is None else f"{tok_all:.3g}",
                     ", ".join(shared) or "-", PAPER.get(model, {}).get("s_per_step", "")])
    with open(HERE / "throughput.md", "w") as f:
        f.write("# Throughput (B=128, T=128)\n\n")
        f.write("s/step = median wall time of one training iteration (batch sampling + jitted step + block_until_ready) over "
                "iterations 1000-1999, excluding validation iterations; tokens/s = B*T / (s/step). 'solo' = no other process "
                "on the GPU during that timing window (scripts/gpu_sampler.py log, or hand-recorded evidence for the first "
                "runs; see the per-run list below).\n\n")
        f.write(md_table(["ID", "GPU", "solo runs", "s/step (solo)", "tokens/s (solo)", "all runs", "s/step (all)",
                          "tokens/s (all)", "runs that shared the GPU", "paper s/step (8xA5000)"], rows))
        f.write("\n## Per run\n\n")
        f.write(md_table(["run", "s/step", "shared GPU in timing window", "evidence"],
                         [[k, f"{e['throughput']['s_per_step_median']:.4f}", status.get(k, {}).get("shared"),
                           status.get(k, {}).get("evidence", "")] for k, e in sorted(runs.items())]))


def write_per_run(runs, status):
    fields = ["run", "seed", "legacy_dropout_rng", "decay_mask", "num_params", "best_step", "val_loss",
              "fc_bpc", "fc_ppl", "fc_acc", "fc_ece", "fc_acc_rare", "fc_acc_common", "fc_stride", "fc_seconds",
              "c128_bpc", "c128_acc", "c128_acc_last", "l32_bpc", "l32_acc", "l32_acc_last", "l32_acc_rare", "l32_acc_common",
              "self_bleu_4", "s_per_step", "tokens_per_s", "shared_gpu_in_timing_window", "git_commit"]
    with open(HERE / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, e in sorted(runs.items()):
            c, fc, c128, l32 = e["ckpt"], e["ckpt"]["full_context"], e["ckpt"]["chunked128"], e["ckpt"]["legacy32"]
            w.writerow({
                "run": name, "seed": e["seed"], "legacy_dropout_rng": e["legacy_dropout_rng"], "decay_mask": e["decay_mask"],
                "num_params": e["num_params"], "best_step": c["step"], "val_loss": c["val_loss"],
                "fc_bpc": fc["bpc"], "fc_ppl": fc["perplexity"], "fc_acc": fc["acc"], "fc_ece": fc["ece"],
                "fc_acc_rare": fc["acc_rare"], "fc_acc_common": fc["acc_common"], "fc_stride": fc["stride"], "fc_seconds": fc["seconds"],
                "c128_bpc": c128["bpc"], "c128_acc": c128["acc"], "c128_acc_last": c128["acc_last"],
                "l32_bpc": l32["bpc"], "l32_acc": l32["acc"], "l32_acc_last": l32["acc_last"],
                "l32_acc_rare": l32["acc_rare"], "l32_acc_common": l32["acc_common"],
                "self_bleu_4": c["self_bleu_4"], "s_per_step": (e.get("throughput") or {}).get("s_per_step_median"),
                "tokens_per_s": (e.get("throughput") or {}).get("tokens_per_s"),
                "shared_gpu_in_timing_window": status.get(name, {}).get("shared"),
                "git_commit": e.get("git_commit"),
            })


def write_claims(runs, status):
    """Numbers behind the paper's claims (a)-(f); the verdicts are written in REPORT.md."""
    def g(model, getter):
        return stat(group(runs, model), getter)

    def paired(m1, m2, getter):
        """Per-seed differences m1 - m2 (seed-matched)."""
        a = {e["seed"]: getter(e) for e in group(runs, m1).values()}
        b = {e["seed"]: getter(e) for e in group(runs, m2).values()}
        diffs = [a[s] - b[s] for s in sorted(set(a) & set(b))]
        return mean_std(diffs) + (diffs,)

    fc = lambda k: (lambda e: e["ckpt"]["full_context"][k])  # noqa: E731
    lines = ["# Numbers behind the paper's claims\n",
             "Mean ± sample std over seeds; 'diff' = seed-matched difference (same seed index), mean ± std over the 3 pairs.\n"]

    def block(title, metric_name, getter, models, pairs, scale=1, dec=4):
        lines.append(f"\n## {title}\n")
        rows = [[m, fmt(*g(m, getter)[:2], scale, dec)] for m in models]
        lines.append(md_table(["model", metric_name], rows))
        for a, b in pairs:
            m, s, n, diffs = paired(a, b, getter)
            lines.append(f"\n- {a} − {b}: {fmt(m, s, scale, dec)} (per seed: {', '.join(f'{d * scale:+.{dec}f}' for d in diffs)})\n")

    block("(a) TR2o vs parameter-matched LR-2: BPC (full_context)", "BPC", fc("bpc"), ["TR2o", "LR-2"], [("TR2o", "LR-2")])
    block("(a') same, legacy32 BPC", "BPC", lambda e: e["ckpt"]["legacy32"]["bpc"], ["TR2o", "LR-2"], [("TR2o", "LR-2")])
    block("(b) Calibration: ECE (full_context)", "ECE", fc("ece"), TABLE_V_MODELS,
          [("TR2o", "LR-1"), ("TR2o", "LR-2"), ("TR8o", "LR-1"), ("TR8o", "LR-2")])
    block("(c) Self-BLEU-4 (lower = more diverse)", "Self-BLEU-4", lambda e: e["ckpt"]["self_bleu_4"], TABLE_V_MODELS,
          [("TR2o", "TR8o"), ("TR2o", "LR-1"), ("TR2o", "LR-2")])
    block("(d) RoPE (TR2o) vs relative (TR8o): BPC (full_context)", "BPC", fc("bpc"), ["TR2o", "TR8o"], [("TR2o", "TR8o")])
    block("(d') same, legacy32 BPC", "BPC", lambda e: e["ckpt"]["legacy32"]["bpc"], ["TR2o", "TR8o"], [("TR2o", "TR8o")])
    for mode in ("legacy32", "chunked128", "full_context"):
        block(f"(e) Ablation, {mode} BPC", "BPC", lambda e, m=mode: e["ckpt"][m]["bpc"], [a for a, _ in ABLATION],
              [("TR2o-rngfix", "TR2o-legacy"), ("TR2o", "TR2o-rngfix"), ("TR2o", "TR2o-legacy")])
    block("(e') Evaluation context: TR2o BPC by mode", "BPC", fc("bpc"), ["TR2o"], [])
    for mode in ("legacy32", "chunked128", "full_context"):
        m, s, _ = g("TR2o", lambda e, mm=mode: e["ckpt"][mm]["bpc"])
        lines.append(f"- TR2o {mode}: {fmt(m, s, 1, 4)}\n")
    lines.append("\n## (f) Per-step speed: LSTM s/step ÷ Transformer s/step (solo runs only; see throughput.md)\n")
    solo = {k: e for k, e in runs.items() if status.get(k, {}).get("shared") is False}
    sps = {m: stat(group(solo, m), lambda e: e["throughput"]["s_per_step_median"])[0] for m in TABLE_V_MODELS}
    lines.append(md_table(["model", "solo runs", "s/step (solo mean)"],
                          [[m, len(group(solo, m)), "n/a" if sps[m] is None else f"{sps[m]:.4f}"] for m in TABLE_V_MODELS]))
    for t in ("TR2o", "TR8o"):
        for lm in ("LR-1", "LR-2"):
            if sps[t] and sps[lm]:
                lines.append(f"- {lm} / {t}: {sps[lm] / sps[t]:.2f}x\n")
    lines.append("\n## ECE provenance check: validation ECE at iteration 0 (untrained model) vs at the selected checkpoint\n")
    lines.append("The original Logger.print_metrics printed `self.ece[0]` (the iteration-0 value) at every evaluation.\n\n")
    rows = []
    for m in TABLE_V_MODELS + ["TR2o-legacy"]:
        it0, ck = [], []
        for name, e in group(runs, m).items():
            h = RAW / name / "val_history.json"
            if h.exists():
                it0.append(json.load(open(h))["ece"][0])
            ck.append(e["ckpt"]["ECE"])
        rows.append([m, ", ".join(f"{x:.4f}" for x in it0), fmt(*mean_std(ck)[:2], 1, 4),
                     fmt(*stat(group(runs, m), fc("ece"))[:2], 1, 4), PAPER.get(m, PAPER["TR2o"])["ece"]])
    lines.append(md_table(["model", "val ECE at iteration 0 (per seed)", "val ECE at best checkpoint", "test ECE (full_context)",
                           "paper Table V ECE"], rows))
    with open(HERE / "claims.md", "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.sync:
        sync()
    runs, status = load_runs(), load_status()
    print(f"{len(runs)} runs: {', '.join(sorted(runs))}")
    write_table_v(runs, status)
    write_ablation(runs)
    m, s, n, passed = write_sanity(runs)
    write_throughput(runs, status)
    write_per_run(runs, status)
    write_claims(runs, status)
    for name in ("table_v.md", "table_v.tex", "table_v.csv", "ablation.md", "sanity_gate.md", "throughput.md", "per_run.csv",
                 "claims.md"):  # exactly one final newline (pre-commit's end-of-file-fixer runs in CI)
        p = HERE / name
        p.write_text(p.read_text().rstrip("\n") + "\n")
    if m is not None:
        print(f"sanity gate: TR2o-legacy legacy32 BPC {m:.4f} ± {s:.4f} (n={n}) vs {SANITY_TARGET}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
