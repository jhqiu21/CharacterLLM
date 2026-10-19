"""
Tables and figures of the v5 long-context experiments (results/v5/REPORT.md).

    python results/v5/aggregate_v5.py            # tables (csv/md/tex) and figures (pdf/png)
    python results/v5/aggregate_v5.py --sync     # first copy the v5 outputs from runs/ into results/v5/raw/

Reads results/v5/raw/ (this round), and results/v4/raw/ and results/v3/raw/ for the reference rows (plain RoPE, RoPE + PI and
ALiBi on the test split; never written). A per-run file is looked up in v5, then v4, then v3.
Aggregates are mean ± sample std (ddof 1) over seeds 42, 43, 44; seed-matched differences are mean ± std of the per-seed
differences. Sliding-window BPC uses stride 128 (every target after the first window sees >= W - 127 characters);
"W = 128, stride 1" is the full-context metric of the main results.
"""

import argparse
import csv
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAWS = [HERE / "raw", ROOT / "results" / "v4" / "raw", ROOT / "results" / "v3" / "raw"]
TABLES, FIGS = HERE / "tables", HERE / "figures"
SEEDS = [42, 43, 44]
LN2 = np.log(2)
LENGTHS = [128, 256, 512, 1024]
sys.path.insert(0, str(ROOT / "scripts"))
from launch_v5 import BUDGET_ITERS, CONDITIONS, LR_GRID, SE_NEAR, chosen_lr as launcher_chosen_lr, ft_run  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
ROPE, ALIBI = "#2a78d6", "#e87ba4"

# ---------------------------------------------------------------------------------------------- entities
# evaluation-only variants of the RoPE runs: key -> (label, file stem); test file = stem + ".json", validation = stem + "_val.json"
INFER = {"rope": ("RoPE", "extrapolation"), "rope_pi": ("RoPE + PI", "extrapolation_pi")}
INFER.update({f"local{n}": (f"RoPE + local attention {n}", f"extrapolation_local{n}") for n in (32, 64, 128)})
INFER.update({f"se{m}{n}": (f"RoPE + SelfExtend {m}, near {n}", f"extrapolation_se{m}{n}")
              for m in ("floor", "linear") for n in SE_NEAR})
FAMILY = {"local": [f"local{n}" for n in (32, 64, 128)], "sefloor": [f"sefloor{n}" for n in SE_NEAR],
          "selinear": [f"selinear{n}" for n in SE_NEAR]}
FAMILY_LABEL = {"local": "local attention", "sefloor": "SelfExtend (floor)", "selinear": "SelfExtend (linear)"}
SELECT_W = 1024  # variants are chosen by validation BPC at this window (fixed before the test results were read)

FT_LABEL = {"pi512": "RoPE + PI + 512-window fine-tuning", "long512": "RoPE + 512-window fine-tuning",
            "short128": "RoPE + 128-window fine-tuning", "alibi512": "ALiBi + 512-window fine-tuning"}
FT_COLORS = {"pi512": "#eb6834", "long512": "#1baf7a", "short128": "#eda100", "alibi512": ALIBI}


def base_run(pe, seed):
    return f"TR-final_s{seed}" if pe == "rope" else f"TR-final-{pe}_s{seed}"


# ---------------------------------------------------------------------------------------------- sync
FT_FILES = ["config.yaml", "val_eval.json", "timing.json", "training_results.json", "extrapolation.json", "extrapolation_pi.json"]


def sync():
    raw = RAWS[0]
    raw.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted((ROOT / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        if re.match(r"^ft(\d+k)?-", run_dir.name):
            if not (run_dir / "val_eval.json").exists():
                continue
            dst = raw / run_dir.name
            dst.mkdir(exist_ok=True)
            for name in FT_FILES:
                if (run_dir / name).exists():
                    shutil.copy2(run_dir / name, dst / name)
            h = json.load(open(run_dir / "metrics_history.json"))
            out = {k: h[k] for k in ("iteration", "loss_all_val", "acc", "bpc")}
            out["train_loss_all"] = h["loss_all_train"]
            (dst / "val_history.json").write_text(json.dumps(out))
            atts = list(run_dir.glob("attention_W*"))
        elif re.match(r"^TR-final_s\d+$", run_dir.name):  # v3 RoPE runs: only the v5 evaluation files
            files = [p for p in run_dir.glob("extrapolation*.json")
                     if p.name.endswith("_val.json") or re.search(r"_(local|se)", p.name)]
            if files:
                (raw / run_dir.name).mkdir(exist_ok=True)
            for p in files:
                shutil.copy2(p, raw / run_dir.name / p.name)
            atts = []
        else:
            continue
        for att in atts:
            (raw / run_dir.name / att.name).mkdir(parents=True, exist_ok=True)
            if (att / "stats.json").exists():
                shutil.copy2(att / "stats.json", raw / run_dir.name / att.name / "stats.json")
    for name in ("v5_status.jsonl", "launch_v5.txt", "launch_v5_b.txt"):
        if (ROOT / "runs" / name).exists():
            shutil.copy2(ROOT / "runs" / name, raw / name)
    for p in raw.rglob("*"):  # end text files with a newline so pre-commit's end-of-file-fixer leaves them alone
        if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml", ".txt") and p.read_bytes()[-1:] != b"\n":
            with open(p, "a") as f:
                f.write("\n")


# ---------------------------------------------------------------------------------------------- loading, statistics
def path(run, fname):
    for raw in RAWS:
        if (raw / run / fname).exists():
            return raw / run / fname
    return None


def load(run, fname):
    p = path(run, fname)
    return json.load(open(p)) if p else None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def fmt(m, s, dec=4, signed=False):
    if m is None:
        return "n/a"
    sign = "+" if signed else ""
    if s is None or s != s:
        return f"{m:{sign}.{dec}f}"
    return f"{m:{sign}.{dec}f} ± {s:.{dec}f}"


def sliding(run, fname, W):
    x = load(run, fname)
    if x is None or str(W) not in x["lengths"] or "na" in x["lengths"][str(W)]:
        return None
    return x["lengths"][str(W)]["sliding"]["bpc"]


def stride1_128(run):
    """W = 128, stride 1 BPC: the run's eval.json full-context value, or the --stride1 128 entry of extrapolation.json."""
    e = load(run, "eval.json")
    if e:
        return e["checkpoints"]["best_loss_all_"]["full_context"]["bpc"]
    x = load(run, "extrapolation.json")
    return x["stride1"]["128"]["bpc"] if x and "128" in x.get("stride1", {}) else None


def per_seed(fn):
    return {s: v for s in SEEDS if (v := fn(s)) is not None}


def stat(d):
    return mean_std(list(d.values()))


def diff(a, b):
    return mean_std([a[s] - b[s] for s in SEEDS if s in a and s in b])


# entity -> {seed: value} at (W, metric)
def infer_vals(key, W, split="test"):
    stem = INFER[key][1] + ("_val" if split == "val" else "")
    return per_seed(lambda s: sliding(base_run("rope", s), stem + ".json", W))


def alibi_vals(W):
    return per_seed(lambda s: sliding(base_run("alibi", s), "extrapolation.json", W))


def ft_vals(cond, W, fname="extrapolation.json", iters=1000):
    lr = chosen_lr(cond)
    return {} if not continued(cond) else per_seed(lambda s: sliding(ft_run(cond, lr, s, iters), fname, W))


def ft_stride1(cond, iters=1000):
    lr = chosen_lr(cond)
    return {} if not continued(cond) else per_seed(lambda s: stride1_128(ft_run(cond, lr, s, iters)))


def grid_val_bpc(cond, lr):
    e = load(ft_run(cond, lr, 42), "val_eval.json")
    return None if e is None else e["checkpoints"]["best_loss_all_"]["val_loss"] / LN2


def chosen_lr(cond):
    """Lowest best fixed-validation loss of the seed-42 grid (the launcher's rule), from the raw copies."""
    losses = {lr: grid_val_bpc(cond, lr) for lr in LR_GRID}
    if any(v is None for v in losses.values()):
        return None
    return min(LR_GRID, key=lambda lr: losses[lr])


def grid_best_step(cond, lr):
    e = load(ft_run(cond, lr, 42), "val_eval.json")
    return None if e is None else int(e["checkpoints"]["best_loss_all_"]["step"])


def continued(cond):
    """False when no lr of the grid improved on the starting validation loss (the launcher then ran no further seeds)."""
    lr = chosen_lr(cond)
    return lr is not None and grid_best_step(cond, lr) != 0


def selected(family):
    """Setting of a variant family with the lowest validation BPC at SELECT_W (mean over seeds)."""
    scores = {k: stat(infer_vals(k, SELECT_W, "val"))[0] for k in FAMILY[family]}
    scores = {k: v for k, v in scores.items() if v is not None}
    return min(scores, key=scores.get) if scores else None


# ---------------------------------------------------------------------------------------------- table writer
def tex_cell(c):
    s = str(c)
    for a, b in (("—", "--"), ("±", "$\\pm$"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("≥", "$\\geq$"), ("−", "$-$"),
                 ("Δ", "$\\Delta$"), ("†", "$^\\dagger$"), ("×", "$\\times$")):
        s = s.replace(a, b)
    return s


def write_table(stem, header, rows, caption, csv_rows=None, csv_header=None, notes="", align=None):
    """Markdown, CSV and a booktabs .tex (a bare tabular). A row "midrule" is a rule in the .tex and skipped elsewhere."""
    TABLES.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    tex = [f"% Generated by results/v5/aggregate_v5.py -- do not edit by hand. Requires booktabs. {caption}",
           "\\begin{tabular}{" + (align or "l" + "c" * (len(header) - 1)) + "}", "\\toprule",
           " & ".join(re.sub(r"\bW=(\d+)", r"$W{=}\1$", tex_cell(h)) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append("\\midrule" if r == "midrule" else " & ".join(tex_cell(c) for c in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / f"{stem}.tex").write_text("\n".join(tex) + "\n")
    md_rows = [r for r in rows if r != "midrule"]
    (TABLES / f"{stem}.md").write_text(
        f"{caption}\n\n" + "\n".join(md[:2] + ["| " + " | ".join(str(c) for c in r) + " |" for r in md_rows]) + "\n"
        + (f"\n{notes}\n" if notes else ""))
    with open(TABLES / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header or header)
        w.writerows(csv_rows or md_rows)


# ---------------------------------------------------------------------------------------------- tables
def write_inference(split):
    rows, csv_rows = [], []
    sel = {f: selected(f) for f in FAMILY}
    keys = ["rope", "rope_pi"] + [k for f in FAMILY for k in FAMILY[f]]
    for key in keys:
        if key in ("local32",) or key.startswith("se") and key.endswith("16") and key[2:].startswith("floor"):
            rows.append("midrule")
        vals = [stat(infer_vals(key, W, split)) for W in LENGTHS]
        if all(v[0] is None for v in vals):
            continue
        mark = " †" if key in sel.values() else ""
        rows.append([INFER[key][0] + mark] + [fmt(m, s) for m, s, _ in vals])
        csv_rows.append([key] + [m for m, _, _ in vals] + [s for _, s, _ in vals] + [n for _, _, n in vals])
    if split == "test":
        rows.append("midrule")
        vals = [stat(alibi_vals(W)) for W in LENGTHS]
        rows.append(["ALiBi (reference)"] + [fmt(m, s) for m, s, _ in vals])
        csv_rows.append(["alibi"] + [m for m, _, _ in vals] + [s for _, s, _ in vals] + [n for _, _, n in vals])
    ref = stat(per_seed(lambda s: stride1_128(base_run("rope", s))))
    note = f"Reference: RoPE at W=128 with stride 1 (every target sees 127 characters; test): {fmt(*ref[:2])}."
    split_name = "test" if split == "test" else "validation"
    write_table(f"inference_variants{'_val' if split == 'val' else ''}", ["Model"] + [f"W={W}" for W in LENGTHS], rows,
                f"Evaluation-only long-context variants of the RoPE model (no fine-tuning), {split_name} BPC, sliding window "
                f"with stride 128, mean ± std over 3 seeds. † = the setting of each family with the lowest validation BPC at "
                f"W={SELECT_W} (chosen on the validation split). Local attention: every layer sees the last N characters. "
                "SelfExtend: distances below the near window keep their exact RoPE positions, farther ones are compressed "
                "(floor: grouped positions as in SelfExtend; linear: continuous compression) with the smallest group size "
                "that keeps every relative position below 128.",
                csv_rows, ["key"] + [f"mean_W{W}" for W in LENGTHS] + [f"std_W{W}" for W in LENGTHS] + [f"n_W{W}" for W in LENGTHS],
                notes=note if split == "test" else "")
    return sel


def write_ft_grid():
    rows, csv_rows = [], []
    for cond in CONDITIONS:
        lr_best = chosen_lr(cond) if continued(cond) else None
        cells = []
        for lr in LR_GRID:
            v = grid_val_bpc(cond, lr)
            h = load(ft_run(cond, lr, 42), "val_history.json")
            start = h["bpc"][0] if h else None
            step = grid_best_step(cond, lr)
            cells.append("n/a" if v is None else f"{v:.4f}" + (" (it. 0)" if step == 0 else "") + (" *" if lr == lr_best else ""))
            csv_rows.append([cond, lr, v, start, step, lr == lr_best])
        h0 = load(ft_run(cond, LR_GRID[0], 42), "val_history.json")
        rows.append([FT_LABEL[cond], "n/a" if h0 is None else f"{h0['bpc'][0]:.4f}"] + cells)
    write_table("ft_lr_grid", ["Condition", "before fine-tuning"] + [f"lr {lr}" for lr in LR_GRID], rows,
                "Fine-tuning learning-rate grid, seed 42, 1,000 iterations (warmup 50, cosine to 0): best fixed-validation BPC "
                "(1024 validation windows of the condition's training length: 512, or 128 for the 128-window condition). "
                "* = chosen (lowest). 'before fine-tuning': the same validation set at iteration 0 (the first step has lr 0, so "
                "this is the source model). (it. 0) = the best checkpoint is iteration 0, i.e. fine-tuning never improved on "
                "the source model; 3e-3 was added after the first four (the PI condition's best was 1e-3, the upper edge).",
                csv_rows, ["condition", "lr", "best_val_bpc", "val_bpc_iter0", "best_step", "chosen"])


FT_ROWS = [  # (label, {seed: stride-1 W=128}, fn(W) -> {seed: BPC})
    ("rope", "RoPE (no fine-tuning)"), ("rope_pi", "RoPE + PI (no fine-tuning)"), ("long512", None), ("pi512", None),
    ("short128", None), ("alibi", "ALiBi (no fine-tuning)"), ("alibi512", None)]


def ft_row_values(key):
    if key in ("rope", "rope_pi"):
        s1 = per_seed(lambda s: stride1_128(base_run("rope", s)))
        return s1, {W: infer_vals(key, W) for W in LENGTHS}
    if key == "alibi":
        return per_seed(lambda s: stride1_128(base_run("alibi", s))), {W: alibi_vals(W) for W in LENGTHS}
    return ft_stride1(key), {W: ft_vals(key, W) for W in LENGTHS}


def write_finetune():
    rows, csv_rows = [], []
    for key, label in FT_ROWS:
        if key in CONDITIONS and not continued(key):
            rows.append([FT_LABEL[key], "any", "= RoPE (no fine-tuning): no lr improved the validation loss", "", "", "", ""])
            csv_rows.append([key, None] + [None] * 10)
            if key == "short128":
                rows.append("midrule")
            continue
        s1, by_w = ft_row_values(key)
        lr = chosen_lr(key) if key in CONDITIONS else None
        cells = [fmt(*stat(s1)[:2])] + [fmt(*stat(by_w[W])[:2]) for W in LENGTHS]
        rows.append([label or FT_LABEL[key], lr or "—"] + cells)
        csv_rows.append([key, lr] + [stat(s1)[0]] + [stat(by_w[W])[0] for W in LENGTHS] + [stat(s1)[1]]
                        + [stat(by_w[W])[1] for W in LENGTHS])
        if key == "pi512":
            v = stat(ft_vals("pi512", 1024, "extrapolation_pi.json"))
            rows.append(["  … + PI again at W=1024 (pos_scale 0.125)", lr or "—", "", "", "", "", fmt(*v[:2])])
            csv_rows.append(["pi512_pi1024", lr, None, None, None, None, v[0], None, None, None, None, v[1]])
        if key in ("short128", "rope_pi"):
            rows.append("midrule")
    write_table("finetune", ["Model", "lr", "W=128, stride 1"] + [f"W={W}" for W in LENGTHS], rows,
                "Long-window fine-tuning, test BPC, mean ± std over 3 seeds. Every fine-tuned run starts from the 50k checkpoint "
                "of the same seed and trains 1,000 more iterations of 16,384 characters (32 × 512 or 128 × 128; 2% of the "
                "pre-training characters); lr chosen on the validation split (seed 42). W columns: sliding window, stride 128. "
                "PI fine-tuning trains with pos_scale 0.25 (512 positions onto the 128 trained ones) and keeps it at every W.",
                csv_rows, ["key", "lr", "mean_s1_128"] + [f"mean_W{W}" for W in LENGTHS] + ["std_s1_128"] + [f"std_W{W}" for W in LENGTHS])


def write_differences(sel):
    """Seed-matched differences that answer the questions of the plan."""
    rope_s1 = per_seed(lambda s: stride1_128(base_run("rope", s)))
    alibi_s1 = per_seed(lambda s: stride1_128(base_run("alibi", s)))
    comps = [
        ("PI fine-tuning − 512-window fine-tuning (does PI add anything after adaptation?)",
         lambda W: ft_vals("pi512", W), lambda W: ft_vals("long512", W)),
        ("512-window fine-tuning − RoPE", lambda W: ft_vals("long512", W), lambda W: infer_vals("rope", W)),
        ("PI fine-tuning − RoPE + PI", lambda W: ft_vals("pi512", W), lambda W: infer_vals("rope_pi", W)),
        ("512-window fine-tuning − RoPE at W=128, stride 1", lambda W: ft_vals("long512", W), lambda W: rope_s1),
        ("PI fine-tuning − RoPE at W=128, stride 1", lambda W: ft_vals("pi512", W), lambda W: rope_s1),
        ("PI fine-tuning − ALiBi", lambda W: ft_vals("pi512", W), lambda W: alibi_vals(W)),
        ("512-window fine-tuning − ALiBi", lambda W: ft_vals("long512", W), lambda W: alibi_vals(W)),
        ("512-window fine-tuning − ALiBi + 512-window fine-tuning", lambda W: ft_vals("long512", W), lambda W: ft_vals("alibi512", W)),
        ("PI fine-tuning − ALiBi + 512-window fine-tuning", lambda W: ft_vals("pi512", W), lambda W: ft_vals("alibi512", W)),
        ("ALiBi + 512-window fine-tuning − ALiBi", lambda W: ft_vals("alibi512", W), lambda W: alibi_vals(W)),
    ]
    for fam, key in sel.items():
        if key:
            comps.append((f"{INFER[key][0]} − RoPE + PI", lambda W, k=key: infer_vals(k, W), lambda W: infer_vals("rope_pi", W)))
            comps.append((f"{INFER[key][0]} − RoPE at W=128, stride 1",
                          lambda W, k=key: infer_vals(k, W), lambda W: rope_s1))
    rows, csv_rows = [], []
    for label, fa, fb in comps:
        vals = [diff(fa(W), fb(W)) for W in LENGTHS]
        rows.append([label] + [fmt(m, s, signed=True) for m, s, _ in vals])
        csv_rows.append([label] + [m for m, _, _ in vals] + [s for _, s, _ in vals] + [n for _, _, n in vals])
    # stride-1 comparisons at W=128 (does fine-tuning cost short-context accuracy?)
    rows.append("midrule")
    for cond in [c for c in CONDITIONS if continued(c)]:
        ref = alibi_s1 if cond == "alibi512" else rope_s1
        m, s, n = diff(ft_stride1(cond), ref)
        rows.append([f"{FT_LABEL[cond]} − its base model, W=128 stride 1", fmt(m, s, signed=True), "", "", ""])
        csv_rows.append([f"{cond}_stride1_128", m, None, None, None, s, None, None, None, n, None, None, None])
    write_table("differences", ["Comparison (seed-matched)"] + [f"W={W}" for W in LENGTHS], rows,
                "Seed-matched test-BPC differences (mean ± std over 3 seeds; negative = the first model is better). Sliding "
                "window, stride 128, except the last block (W=128, stride 1).",
                csv_rows, ["comparison"] + [f"mean_W{W}" for W in LENGTHS] + [f"std_W{W}" for W in LENGTHS] + [f"n_W{W}" for W in LENGTHS],
                align="lcccc")


def write_budget():
    """1,000 vs 4,000 fine-tuning iterations (same lr), test BPC, and the seed-matched differences that matter."""
    rows, csv_rows = [], []
    for cond in [c for c in CONDITIONS if continued(c)]:
        for iters in (1000, BUDGET_ITERS):
            s1 = ft_stride1(cond, iters)
            by_w = {W: ft_vals(cond, W, iters=iters) for W in LENGTHS}
            rows.append([FT_LABEL[cond], f"{iters:,}", fmt(*stat(s1)[:2])] + [fmt(*stat(by_w[W])[:2]) for W in LENGTHS])
            csv_rows.append([cond, iters, stat(s1)[0]] + [stat(by_w[W])[0] for W in LENGTHS] + [stat(s1)[1]]
                            + [stat(by_w[W])[1] for W in LENGTHS])
        rows.append("midrule")
    vals = [stat(alibi_vals(W)) for W in LENGTHS]
    rows.append(["ALiBi (no fine-tuning)", "0", fmt(*stat(per_seed(lambda s: stride1_128(base_run("alibi", s))))[:2])]
                + [fmt(m, s) for m, s, _ in vals])
    rope_s1 = per_seed(lambda s: stride1_128(base_run("rope", s)))
    rows.append(["RoPE (no fine-tuning)", "0", fmt(*stat(rope_s1)[:2]), "", "", "", ""])
    write_table("budget", ["Model", "Iterations", "W=128, stride 1"] + [f"W={W}" for W in LENGTHS], rows,
                f"Fine-tuning budget: 1,000 vs {BUDGET_ITERS:,} iterations (16,384 characters each; warmup and validation "
                "interval 5% of the run; the lr chosen for 1,000), test BPC, mean ± std over 3 seeds; sliding window, stride 128.",
                csv_rows, ["condition", "iterations", "mean_s1_128"] + [f"mean_W{W}" for W in LENGTHS] + ["std_s1_128"]
                + [f"std_W{W}" for W in LENGTHS])
    B = BUDGET_ITERS
    comps = [
        (f"PI fine-tuning − 512-window fine-tuning, {B:,} it.", lambda W: ft_vals("pi512", W, iters=B),
         lambda W: ft_vals("long512", W, iters=B)),
        (f"512-window fine-tuning: {B:,} − 1,000 it.", lambda W: ft_vals("long512", W, iters=B), lambda W: ft_vals("long512", W)),
        (f"PI fine-tuning: {B:,} − 1,000 it.", lambda W: ft_vals("pi512", W, iters=B), lambda W: ft_vals("pi512", W)),
        (f"ALiBi fine-tuning: {B:,} − 1,000 it.", lambda W: ft_vals("alibi512", W, iters=B), lambda W: ft_vals("alibi512", W)),
        (f"512-window fine-tuning, {B:,} it. − ALiBi", lambda W: ft_vals("long512", W, iters=B), alibi_vals),
        (f"512-window fine-tuning, {B:,} it. − ALiBi fine-tuning, {B:,} it.", lambda W: ft_vals("long512", W, iters=B),
         lambda W: ft_vals("alibi512", W, iters=B)),
        (f"PI fine-tuning, {B:,} it. − ALiBi", lambda W: ft_vals("pi512", W, iters=B), alibi_vals),
        (f"512-window fine-tuning, {B:,} it. − RoPE at W=128, stride 1", lambda W: ft_vals("long512", W, iters=B), lambda W: rope_s1),
        (f"PI fine-tuning, {B:,} it. − RoPE at W=128, stride 1", lambda W: ft_vals("pi512", W, iters=B), lambda W: rope_s1),
    ]
    rows, csv_rows = [], []
    for label, fa, fb in comps:
        vals = [diff(fa(W), fb(W)) for W in LENGTHS]
        rows.append([label] + [fmt(m, s, signed=True) for m, s, _ in vals])
        csv_rows.append([label] + [m for m, _, _ in vals] + [s for _, s, _ in vals] + [n for _, _, n in vals])
    rows.append("midrule")
    for cond in [c for c in CONDITIONS if continued(c)]:
        ref = per_seed(lambda s: stride1_128(base_run("alibi" if cond == "alibi512" else "rope", s)))
        m, sd, n = diff(ft_stride1(cond, B), ref)
        rows.append([f"{FT_LABEL[cond]}, {B:,} it. − its base model, W=128 stride 1", fmt(m, sd, signed=True), "", "", ""])
        csv_rows.append([f"{cond}_{B}_stride1_128", m, None, None, None, sd, None, None, None, n, None, None, None])
    write_table("budget_differences", ["Comparison (seed-matched)"] + [f"W={W}" for W in LENGTHS], rows,
                "Seed-matched test-BPC differences for the fine-tuning budget (mean ± std over 3 seeds; negative = the first "
                "is better). Sliding window, stride 128, except the last block (W=128, stride 1).",
                csv_rows, ["comparison"] + [f"mean_W{W}" for W in LENGTHS] + [f"std_W{W}" for W in LENGTHS] + [f"n_W{W}" for W in LENGTHS],
                align="lcccc")


def write_paper_repair(sel):
    """Report table: every repair of RoPE beyond its training length next to ALiBi, test BPC (results/v5/REPORT.md, Summary)."""
    rope_s1 = per_seed(lambda s: stride1_128(base_run("rope", s)))
    alibi_s1 = per_seed(lambda s: stride1_128(base_run("alibi", s)))
    same = "= RoPE"
    entries = [  # (label, extra training characters, {seed: W=128 stride-1 BPC} or a string, fn(W) -> {seed: BPC})
        ("RoPE", "--", rope_s1, lambda W: infer_vals("rope", W)),
        ("RoPE + PI", "--", same, lambda W: infer_vals("rope_pi", W)),
        "midrule",
        ("RoPE + local attention 128", "--", same, lambda W: infer_vals(sel["local"], W)),
        ("RoPE + SelfExtend (near 64)", "--", same, lambda W: infer_vals(sel["sefloor"], W)),
        "midrule"]
    for iters in (1000, BUDGET_ITERS):
        chars = f"{iters * 16384 / 1e6:.1f}M"
        entries += [("RoPE + 512-window fine-tuning", chars, ft_stride1("long512", iters),
                     lambda W, i=iters: ft_vals("long512", W, iters=i)),
                    ("RoPE + PI + 512-window fine-tuning", chars, ft_stride1("pi512", iters),
                     lambda W, i=iters: ft_vals("pi512", W, iters=i)),
                    "midrule"]
    entries += [("ALiBi", "--", alibi_s1, alibi_vals),
                ("ALiBi + 512-window fine-tuning", f"{BUDGET_ITERS * 16384 / 1e6:.1f}M", ft_stride1("alibi512", BUDGET_ITERS),
                 lambda W: ft_vals("alibi512", W, iters=BUDGET_ITERS))]
    rows, csv_rows = [], []
    for e in entries:
        if e == "midrule":
            rows.append(e)
            continue
        label, chars, s1, fn = e
        s1_cell = s1 if isinstance(s1, str) else fmt(*stat(s1)[:2])
        vals = [stat(fn(W)) for W in LENGTHS[1:]]
        rows.append([label, chars, s1_cell] + [fmt(m, sd) for m, sd, _ in vals])
        csv_rows.append([label, chars, None if isinstance(s1, str) else stat(s1)[0]] + [m for m, _, _ in vals] + [sd for _, sd, _ in vals])
    write_table("paper_repair", ["Model", "Extra training", "W=128, stride 1"] + [f"W={W}" for W in LENGTHS[1:]], rows,
                "Repairs of RoPE beyond its training length, test BPC, mean ± std over 3 seeds. W=128, stride 1: full-context "
                "evaluation at the training length (= RoPE: the same model there). Other columns: sliding window, stride 128. "
                "Extra training: characters of 512-window fine-tuning after the 819.2M characters of pre-training.",
                csv_rows, ["model", "extra_training", "mean_s1_128"] + [f"mean_W{W}" for W in LENGTHS[1:]]
                + [f"std_W{W}" for W in LENGTHS[1:]], align="llcccc")


def per_position_curves(run_fname_pairs, W):
    curves = []
    for run, fname in run_fname_pairs:
        x = load(run, fname)
        if x and str(W) in x["lengths"] and "per_position" in x["lengths"][str(W)]:
            curves.append(np.asarray(x["lengths"][str(W)]["per_position"]["nll_per_position"]) / LN2)
    return np.asarray(curves)


def position_entities(sel):
    ents = [("RoPE", ROPE, "-", [(base_run("rope", s), "extrapolation.json") for s in SEEDS]),
            ("RoPE + PI", ROPE, "--", [(base_run("rope", s), "extrapolation_pi.json") for s in SEEDS])]
    for fam, key in sel.items():
        if key:
            ents.append((INFER[key][0], {"local": "#3d3c39", "sefloor": "#6250d6", "selinear": "#e34948"}[fam], ":",
                         [(base_run("rope", s), INFER[key][1] + ".json") for s in SEEDS]))
    for cond in ("long512", "pi512", "alibi512"):
        lr = chosen_lr(cond)
        if lr:
            ents.append((FT_LABEL[cond], FT_COLORS[cond], "-", [(ft_run(cond, lr, s), "extrapolation.json") for s in SEEDS]))
    ents.append(("ALiBi", ALIBI, "--", [(base_run("alibi", s), "extrapolation.json") for s in SEEDS]))
    return ents


def write_per_position(sel, W=1024):
    edges = [0, 64, 128, 192, 256, 448, 512, 576, 960, 1024]
    buckets = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if (a, b) not in ((256, 448), (576, 960))]
    rows, csv_rows = [], []
    for label, _, _, pairs in position_entities(sel):
        c = per_position_curves(pairs, W)
        if not len(c):
            continue
        m = c.mean(0)
        vals = [float(m[a:b].mean()) for a, b in buckets]
        rows.append([label] + [f"{v:.3f}" for v in vals])
        csv_rows.append([label, len(c)] + vals)
    write_table("per_position_W1024", ["Model"] + [f"{a}–{b - 1}" for a, b in buckets], rows,
                f"Per-position test BPC at W={W}: mean NLL over 2,048 non-overlapping test chunks (seed 0), averaged over the "
                "positions of each bucket and over 3 seeds.",
                csv_rows, ["model", "n_seeds"] + [f"bpc_{a}_{b - 1}" for a, b in buckets])


def attention_short_range():
    """Peak distance and profile mass on d = 0-2 per layer for the fine-tuned RoPE models at W = 512 vs the v4 references."""
    ents = [("RoPE, W=128", base_run("rope", 42), "attention_W128"),
            ("RoPE + PI, W=512", base_run("rope", 42), "attention_W512_pi"),
            ("RoPE, W=512", base_run("rope", 42), "attention_W512")]
    for cond in ("pi512", "long512"):
        lr = chosen_lr(cond)
        if lr:
            ents.append((f"{FT_LABEL[cond]}, W=512", ft_run(cond, lr, 42), "attention_W512"))
    out = {}
    for label, run, d in ents:
        st = load(run, f"{d}/stats.json")
        if st:
            out[label] = (np.asarray(st["distance_profile"]).mean(1), st)
    return out


def write_attention():
    rows, csv_rows = [], []
    for label, (prof, st) in attention_short_range().items():
        peak = prof[:, :17].argmax(1)
        near = prof[:, :3].sum(1)
        s = st["all_queries"]
        span = np.asarray(s["span_mean"]).mean(1)
        rows.append([label, " / ".join(str(int(v)) for v in peak), " / ".join(f"{v:.2f}" for v in near),
                     " / ".join(f"{v:.1f}" for v in span), f"{np.asarray(s['space_mass']).max():.3f}"])
        csv_rows.append([label] + peak.tolist() + near.tolist() + span.tolist() + [float(np.asarray(s["space_mass"]).max())])
    write_table("attention_short_range", ["Model", "Peak distance (chars)", "Profile summed over d = 0-2", "Effective span (chars)",
                                          "Best word-boundary head"], rows,
                "Short-range attention, seed 42, 512 test windows, all queries of the window: head-mean distance profile per "
                "layer (layers 0 / 1 / 2 / 3), its peak over distances 0-16, its sum over d = 0-2, the mean 90%-mass span, and "
                "the largest per-head weight on the most recent space. Reference rows from results/v4.",
                csv_rows, ["model"] + [f"peak_L{i}" for i in range(4)] + [f"mass_d0to2_L{i}" for i in range(4)]
                + [f"span_L{i}" for i in range(4)] + ["best_space_head"], align="lllll")


# ---------------------------------------------------------------------------------------------- figures
def style(ax, labelsize=10):
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=labelsize)


def save(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGS / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lines(ax, series):
    for label, color, ls, marker, vals_by_w in series:
        pts = [(W, *stat(vals_by_w(W))[:2]) for W in LENGTHS]
        pts = [p for p in pts if p[1] is not None]
        if not pts:
            continue
        W, m, s = zip(*pts)
        ax.errorbar(W, m, yerr=[0 if x != x else x for x in s], color=color, ls=ls, marker=marker, ms=5, lw=1.6, capsize=3,
                    mfc="white" if ls != "-" else color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS, [str(W) for W in LENGTHS])
    ax.set_xlabel("evaluation window W", color=INK, fontsize=11)
    ax.set_ylabel("test BPC (stride 128)", color=INK, fontsize=11)


def fig_bpc_vs_w(sel):
    rope_s1 = stat(per_seed(lambda s: stride1_128(base_run("rope", s))))[0]
    infer = [("RoPE", ROPE, "-", "o", lambda W: infer_vals("rope", W)),
             ("RoPE + PI", ROPE, "--", "o", lambda W: infer_vals("rope_pi", W))]
    for fam, key in sel.items():
        if key:
            infer.append((INFER[key][0], {"local": "#3d3c39", "sefloor": "#6250d6", "selinear": "#e34948"}[fam], "-", "s",
                          lambda W, k=key: infer_vals(k, W)))
    infer.append(("ALiBi", ALIBI, "-", "v", alibi_vals))
    panels = [(infer, "evaluation-only variants (no training)")]
    for iters in (1000, BUDGET_ITERS):
        ft = [(FT_LABEL[c], FT_COLORS[c], "-", "D", lambda W, c=c, i=iters: ft_vals(c, W, iters=i)) for c in CONDITIONS if continued(c)]
        ft.append(("ALiBi (no fine-tuning)", ALIBI, "--", "v", alibi_vals))
        panels.append((ft, f"fine-tuning, {iters:,} iterations"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), sharey=True)
    for ax, (series, title) in zip(axes, panels):
        plot_lines(ax, series)
        if rope_s1:
            ax.axhline(rope_s1, color=MUTED, lw=0.9, ls=":")
            ax.annotate("RoPE, W=128, stride 1", (1024, rope_s1), xytext=(0, 3), textcoords="offset points", fontsize=8,
                        color=MUTED, ha="right")
        ax.set_ylim(1.38, 1.66)
        ax.set_title(title, fontsize=11, color=INK)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    for ax in axes[1:]:
        ax.set_ylabel("")
    save(fig, "bpc_vs_W")


def fig_ft_curves():
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3.2 * len(CONDITIONS), 3.0), sharey=False)
    shades = ["#c8ddf7", "#9ec5f4", "#5a9be6", "#2a78d6", "#0b3f7e"]
    for ax, cond in zip(axes, CONDITIONS):
        for lr, c in zip(LR_GRID, shades):
            h = load(ft_run(cond, lr, 42), "val_history.json")
            if h:
                ax.plot(h["iteration"], h["bpc"], color=c, lw=1.5, label=f"lr {lr}" + (" (chosen)" if lr == chosen_lr(cond) else ""))
        ax.set_title(FT_LABEL[cond].replace(" fine-tuning", "\nfine-tuning"), fontsize=9.5, color=INK)
        if cond in ("pi512", "long512"):
            ax.set_ylim(1.45, 2.6)
        ax.set_xlabel("fine-tuning iteration", fontsize=10, color=INK)
        style(ax, labelsize=9)
        ax.legend(fontsize=7.5, frameon=False)
    axes[0].set_ylabel("validation BPC (fixed set)", fontsize=10, color=INK)
    save(fig, "ft_validation_curves")


def fig_per_position(sel, W=1024, smooth=16):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for label, color, ls, pairs in position_entities(sel):
        c = per_position_curves(pairs, W)
        if not len(c):
            continue
        m = np.convolve(c.mean(0), np.ones(smooth) / smooth, mode="valid")
        ax.plot(np.arange(len(m)) + smooth / 2, m, color=color, ls=ls, lw=1.4, label=label)
    for x in (128, 512):
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(1.3, 2.4)
    ax.set_xlabel("position in the window", fontsize=11, color=INK)
    ax.set_ylabel(f"test BPC (running mean over {smooth})", fontsize=11, color=INK)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "per_position_W1024")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.sync:
        sync()
    for cond in CONDITIONS:
        assert launcher_chosen_lr(cond) in (None, chosen_lr(cond)), cond  # raw copies agree with runs/
    sel = write_inference("val")
    write_inference("test")
    write_ft_grid()
    write_finetune()
    write_differences(sel)
    write_budget()
    write_per_position(sel)
    write_attention()
    write_paper_repair(sel)
    fig_bpc_vs_w(sel)
    fig_ft_curves()
    fig_per_position(sel)
    print("selected:", sel, "chosen lr:", {c: chosen_lr(c) for c in CONDITIONS})


if __name__ == "__main__":
    main()
