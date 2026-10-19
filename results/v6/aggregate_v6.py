"""
Tables and figures of the v6 capacity study (results/v6/REPORT.md).

    python results/v6/aggregate_v6.py            # tables (csv/md) and figures (pdf/png)
    python results/v6/aggregate_v6.py --sync     # first copy the v6 outputs from runs/ into results/v6/raw/

Reads results/v6/raw/ (this round) and, for the reference rows, results/v5/raw/ and results/v3/raw/ (never written).
A per-run file is looked up in v6, then v5, v4, v3. Aggregates are mean ± sample std (ddof 1) over the seeds present;
seed-matched differences are mean ± std of the per-seed differences.
Test and validation numbers are sliding-window metrics (utils/full_context_eval.py) with stride 128: every target
character of the split is scored exactly once, and every target after the first window sees >= W - 127 characters.
Acc_all = top-1 (argmax == target) over all targets; Acc_common / Acc_rare = top-1 on the 5 most / least frequent
training characters; Top-k = the target is among the k highest logits.
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
RAWS = [HERE / "raw"] + [ROOT / "results" / v / "raw" for v in ("v5", "v4", "v3")]
TABLES, FIGS = HERE / "tables", HERE / "figures"
sys.path.insert(0, str(ROOT / "scripts"))
from launch_v6 import (  # noqa: E402
    BRACKET_LR,
    EXTRA_SEEDS,
    FORMAL_ITERS,
    FORMAL_SEED,
    GAIN_RULE_PP,
    LR_GRID,
    MODELS,
    SEARCH_ITERS,
    SEARCH_SEED,
    V5_RUNS,
    formal_run,
    search_run,
)

SEEDS = [FORMAL_SEED] + EXTRA_SEEDS
LENGTHS = [128, 256, 512, 1024]
MAIN_W = 512
LN2 = np.log(2)
TOKENS_PER_UPDATE = 16384
TARGET_ACC = 80.0
TRAIN_MEAN_EVERY = 100  # training loss is stored as means over this many iterations
GAP_MARKS = [5000, 10000, 30000, FORMAL_ITERS - 1]  # iterations at which the validation - training gap is reported
METRICS = [("bpc", "BPC", 4, 1), ("acc", "Acc_all (%)", 2, 100), ("acc_common", "Acc_common (%)", 2, 100),
           ("acc_rare", "Acc_rare (%)", 2, 100), ("acc_top2", "Top-2 (%)", 2, 100), ("acc_top3", "Top-3 (%)", 2, 100),
           ("acc_top5", "Top-5 (%)", 2, 100)]
PARAMS = {"AL-3M": 3_169_280, "AL-14M": 14_204_928}
NAMES = {"AL-3M": "ALiBi-3.2M", "AL-14M": "ALiBi-14.2M"}  # the report's names of the two models (run names AL-3M / AL-14M)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"AL-3M": "#2a78d6", "AL-14M": "#eb6834", "v5-alibi": "#1baf7a", "v5-rope": "#eda100"}

# test-set entities: key -> (label, run name of a seed, file with top-k, fallback file without top-k)
ENTITIES = {
    "AL-3M": ("ALiBi-3.2M (512, from scratch)", lambda s: formal_run("AL-3M", s), "extrapolation.json", None),
    "AL-14M": ("ALiBi-14.2M (512, from scratch)", lambda s: formal_run("AL-14M", s), "extrapolation.json", None),
    "v5-alibi": ("Transformer (ALiBi) + 512-window fine-tuning", lambda s: f"ft4k-alibi512_lr3e-5_s{s}",
                 "extrapolation_v6.json", "extrapolation.json"),
    "v5-rope": ("Transformer (RoPE) + 512-window fine-tuning", lambda s: f"ft4k-long512_lr3e-4_s{s}",
                "extrapolation_v6.json", "extrapolation.json"),
    "v3-alibi": ("Transformer (ALiBi), 128 only", lambda s: f"TR-final-alibi_s{s}", "extrapolation_v6.json", "extrapolation.json"),
}


# ---------------------------------------------------------------------------------------------- sync
RUN_FILES = ["config.yaml", "val_eval.json", "timing.json", "training_results.json", "extrapolation.json", "extrapolation_val.json",
             "errors_W512.json"]


def sync():
    raw = RAWS[0]
    raw.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted((ROOT / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        dst = raw / run_dir.name
        if re.match(r"^v6(search)?-", run_dir.name):
            if not (run_dir / "val_eval.json").exists():
                continue
            dst.mkdir(exist_ok=True)
            for name in RUN_FILES:
                if (run_dir / name).exists():
                    shutil.copy2(run_dir / name, dst / name)
            h = json.load(open(run_dir / "metrics_history.json"))
            out = {k: h[k] for k in ("iteration", "loss_all_val", "acc", "bpc")}
            tr = np.asarray(h["loss_all_train"], np.float64)
            n = len(tr) // TRAIN_MEAN_EVERY * TRAIN_MEAN_EVERY
            out["train_mean_every"] = TRAIN_MEAN_EVERY
            out["train_loss_all_mean"] = [float(v) for v in tr[:n].reshape(-1, TRAIN_MEAN_EVERY).mean(1)]
            (dst / "val_history.json").write_text(json.dumps(out))
        elif run_dir.name in V5_RUNS:
            for name in ("extrapolation_v6.json", "errors_W512.json"):
                if (run_dir / name).exists():
                    dst.mkdir(exist_ok=True)
                    shutil.copy2(run_dir / name, dst / name)
    for name in ("v6_status.jsonl", "launch_v6.txt", "launch_v6_b.txt"):
        if (ROOT / "runs" / name).exists():
            shutil.copy2(ROOT / "runs" / name, raw / name)
    for p in raw.rglob("*"):  # end text files with a newline so pre-commit's end-of-file-fixer leaves them alone
        if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml", ".txt") and p.read_bytes()[-1:] != b"\n":
            with open(p, "a") as f:
                f.write("\n")


# ---------------------------------------------------------------------------------------------- loading, statistics
def load(run, fname):
    for raw in RAWS:
        if (raw / run / fname).exists():
            return json.load(open(raw / run / fname))
    return None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def fmt(m, s=None, dec=4, signed=False):
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
    return x["lengths"][str(W)]["sliding"]


def entity_metric(key, W, metric, split="test"):
    """{seed: value} of `metric` at window W."""
    _, run_of, fname, fallback = ENTITIES[key]
    if split == "val":
        fname, fallback = "extrapolation_val.json", None
    out = {}
    for s in SEEDS:
        e = sliding(run_of(s), fname, W)
        if (e is None or metric not in e) and fallback:
            e = sliding(run_of(s), fallback, W)
        if e is not None and metric in e:
            out[s] = e[metric]
    return out


def load_yaml(run):
    import yaml

    for raw in RAWS:
        if (raw / run / "config.yaml").exists():
            return yaml.safe_load(open(raw / run / "config.yaml"))
    return None


def selected_val(run):
    """(acc, bpc, step, checkpoint) of the selected checkpoint on the whole validation split (val_eval.json)."""
    e = load(run, "val_eval.json")
    if e is None:
        return None
    prefix = e.get("selected_checkpoint", "best_loss_all_")
    c = e["checkpoints"][prefix]
    return c["full_context"]["acc"], c["full_context"]["bpc"], int(c["step"]), prefix


def chosen_lr(model):
    vals = {lr: selected_val(search_run(model, lr)) for lr in LR_GRID}
    if any(v is None for v in vals.values()):
        return None
    return max(LR_GRID, key=lambda lr: (vals[lr][0], -vals[lr][1]))


def paired(a, b):
    """mean ± std of a[s] - b[s] over the common seeds, and the per-seed differences."""
    d = {s: a[s] - b[s] for s in SEEDS if s in a and s in b}
    m, sd, n = mean_std(list(d.values()))
    return m, sd, n, d


def status_records():
    p = RAWS[0] / "v6_status.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


# ---------------------------------------------------------------------------------------------- table writer
def tex_cell(c):
    """LaTeX for a table cell; the accuracies get their math names, as in results/v4."""
    s = str(c)
    for x, y in (("—", "--"), ("±", "$\\pm$"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("≥", "$\\geq$"), ("−", "$-$"),
                 ("Δ", "$\\Delta$"), ("×", "$\\times$"), ("←", "$\\leftarrow$"), (" †", "$^\\dagger$"), ("†", "$^\\dagger$"),
                 (" *", "$^*$")):
        s = s.replace(x, y)
    return re.sub(r"Acc\\_(all|comm|rare)", r"$\\mathrm{Acc}_{\\text{\1}}$", s)


def tex_header(h):
    """LaTeX for a Markdown header ($W{=}512$, as in results/v4)."""
    return re.sub(r"\bW=(\d+)", r"$W{=}\1$", tex_cell(h))


def write_table(stem, header, rows, caption, notes="", csv_rows=None, csv_header=None, align=None):
    """Markdown, CSV and a booktabs .tex (a bare tabular). A row "midrule" is a rule in the .tex and skipped elsewhere."""
    TABLES.mkdir(parents=True, exist_ok=True)
    md_rows = [r for r in rows if r != "midrule"]
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in md_rows]
    (TABLES / f"{stem}.md").write_text(f"{caption}\n\n" + "\n".join(md) + "\n" + (f"\n{notes}\n" if notes else ""))
    with open(TABLES / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header or header)
        w.writerows(csv_rows or md_rows)
    tex = [f"% Generated by results/v6/aggregate_v6.py -- do not edit by hand. Requires booktabs. {caption}",
           "\\begin{tabular}{" + (align or "l" + "c" * (len(header) - 1)) + "}", "\\toprule",
           " & ".join(tex_header(h) for h in header) + " \\\\", "\\midrule"]
    tex += ["\\midrule" if r == "midrule" else " & ".join(tex_cell(c) for c in r) + " \\\\" for r in rows]
    (TABLES / f"{stem}.tex").write_text("\n".join(tex + ["\\bottomrule", "\\end{tabular}"]) + "\n")


def pct(m, s=None, signed=False):
    return fmt(None if m is None else 100 * m, None if s is None else 100 * s, 2, signed)


# ---------------------------------------------------------------------------------------------- tables
def write_reuse_check():
    """The v6 re-evaluation of the v5 / v3 runs must reproduce their BPC and Acc_all (it only adds top-k)."""
    rows, worst = [], {"bpc": 0.0, "acc": 0.0}
    for run in V5_RUNS:
        new, old = load(run, "extrapolation_v6.json"), load(run, "extrapolation.json")
        if new is None or old is None:
            continue
        for W in LENGTHS:
            a, b = new["lengths"][str(W)]["sliding"], old["lengths"][str(W)]["sliding"]
            assert a["n_scored"] == b["n_scored"] == 4_999_999 and a["stride"] == b["stride"] == 128
            d = {k: abs(a[k] - b[k]) for k in worst}
            worst = {k: max(worst[k], d[k]) for k in worst}
            rows.append([run, W, f"{b['bpc']:.6f}", f"{a['bpc']:.6f}", f"{100 * b['acc']:.4f}", f"{100 * a['acc']:.4f}",
                         f"{d['bpc']:.1e}", f"{100 * d['acc']:.1e}"])
    write_table("reuse_check", ["run", "W", "BPC (v5/v3 file)", "BPC (v6 re-eval)", "Acc (v5/v3)", "Acc (v6)", "|ΔBPC|", "|ΔAcc| (pp)"],
                rows, "Re-evaluation of the v5 / v3 runs for top-k: BPC and Acc_all against the existing files (test, stride 128).",
                f"Largest differences: BPC {worst['bpc']:.1e}, Acc_all {100 * worst['acc']:.1e} pp.")
    return worst


def write_metrics_at(W, keys, stem, caption, split="test"):
    rows, csv_rows = [], []
    for key in keys:
        vals = [mean_std(list(entity_metric(key, W, m, split).values())) for m, *_ in METRICS]
        if vals[0][0] is None:
            continue
        rows.append([ENTITIES[key][0], vals[0][2]] + [fmt(None if m is None else m * sc, None if s is None else s * sc, dec)
                                                       for (m, s, _), (_, _, dec, sc) in zip(vals, METRICS)])
        csv_rows.append([key, vals[0][2]] + [v[0] for v in vals] + [v[1] for v in vals])
    write_table(stem, ["Model", "Seeds"] + [lab for _, lab, *_ in METRICS], rows, caption,
                csv_rows=csv_rows, csv_header=["key", "n_seeds"] + [m for m, *_ in METRICS] + [f"{m}_std" for m, *_ in METRICS])
    return rows


def write_by_w(split="test"):
    keys = ["AL-3M", "AL-14M"] + (["v5-alibi", "v5-rope", "v3-alibi"] if split == "test" else [])
    rows, csv_rows = [], []
    for key in keys:
        for metric, lab, dec, sc in METRICS[:2]:
            vals = [mean_std(list(entity_metric(key, W, metric, split).values())) for W in LENGTHS]
            if all(v[0] is None for v in vals):
                continue
            rows.append([ENTITIES[key][0], lab] + [fmt(None if m is None else m * sc, None if s is None else s * sc, dec)
                                                    for m, s, _ in vals])
            csv_rows.append([key, metric] + [v[0] for v in vals] + [v[1] for v in vals] + [v[2] for v in vals])
    write_table(f"{split}_by_W", ["Model", "Metric"] + [f"W={W}" for W in LENGTHS], rows,
                f"{'Test' if split == 'test' else 'Validation'} BPC and Acc_all at each window (stride 128), mean ± std over seeds.",
                csv_rows=csv_rows, csv_header=["key", "metric"] + [f"W{W}" for W in LENGTHS] + [f"W{W}_std" for W in LENGTHS]
                + [f"W{W}_n" for W in LENGTHS])


def write_search():
    rows = []
    for model in MODELS:
        best = chosen_lr(model)
        for lr in LR_GRID + [BRACKET_LR]:
            v = selected_val(search_run(model, lr))
            if v is None:
                if lr != BRACKET_LR:
                    rows.append([NAMES[model], lr, "running", "", "", ""])
                continue
            acc, bpc, step, _ = v
            loss_ckpt = load(search_run(model, lr), "val_eval.json")["checkpoints"]["best_loss_all_"]
            mark = " ← chosen" if lr == best else " (post-hoc bracket check)" if lr == BRACKET_LR else ""
            rows.append([NAMES[model], lr + mark, f"{100 * acc:.3f}", f"{bpc:.4f}", step,
                         f"{100 * loss_ckpt['full_context']['acc']:.3f}"])
    write_table("search", ["Model", "lr", "val Acc_all (%)", "val BPC", "selected step", "Acc_all of the best-loss checkpoint (%)"],
                rows, f"Learning-rate search: {SEARCH_ITERS:,} iterations ({SEARCH_ITERS * TOKENS_PER_UPDATE / 1e6:.2f}M training "
                f"characters), seed {SEARCH_SEED}, warmup 2,000, cosine to 0. Whole validation split, W = 512, stride 128, at the "
                "checkpoint selected by fixed-validation accuracy (BPC tie-break). Chosen: highest val Acc_all, ties -> lower BPC.")


def formal_val(model):
    """{seed: (acc, bpc, step)}"""
    out = {}
    for s in SEEDS:
        v = selected_val(formal_run(model, s))
        if v is not None:
            out[s] = v[:3]
    return out


def write_formal_val():
    rows = []
    v3, v14 = formal_val("AL-3M"), formal_val("AL-14M")
    for s in SEEDS:
        if s in v3 or s in v14:
            a = v3.get(s, (None, None, None))
            b = v14.get(s, (None, None, None))
            rows.append([s, pct(a[0]), fmt(a[1]), a[2], pct(b[0]), fmt(b[1]), b[2],
                         pct(b[0] - a[0], signed=True) if a[0] is not None and b[0] is not None else "",
                         fmt(b[1] - a[1], signed=True) if a[1] is not None and b[1] is not None else ""])
    accd = paired({s: v[0] for s, v in v14.items()}, {s: v[0] for s, v in v3.items()})
    bpcd = paired({s: v[1] for s, v in v14.items()}, {s: v[1] for s, v in v3.items()})
    if accd[2] > 1:
        m3, s3, _ = mean_std([v[0] for v in v3.values()])
        m14, s14, _ = mean_std([v[0] for v in v14.values()])
        b3, sb3, _ = mean_std([v[1] for v in v3.values()])
        b14, sb14, _ = mean_std([v[1] for v in v14.values()])
        rows.append(["mean ± std", pct(m3, s3), fmt(b3, sb3), "", pct(m14, s14), fmt(b14, sb14), "",
                     pct(accd[0], accd[1], True), fmt(bpcd[0], bpcd[1], signed=True)])
    gain = None
    if FORMAL_SEED in v3 and FORMAL_SEED in v14:
        g = 100 * (v14[FORMAL_SEED][0] - v3[FORMAL_SEED][0])
        gain = g >= GAIN_RULE_PP and v14[FORMAL_SEED][1] < v3[FORMAL_SEED][1]
    note = (f"Rule for seeds 43 / 44 (fixed before the formal runs): seed-42 AL-14M − AL-3M validation Acc_all ≥ {GAIN_RULE_PP} pp "
            f"and lower validation BPC. Seed 42: {'met' if gain else 'not met' if gain is not None else 'pending'}.")
    write_table("formal_val", ["seed", "ALiBi-3.2M val Acc_all (%)", "ALiBi-3.2M val BPC", "step", "ALiBi-14.2M val Acc_all (%)",
                               "ALiBi-14.2M val BPC",
                               "step", "Δ Acc (pp)", "Δ BPC"], rows,
                f"Formal runs ({FORMAL_ITERS:,} iterations, {FORMAL_ITERS * TOKENS_PER_UPDATE:,} training characters): whole "
                "validation split, W = 512, stride 128, at the selected checkpoint.", note)
    return gain, accd, bpcd


def write_paired_test():
    rows, csv_rows = [], []
    for W in LENGTHS:
        for metric, lab, dec, sc in METRICS:
            m, sd, n, d = paired(entity_metric("AL-14M", W, metric), entity_metric("AL-3M", W, metric))
            if not n:
                continue
            signs = "".join("+" if v > 0 else "−" if v < 0 else "0" for v in d.values())
            rows.append([W, lab, fmt(None if m is None else m * sc, None if sd is None else sd * sc, dec, True), n,
                         " / ".join(f"{v * sc:+.{dec}f}" for v in d.values()), signs])
            csv_rows.append([W, metric, m, sd, n] + [d.get(s) for s in SEEDS])
    write_table("paired_test", ["W", "metric", "ALiBi-14.2M − ALiBi-3.2M", "seeds", "per seed (42 / 43 / 44)", "signs"], rows,
                "Seed-matched test differences ALiBi-14.2M − ALiBi-3.2M (stride 128).",
                csv_rows=csv_rows, csv_header=["W", "metric", "mean", "std", "n"] + [f"s{s}" for s in SEEDS])


def history(run):
    return load(run, "val_history.json")


def at_iteration(h, it, key):
    i = h["iteration"].index(it) if it in h["iteration"] else None
    return None if i is None else h[key][i]


def train_bpc_near(h, it, span=1000):
    """Mean training BPC (unweighted loss on the training batches) over the `span` iterations ending at `it`."""
    k = TRAIN_MEAN_EVERY
    tr = h["train_loss_all_mean"][max(0, (it + 1 - span) // k):(it + 1) // k]
    return float(np.mean(tr) / LN2) if tr else None


def write_curves():
    """Fixed-validation-set trajectory and the train / validation gap of every formal run."""
    marks = [5000, 10000, 20000, 30000, 40000, FORMAL_ITERS - 1]
    rows, csv_rows = [], []
    for model in MODELS:
        for s in SEEDS:
            h = history(formal_run(model, s))
            if h is None:
                continue
            accs = [at_iteration(h, it, "acc") for it in marks]
            bpcs = [at_iteration(h, it, "bpc") for it in marks]
            gaps = [b - train_bpc_near(h, it) for it, b in zip(marks, bpcs)]
            best_bpc_it = h["iteration"][int(np.argmin(h["bpc"]))]
            rows.append([model, s] + [pct(a) for a in accs] + [fmt(bpcs[-1]), best_bpc_it]
                        + [fmt(gaps[marks.index(it)], signed=True) for it in GAP_MARKS])
            csv_rows.append([model, s] + accs + bpcs + [best_bpc_it] + gaps)
    write_table("curves", ["Model", "Seed"] + [f"val Acc @ {it // 1000 if it % 1000 == 0 else round((it + 1) / 1000)}k (%)" for it in marks]
                + ["val BPC @ 50k", "iteration of lowest val BPC"] + [f"val − train BPC @ {it // 1000}k" for it in GAP_MARKS[:-1]]
                + ["val − train BPC @ 50k"], rows,
                "Fixed validation set (1,024 windows × 512, all positions) during the formal runs. Train BPC = mean unweighted loss "
                "on the training batches over the last 1,000 iterations (no dropout, so the same model as at validation). "
                "Passes over the 85.5M-character training split: 0.96 at 5k iterations, 1.9 at 10k, 9.6 at 50k.",
                csv_rows=csv_rows, csv_header=["Model", "Seed"] + [f"acc_{it}" for it in marks] + [f"bpc_{it}" for it in marks]
                + ["best_bpc_iteration"] + [f"gap_{it}" for it in marks])


def write_budget_scaling():
    """2 x 2: model size x training characters (search run at the chosen lr = 10k iterations; formal seed 42 = 50k)."""
    rows, cells = [], {}
    for model in MODELS:
        lr = chosen_lr(model)
        short = selected_val(search_run(model, lr)) if lr else None
        long = selected_val(formal_run(model, FORMAL_SEED))
        cells[model] = (short, long)
        rows.append([NAMES[model], lr,
                     pct(short[0]) if short else "n/a", fmt(short[1]) if short else "n/a",
                     pct(long[0]) if long else "n/a", fmt(long[1]) if long else "n/a",
                     pct(long[0] - short[0], signed=True) if short and long else "n/a"])
    s3, l3 = cells["AL-3M"]
    s14, l14 = cells["AL-14M"]
    if s3 and s14:
        rows.append(["ALiBi-14.2M − ALiBi-3.2M", "", pct(s14[0] - s3[0], signed=True), fmt(s14[1] - s3[1], signed=True),
                     pct(l14[0] - l3[0], signed=True) if l3 and l14 else "n/a",
                     fmt(l14[1] - l3[1], signed=True) if l3 and l14 else "n/a", ""])
    short_m, long_m = SEARCH_ITERS * TOKENS_PER_UPDATE / 1e6, FORMAL_ITERS * TOKENS_PER_UPDATE / 1e6
    write_table("budget_scaling", ["Model", "lr", f"val Acc_all, {short_m:.0f}M chars (%)", "val BPC",
                                   f"val Acc_all, {long_m:.0f}M chars (%)", "val BPC", "5× characters: Δ Acc (pp)"],
                rows, "Model size × training characters, whole validation split (W = 512, stride 128), selected checkpoints. "
                f"{short_m:.0f}M = the search run at the chosen lr (seed {SEARCH_SEED}, complete "
                f"{SEARCH_ITERS:,}-iteration schedule); {long_m:.0f}M = the formal run, seed {FORMAL_SEED}.",
                "The training split holds 85,500,000 characters, so 164M characters are 1.9 passes and 819M are 9.6 passes over it.")


def write_paper_search():
    """Report table: the learning-rate search of both models (validation split, W = 512), compact."""
    rows = []
    for model in MODELS:
        best = chosen_lr(model)
        for lr in LR_GRID + [BRACKET_LR]:
            v = selected_val(search_run(model, lr))
            if v is None:
                continue
            mark = " *" if lr == best else " †" if lr == BRACKET_LR else ""
            rows.append([NAMES[model], lr + mark, f"{100 * v[0]:.2f}", f"{v[1]:.4f}"])
        rows.append("midrule")
    write_table("paper_search", ["Model", "Peak lr", "Acc_all (%)", "BPC"], rows[:-1],
                f"Learning-rate search, validation split (W = {MAIN_W}, stride 128), {SEARCH_ITERS:,} iterations, seed {SEARCH_SEED}, "
                "selected checkpoint. * = chosen; † = post-hoc check, not used for the choice.", align="llcc")


def write_paper_capacity():
    """Report table: the best v5 models and the two v6 models at W = 512 (test), and the seed-matched capacity gain."""
    cols = [("bpc", 4, 1), ("acc", 2, 100), ("acc_common", 2, 100), ("acc_rare", 2, 100), ("acc_top2", 2, 100), ("acc_top5", 2, 100)]
    training = {"v5-alibi": "128, then 512 (+65.5M)", "v5-rope": "128, then 512 (+65.5M)",
                "AL-3M": "512 from scratch", "AL-14M": "512 from scratch"}
    rows, csv_rows = [], []
    for key in ("v5-alibi", "v5-rope", "midrule", "AL-3M", "AL-14M"):
        if key == "midrule":
            rows.append(key)
            continue
        vals = [mean_std(list(entity_metric(key, MAIN_W, m).values())) for m, _, _ in cols]
        if vals[0][0] is None:
            continue
        params = PARAMS.get(key, PARAMS["AL-3M"])
        label = NAMES.get(key, ENTITIES[key][0])
        rows.append([label, f"{params / 1e6:.1f}M", training[key]]
                    + [fmt(m * sc, sd * sc, dec) for (m, sd, _), (_, dec, sc) in zip(vals, cols)])
        csv_rows.append([key, params, training[key]] + [v[0] for v in vals] + [v[1] for v in vals])
    diffs = [paired(entity_metric("AL-14M", MAIN_W, m), entity_metric("AL-3M", MAIN_W, m)) for m, _, _ in cols]
    if all(d[2] for d in diffs):
        rows.append("midrule")
        rows.append(["ALiBi-14.2M − ALiBi-3.2M", "", "seed-matched"]
                    + [fmt(m * sc, sd * sc, dec, signed=True) for (m, sd, _, _), (_, dec, sc) in zip(diffs, cols)])
        csv_rows.append(["diff", None, None] + [d[0] for d in diffs] + [d[1] for d in diffs])
    write_table("paper_capacity", ["Model", "Params", "Training windows", "BPC", "Acc_all (%)", "Acc_comm (%)", "Acc_rare (%)",
                                   "Top-2 (%)", "Top-5 (%)"], rows,
                f"Test set, W = {MAIN_W}, stride 128, mean ± std over 3 seeds; every model trained on 819.2M characters "
                "(+ 65.5M of fine-tuning for the first two). First two rows: best-validation-loss checkpoints; ALiBi rows: "
                "best-validation-accuracy checkpoints.", csv_rows=csv_rows,
                csv_header=["key", "params", "training"] + [m for m, _, _ in cols] + [f"{m}_std" for m, _, _ in cols], align="lllcccccc")


def write_paper_errors():
    """Report table: test top-1 accuracy by the target's position in its word, ALiBi-3.2M vs ALiBi-14.2M (W = 512)."""
    errs = {k: [e for s in SEEDS if (e := load(ENTITIES[k][1](s), "errors_W512.json"))] for k in ("AL-3M", "AL-14M")}
    if not all(errs.values()):
        return
    rows, csv_rows = [], []
    for g in list(errs["AL-14M"][0]["groups"]) + ["all"]:
        def field(k, f):
            return mean_std([e[f] if g == "all" else e["groups"][g][f] for e in errs[k]])[0]
        share = 1.0 if g == "all" else field("AL-14M", "share_of_targets")
        err_share = 1.0 if g == "all" else field("AL-14M", "share_of_errors")
        a3, a14, b14 = field("AL-3M", "acc"), field("AL-14M", "acc"), field("AL-14M", "bpc")
        rows.append([g, f"{100 * share:.1f}", f"{100 * a3:.2f}", f"{100 * a14:.2f}", f"{100 * (a14 - a3):+.2f}", f"{b14:.3f}",
                     f"{100 * err_share:.1f}"])
        csv_rows.append([g, share, a3, a14, a14 - a3, b14, err_share])
    write_table("paper_errors", ["Target", "Targets (%)", "Acc, 3.2M (%)", "Acc, 14.2M (%)", "Δ", "BPC, 14.2M",
                                 "Errors, 14.2M (%)"], rows,
                f"Test set, W = {MAIN_W}, stride 128: top-1 accuracy by the target's position in its word, mean over 3 seeds "
                "(std ≤ 0.06 points). 'space': the target is a space (the end of a word); '1st letter': the previous "
                "character is a space.", csv_rows=csv_rows,
                csv_header=["group", "share_of_targets", "acc_3M", "acc_14M", "diff", "bpc_14M", "share_of_errors_14M"])


def write_selection():
    """Accuracy-selected (best_acc_bpc_) vs loss-selected (best_loss_all_) checkpoint of every formal run."""
    rows = []
    for model in MODELS:
        for s in SEEDS:
            e = load(formal_run(model, s), "val_eval.json")
            if e is None:
                continue
            a, b = e["checkpoints"]["best_acc_bpc_"], e["checkpoints"]["best_loss_all_"]
            rows.append([formal_run(model, s), a["step"], b["step"], pct(a["val_acc"]), pct(b["val_acc"]),
                         pct(a["full_context"]["acc"]), pct(b["full_context"]["acc"]), fmt(a["full_context"]["bpc"]),
                         fmt(b["full_context"]["bpc"])])
    write_table("selection", ["run", "step (acc)", "step (loss)", "fixed-set acc, acc ckpt (%)", "fixed-set acc, loss ckpt (%)",
                              "val-split acc, acc ckpt (%)", "val-split acc, loss ckpt (%)", "val-split BPC, acc ckpt",
                              "val-split BPC, loss ckpt"], rows,
                "Checkpoint selected by fixed-validation accuracy (BPC tie-break; used for every v6 number) vs the one with the "
                "lowest fixed-validation loss. Fixed set: 1,024 × 512 windows (selection); val split: whole validation split, W = 512, "
                "stride 128.")


def write_errors():
    """Test top-1 accuracy by the target's position in its word (scripts/analyze_errors.py, W = 512, stride 128)."""
    rows, csv_rows = [], []
    for key in ("AL-3M", "AL-14M", "v5-alibi", "v5-rope"):
        runs = {s: load(ENTITIES[key][1](s), "errors_W512.json") for s in SEEDS}
        runs = {s: e for s, e in runs.items() if e}
        if not runs:
            continue
        names = list(next(iter(runs.values()))["groups"])
        for g in names:
            col = {f: mean_std([e["groups"][g][f] for e in runs.values()]) for f in ("share_of_targets", "acc", "bpc", "share_of_errors")}
            rows.append([ENTITIES[key][0], g, len(runs), pct(*col["share_of_targets"][:2]), pct(*col["acc"][:2]),
                         fmt(*col["bpc"][:2]), pct(*col["share_of_errors"][:2])])
            csv_rows.append([key, g, len(runs)] + [col[f][0] for f in col] + [col[f][1] for f in col])
        acc, bpc = (mean_std([e[f] for e in runs.values()]) for f in ("acc", "bpc"))
        rows.append([ENTITIES[key][0], "all", len(runs), "100", pct(*acc[:2]), fmt(*bpc[:2]), "100"])
    write_table("errors_by_word_position", ["Model", "Target", "Seeds", "share of targets (%)", "top-1 acc (%)", "BPC",
                                            "share of all errors (%)"], rows,
                "Test set, W = 512, stride 128: top-1 accuracy by the target's position in its word ('space' = the target is a "
                "space, i.e. the end of a word; '1st letter' = the first letter of a word).",
                csv_rows=csv_rows, csv_header=["key", "group", "n_seeds", "share_of_targets", "acc", "bpc", "share_of_errors",
                                               "share_of_targets_std", "acc_std", "bpc_std", "share_of_errors_std"])


def write_gpu_time():
    recs = status_records()
    rows, total = [], 0.0
    by_phase = {}
    for r in recs:
        by_phase.setdefault(r["phase"], []).append(r)
    for phase, rs in by_phase.items():
        sec = sum(r["seconds"] for r in rs)
        total += sec
        rows.append([phase, len(rs), sum(r["complete"] for r in rs), f"{sec / 3600:.2f}",
                     sorted({p for r in rs for p in r["foreign_gpu_pids_seen"]}) or "none"])
    rows.append(["total", len(recs), sum(r["complete"] for r in recs), f"{total / 3600:.2f}", ""])
    for gpu in sorted({r["gpu"] for r in recs}):
        rs = [r for r in recs if r["gpu"] == gpu]
        rows.append([f"of which GPU {gpu}", len(rs), sum(r["complete"] for r in rs), f"{sum(r['seconds'] for r in rs) / 3600:.2f}", ""])
    first, last = min(r["start"] for r in recs), max(r["end"] for r in recs)
    write_table("gpu_time", ["phase", "jobs", "complete", "GPU hours (sum of job wall times)", "foreign GPU processes seen"], rows,
                "GPU time of this round (RTX 3090s; GPU 3 only until 14:43, then GPUs 2 and 3 with one job per GPU); seconds per job "
                f"from runs/v6_status.jsonl. Wall clock from the first job's start to the last job's end: {first} to {last}.")
    rows = []
    for model in MODELS:
        for run in [search_run(model, lr) for lr in LR_GRID + [BRACKET_LR]] + [formal_run(model, s) for s in SEEDS]:
            t, cfg, sel = load(run, "timing.json"), load_yaml(run), selected_val(run)
            rec = [r for r in recs if r["job"] == run]
            if t is None or cfg is None or sel is None:
                continue
            rows.append([run, cfg["run"]["seed"], cfg["training"]["learning_rate"], f"{PARAMS[model]:,}", t["updates"],
                         f"{t['train_tokens']:,}", f"{sel[3]}{sel[2]}", f"{t['s_per_step_median']:.4f}",
                         f"{t['total_train_time_s'] / 60:.1f}", f"{rec[-1]['seconds'] / 60:.1f}" if rec else "",
                         t["cuda_visible_devices"], t["peak_device_memory_gib_training"], cfg["run"]["git_commit"][:7]
                         + ("-dirty" if cfg["run"]["git_commit"].endswith("-dirty") else "")])
    write_table("runs", ["run", "seed", "peak lr", "parameters", "updates", "training characters",
                         "selected checkpoint (checkpoints/<run>/best/)", "s / update (median)", "training min",
                         "job min (incl. validation-split eval)", "GPU", "peak GiB", "code"], rows,
                "Every v6 training run: seed, actual updates and training characters (timing.json), the checkpoint that every "
                "reported number of the run comes from, speed and memory. Code: the commit recorded by train.py; -dirty = with the "
                "uncommitted changes of results/v6/code/ (snapshot of each launch).")
    return total


# ---------------------------------------------------------------------------------------------- figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGS / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_curves():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("bpc", "validation BPC (fixed set)"), ("acc", "validation Acc_all (%)"), ("gap", "validation − training BPC")]
    for ax, (key, title) in zip(axes, panels):
        for model in MODELS:
            for s in SEEDS:
                h = history(formal_run(model, s))
                if h is None:
                    continue
                its = np.asarray(h["iteration"])
                tok = (its + 1) * TOKENS_PER_UPDATE / 1e6
                if key == "gap":
                    y = np.asarray(h["bpc"]) - np.asarray([train_bpc_near(h, int(i), 500) or np.nan for i in its])
                else:
                    y = np.asarray(h[key]) * (100 if key == "acc" else 1)
                keep = its >= 2000
                ax.plot(tok[keep], y[keep], color=COLORS[model], lw=2 if s == FORMAL_SEED else 1,
                        alpha=1 if s == FORMAL_SEED else 0.55, label=NAMES[model] if s == FORMAL_SEED else None)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training characters (M)", fontsize=9, color=MUTED)
        style(ax)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=9)
    fig.text(0.01, -0.04, "thick: seed 42; thin: seeds 43, 44. From iteration 2,000 (end of warmup).", fontsize=8, color=MUTED)
    save(fig, "training_curves")


def fig_search():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for model in MODELS:
        pts = [(float(lr), selected_val(search_run(model, lr))) for lr in LR_GRID]
        pts = [(x, 100 * v[0]) for x, v in pts if v]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=8, lw=2, color=COLORS[model], label=NAMES[model])
            ax.annotate(NAMES[model], pts[0], xytext=(0, 9), textcoords="offset points", fontsize=9, color=INK, ha="center")
        extra = selected_val(search_run(model, BRACKET_LR))
        if pts and extra:  # post-hoc bracket check: hollow marker, dashed
            ax.plot([pts[-1][0], float(BRACKET_LR)], [pts[-1][1], 100 * extra[0]], ls="--", lw=1.5, color=COLORS[model])
            ax.plot(float(BRACKET_LR), 100 * extra[0], marker="o", ms=8, mfc="white", mew=2, color=COLORS[model])
    ax.set_xscale("log")
    ax.set_xticks([float(x) for x in LR_GRID + [BRACKET_LR]], LR_GRID + [BRACKET_LR])
    ax.text(0.99, 0.02, "hollow: post-hoc 4× check (not used for selection)", transform=ax.transAxes, ha="right", fontsize=7.5,
            color=MUTED)
    ax.minorticks_off()
    ax.set_xlabel("peak learning rate", fontsize=9, color=MUTED)
    ax.set_title(f"validation Acc_all (%) after {SEARCH_ITERS // 1000}k iterations", fontsize=10, color=INK, loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9)
    style(ax)
    save(fig, "lr_search")


def fig_acc_vs_w():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    keys = ["AL-3M", "AL-14M", "v5-alibi", "v5-rope"]
    for ax, (metric, title, sc) in zip(axes, [("acc", "test Acc_all (%)", 100), ("bpc", "test BPC", 1)]):
        for key in keys:
            pts = [(W, mean_std(list(entity_metric(key, W, metric).values()))) for W in LENGTHS]
            pts = [(W, m * sc, (sd if sd == sd else 0) * sc) for W, (m, sd, n) in pts if m is not None]
            if not pts:
                continue
            x, y, e = zip(*pts)
            ax.errorbar(x, y, yerr=e, marker="o", ms=7, lw=2, capsize=3, color=COLORS[key], label=ENTITIES[key][0])
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, [str(W) for W in LENGTHS])
        ax.minorticks_off()
        ax.set_xlabel("evaluation window W (stride 128)", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    if axes[0].get_legend_handles_labels()[0]:
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.16))
    save(fig, "test_vs_W")


def fig_compute():
    """Validation Acc_all vs training GPU minutes: 10k-iteration search runs (chosen lr) and the 50k formal runs."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for model in MODELS:
        pts = []
        budgets = ((search_run(model, chosen_lr(model) or LR_GRID[0]), SEARCH_ITERS), (formal_run(model, FORMAL_SEED), FORMAL_ITERS))
        for run, chars in budgets:
            t, v = load(run, "timing.json"), selected_val(run)
            if t and v:
                pts.append((t["total_train_time_s"] / 60, 100 * v[0], chars * TOKENS_PER_UPDATE / 1e6))
        if not pts:
            continue
        x, y, c = zip(*pts)
        ax.plot(x, y, marker="o", ms=8, lw=2, color=COLORS[model], label=NAMES[model])
        for xi, yi, ci in pts:
            ax.annotate(f"{ci:.0f}M chars", (xi, yi), xytext=(7, -10), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("training time on one RTX 3090 (min, log scale)", fontsize=9, color=MUTED)
    ax.set_title("validation Acc_all (%), W = 512", fontsize=10, color=INK, loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    save(fig, "acc_vs_compute")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.sync:
        sync()
    worst = write_reuse_check()
    write_metrics_at(MAIN_W, list(ENTITIES), "test_W512", f"Test set, W = {MAIN_W}, stride 128 (4,999,999 targets, each scored once); "
                     "mean ± std over seeds. v5 / v3 rows: best-validation-loss checkpoints (their selection rule); v6 rows: "
                     "best-validation-accuracy checkpoints (BPC tie-break).")
    for W in (128, 1024):
        write_metrics_at(W, list(ENTITIES), f"test_W{W}", f"Test set, W = {W}, stride 128; as test_W512.")
    write_metrics_at(MAIN_W, ["AL-3M", "AL-14M"], "val_W512", f"Validation split, W = {MAIN_W}, stride 128 (4,499,999 targets).",
                     split="val")
    write_by_w("test")
    write_by_w("val")
    write_search()
    gain, accd, bpcd = write_formal_val()
    write_paired_test()
    write_curves()
    write_budget_scaling()
    write_selection()
    write_errors()
    write_paper_capacity()
    write_paper_errors()
    write_paper_search()
    total = write_gpu_time()
    fig_curves()
    fig_search()
    fig_acc_vs_w()
    fig_compute()
    best = {k: mean_std(list(entity_metric(k, MAIN_W, "acc").values()))[0] for k in ENTITIES}
    best = {k: v for k, v in best.items() if v is not None}
    print(f"re-evaluation vs existing files: max |ΔBPC| {worst['bpc']:.1e}, max |ΔAcc| {100 * worst['acc']:.1e} pp")
    print(f"seed-42 gain rule: {gain}; val Acc 14M − 3M {accd[:3]}, BPC {bpcd[:3]}")
    for k, v in best.items():
        print(f"test Acc_all W=512 {k}: {100 * v:.2f}% -> {TARGET_ACC - 100 * v:.2f} pp below {TARGET_ACC:.0f}%")
    print(f"GPU hours (v6 jobs): {total / 3600:.2f}")


if __name__ == "__main__":
    main()
