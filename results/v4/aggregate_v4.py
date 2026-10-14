"""
Rebuild every final table and figure of the report from results/v3/raw/ and results/v4/raw/ (copies of runs/ outputs).

    python results/v4/aggregate_v4.py            # tables (csv/md/tex) and figures (pdf/png)
    python results/v4/aggregate_v4.py --sync     # first copy the v4 outputs from runs/ into results/v4/raw/

results/v3/ is read, never written. A per-run file is looked up in results/v4/raw/<run>/ first, then in results/v3/raw/<run>/
(v4 adds the NoPE runs and, for the v3 RoPE / sinusoidal runs, the position-interpolation outputs extrapolation_pi.json and
attention_W512_pi/). The lr bracket table also reads results/v3_tuning/tuning.csv (the v3 search, A100).
Aggregates are mean ± sample std (ddof 1) over seeds 42, 43, 44; seed-matched differences are the mean ± std of the per-seed
differences. Names follow the report: Transformer (RoPE) etc., LSTM-3.2M (formerly LR-2), LSTM-2.1M (formerly LR-1).
"""

import argparse
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW4, RAW3 = HERE / "raw", ROOT / "results" / "v3" / "raw"
TUNING_CSV = ROOT / "results" / "v3_tuning" / "tuning.csv"
TABLES, FIGS = HERE / "tables", HERE / "figures"
SEEDS = [42, 43, 44]
LN2 = np.log(2)
LENGTHS = [128, 256, 512, 1024]

# encodings, in the order of the report; key = model.pos_encoding_type
PES = ["rope", "alibi", "sinusoidal", "hybrid", "learned", "relative", "none"]
NAME = {"rope": "RoPE", "alibi": "ALiBi", "sinusoidal": "Sinusoidal", "hybrid": "Hybrid", "learned": "Learned",
        "relative": "Relative bias", "none": "NoPE", "rope_pi": "RoPE + PI", "sinusoidal_pi": "Sinusoidal + PI",
        "LSTM-3.2M": "LSTM-3.2M", "LSTM-2.1M": "LSTM-2.1M", "TR": "Transformer (RoPE)"}
PREFIX = {pe: "TR-final" if pe == "rope" else f"TR-final-{'nope' if pe == 'none' else pe}" for pe in PES}
PREFIX.update({"LSTM-3.2M": "LSTM-final", "LSTM-2.1M": "LR-1-final", "TR": "TR-final"})
TIMING = {"TR": "timing_TR", "LSTM-2.1M": "timing_LR-1", "LSTM-3.2M": "timing_LSTM"}
# Colour follows the entity in every figure: the reference categorical palette (fixed order) for the eight v3 entities;
# NoPE (no positional signal) is neutral dark grey with its own marker. PI variants: the base encoding's colour, dashed,
# hollow markers.
COLORS = {"rope": "#2a78d6", "learned": "#eb6834", "sinusoidal": "#1baf7a", "hybrid": "#eda100", "alibi": "#e87ba4",
          "relative": "#008300", "LSTM-3.2M": "#6250d6", "LSTM-2.1M": "#e34948", "TR": "#2a78d6", "none": "#3d3c39"}
COLORS.update(rope_pi=COLORS["rope"], sinusoidal_pi=COLORS["sinusoidal"])
MARKERS = {"rope": "o", "learned": "s", "sinusoidal": "^", "hybrid": "D", "alibi": "v", "relative": "P", "LSTM-3.2M": "X",
           "LSTM-2.1M": "*", "TR": "o", "none": "h", "rope_pi": "o", "sinusoidal_pi": "^"}
INK, MUTED = "#0b0b0b", "#52514e"

RUN_FILES = ["eval.json", "val_eval.json", "config.yaml", "timing.json", "training_results.json", "extrapolation.json",
             "extrapolation_pi.json", "pe_tables.json"]


# ----------------------------------------------------------------------------------------------
# sync (runs/ -> results/v4/raw/)
# ----------------------------------------------------------------------------------------------
def sync():
    RAW4.mkdir(parents=True, exist_ok=True)
    new_runs = re.compile(r"^TR-final-nope_s\d+$|^bracket_")
    pi_runs = re.compile(r"^TR-final(-sinusoidal)?_s\d+$")
    for run_dir in sorted((ROOT / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        if new_runs.match(run_dir.name):
            if not any((run_dir / f).exists() for f in ("eval.json", "val_eval.json")):
                continue
            dst = RAW4 / run_dir.name
            dst.mkdir(exist_ok=True)
            for name in RUN_FILES:
                if (run_dir / name).exists():
                    shutil.copy2(run_dir / name, dst / name)
            atts = list(run_dir.glob("attention_W*"))
            hist_path = run_dir / "metrics_history.json"
            if hist_path.exists():
                h = json.load(open(hist_path))
                train = np.asarray(h["loss_all_train"], float)
                out = {k: h[k] for k in ("iteration", "loss_all_val", "acc", "acc_last", "bpc", "ece")}
                n = len(train) // 500 * 500
                out["train_loss_all_mean_per_500"] = train[:n].reshape(-1, 500).mean(1).tolist()
                out["train_loss_all_last500"] = float(train[-500:].mean())
                (dst / "val_history.json").write_text(json.dumps(out))
        elif pi_runs.match(run_dir.name):  # v3 runs: only the v4 outputs (position interpolation, report sample)
            atts = list(run_dir.glob("attention_W*_pi"))
            if run_dir.name == "TR-final_s42" and (run_dir / "bestlossall.txt").exists():  # generated sample in the appendix
                (RAW4 / run_dir.name).mkdir(exist_ok=True)
                shutil.copy2(run_dir / "bestlossall.txt", RAW4 / run_dir.name / "sample_best_loss.txt")
            if (run_dir / "extrapolation_pi.json").exists() or atts:
                (RAW4 / run_dir.name).mkdir(exist_ok=True)
            if (run_dir / "extrapolation_pi.json").exists():
                shutil.copy2(run_dir / "extrapolation_pi.json", RAW4 / run_dir.name / "extrapolation_pi.json")
        else:
            continue
        for att in atts:
            (RAW4 / run_dir.name / att.name).mkdir(parents=True, exist_ok=True)
            for name in ("stats.json", "example.npz"):
                if (att / name).exists():
                    shutil.copy2(att / name, RAW4 / run_dir.name / att.name / name)
    for name in ("v4_status.jsonl", "launch_v4.txt"):
        if (ROOT / "runs" / name).exists():
            shutil.copy2(ROOT / "runs" / name, RAW4 / name)
    for p in RAW4.rglob("*"):  # end text files with a newline so pre-commit's end-of-file-fixer leaves them alone
        if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml", ".txt") and p.read_bytes()[-1:] != b"\n":
            with open(p, "a") as f:
                f.write("\n")


# ----------------------------------------------------------------------------------------------
# loading and statistics
# ----------------------------------------------------------------------------------------------
def path(run, fname):
    for raw in (RAW4, RAW3):
        if (raw / run / fname).exists():
            return raw / run / fname
    return None


def load(run, fname):
    p = path(run, fname)
    return json.load(open(p)) if p else None


def runs_of(prefix, suffix=""):
    """{seed: run name} of a group's runs that have a test evaluation."""
    return {s: f"{prefix}_s{s}{suffix}" for s in SEEDS if path(f"{prefix}_s{s}{suffix}", "eval.json")}


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def fmt(m, s, scale=1, dec=4, signed=False):
    if m is None:
        return "n/a"
    sign = "+" if signed else ""
    if s is None or s != s:
        return f"{m * scale:{sign}.{dec}f}"
    return f"{m * scale:{sign}.{dec}f} ± {s * scale:.{dec}f}"


def metric(run, key):
    e = load(run, "eval.json")
    c = e["checkpoints"]["best_loss_all_"]
    fc, ch = c["full_context"], c["chunked128"]
    return {"bpc": fc["bpc"], "ppl": fc["perplexity"], "acc": fc["acc"], "ece": fc["ece"], "acc_rare": fc["acc_rare"],
            "acc_common": fc["acc_common"], "acc_last": ch["acc_last"], "bleu": c["self_bleu_4"],
            "bleu_multi": c["self_bleu_4_multi"]["mean"], "params": e["num_params"], "step": c["step"]}[key]


def group_stat(runs, key):
    return mean_std([metric(r, key) for r in runs])


def seed_matched_diff(a, b, key="bpc"):
    """mean ± std over seeds of metric(a[s]) - metric(b[s]) for the seeds both groups have."""
    return mean_std([metric(a[s], key) - metric(b[s], key) for s in SEEDS if s in a and s in b])


def learning_rate(run):
    return float(yaml.safe_load(open(path(run, "config.yaml")))["training"]["learning_rate"])


def gap(run):
    """Final fixed-set validation loss minus mean training loss over the last 500 iterations (nats)."""
    h = load(run, "val_history.json")
    return None if h is None else h["loss_all_val"][-1] - h["train_loss_all_last500"]


# ----------------------------------------------------------------------------------------------
# table writer
# ----------------------------------------------------------------------------------------------
def tex_cell(c):
    s = str(c)
    for a, b in (("—", "--"), ("ε", "$\\varepsilon$"), ("±", "$\\pm$"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("≥", "$\\geq$"), ("−", "$-$"), ("Δ", "$\\Delta$"),
                 ("†", "$^\\dagger$")):
        s = s.replace(a, b)
    return s


# LaTeX for Markdown header texts (the .tex tables are \input by the report)
TEX_HEADER = {
    "Acc_all (%)": "$\\mathrm{Acc}_{\\text{all}}$ (\\%)", "Acc_last (%)": "$\\mathrm{Acc}_{\\text{last}}$ (\\%)",
    "Acc_rare (%)": "$\\mathrm{Acc}_{\\text{rare}}$ (\\%)", "Acc_comm (%)": "$\\mathrm{Acc}_{\\text{comm}}$ (\\%)",
    "ΔBPC vs RoPE": "$\\Delta$BPC vs RoPE", "val − train gap (nats)": "val $-$ train gap (nats)",
    "Label smoothing ε": "$\\varepsilon$", "Mass on last 8 chars": "Mass, last 8 chars",
    "Mean weight on distances 0-2": "Weight on $d \\le 2$",
}


def tex_header(h):
    if h in TEX_HEADER:
        return TEX_HEADER[h]
    if h.startswith("W="):
        return f"$W{{=}}{h[2:]}$"
    return tex_cell(h)


def write_table(stem, header, rows, caption, csv_rows=None, csv_header=None, notes="", align=None):
    """Markdown, CSV and a booktabs .tex (a bare tabular, to be wrapped in a table float by the report).
    A row "midrule" is a rule in the .tex and skipped elsewhere."""
    TABLES.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    tex = [f"% Generated by results/v4/aggregate_v4.py -- do not edit by hand. Requires booktabs. {caption}",
           "\\begin{tabular}{" + (align or "l" + "c" * (len(header) - 1)) + "}", "\\toprule",
           " & ".join(tex_header(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "midrule":
            tex.append("\\midrule")
            continue
        tex.append(" & ".join(tex_cell(c) for c in r) + " \\\\")
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


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------
TV_COLS = [("bpc", "BPC", 1, 4), ("ppl", "PPL", 1, 4), ("acc", "Acc_all (%)", 100, 2), ("ece", "ECE", 1, 4),
           ("acc_rare", "Acc_rare (%)", 100, 2), ("acc_common", "Acc_comm (%)", 100, 2), ("acc_last", "Acc_last (%)", 100, 2),
           ("bleu", "Self-BLEU-4", 1, 4), ("bleu_multi", "Self-BLEU-4 multi", 1, 4)]
FAMILIES = ["TR", "LSTM-3.2M", "LSTM-2.1M"]


def write_table_v():
    for version, suffix, label in [("tie", "", "tie-rule lr"), ("bestlr", "_bestlr", "lowest-BPC lr")]:
        rows, csv_rows = [], []
        for fam in FAMILIES:
            runs = list(runs_of(PREFIX[fam], suffix).values())
            if not runs:
                continue
            lr = learning_rate(runs[0])
            row, crow = [NAME[fam], f"{lr:g}"], [NAME[fam], lr, len(runs)]
            for key, _, scale, dec in TV_COLS:
                m, s, _ = group_stat(runs, key)
                row.append(fmt(m, s, scale, dec))
                crow += [m, s]
            params = group_stat(runs, "params")[0]
            t = load(TIMING[fam], "timing.json")
            t = None if t is None or suffix else t["s_per_step_median"]
            row.append(f"{params:,.0f}")
            if not suffix:
                row.append("n/a" if t is None else f"{t:.4f}")
            crow += [params, t, " ".join(runs)]
            rows.append(row)
            csv_rows.append(crow)
        header = ["Model", "lr"] + [c[1] for c in TV_COLS] + ["#Params"] + ([] if suffix else ["s/step"])
        csv_header = ["model", "lr", "seeds"] + [f"{c[0]}_{x}" for c in TV_COLS for x in ("mean", "std")] + ["params", "s_per_step", "runs"]
        tr = runs_of(PREFIX["TR"], suffix)
        notes = []
        for fam in ("LSTM-3.2M", "LSTM-2.1M"):
            d = seed_matched_diff(tr, runs_of(PREFIX[fam], suffix))
            if d[2]:
                notes.append(f"Seed-matched BPC difference Transformer (RoPE) - {NAME[fam]}: {fmt(d[0], d[1], signed=True)}.")
        if suffix:
            for fam in FAMILIES:
                d = seed_matched_diff(runs_of(PREFIX[fam], suffix), runs_of(PREFIX[fam]))
                if d[2]:
                    notes.append(f"{NAME[fam]}: lowest-BPC lr - tie-rule lr: {fmt(d[0], d[1], signed=True)} BPC (seed-matched).")
        write_table(f"table_v_{version}", header, rows,
                    f"Transformer vs LSTMs, {label}: test set, best-validation-loss checkpoint, mean ± std over seeds 42-44. "
                    "BPC/PPL/Acc/ECE: full-context evaluation (W=128, stride 1); Acc_last: last position of non-overlapping "
                    "128-character blocks; s/step: separate solo timing run (median over iterations 1000-1999, RTX 3090).",
                    csv_rows, csv_header, notes=" ".join(notes))


def write_paper_main():
    """Table of the report: both learning-rate settings in one booktabs tabular (tables/paper_main.tex only)."""
    cols = [("bpc", 1, 4), ("ppl", 1, 4), ("acc", 100, 2), ("acc_last", 100, 2), ("acc_rare", 100, 2), ("acc_common", 100, 2),
            ("ece", 1, 4), ("bleu", 1, 4)]
    head = ["Model", "lr", "\\#Params", "BPC", "PPL", "$\\mathrm{Acc}_{\\text{all}}$ (\\%)", "$\\mathrm{Acc}_{\\text{last}}$ (\\%)",
            "$\\mathrm{Acc}_{\\text{rare}}$ (\\%)", "$\\mathrm{Acc}_{\\text{comm}}$ (\\%)", "ECE", "Self-BLEU-4", "s/step"]
    tex = ["% Generated by results/v4/aggregate_v4.py -- do not edit by hand. Same numbers as table_v_tie / table_v_bestlr.",
           "\\begin{tabular}{llr" + "c" * (len(head) - 3) + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for suffix, label in [("", "Tie-rule learning rate (selected configuration)"), ("_bestlr", "Lowest-BPC learning rate")]:
        tex.append(f"\\multicolumn{{{len(head)}}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for fam in FAMILIES:
            runs = list(runs_of(PREFIX[fam], suffix).values())
            if not runs:
                continue
            lr = learning_rate(runs[0])
            row = [NAME[fam], f"{lr:.1e}".replace("e-0", "e-"), f"{group_stat(runs, 'params')[0]:,.0f}"]
            row += [tex_cell(fmt(*group_stat(runs, k)[:2], scale, dec)) for k, scale, dec in cols]
            t = load(TIMING[fam], "timing.json")
            row.append("--" if suffix or t is None else f"{t['s_per_step_median']:.4f}")
            tex.append(" & ".join(row) + " \\\\")
        if not suffix:
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "paper_main.tex").write_text("\n".join(tex) + "\n")


def write_hpsearch():
    """The v3 search grid, both families side by side (tables/hpsearch.*), from results/v3_tuning/tuning.csv."""
    rows_all = [r for r in tuning_rows() if r["gpu"] == "A100"]
    fams = [("TR", "Transformer (RoPE)"), ("LSTM-3.2M", "LSTM-3.2M")]
    grids = {}
    for fam, _ in fams:
        fr = [r for r in rows_all if r["family"] == fam and r["ls_eps"] > 0]
        lrs = sorted({r["lr"] for r in fr})
        best = min(r["bpc"] for r in fr)
        grid = []
        for lr in lrs:
            cell = []
            for d in (0.0, 0.1):
                m = [r for r in fr if r["lr"] == lr and r["dropout"] == d]
                cell.append("--" if not m else f"{m[0]['bpc']:.4f}" + ("†" if m[0]["bpc"] == best else ""))
            grid.append([f"{lr:.1e}".replace("e-0", "e-")] + cell)
        sel = [r for r in rows_all if r["family"] == fam and r["ls_eps"] == 0.0]
        grids[fam] = (grid, sel[0] if sel else None)
    n = max(len(g) for g, _ in grids.values())
    md_rows, tex_rows = [], []
    for i in range(n):
        row = []
        for fam, _ in fams:
            g = grids[fam][0]
            row += g[i] if i < len(g) else ["", "", ""]
        md_rows.append(row)
        tex_rows.append(" & ".join(tex_cell(c) for c in row) + " \\\\")
    sel_cells = []
    for fam, _ in fams:
        r = grids[fam][1]
        sel_cells.append(f"selected lr {r['lr']:.1e}, dropout 0, ε = 0: {r['bpc']:.4f}".replace("e-0", "e-") if r else "")
    header = ["lr", "dropout 0", "dropout 0.1", "lr ", "dropout 0 ", "dropout 0.1 "]
    caption = ("Hyperparameter search (A100): validation BPC (full context, stride 8) after 10,000 iterations, seed 4212, label "
               "smoothing ε = 0.05. † lowest BPC; --: not run. Left: Transformer (RoPE); right: LSTM-3.2M.")
    (TABLES / "hpsearch.md").write_text(f"{caption}\n\n| " + " | ".join(header) + " |\n|" + "---|" * 6 + "\n"
                                        + "\n".join("| " + " | ".join(r) + " |" for r in md_rows) + "\n\n" + "; ".join(sel_cells) + "\n")
    tex = ["% Generated by results/v4/aggregate_v4.py -- do not edit by hand. " + caption,
           "\\begin{tabular}{lcc|lcc}", "\\toprule",
           "\\multicolumn{3}{c|}{Transformer (RoPE)} & \\multicolumn{3}{c}{LSTM-3.2M} \\\\",
           "lr & dropout 0 & dropout 0.1 & lr & dropout 0 & dropout 0.1 \\\\", "\\midrule"] + tex_rows + ["\\midrule"]
    tex.append(" & ".join(f"\\multicolumn{{3}}{{l{'|' if i == 0 else ''}}}{{{tex_cell(c)}}}" for i, c in enumerate(sel_cells)) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "hpsearch.tex").write_text("\n".join(tex) + "\n")


def write_pe_comparison():
    rope = runs_of(PREFIX["rope"])
    rows, csv_rows = [], []
    ents = []
    for pe in PES:
        runs = runs_of(PREFIX[pe])
        if runs:
            ents.append((group_stat(list(runs.values()), "bpc")[0], pe, runs))
    for _, pe, runs in sorted(ents, key=lambda e: e[0]):
        names = list(runs.values())
        row, crow = [NAME[pe]], [NAME[pe], len(names)]
        for key, scale, dec in [("bpc", 1, 4), ("acc", 100, 2), ("ece", 1, 4), ("acc_last", 100, 2)]:
            m, s, _ = group_stat(names, key)
            row.append(fmt(m, s, scale, dec))
            crow += [m, s]
        d = seed_matched_diff(runs, rope) if pe != "rope" else (None, None, 0)
        row.append("—" if pe == "rope" else fmt(d[0], d[1], 1, 4, signed=True))
        p = group_stat(names, "params")[0]
        row.append(f"{p:,.0f}")
        per_seed = [metric(runs[s], "bpc") if s in runs else None for s in SEEDS]
        crow += [d[0], d[1], p] + per_seed + [" ".join(names)]
        rows.append(row)
        csv_rows.append(crow)
    write_table("pe_comparison", ["Encoding", "BPC", "Acc_all (%)", "ECE", "Acc_last (%)", "ΔBPC vs RoPE", "#Params"], rows,
                "Positional encodings at the training length (Transformer, final config tuned for RoPE): test set, full-context "
                "W=128 stride 1, mean ± std over seeds 42-44. ΔBPC: seed-matched difference from RoPE (positive = worse).",
                csv_rows, ["encoding", "seeds", "bpc_mean", "bpc_std", "acc_mean", "acc_std", "ece_mean", "ece_std", "acc_last_mean",
                           "acc_last_std", "delta_vs_rope_mean", "delta_vs_rope_std", "params", "bpc_s42", "bpc_s43", "bpc_s44", "runs"])


TRAIN_GROUPS = [("Transformer (RoPE)", "TR-final", ""), ("LSTM-3.2M", "LSTM-final", ""), ("LSTM-2.1M", "LR-1-final", ""),
                ("Transformer (RoPE), lowest-BPC lr", "TR-final", "_bestlr"), ("LSTM-3.2M, lowest-BPC lr", "LSTM-final", "_bestlr"),
                ("LSTM-2.1M, lowest-BPC lr", "LR-1-final", "_bestlr")] + \
               [(f"Transformer ({NAME[pe]})", PREFIX[pe], "") for pe in PES if pe != "rope"] + \
               [("Transformer (RoPE), dropout 0.1", "TR-final-d0.1", "")]


def write_training_summary():
    """Best-checkpoint iteration, late validation trend and train-validation gap of every 50k group."""
    rows, csv_rows, all_steps = [], [], []
    for label, prefix, suffix in TRAIN_GROUPS:
        runs = list(runs_of(prefix, suffix).values())
        hs = [load(r, "val_history.json") for r in runs]
        if not runs or any(h is None for h in hs):
            continue
        steps = [metric(r, "step") for r in runs]
        all_steps += steps
        late = mean_std([h["loss_all_val"][-1] - h["loss_all_val"][h["iteration"].index(45000)] for h in hs])
        g2k = mean_std([h["loss_all_val"][h["iteration"].index(2000)] - h["train_loss_all_mean_per_500"][3] for h in hs])
        gend = mean_std([gap(r) for r in runs])
        rows.append([label, f"{min(steps):,}-{max(steps):,}", fmt(late[0], late[1], signed=True), fmt(g2k[0], g2k[1]),
                     fmt(gend[0], gend[1])])
        csv_rows.append([label, len(runs), min(steps), max(steps), late[0], late[1], g2k[0], g2k[1], gend[0], gend[1]])
    write_table("training_summary", ["Group", "Best-loss iteration", "Val. loss change 45k-end (nats)", "Gap at 2k (nats)",
                                     "Gap at end (nats)"], rows,
                "Training dynamics of the 50,000-iteration runs (seeds 42-44). Best-loss iteration: range over seeds of the "
                "checkpoint with the lowest fixed-set validation loss. Val. loss change: from iteration 45,000 to the end. Gap: "
                "fixed-set validation loss minus the mean training loss of the preceding 500 iterations (dropout runs: training "
                "loss measured with dropout on).",
                csv_rows, ["group", "seeds", "best_step_min", "best_step_max", "late_change_mean", "late_change_std",
                           "gap2k_mean", "gap2k_std", "gap_end_mean", "gap_end_std"],
                notes=f"All {len(all_steps)} runs: best-loss iteration {min(all_steps):,}-{max(all_steps):,}.")


def write_dropout_ablation():
    rows, csv_rows = [], []
    for label, prefix in [("0.0 (chosen)", "TR-final"), ("0.1", "TR-final-d0.1")]:
        names = list(runs_of(prefix).values())
        if not names:
            continue
        row, crow = [label], [label, len(names)]
        for key, scale, dec in [("bpc", 1, 4), ("acc", 100, 2), ("ece", 1, 4), ("acc_last", 100, 2)]:
            m, s, _ = group_stat(names, key)
            row.append(fmt(m, s, scale, dec))
            crow += [m, s]
        g = mean_std([gap(n) for n in names])
        st = mean_std([metric(n, "step") for n in names])
        row += [fmt(g[0], g[1], 1, 4), f"{st[0]:,.0f}"]
        crow += [g[0], g[1], st[0], " ".join(names)]
        rows.append(row)
        csv_rows.append(crow)
    d = seed_matched_diff(runs_of("TR-final-d0.1"), runs_of("TR-final"))
    write_table("dropout_ablation", ["Dropout", "BPC", "Acc_all (%)", "ECE", "Acc_last (%)", "val − train gap (nats)",
                                     "best-loss step"], rows,
                "Dropout ablation, Transformer (RoPE): test set, full-context W=128 stride 1, mean ± std over seeds 42-44. "
                "Gap: final validation loss minus the mean training loss of the last 500 iterations (measured with dropout on).",
                csv_rows, ["dropout", "seeds", "bpc_mean", "bpc_std", "acc_mean", "acc_std", "ece_mean", "ece_std", "acc_last_mean",
                           "acc_last_std", "gap_mean", "gap_std", "best_step_mean", "runs"],
                notes=f"Seed-matched BPC difference (0.1 - 0.0): {fmt(d[0], d[1], 1, 4, signed=True)}.")


# extrapolation entities: (key, run prefix, file)
EXTRAP = [("rope", "TR-final", "extrapolation.json"), ("rope_pi", "TR-final", "extrapolation_pi.json"),
          ("sinusoidal", "TR-final-sinusoidal", "extrapolation.json"),
          ("sinusoidal_pi", "TR-final-sinusoidal", "extrapolation_pi.json"),
          ("learned", "TR-final-learned", "extrapolation.json"), ("hybrid", "TR-final-hybrid", "extrapolation.json"),
          ("relative", "TR-final-relative", "extrapolation.json"), ("none", "TR-final-nope", "extrapolation.json"),
          ("alibi", "TR-final-alibi", "extrapolation.json"),
          ("LSTM-3.2M", "LSTM-final", "extrapolation.json"), ("LSTM-2.1M", "LR-1-final", "extrapolation.json")]
EXTRAP_LABEL = dict(NAME, relative="Relative bias (clamped)")


def extrap_values(key, W):
    """{seed: sliding-window BPC} and the N/A reason, if any."""
    _, prefix, fname = next(e for e in EXTRAP if e[0] == key)
    vals, na = {}, None
    for s, run in runs_of(prefix).items():
        x = load(run, fname)
        if x is None or str(W) not in x["lengths"]:
            continue
        ent = x["lengths"][str(W)]
        if "na" in ent:
            na = ent["na"]
            continue
        vals[s] = ent["sliding"]["bpc"]
    return vals, na


def write_extrapolation():
    rows, csv_rows, sanity, pi128 = [], [], [], []
    for key, prefix, fname in EXTRAP:
        if not runs_of(prefix):
            continue
        row, crow, n_eval = [EXTRAP_LABEL[key]], [EXTRAP_LABEL[key]], 0
        for W in LENGTHS:
            vals, na = extrap_values(key, W)
            n_eval = max(n_eval, len(vals))
            m, s, _ = mean_std(list(vals.values()))
            row.append("N/A" if na and not vals else fmt(m, s))
            crow += [m, s]
        if n_eval == 0:
            continue
        for run in runs_of(prefix).values():
            x = load(run, fname)
            if x and "sanity" in x:
                sanity.append((run, x["sanity"]["diff"], x["sanity"]["pass_1e-4"]))
        if key.endswith("_pi"):
            base = extrap_values(key[:-3], 128)[0]
            pi128 += [abs(v - base[s]) for s, v in extrap_values(key, 128)[0].items() if s in base]
        rows.append(row)
        csv_rows.append(crow[:1] + [n_eval] + crow[1:])
    alibi = extrap_values("alibi", 1024)[0]
    vs_alibi = []
    for key in ("rope_pi", "sinusoidal_pi", "none"):
        v = extrap_values(key, 1024)[0]
        d = mean_std([v[s] - alibi[s] for s in v if s in alibi])
        if d[2]:
            vs_alibi.append(f"{EXTRAP_LABEL[key]} {fmt(d[0], d[1], signed=True)}")
    worst = max((abs(d) for _, d, _ in sanity), default=None)
    notes = (f"Sanity check (W=128, stride 1 vs eval.json full_context): {sum(p for *_, p in sanity)}/{len(sanity)} runs within 1e-4"
             + (f", largest |difference| {worst:.1e} BPC." if worst is not None else ".")
             + (f" PI at W=128 (pos_scale 1) vs plain W=128: largest |difference| {max(pi128):.1e} BPC over {len(pi128)} runs."
                if pi128 else "")
             + (" Seed-matched difference from ALiBi at W=1024: " + "; ".join(vs_alibi) + "." if vs_alibi else ""))
    write_table("extrapolation", ["Model"] + [f"W={W}" for W in LENGTHS], rows,
                "Length extrapolation: test-set sliding-window BPC with window W and stride 128 (all models trained at 128), "
                "mean ± std over seeds 42-44. PI: position interpolation at evaluation time (positions scaled by 128/W), no "
                "fine-tuning. N/A: learned table of 128 positions.",
                csv_rows, ["model", "seeds"] + [f"W{W}_{x}" for W in LENGTHS for x in ("mean", "std")], notes)


BUCKETS_SHOWN = [0, 1, 2, 3, 7, 15]  # 64-position buckets: 0-63, 64-127, 128-191, 192-255, 448-511, 960-1023


def write_per_position_buckets():
    rows, csv_rows = [], []
    for key, prefix, fname in EXTRAP:
        vals = []
        for run in runs_of(prefix).values():
            x = load(run, fname)
            if x and "1024" in x["lengths"] and "per_position" in x["lengths"]["1024"]:
                vals.append(x["lengths"]["1024"]["per_position"]["bpc_buckets"])
        if not vals:
            continue
        v = np.asarray(vals)
        m, sd = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(v.shape[1]))
        rows.append([EXTRAP_LABEL[key]] + [fmt(m[b], sd[b], dec=3) for b in BUCKETS_SHOWN])
        csv_rows.append([EXTRAP_LABEL[key], len(v)] + m.tolist() + sd.tolist())
    header = ["Model"] + [f"{64 * b}-{64 * b + 63}" for b in BUCKETS_SHOWN]
    write_table("per_position_buckets_W1024", header, rows,
                "Per-position test BPC at W=1024 (2,048 non-overlapping test chunks, seed 0), mean over 64-position buckets, "
                "mean ± std over seeds 42-44. Trained at 128: buckets from 128-191 on are beyond the training length.",
                csv_rows, ["model", "seeds"] + [f"bucket{b}_mean" for b in range(16)] + [f"bucket{b}_std" for b in range(16)])


# attention entities: (key, encoding, W, directory)
ATTN = [(pe, pe, 128, "attention_W128") for pe in PES] + \
       [(pe, pe, 512, "attention_W512") for pe in ["rope", "sinusoidal", "relative", "none", "alibi"]] + \
       [("rope_pi", "rope", 512, "attention_W512_pi")]


def attention_stats(pe, directory):
    return load(f"{PREFIX[pe]}_s42", f"{directory}/stats.json")


def attn_entities(W):
    out = {}
    for key, pe, w, d in ATTN:
        if w == W:
            st = attention_stats(pe, d)
            if st:
                out[key] = st
    return out


def write_attention_tables():
    rows, csv_rows = [], []
    for W in (128, 512):
        if rows:
            rows.append("midrule")
        for key, st in attn_entities(W).items():
            part = "all_queries" if W == 128 else "past_train_len"
            s = st[part]
            span = np.asarray(s["span_mean"]).mean(1)
            l8 = np.asarray(s["local8"]).mean(1)
            ent = np.asarray(s["entropy"]).mean(1)
            sm = np.asarray(s["space_mass"])
            best = f"{sm.max():.3f}"
            q = "all" if part == "all_queries" else f"≥ {st['train_len']}"
            rows.append([NAME[key], W, q, " / ".join(f"{v:.1f}" for v in span), " / ".join(f"{v:.2f}" for v in l8),
                         " / ".join(f"{v:.2f}" for v in ent), best])
            csv_rows.append([NAME[key], W, q] + span.tolist() + l8.tolist() + ent.tolist()
                            + [float(sm.max()), int(np.argmax(sm) // sm.shape[1]), s["space_mass_uniform"]])
    write_table("attention_summary", ["Encoding", "W", "Queries", "Effective span (chars)", "Mass on last 8 chars", "Entropy (nats)",
                                      "Best word-boundary head"], rows,
                "Attention statistics, seed 42, 512 test windows. Per-layer values (layers 0 / 1 / 2 / 3) are means over the "
                "4 heads. Effective span: smallest distance holding 90% of a query's attention. Word-boundary head: largest "
                "per-head attention on the most recent space before the query (uniform causal attention: 0.028 at W=128, "
                "0.004 for queries >= 128 at W=512). W=512 rows use only queries at positions >= 128.",
                csv_rows, ["encoding", "W", "queries"] + [f"span_L{i}" for i in range(4)] + [f"local8_L{i}" for i in range(4)]
                + [f"entropy_L{i}" for i in range(4)] + ["best_space_head", "best_space_head_layer", "space_uniform"],
                align="lcllllc")


SHORT = [("rope", "attention_W128", "RoPE, W=128"), ("rope_pi", "attention_W512_pi", "RoPE + PI, W=512"),
         ("rope", "attention_W512", "RoPE, W=512"), ("none", "attention_W128", "NoPE, W=128"),
         ("none", "attention_W512", "NoPE, W=512")]


def short_profiles():
    """{label: (key, head-mean distance profile per layer)} from distance_profile (all queries of the window)."""
    out = {}
    for key, d, label in SHORT:
        st = attention_stats("rope" if key == "rope_pi" else key, d)
        if st:
            out[label] = (key, np.asarray(st["distance_profile"]).mean(1))  # (Lyr, W)
    return out


def write_attention_short_range():
    rows, csv_rows = [], []
    for label, (_, prof) in short_profiles().items():
        peak = prof[:, :17].argmax(1)
        near = prof[:, :3].sum(1)
        rows.append([label, " / ".join(str(int(v)) for v in peak), " / ".join(f"{v:.2f}" for v in near)])
        csv_rows.append([label] + peak.tolist() + near.tolist())
    write_table("attention_short_range", ["Model", "Peak distance (chars)", "Mean weight on distances 0-2"], rows,
                "Short-range attention, seed 42: head-mean distance profile per layer (layers 0 / 1 / 2 / 3), all queries of "
                "the window. Peak: distance (0-16) with the largest mean weight. Position interpolation at W=512 maps 4 "
                "characters onto one trained position step.",
                csv_rows, ["model"] + [f"peak_L{i}" for i in range(4)] + [f"mass_d0to2_L{i}" for i in range(4)])


def fig_attention_short_range():
    prof = short_profiles()
    keep = {k: v for k, v in prof.items() if k in ("RoPE, W=128", "RoPE + PI, W=512", "RoPE, W=512")}
    if not keep:
        return
    n_l = next(iter(keep.values()))[1].shape[0]
    fig, axes = plt.subplots(1, n_l, figsize=(3.0 * n_l, 2.9), sharey=True)
    for li, ax in enumerate(axes):
        for label, (key, p) in keep.items():
            ls = "--" if key == "rope_pi" else (":" if "512" in label else "-")
            ax.plot(np.arange(17), p[li, :17], color=COLORS["rope"], ls=ls, marker=MARKERS["rope"], ms=4,
                    mfc="white" if key == "rope_pi" else COLORS["rope"], lw=1.4, label=label)
        ax.set_title(f"layer {li}", fontsize=9, color=INK)
        ax.set_xlabel("distance i - j (characters)", fontsize=8, color=INK)
        style(ax)
    axes[0].set_ylabel("mean attention weight (mean over heads)", fontsize=8, color=INK)
    axes[-1].legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Short-range attention: position interpolation stretches the local pattern (seed 42)", fontsize=10, color=INK,
                 y=1.04)
    save(fig, "attention_short_range")


def tuning_rows():
    rows = list(csv.DictReader(open(TUNING_CSV)))
    out = []
    for r in rows:
        fam = "TR" if r["family"] == "TR" else "LSTM-3.2M"
        out.append({"family": fam, "lr": float(r["lr"]), "dropout": float(r["dropout"]), "ls_eps": float(r["ls_eps"]),
                    "bpc": float(r["bpc"]), "gpu": "A100", "run": r["run"], "source": "v3 search", "diverged": r["status"] != "done"})
    for run, fam in [("xcheck_TR_ls0", "TR")]:
        v = load(run, "val_eval.json")
        if v:
            out.append(row_from_val_eval(run, fam, v, "v3 cross-machine check"))
    for d in sorted(RAW4.glob("bracket_*")):
        v = load(d.name, "val_eval.json")
        fam = "TR" if "_TR_" in d.name else "LSTM-3.2M"
        if v:
            out.append(row_from_val_eval(d.name, fam, v, "v4 bracket"))
        elif (d / "config.yaml").exists():
            out.append({"family": fam, "lr": learning_rate(d.name), "dropout": 0.0, "ls_eps": 0.0, "bpc": float("inf"), "gpu": "RTX 3090",
                        "run": d.name, "source": "v4 bracket", "diverged": True})
    return out


def row_from_val_eval(run, fam, v, source):
    cfg = yaml.safe_load(open(path(run, "config.yaml")))
    m = cfg["model"]
    drop = m.get("dropout", m.get("attn_dropout", 0.0))
    bpc = v["checkpoints"]["best_loss_all_"]["full_context"]["bpc"]
    return {"family": fam, "lr": float(cfg["training"]["learning_rate"]), "dropout": float(drop), "ls_eps": float(cfg["loss"]["ls_eps"]),
            "bpc": bpc, "gpu": "RTX 3090", "run": run, "source": source, "diverged": not np.isfinite(bpc)}


BRACKET_TIE = 0.02  # decision rule, fixed before the runs (results/v4/REPORT.md)


def bracket_decision(rows, fam):
    new = [r for r in rows if r["family"] == fam and r["source"] == "v4 bracket"]
    ref_lr = {"TR": 2.7e-3, "LSTM-3.2M": 7.2e-3}[fam]
    ref = [r for r in new if abs(r["lr"] - ref_lr) < 1e-12]
    higher = [r for r in new if r["lr"] > ref_lr * 1.0001]
    if not ref or len(higher) < 2:
        return None
    ref = ref[0]
    gains = [(r["lr"], ref["bpc"] - r["bpc"] if not r["diverged"] else -np.inf) for r in higher]
    beat = [g for g in gains if g[1] > BRACKET_TIE]
    return {"family": fam, "ref_lr": ref_lr, "ref_bpc": ref["bpc"], "gains": gains,
            "bracketed": not beat, "best_new": min(higher, key=lambda r: r["bpc"])}


def write_lr_bracket():
    rows_all = tuning_rows()
    rows, csv_rows, notes = [], [], []
    for fam in ("TR", "LSTM-3.2M"):
        fr = sorted([r for r in rows_all if r["family"] == fam],
                    key=lambda r: (r["gpu"] != "A100", r["ls_eps"] == 0.0, r["dropout"], r["lr"]))
        if rows:
            rows.append("midrule")
        best_a100 = min((r["bpc"] for r in fr if r["gpu"] == "A100" and r["ls_eps"] > 0 and r["dropout"] == 0.0), default=None)
        for r in fr:
            mark = "†" if r["gpu"] == "RTX 3090" else ""
            bpc = "diverged" if r["diverged"] else f"{r['bpc']:.4f}"
            if r["bpc"] == best_a100:
                bpc += " (lowest of search)"
            rows.append([NAME[fam], f"{r['lr']:g}", f"{r['dropout']:g}", f"{r['ls_eps']:g}", r["gpu"] + mark, bpc])
            csv_rows.append([fam, r["lr"], r["dropout"], r["ls_eps"], r["gpu"], r["bpc"], r["source"], r["run"]])
        d = bracket_decision(rows_all, fam)
        if d:
            g = ", ".join(f"lr {lr:g}: {gain:+.4f}" for lr, gain in d["gains"])
            notes.append(f"{NAME[fam]}: improvement over the 3090 rerun at lr {d['ref_lr']:g} (BPC {d['ref_bpc']:.4f}): {g}. "
                         + ("Bracketed (no new lr is better by more than 0.02)." if d["bracketed"]
                            else "NOT bracketed: a new lr is better by more than 0.02 (stop condition)."))
    write_table("lr_bracket", ["Model", "lr", "Dropout", "Label smoothing ε", "GPU", "Validation BPC"], rows,
                "Learning-rate search and bracket check: full-context BPC on the validation split (stride 8), seed 4212, 10,000 "
                "iterations, best-validation-loss checkpoint. A100 rows: the original search (label smoothing 0.05 except the "
                "last row per family); † RTX 3090 runs with the final configuration (dropout 0, ε = 0): the rerun of the "
                "lowest-BPC lr and two further steps of factor 1.5.",
                csv_rows, ["family", "lr", "dropout", "ls_eps", "gpu", "val_bpc", "source", "run"], "\n".join(notes))
    return rows_all


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # reproducible bytes
    fig.savefig(FIGS / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def line_style(key):
    """(linestyle, marker face colour) per entity: PI dashed with hollow markers, LSTMs dash-dot, the rest solid."""
    if key.endswith("_pi"):
        return "--", "white"
    if key.startswith("LSTM"):
        return "-.", COLORS[key]
    return "-", COLORS[key]


FIG_EXTRAP = ["rope", "rope_pi", "sinusoidal", "sinusoidal_pi", "relative", "none", "alibi", "LSTM-3.2M", "LSTM-2.1M"]


def fig_extrapolation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, zoom in zip(axes, (False, True)):
        for key in FIG_EXTRAP:
            pts = [(W, *mean_std(list(extrap_values(key, W)[0].values()))[:2]) for W in LENGTHS]
            pts = [p for p in pts if p[1] is not None]
            if not pts:
                continue
            W, m, s = zip(*pts)
            s = [0 if x != x else x for x in s]
            ls, mfc = line_style(key)
            ax.errorbar(W, m, yerr=s, color=COLORS[key], marker=MARKERS[key], mfc=mfc, ms=6, lw=1.8, ls=ls, capsize=3,
                        label=EXTRAP_LABEL[key])
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, [str(W) for W in LENGTHS])
        ax.axvline(128, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("evaluation window W (trained at 128)", color=INK, fontsize=9)
        ax.set_ylabel("test BPC (sliding window, stride 128)", color=INK, fontsize=9)
        ax.set_title("zoom: BPC 1.35-1.75" if zoom else "all models", fontsize=9, color=INK)
        if zoom:
            ax.set_ylim(1.35, 1.75)
        style(ax)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "extrapolation_bpc_vs_W")


def per_position(key, W):
    _, prefix, fname = next(e for e in EXTRAP if e[0] == key)
    curves = []
    for run in runs_of(prefix).values():
        x = load(run, fname)
        if x and str(W) in x["lengths"] and "per_position" in x["lengths"][str(W)]:
            curves.append(np.asarray(x["lengths"][str(W)]["per_position"]["nll_per_position"]) / LN2)
    return np.asarray(curves)


def fig_per_position():
    for W, smooth, keys in [(1024, 16, FIG_EXTRAP), (128, 1, ["rope", "alibi", "sinusoidal", "hybrid", "learned", "relative", "none",
                                                               "LSTM-3.2M", "LSTM-2.1M"])]:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for key in keys:
            c = per_position(key, W)
            if not len(c):
                continue
            m = c.mean(0)
            if smooth > 1:  # running mean over `smooth` positions, for readability; buckets are in the json files
                m = np.convolve(m, np.ones(smooth) / smooth, mode="valid")
            x = np.arange(len(m)) + (smooth - 1) / 2
            ls, _ = line_style(key)
            ax.plot(x, m, color=COLORS[key], lw=1.5, ls=ls, label=f"{EXTRAP_LABEL[key]} ({len(c)} seeds)")
        if W > 128:
            ax.axvline(128, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("position in the window", color=INK, fontsize=9)
        ax.set_ylabel("mean test BPC at position", color=INK, fontsize=9)
        ax.set_title(f"Per-position loss, W={W}, 2,048 test chunks" + (f" (running mean over {smooth})" if smooth > 1 else ""),
                     fontsize=9, color=INK)
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, f"per_position_loss_W{W}")


def fig_attention_profiles():
    for W in (128, 512):
        stats = attn_entities(W)
        if not stats:
            continue
        n_l = next(iter(stats.values()))["n_layers"]
        fig, axes = plt.subplots(1, n_l, figsize=(3.2 * n_l, 3.2), sharey=True)
        for li, ax in enumerate(axes):
            for key, st in stats.items():
                prof = np.asarray(st["distance_profile"])[li].mean(0)  # mean over heads
                ax.plot(np.arange(len(prof)), np.maximum(prof, 1e-8), color=COLORS[key], lw=1.3, ls=line_style(key)[0], label=NAME[key])
            ax.set_yscale("log")
            if W > 128:
                ax.axvline(128, color=MUTED, lw=0.8, ls=":")
            ax.set_title(f"layer {li}", fontsize=9, color=INK)
            ax.set_xlabel("distance i - j", fontsize=8, color=INK)
            style(ax)
        axes[0].set_ylabel("mean attention weight (mean over heads)", fontsize=8, color=INK)
        axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"Attention distance profiles, W={W}, seed 42, 512 test windows", fontsize=10, color=INK)
        save(fig, f"attention_distance_profiles_W{W}")


def fig_attention_span():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (W, part) in zip(axes, [(128, "all_queries"), (512, "past_train_len")]):
        for key, st in attn_entities(W).items():
            if part not in st:
                continue
            span = np.asarray(st[part]["span_mean"])
            ls, mfc = line_style(key)
            ax.plot(range(span.shape[0]), span.mean(1), color=COLORS[key], marker=MARKERS[key], mfc=mfc, ms=6, lw=1.8, ls=ls,
                    label=NAME[key])
        ax.set_xticks(range(4))
        ax.set_xlabel("layer", fontsize=9, color=INK)
        ax.set_ylabel("mean effective span (90% mass), chars", fontsize=9, color=INK)
        ax.set_title(f"W={W}" + (", queries at positions >= 128" if part == "past_train_len" else ", all queries"), fontsize=9, color=INK)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False)
    save(fig, "attention_effective_span")


def fig_attention_heatmaps():
    for W in (128, 512):
        ex = {}
        for key, pe, w, d in ATTN:
            p = path(f"{PREFIX[pe]}_s42", f"{d}/example.npz")
            if w == W and p:
                z = np.load(p)
                ex[key] = z["attn"].astype(np.float32).mean(1)  # (Lyr, W, W)
        if not ex:
            continue
        n_l = next(iter(ex.values())).shape[0]
        layers = [0, n_l - 1]
        fig, axes = plt.subplots(len(layers), len(ex), figsize=(2.4 * len(ex), 2.6 * len(layers)), squeeze=False)
        for j, (key, a) in enumerate(ex.items()):
            for i, li in enumerate(layers):
                ax = axes[i, j]
                im = ax.imshow(np.log10(np.maximum(a[li], 1e-6)), cmap="viridis", vmin=-4, vmax=0, interpolation="nearest")
                ax.set_title(f"{NAME[key]}, layer {li}", fontsize=8, color=INK)
                ax.set_xticks([])
                ax.set_yticks([])
                if W > 128:
                    ax.axvline(127.5, color="white", lw=0.5, ls=":")
                    ax.axhline(127.5, color="white", lw=0.5, ls=":")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="log10 attention weight (mean over heads)")
        fig.suptitle(f"Attention on one test window (W={W}, same text for every encoding); rows: query, columns: key",
                     fontsize=9, color=INK)
        save(fig, f"attention_heatmaps_W{W}")


def fig_space_attention():
    stats = attn_entities(128)
    if not stats:
        return
    fig, axes = plt.subplots(1, len(stats), figsize=(2.6 * len(stats), 2.9), squeeze=False)
    vmax = max(np.max(s["all_queries"]["space_mass"]) for s in stats.values())
    for ax, (key, st) in zip(axes[0], stats.items()):
        m = np.asarray(st["all_queries"]["space_mass"])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > vmax * 0.6 else INK)
        ax.set_title(NAME[key], fontsize=9, color=INK)
        ax.set_xlabel("head", fontsize=8)
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
        if ax is axes[0][0]:
            ax.set_ylabel("layer", fontsize=8)
        else:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="weight on last space")
    u = next(iter(stats.values()))["all_queries"]["space_mass_uniform"]
    fig.suptitle(f"Word-boundary attention per head, W=128, seed 42 (uniform causal attention: {u:.3f})", fontsize=10, color=INK)
    save(fig, "attention_space_mass")


def pe_table(pe):
    return load(f"{PREFIX[pe]}_s42", "pe_tables.json")


def cos_sim(t):
    t = np.asarray(t, float)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return t @ t.T


def fig_pe_similarity_and_pca():
    tabs = {pe: pe_table(pe) for pe in ("learned", "hybrid", "sinusoidal")}
    tabs = {k: v for k, v in tabs.items() if v}
    if not tabs:
        return
    fig, axes = plt.subplots(1, len(tabs), figsize=(3.6 * len(tabs), 3.4), squeeze=False)
    for ax, (pe, t) in zip(axes[0], tabs.items()):
        im = ax.imshow(cos_sim(t["table"]), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(NAME[pe] + (" (table added)" if pe == "hybrid" else ""), fontsize=9, color=INK)
        ax.set_xlabel("position", fontsize=8)
        if ax is axes[0][0]:
            ax.set_ylabel("position", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="cosine similarity")
    fig.suptitle("Position-embedding similarity (seed 42)", fontsize=10, color=INK)
    save(fig, "pe_similarity")

    fig, axes = plt.subplots(1, len(tabs), figsize=(3.6 * len(tabs), 3.4), squeeze=False)
    for ax, (pe, t) in zip(axes[0], tabs.items()):
        x = np.asarray(t["table"], float)
        x = x - x.mean(0)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        pc = u[:, :2] * s[:2]
        ev = s ** 2 / (s ** 2).sum()
        ax.plot(pc[:, 0], pc[:, 1], color="#c3c2b7", lw=0.8, zorder=1)
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=np.arange(len(pc)), cmap="viridis", s=12, zorder=2)
        ax.set_title(f"{NAME[pe]}: PC1 {100 * ev[0]:.0f}%, PC2 {100 * ev[1]:.0f}%", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="position")
    fig.suptitle("Position embeddings projected on their first two principal components (seed 42)", fontsize=10, color=INK)
    save(fig, "pe_pca")


def fig_relative_vs_alibi():
    rel, ali = pe_table("relative"), pe_table("alibi")
    if not rel and not ali:
        return
    n_l = rel["n_layers"] if rel else 4
    fig, axes = plt.subplots(1, n_l, figsize=(3.2 * n_l, 3.2), sharey=True)
    heads_ls = ["-", "--", "-.", ":"]
    for li, ax in enumerate(axes):
        if rel:
            b = np.asarray(rel["bias_by_distance"])[li]
            for h in range(b.shape[0]):
                ax.plot(np.arange(b.shape[1]), b[h], color=COLORS["relative"], lw=1.4, ls=heads_ls[h % 4], label=f"relative bias, head {h}")
        if ali:
            a = np.asarray(ali["bias_by_distance"])
            for h in range(a.shape[0]):
                ax.plot(np.arange(a.shape[1]), a[h], color=COLORS["alibi"], lw=1.2, ls=heads_ls[h % 4], label=f"ALiBi, head {h}")
        ax.set_title(f"layer {li}", fontsize=9, color=INK)
        ax.set_xlabel("distance i - j", fontsize=8, color=INK)
        style(ax)
    axes[0].set_ylabel("bias added to the attention score", fontsize=8, color=INK)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Learned relative bias (seed 42) vs fixed ALiBi bias", fontsize=10, color=INK)
    save(fig, "pe_relative_vs_alibi")


def fig_val_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, suffix, title in zip(axes, ("", "_bestlr"), ("tie-rule lr", "lowest-BPC lr")):
        for fam in FAMILIES:
            for i, n in enumerate(runs_of(PREFIX[fam], suffix).values()):
                h = load(n, "val_history.json")
                if not h:
                    continue
                it = np.asarray(h["iteration"])
                tr = np.asarray(h["train_loss_all_mean_per_500"])
                ax.plot(it[1:], np.asarray(h["loss_all_val"])[1:] / LN2, color=COLORS[fam], lw=1.3,
                        label=f"{NAME[fam]} validation" if i == 0 else None)
                ax.plot((np.arange(len(tr)) + 1) * 500, tr / LN2, color=COLORS[fam], lw=0.9, ls=":",
                        label=f"{NAME[fam]} training (500-iteration mean)" if i == 0 else None)
        ax.set_ylim(1.45, 2.0)
        ax.set_xlabel("iteration", fontsize=9, color=INK)
        ax.set_ylabel("loss (bits per character)", fontsize=9, color=INK)
        ax.set_title(f"Training and fixed-set validation loss, {title}", fontsize=9, color=INK)
        style(ax)
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    if handles:
        fig.legend(list(handles.values()), list(handles.keys()), fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.99, 0.9))
    save(fig, "val_curves")


def fig_lr_bracket(rows_all):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, fam in zip(axes, ("TR", "LSTM-3.2M")):
        fr = [r for r in rows_all if r["family"] == fam and not r["diverged"]]
        col = COLORS[fam]
        for (gpu, eps, drop), ls, mk, lab in [(("A100", 0.05, 0.0), "-", "o", "A100, ε 0.05, dropout 0"),
                                              (("A100", 0.05, 0.1), ":", "s", "A100, ε 0.05, dropout 0.1"),
                                              (("RTX 3090", 0.0, 0.0), "--", "D", "RTX 3090, ε 0, dropout 0 (final config)")]:
            pts = sorted((r["lr"], r["bpc"]) for r in fr if r["gpu"] == gpu and r["ls_eps"] == eps and r["dropout"] == drop)
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, color=col, ls=ls, marker=mk, ms=6, lw=1.6, mfc="white" if gpu == "RTX 3090" else col, label=lab)
        a100_eps0 = [(r["lr"], r["bpc"]) for r in fr if r["gpu"] == "A100" and r["ls_eps"] == 0.0]
        if a100_eps0:
            x, y = zip(*a100_eps0)
            ax.plot(x, y, color=col, ls="none", marker="*", ms=9, label="A100, ε 0, dropout 0 (chosen lr)")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate", fontsize=9, color=INK)
        ax.set_ylabel("validation BPC (10k iterations)", fontsize=9, color=INK)
        ax.set_title(NAME[fam], fontsize=9, color=INK)
        style(ax)
        ax.legend(fontsize=7, frameon=False)
    save(fig, "lr_bracket")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.sync:
        sync()
    write_table_v()
    write_paper_main()
    write_hpsearch()
    write_pe_comparison()
    write_dropout_ablation()
    write_training_summary()
    write_extrapolation()
    write_per_position_buckets()
    write_attention_tables()
    write_attention_short_range()
    rows_all = write_lr_bracket()
    for f in (fig_extrapolation, fig_per_position, fig_attention_profiles, fig_attention_span, fig_attention_heatmaps,
              fig_pe_similarity_and_pca, fig_relative_vs_alibi, fig_space_attention, fig_val_curves):
        f()
    fig_lr_bracket(rows_all)
    fig_attention_short_range()
    print(f"tables -> {TABLES}, figures -> {FIGS}")


if __name__ == "__main__":
    main()
