"""
Rebuild every v3 table and figure from results/v3/raw/<run>/ (copies of runs/<run>/ outputs).

    python results/v3/aggregate_v3.py            # tables (csv/md/tex) and figures (pdf/png) from raw/
    python results/v3/aggregate_v3.py --sync     # first copy runs/<run>/ outputs into raw/

Aggregates are mean ± sample std (ddof=1) over seeds 42, 43, 44; the number of seeds found is shown.
Runs missing from raw/ are left out, so partial results aggregate too.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW, TABLES, FIGS = HERE / "raw", HERE / "tables", HERE / "figures"
SEEDS = [42, 43, 44]
LN2 = np.log(2)

PES = ["rope", "learned", "sinusoidal", "hybrid", "alibi", "relative"]
EXTRAPOLATING = ["rope", "alibi", "sinusoidal", "relative"]
LENGTHS = [128, 256, 512, 1024]
# Fixed colour and marker per entity in every figure (reference categorical palette, fixed order).
COLORS = {"rope": "#2a78d6", "learned": "#eb6834", "sinusoidal": "#1baf7a", "hybrid": "#eda100", "alibi": "#e87ba4",
          "relative": "#008300", "LR-2": "#6250d6", "LR-1": "#e34948", "TR": "#2a78d6"}
MARKERS = {"rope": "o", "learned": "s", "sinusoidal": "^", "hybrid": "D", "alibi": "v", "relative": "P", "LR-2": "X", "LR-1": "*",
           "TR": "o"}
INK, MUTED = "#0b0b0b", "#52514e"

# group id -> run-name prefix (runs are <prefix>_s<seed>[suffix])
TABLE_V = {"TR": "TR-final", "LR-1": "LR-1-final", "LR-2": "LSTM-final"}
PE_PREFIX = {pe: ("TR-final" if pe == "rope" else f"TR-final-{pe}") for pe in PES}
TIMING = {"TR": "timing_TR", "LR-1": "timing_LR-1", "LR-2": "timing_LSTM"}
TABLE_V1 = {  # results/table_v.csv (tag tableV-v1 configs, seeds 4212-4214), for reference only
    "TR": ("TR2o", 1.6132, 0.0174), "LR-1": ("LR-1", 1.7460, 0.0018), "LR-2": ("LR-2", 1.7009, 0.0011)}

RUN_FILES = ["eval.json", "val_eval.json", "config.yaml", "timing.json", "training_results.json", "extrapolation.json",
             "pe_tables.json"]


# ----------------------------------------------------------------------------------------------
# sync
# ----------------------------------------------------------------------------------------------
def sync():
    RAW.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^(TR-final|LSTM-final|LR-1-final)(-[\w.]+)?_s\d+(_bestlr)?$|^timing_|^xcheck_")
    for run_dir in sorted((ROOT / "runs").iterdir()):
        if not run_dir.is_dir() or not pattern.match(run_dir.name):
            continue
        if not any((run_dir / f).exists() for f in ("eval.json", "val_eval.json")):
            continue
        dst = RAW / run_dir.name
        dst.mkdir(exist_ok=True)
        for name in RUN_FILES:
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, dst / name)
        for att in run_dir.glob("attention_W*"):
            (dst / att.name).mkdir(exist_ok=True)
            for name in ("stats.json", "example.npz"):
                if (att / name).exists():
                    shutil.copy2(att / name, dst / att.name / name)
        hist_path = run_dir / "metrics_history.json"
        if hist_path.exists():
            h = json.load(open(hist_path))
            train = np.asarray(h["loss_all_train"], float)
            out = {k: h[k] for k in ("iteration", "loss_all_val", "acc", "acc_last", "bpc", "ece")}
            # training loss (unweighted, all positions) averaged over blocks of 500 iterations
            n = len(train) // 500 * 500
            out["train_loss_all_mean_per_500"] = train[:n].reshape(-1, 500).mean(1).tolist()
            out["train_loss_all_last500"] = float(train[-500:].mean())
            (dst / "val_history.json").write_text(json.dumps(out))
    for name in ("v3_status.jsonl", "v3_evals_status.jsonl"):
        if (ROOT / "runs" / name).exists():
            shutil.copy2(ROOT / "runs" / name, RAW / name)
    for p in RAW.rglob("*"):  # end text files with a newline so pre-commit's end-of-file-fixer leaves them alone
        if p.is_file() and p.suffix in (".json", ".jsonl", ".yaml") and p.read_bytes()[-1:] != b"\n":
            with open(p, "a") as f:
                f.write("\n")


# ----------------------------------------------------------------------------------------------
# loading and statistics
# ----------------------------------------------------------------------------------------------
def load(name, fname):
    p = RAW / name / fname
    return json.load(open(p)) if p.exists() else None


def runs_of(prefix, suffix=""):
    """{seed: run name} for the runs of a group present in raw/."""
    return {s: f"{prefix}_s{s}{suffix}" for s in SEEDS if (RAW / f"{prefix}_s{s}{suffix}" / "eval.json").exists()}


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def fmt(m, s, scale=1, dec=4, pm="±"):
    if m is None:
        return "n/a"
    if dec == 0:
        return f"{m * scale:,.0f}"
    if s is None or s != s:
        return f"{m * scale:.{dec}f}"
    return f"{m * scale:.{dec}f} {pm} {s * scale:.{dec}f}"


def best(name):
    e = load(name, "eval.json")
    return e, e["checkpoints"]["best_loss_all_"]


def get(e, c, key):
    fc, ch = c["full_context"], c["chunked128"]
    return {
        "bpc": fc["bpc"], "ppl": fc["perplexity"], "acc": fc["acc"], "ece": fc["ece"], "acc_rare": fc["acc_rare"],
        "acc_common": fc["acc_common"], "acc_last": ch["acc_last"], "bleu": c["self_bleu_4"],
        "bleu_multi": c["self_bleu_4_multi"]["mean"], "params": e["num_params"], "step": c["step"],
    }[key]


def group_stat(names, key):
    vals = []
    for n in names:
        e, c = best(n)
        vals.append(get(e, c, key))
    return mean_std(vals)


def gap(name):
    """Final fixed-set validation loss minus mean training loss over the last 500 iterations (nats)."""
    h = load(name, "val_history.json")
    return None if h is None else h["loss_all_val"][-1] - h["train_loss_all_last500"]


# ----------------------------------------------------------------------------------------------
# table writers
# ----------------------------------------------------------------------------------------------
def write_table(stem, header, rows, caption, csv_rows=None, csv_header=None, notes=""):
    TABLES.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (TABLES / f"{stem}.md").write_text(f"{caption}\n\n" + "\n".join(md) + "\n" + (f"\n{notes}\n" if notes else ""))

    def tex_cell(c):
        return str(c).replace("±", "$\\pm$").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")

    tex = ["% Generated by results/v3/aggregate_v3.py -- do not edit by hand. Requires \\usepackage{booktabs}.",
           "\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{tex_cell(caption)}}}", f"\\label{{tab:{stem}}}",
           "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\toprule", " & ".join(tex_cell(h) for h in header) + " \\\\",
           "\\midrule"]
    tex += [" & ".join(tex_cell(c) for c in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / f"{stem}.tex").write_text("\n".join(tex) + "\n")
    with open(TABLES / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header or header)
        w.writerows(csv_rows or rows)


TV_COLS = [("bpc", "BPC", 1, 4), ("ppl", "PPL", 1, 4), ("acc", "Acc_all (%)", 100, 2), ("ece", "ECE", 1, 4),
           ("acc_rare", "Acc_rare (%)", 100, 2), ("acc_common", "Acc_comm (%)", 100, 2), ("acc_last", "Acc_last@128 (%)", 100, 2),
           ("bleu", "Self-BLEU-4", 1, 4), ("bleu_multi", "Self-BLEU-4 multi", 1, 4)]


def timing(group_id):
    t = load(TIMING[group_id], "timing.json")
    return None if t is None else t["s_per_step_median"]


def write_table_v():
    out = {}
    for version, suffix, label in [("tie", "", "tie-rule lr (block 1)"), ("bestlr", "_bestlr", "lowest-BPC lr (block 1b)")]:
        rows, csv_rows = [], []
        for gid, prefix in TABLE_V.items():
            names = list(runs_of(prefix, suffix).values())
            if not names:
                continue
            cfg_lr = learning_rate(names[0])
            row = [gid, f"{cfg_lr:g}", len(names)]
            crow = [gid, cfg_lr, len(names)]
            for key, _, scale, dec in TV_COLS:
                m, s, _ = group_stat(names, key)
                row.append(fmt(m, s, scale, dec))
                crow += [m, s]
            m, _, _ = group_stat(names, "params")
            t = timing(gid)
            row += [f"{m:,.0f}", "n/a" if t is None else f"{t:.4f}", fmt(*mean_std([group_stat([n], "step")[0] for n in names])[:2], 1, 0)]
            crow += [m, t, " ".join(names)]
            rows.append(row)
            csv_rows.append(crow)
            out[(version, gid)] = names
        header = ["ID", "lr", "seeds"] + [c[1] for c in TV_COLS] + ["#Params", "s/step (solo)", "best-loss step"]
        csv_header = ["ID", "lr", "seeds"] + [f"{c[0]}_{x}" for c in TV_COLS for x in ("mean", "std")] + ["params", "s_per_step", "runs"]
        write_table(f"table_v_{version}", header, rows,
                    f"Table V, {label}: test set, checkpoint with minimum validation loss, mean ± std over seeds 42-44. "
                    "BPC/PPL/Acc/ECE: full-context (W=128, stride 1); Acc_last: chunked128 position 127; "
                    "s/step: separate solo timing run (2,000 iterations, median over iterations 1000-1999, RTX 3090).",
                    csv_rows, csv_header)
    return out


def learning_rate(name):
    import yaml
    return float(yaml.safe_load(open(RAW / name / "config.yaml"))["training"]["learning_rate"])


def write_pe_comparison():
    rows, csv_rows = [], []
    for pe in PES:
        names = list(runs_of(PE_PREFIX[pe]).values())
        if not names:
            continue
        row, crow = [pe, len(names)], [pe, len(names)]
        for key, _, scale, dec in [("bpc", "BPC", 1, 4), ("acc", "Acc_all (%)", 100, 2), ("ece", "ECE", 1, 4),
                                   ("acc_last", "Acc_last@128 (%)", 100, 2)]:
            m, s, _ = group_stat(names, key)
            row.append(fmt(m, s, scale, dec))
            crow += [m, s]
        p = group_stat(names, "params")[0]
        row.append(f"{p:,.0f}")
        crow += [p, " ".join(names)]
        rows.append(row)
        csv_rows.append(crow)
    write_table("pe_comparison", ["encoding", "seeds", "BPC", "Acc_all (%)", "ECE", "Acc_last@128 (%)", "#Params"], rows,
                "Positional encodings (TR size, final config): test set, full-context W=128 stride 1, mean ± std over seeds.",
                csv_rows, ["encoding", "seeds", "bpc_mean", "bpc_std", "acc_mean", "acc_std", "ece_mean", "ece_std",
                           "acc_last_mean", "acc_last_std", "params", "runs"])


def write_dropout_ablation():
    rows, csv_rows = [], []
    for label, prefix in [("0.0 (chosen)", "TR-final"), ("0.1", "TR-final-d0.1")]:
        names = list(runs_of(prefix).values())
        if not names:
            continue
        row, crow = [label, len(names)], [label, len(names)]
        for key, scale, dec in [("bpc", 1, 4), ("acc", 100, 2), ("ece", 1, 4), ("acc_last", 100, 2)]:
            m, s, _ = group_stat(names, key)
            row.append(fmt(m, s, scale, dec))
            crow += [m, s]
        g = mean_std([gap(n) for n in names])
        st = mean_std([group_stat([n], "step")[0] for n in names])
        row += [fmt(g[0], g[1], 1, 4), fmt(st[0], st[1], 1, 0)]
        crow += [g[0], g[1], st[0], " ".join(names)]
        rows.append(row)
        csv_rows.append(crow)
    write_table("dropout_ablation", ["dropout", "seeds", "BPC", "Acc_all (%)", "ECE", "Acc_last@128 (%)", "val - train gap (nats)",
                                     "best-loss step"], rows,
                "Dropout ablation (TR, RoPE): test set, full-context W=128 stride 1. Gap: final fixed-set validation loss minus "
                "the mean training loss of the last 500 iterations (training loss measured with dropout on).",
                csv_rows, ["dropout", "seeds", "bpc_mean", "bpc_std", "acc_mean", "acc_std", "ece_mean", "ece_std", "acc_last_mean",
                           "acc_last_std", "gap_mean", "gap_std", "best_step_mean", "runs"])


def extrap_entities():
    ents = [(pe, PE_PREFIX[pe]) for pe in PES] + [("LR-2", "LSTM-final"), ("LR-1", "LR-1-final")]
    return [(label, list(runs_of(prefix).values())) for label, prefix in ents]


def extrap_values(names, W):
    vals, na = [], None
    for n in names:
        x = load(n, "extrapolation.json")
        if x is None or str(W) not in x["lengths"]:
            continue
        ent = x["lengths"][str(W)]
        if "na" in ent:
            na = ent["na"]
            continue
        vals.append(ent["sliding"]["bpc"])
    return vals, na


def write_extrapolation():
    rows, csv_rows, sanity = [], [], []
    for label, names in extrap_entities():
        if not names:
            continue
        row, crow = [label], [label]
        n_eval = 0
        for W in LENGTHS:
            vals, na = extrap_values(names, W)
            n_eval = max(n_eval, len(vals))
            m, s, _ = mean_std(vals)
            row.append("N/A" if na and not vals else fmt(m, s, 1, 4))
            crow += [m, s]
        for n in names:
            x = load(n, "extrapolation.json")
            if x and "sanity" in x:
                sanity.append((n, x["sanity"]["diff"], x["sanity"]["pass_1e-4"]))
        rows.append(row[:1] + [n_eval] + row[1:])
        csv_rows.append(crow[:1] + [n_eval] + crow[1:])
    worst = max((abs(d) for _, d, _ in sanity), default=None)
    notes = (f"Sanity check (W=128, stride 1 vs eval.json full_context): {sum(p for *_, p in sanity)}/{len(sanity)} runs within 1e-4"
             + (f", largest |difference| {worst:.1e} BPC." if worst is not None else "."))
    write_table("extrapolation", ["model", "seeds"] + [f"W={W}" for W in LENGTHS], rows,
                "Length extrapolation: test-set sliding-window BPC with window W and stride 128 (trained at 128), "
                "mean ± std over seeds. N/A: learned table of 128 positions.",
                csv_rows, ["model", "seeds"] + [f"W{W}_{x}" for W in LENGTHS for x in ("mean", "std")], notes)
    return sanity


def attention_stats(pe, W):
    return load(f"{PE_PREFIX[pe]}_s42", f"attention_W{W}/stats.json")


def write_attention_tables():
    rows = []
    for W, pes in [(128, PES), (512, EXTRAPOLATING)]:
        for pe in pes:
            st = attention_stats(pe, W)
            if st is None:
                continue
            for part in ["all_queries"] + (["past_train_len"] if "past_train_len" in st else []):
                s = st[part]
                span = np.asarray(s["span_mean"]).mean(1)
                rows.append([pe, W, "all" if part == "all_queries" else f">= {st['train_len']}",
                             " / ".join(f"{v:.1f}" for v in span),
                             " / ".join(f"{v:.2f}" for v in np.asarray(s["local8"]).mean(1)),
                             " / ".join(f"{v:.2f}" for v in np.asarray(s["local32"]).mean(1)),
                             " / ".join(f"{v:.2f}" for v in np.asarray(s["entropy"]).mean(1)),
                             f"{np.max(s['space_mass']):.3f} (L{np.argmax(s['space_mass']) // np.shape(s['space_mass'])[1]})",
                             f"{s['space_mass_uniform']:.4f}"])
    write_table("attention_summary", ["encoding", "W", "queries", "mean span per layer", "local mass d<8 per layer",
                                      "local mass d<32 per layer", "entropy per layer (nats)", "max-head space mass (layer)",
                                      "uniform space mass"], rows,
                "Attention statistics, seed 42, 512 test windows (seed 0). Per-layer values are means over the layer's heads. "
                "Span: smallest distance holding 90% of a query's attention. Space mass: weight on the most recent space before "
                "the query (word boundary); 'uniform' is the value for uniform causal attention.")


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
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_extrapolation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, zoom in zip(axes, (False, True)):
        for label, names in extrap_entities():
            pts = [(W, *mean_std(extrap_values(names, W)[0])[:2]) for W in LENGTHS]
            pts = [p for p in pts if p[1] is not None]
            if not pts:
                continue
            W, m, s = zip(*pts)
            s = [0 if x != x else x for x in s]
            ls = "--" if label.startswith("LR") else "-"
            ax.errorbar(W, m, yerr=s, color=COLORS[label], marker=MARKERS[label], ms=6, lw=2, ls=ls, capsize=3, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, [str(W) for W in LENGTHS])
        ax.axvline(128, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("evaluation window W (trained at 128)", color=INK, fontsize=9)
        ax.set_ylabel("test BPC (sliding window, stride 128)", color=INK, fontsize=9)
        if zoom:
            ax.set_ylim(1.35, 1.75)
            ax.set_title("zoom: BPC 1.35-1.75", fontsize=9, color=INK)
        else:
            ax.set_title("all models (learned / hybrid: W=128 only)", fontsize=9, color=INK)
        style(ax)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "extrapolation_bpc_vs_W")


def per_position(names, W):
    curves = []
    for n in names:
        x = load(n, "extrapolation.json")
        if x and str(W) in x["lengths"] and "per_position" in x["lengths"][str(W)]:
            curves.append(np.asarray(x["lengths"][str(W)]["per_position"]["nll_per_position"]) / LN2)
    return np.asarray(curves)


def fig_per_position():
    for W, smooth in [(1024, 16), (128, 1)]:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for label, names in extrap_entities():
            c = per_position(names, W)
            if not len(c):
                continue
            m = c.mean(0)
            if smooth > 1:  # running mean over `smooth` positions, for readability; buckets are in extrapolation.json
                m = np.convolve(m, np.ones(smooth) / smooth, mode="valid")
            x = np.arange(len(m)) + (smooth - 1) / 2
            ls = "--" if label.startswith("LR") else "-"
            ax.plot(x, m, color=COLORS[label], lw=1.6, ls=ls, label=f"{label} ({len(c)} seeds)")
        if W > 128:
            ax.axvline(128, color=MUTED, lw=0.8, ls=":")
            ax.set_yscale("log")
        ax.set_xlabel("position in the window", color=INK, fontsize=9)
        ax.set_ylabel("mean test BPC at position", color=INK, fontsize=9)
        ax.set_title(f"Per-position loss, W={W}, 2,048 test chunks" + (f" (running mean over {smooth})" if smooth > 1 else ""),
                     fontsize=9, color=INK)
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, f"per_position_loss_W{W}")


def fig_attention_profiles():
    for W, pes in [(128, PES), (512, EXTRAPOLATING)]:
        stats = {pe: attention_stats(pe, W) for pe in pes}
        stats = {k: v for k, v in stats.items() if v}
        if not stats:
            continue
        n_l = next(iter(stats.values()))["n_layers"]
        fig, axes = plt.subplots(1, n_l, figsize=(3.2 * n_l, 3.2), sharey=True)
        for li, ax in enumerate(axes):
            for pe, st in stats.items():
                prof = np.asarray(st["distance_profile"])[li].mean(0)  # mean over heads
                ax.plot(np.arange(len(prof)), np.maximum(prof, 1e-8), color=COLORS[pe], lw=1.4, label=pe)
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
    for ax, (W, pes, part) in zip(axes, [(128, PES, "all_queries"), (512, EXTRAPOLATING, "past_train_len")]):
        any_ = False
        for pe in pes:
            st = attention_stats(pe, W)
            if not st or part not in st:
                continue
            span = np.asarray(st[part]["span_mean"])
            ax.plot(range(span.shape[0]), span.mean(1), color=COLORS[pe], marker=MARKERS[pe], ms=6, lw=2, label=pe)
            any_ = True
        ax.set_xticks(range(4))
        ax.set_xlabel("layer", fontsize=9, color=INK)
        ax.set_ylabel("mean effective span (90% mass), chars", fontsize=9, color=INK)
        ax.set_title(f"W={W}" + (", queries at positions >= 128" if part == "past_train_len" else ", all queries"),
                     fontsize=9, color=INK)
        style(ax)
        if any_:
            ax.legend(fontsize=8, frameon=False)
    save(fig, "attention_effective_span")


def head_mean_example(pe, W):
    p = RAW / f"{PE_PREFIX[pe]}_s42" / f"attention_W{W}" / "example.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["attn"].astype(np.float32).mean(1), str(z["text"])  # (Lyr, W, W)


def fig_attention_heatmaps():
    for W, pes, stem in [(128, PES, "attention_heatmaps_W128"), (512, EXTRAPOLATING, "attention_heatmaps_W512")]:
        ex = {pe: head_mean_example(pe, W) for pe in pes}
        ex = {k: v for k, v in ex.items() if v is not None}
        if not ex:
            continue
        n_l = next(iter(ex.values()))[0].shape[0]
        layers = [0, n_l - 1]
        fig, axes = plt.subplots(len(layers), len(ex), figsize=(2.6 * len(ex), 2.7 * len(layers)), squeeze=False)
        for j, (pe, (a, _)) in enumerate(ex.items()):
            for i, li in enumerate(layers):
                ax = axes[i, j]
                im = ax.imshow(np.log10(np.maximum(a[li], 1e-6)), cmap="viridis", vmin=-4, vmax=0, interpolation="nearest")
                ax.set_title(f"{pe}, layer {li}", fontsize=8, color=INK)
                ax.set_xticks([])
                ax.set_yticks([])
                if W > 128:
                    ax.axvline(127.5, color="white", lw=0.5, ls=":")
                    ax.axhline(127.5, color="white", lw=0.5, ls=":")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="log10 attention weight (mean over heads)")
        fig.suptitle(f"Attention on one test window (W={W}, same text for every encoding); rows: query, columns: key",
                     fontsize=9, color=INK)
        save(fig, stem)


def pe_table(pe):
    return load(f"{PE_PREFIX[pe]}_s42", "pe_tables.json")


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
        ax.set_title(f"{pe}" + (" (table added)" if pe == "hybrid" else ""), fontsize=9, color=INK)
        ax.set_xlabel("position", fontsize=8)
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
        ax.set_title(f"{pe}: PC1 {100 * ev[0]:.0f}%, PC2 {100 * ev[1]:.0f}%", fontsize=9, color=INK)
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
                ax.plot(np.arange(b.shape[1]), b[h], color=COLORS["relative"], lw=1.4, ls=heads_ls[h % 4], label=f"relative, head {h}")
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


def fig_space_attention():
    stats = {pe: attention_stats(pe, 128) for pe in PES}
    stats = {k: v for k, v in stats.items() if v}
    if not stats:
        return
    fig, axes = plt.subplots(1, len(stats), figsize=(2.4 * len(stats), 2.8), squeeze=False)
    vmax = max(np.max(s["all_queries"]["space_mass"]) for s in stats.values())
    for ax, (pe, st) in zip(axes[0], stats.items()):
        m = np.asarray(st["all_queries"]["space_mass"])
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > vmax * 0.6 else INK)
        ax.set_title(pe, fontsize=9, color=INK)
        ax.set_xlabel("head", fontsize=8)
        ax.set_ylabel("layer", fontsize=8)
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="attention on the most recent space")
    u = next(iter(stats.values()))["all_queries"]["space_mass_uniform"]
    fig.suptitle(f"Word-boundary attention per head, W=128, seed 42 (uniform causal attention: {u:.3f})", fontsize=10, color=INK)
    save(fig, "attention_space_mass")


def fig_val_curves():
    groups = [("TR", "TR-final", ""), ("LR-2", "LSTM-final", ""), ("LR-1", "LR-1-final", "")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, suffix, title in zip(axes, ("", "_bestlr"), ("tie-rule lr", "lowest-BPC lr")):
        for label, prefix, _ in groups:
            for i, n in enumerate(runs_of(prefix, suffix).values()):
                h = load(n, "val_history.json")
                if not h:
                    continue
                it = np.asarray(h["iteration"])
                tr = np.asarray(h["train_loss_all_mean_per_500"])
                ax.plot(it[1:], np.asarray(h["loss_all_val"])[1:] / LN2, color=COLORS[label], lw=1.3,
                        label=f"{label} validation" if i == 0 else None)
                ax.plot((np.arange(len(tr)) + 1) * 500, tr / LN2, color=COLORS[label], lw=0.9, ls=":",
                        label=f"{label} training (500-iteration mean)" if i == 0 else None)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.sync:
        sync()
    write_table_v()
    write_pe_comparison()
    write_dropout_ablation()
    write_extrapolation()
    write_attention_tables()
    for f in (fig_extrapolation, fig_per_position, fig_attention_profiles, fig_attention_span, fig_attention_heatmaps,
              fig_pe_similarity_and_pca, fig_relative_vs_alibi, fig_space_attention, fig_val_curves):
        f()
    print(f"tables -> {TABLES}, figures -> {FIGS}")


if __name__ == "__main__":
    main()
