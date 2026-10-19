# Results index

Every number in the report comes from **`v4/`** (which reads `v3/raw/` and `v4/raw/`) and, for the follow-up sections
(repairs of RoPE beyond the training length; capacity and next-character accuracy), from **`v5/`** and **`v6/`**. `v4/claims.md`
traces every number in the report text to one of these folders. The other folders document how the final configuration was
reached and are kept for traceability; their numbers are superseded.

| folder | status | what it holds |
|---|---|---|
| [`v6/`](v6/REPORT.md) | **current** (report: capacity and accuracy) | Capacity study: ALiBi 3.2M vs 14.2M trained from scratch on 512-character windows (819M training characters, checkpoints selected by validation accuracy), Acc_all / Top-k of the best v5 models. `aggregate_v6.py` rebuilds its tables and figures from `v6/raw/` (+ `v5/raw/`, `v3/raw/`); `code/` holds the code snapshots (HEAD + uncommitted diff) taken at every launch. |
| [`v5/`](v5/REPORT.md) | **current** (report: repairs of RoPE) | Fixing RoPE beyond its training length: evaluation-only variants (local attention, SelfExtend) and long-window fine-tuning with and without position interpolation, 1,000 and 4,000 iterations. `aggregate_v5.py` rebuilds its tables and figures from `v5/raw/` (+ `v3/raw/`, `v4/raw/` for reference rows). |
| [`v4/`](v4/REPORT.md) | **current** | Final tables and figures of the report: Transformer vs LSTMs, seven positional encodings (incl. NoPE), dropout, length extrapolation (incl. position interpolation), attention analysis, learning-rate bracket check. `aggregate_v4.py` rebuilds everything; `claims.md` traces every number in the report to a file. |
| [`v3/`](v3/REPORT.md) | current raw data, superseded tables | The 36 final 50k-iteration runs (3 seeds each), extrapolation and attention evaluations. `v3/raw/` is read by `v4/aggregate_v4.py`; `v3/tables` and `v3/figures` use the old model names (TR, LR-1, LR-2). |
| [`v3_tuning/`](v3_tuning/REPORT.md) | current (protocol) | The validation-only hyperparameter search (learning rate, dropout, label smoothing) and its decision rules, fixed before the runs. |
| [`code_changes_v2.md`](code_changes_v2.md), [`v2/`](v2/) | history | Fairness and correctness fixes made before tuning (unified attention, sinusoidal scale, LSTM forget bias, LR schedule, generation). |
| top-level files ([`REPORT.md`](REPORT.md), `table_v.*`, `ablation.md`, `claims.md`, `sanity_gate.md`, `throughput.md`, `per_run.csv`, `raw/`, `checks/`, `aggregate.py`) | history (v1, tag `tableV-v1`) | Re-run of the original course-report models with the dropout-RNG fix, the weight-decay mask and full-context evaluation (old hyperparameters). Not reproducible with the current defaults (`code_changes_v2.md`). |

Model names: the report and `v4/` use **Transformer (RoPE)** (`configs/final/TR.yaml`; run prefix `TR-final`),
**LSTM-3.2M** (`configs/final/LSTM.yaml`, formerly LR-2; run prefix `LSTM-final`) and **LSTM-2.1M** (`configs/final/LR-1.yaml`,
formerly LR-1; run prefix `LR-1-final`). Other encodings are `TR-final-<encoding>`; NoPE runs are `TR-final-nope`.

## Report tables and figures → files

All paths relative to `results/v4/`. Tables exist as `.csv`, `.md` and booktabs `.tex`; figures as `.pdf` and `.png`.

| report item | file | produced from |
|---|---|---|
| Transformer vs LSTMs, both lr settings (report table) | `tables/paper_main.tex` | same as the two rows below |
| Transformer vs LSTMs (tie-rule lr) | `tables/table_v_tie.*` | `v3/raw/{TR,LSTM,LR-1}-final_s*/eval.json`, `v3/raw/timing_*/timing.json` |
| Transformer vs LSTMs (lowest-BPC lr) | `tables/table_v_bestlr.*` | `v3/raw/*_bestlr/eval.json` |
| Positional encodings at the training length | `tables/pe_comparison.*` | `v3/raw/TR-final*_s*/eval.json`, `v4/raw/TR-final-nope_s*/eval.json` |
| Dropout ablation | `tables/dropout_ablation.*` | `v3/raw/TR-final{,-d0.1}_s*/` |
| Length extrapolation | `tables/extrapolation.*`, `figures/extrapolation_bpc_vs_W.*` | `*/raw/<run>/extrapolation.json`, `v4/raw/<run>/extrapolation_pi.json` |
| Per-position loss | `figures/per_position_loss_W1024.*`, `figures/per_position_loss_W128.*` | same |
| Attention statistics | `tables/attention_summary.*`, `figures/attention_distance_profiles_W{128,512}.*`, `figures/attention_effective_span.*`, `figures/attention_heatmaps_W{128,512}.*`, `figures/attention_space_mass.*` | `*/raw/<run>_s42/attention_W*/{stats.json,example.npz}` |
| Positional-encoding tables | `figures/pe_similarity.*`, `figures/pe_pca.*`, `figures/pe_relative_vs_alibi.*` | `v3/raw/<run>_s42/pe_tables.json` |
| Training curves | `figures/val_curves.*` | `v3/raw/<run>/val_history.json` |
| Learning-rate search and bracket check | `tables/lr_bracket.*`, `figures/lr_bracket.*` | `v3_tuning/tuning.csv`, `v3/raw/xcheck_TR_ls0/`, `v4/raw/bracket_*/val_eval.json` |
| Tuning grid (report table) | `tables/hpsearch.*` | `v3_tuning/tuning.csv` |
| Tuning grid and decision rules | `v3_tuning/REPORT.md`, `v3_tuning/tables.md` | `v3_tuning/raw/` |
| Training dynamics (best checkpoint, late trend, train–validation gap) | `tables/training_summary.*` | `*/raw/<run>/val_history.json` |
| Per-position loss in 64-position buckets | `tables/per_position_buckets_W1024.*` | `*/raw/<run>/extrapolation*.json` |
| Short-range attention (position interpolation) | `tables/attention_short_range.*`, `figures/attention_short_range.*` | `*/raw/<run>_s42/attention_W*/stats.json` |
| Generated text sample | — | `v4/raw/TR-final_s42/sample_best_loss.txt` |
| Repairs of RoPE beyond the training length (report table and figure) | `v5/tables/paper_repair.*`, `v5/figures/bpc_vs_W.*` | `v5/raw/`, `v4/raw/`, `v3/raw/` (`aggregate_v5.py`) |
| Local-attention / SelfExtend selection, fine-tuning learning rates | `v5/tables/inference_variants_val.*`, `v5/tables/ft_lr_grid.*` | `v5/raw/` |
| Model capacity at fixed training data | `v6/tables/paper_capacity.*`, `v6/tables/test_by_W.*` | `v6/raw/v6-AL-*`, `v6/raw/<v5 / v3 run>/extrapolation_v6.json` (`aggregate_v6.py`) |
| Accuracy by position in the word | `v6/tables/paper_errors.*` | `v6/raw/<run>/errors_W512.json` |
| Capacity learning rates, size vs training characters, training curves | `v6/tables/paper_search.*`, `v6/tables/budget_scaling.*`, `v6/figures/training_curves.*` | `v6/raw/v6search-*`, `v6/raw/v6-AL-*/val_history.json` |

Regenerate everything with `python results/v4/aggregate_v4.py`, `results/v5/aggregate_v5.py` and `results/v6/aggregate_v6.py`
(add `--sync` to first copy new outputs from `runs/`); `python results/v4/export_paper.py --paper <paper dir>` copies the tables
(`.tex`) and figures (`.pdf`) of v4, v5 and v6 into the LaTeX source, and `python results/v4/check_claims.py` checks the numbers
listed in `v4/claims.md` against the tables.
