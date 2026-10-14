# Results index

Every number in the report comes from **`v4/`** (which reads `v3/raw/` and `v4/raw/`). The other folders document how the
final configuration was reached and are kept for traceability; their numbers are superseded.

| folder | status | what it holds |
|---|---|---|
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

Regenerate everything with `python results/v4/aggregate_v4.py` (add `--sync` to first copy new outputs from `runs/`);
`python results/v4/export_paper.py --paper <paper dir>` copies the tables (`.tex`) and figures (`.pdf`) into the LaTeX source,
and `python results/v4/check_claims.py` checks the numbers listed in `v4/claims.md` against the tables.
