# Numbers behind the report

Every result number in the report (LaTeX source in the separate paper repository), with the file and row/field it comes
from. Tables in the report are `\input` from `results/v4/tables/*.tex` and are not repeated here; this list covers the numbers
in the prose, captions and hand-written appendix tables.

- `file` is a Markdown table under `results/v4/tables/`; `python results/v4/check_claims.py` checks every such row (exact
  match, or the table value rounded half-up to the decimals shown in the text; the sign is ignored when the text gives none).
- `derived`: computed from checked rows; the formula is given. `external`: a published number; the source is given.
  `pilot`: a single pilot run from the original course report (commit `8f26249` of the paper repository,
  `content/4.Experiment.tex` and `content/3.Architecture.tex`); the raw logs were not kept, so these cannot be regenerated.
- Structural numbers (model sizes, 128-character windows, 50,000 iterations, seeds, grid values) come from
  `configs/final/*.yaml` and `results/v3_tuning/REPORT.md` and are not listed.

## Abstract, introduction, conclusion

| id | value | file | row | column |
|---|---|---|---|---|
| A1 | 1.453 | table_v_tie.md | Transformer (RoPE) | BPC |
| A2 | 0.062 | table_v_tie.md | notes | - |
| A3 | 2.4× | derived | 0.0664 / 0.0280 = 2.37 (T8 / T7) | per training step, this JAX implementation |
| A4 | 0.011 | derived | max(+0.0094, +0.0102) = 0.0102 → "within 0.011" (P7, P8) | ALiBi, sinusoidal vs RoPE |
| A5 | 0.167 | pe_comparison.md | NoPE | ΔBPC vs RoPE |
| A6 | 1.413 | extrapolation.md | ALiBi | W=1024 |
| A7 | 8× | derived | 1024 / 128 | - |
| A8 | 0.030--0.039 | derived | hybrid +0.0302, learned +0.0389 (P12, P9) | learned position tables vs RoPE |
| A9 | 3.2M | derived | 3,169,280 (T4) | - |

## Related work

| id | value | file | row | column |
|---|---|---|---|---|
| R1 | 1.27 BPC, about 45M parameters | external | Krause et al., arXiv:1609.07959, Table 2 and text (https://ar5iv.labs.arxiv.org/html/1609.07959) | text8 test |
| R2 | 1.18 BPC, 44M parameters | external | Al-Rfou et al. 2019, Table 1, T12 (https://ar5iv.labs.arxiv.org/html/1808.04444) | text8 test |
| R3 | 1.13 BPC, 235M parameters, 512-character context | external | Al-Rfou et al. 2019, Table 1, T64 | text8 test |
| R4 | 1.08 BPC, 277M parameters | external | Dai et al. 2019, Table 3 (https://ar5iv.labs.arxiv.org/html/1901.02860) | text8 test |
| R5 | 14--87× | derived | 45M / 3.2M = 14; 277M / 3.2M = 87 | - |
| R6 | 1.441--1.453 | derived | T13 (1.4412) and T1 (1.4532) | - |

## Setup

| id | value | file | row | column |
|---|---|---|---|---|
| S1 | 9.6 passes | derived | 50,000 × 128 × 128 / 85,499,999 training characters = 9.58 | - |
| S2 | 1.6377 | lr_bracket.md | Transformer (RoPE) @ 0.0018 @ 0 @ 0 @ A100 | Validation BPC |
| S3 | 1.6370 | lr_bracket.md | Transformer (RoPE) @ 0.0018 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| S4 | 0.02 (tie threshold) | derived | rule fixed in `results/v3_tuning/aggregate_tuning.py`; v1 TR2o seed std 0.0174 (`results/table_v.md`) | - |

## Transformer vs. LSTM

| id | value | file | row | column |
|---|---|---|---|---|
| T1 | 1.4532 ± 0.0055 | table_v_tie.md | Transformer (RoPE) | BPC |
| T2 | 1.515 | table_v_tie.md | LSTM-3.2M | BPC |
| T3 | 1.556 | table_v_tie.md | LSTM-2.1M | BPC |
| T4 | 3,169,280 | table_v_tie.md | Transformer (RoPE) | #Params |
| T5 | 3,177,052 | table_v_tie.md | LSTM-3.2M | #Params |
| T6 | 2,115,072 | table_v_tie.md | LSTM-2.1M | #Params |
| T7 | 0.0280 | table_v_tie.md | Transformer (RoPE) | s/step |
| T8 | 0.0664 | table_v_tie.md | LSTM-3.2M | s/step |
| T9 | 0.0526 | table_v_tie.md | LSTM-2.1M | s/step |
| T10 | 1.9× | derived | 0.0526 / 0.0280 = 1.88 (T9 / T7) | - |
| T11 | -0.062 ± 0.004 | table_v_tie.md | notes | - |
| T12 | -0.103 ± 0.003 | table_v_tie.md | notes | - |
| T13 | 1.4412 ± 0.0003 | table_v_bestlr.md | Transformer (RoPE) | BPC |
| T14 | 1.496 | table_v_bestlr.md | LSTM-3.2M | BPC |
| T15 | 1.5346 ± 0.0023 | table_v_bestlr.md | LSTM-2.1M | BPC |
| T16 | -0.055 ± 0.003 | table_v_bestlr.md | notes | - |
| T17 | 0.012 ± 0.005 | table_v_bestlr.md | notes | - |
| T18 | 0.020 ± 0.002 | table_v_bestlr.md | notes | - |
| T19 | 0.021 ± 0.006 | table_v_bestlr.md | notes | - |
| T20 | 0.0033 | table_v_tie.md | Transformer (RoPE) | ECE |
| T21 | 0.0066 | table_v_tie.md | LSTM-2.1M | ECE |
| T22 | 3.6 points | derived | Acc_rare 50.21 − 46.58 = 3.63 (table_v_tie.md, Transformer (RoPE) and LSTM-3.2M) | - |
| T23 | 0.9 points | derived | Acc_comm 80.00 − 79.15 = 0.85 | - |
| T24 | 0.042 | derived | 1.4957 − 1.4532 = 0.0425 (T14, T1) | - |
| T25 | 0.014 | lr_bracket.md | notes | - |
| T26 | 0.004 | lr_bracket.md | notes | - |
| T27 | 1.6284 | lr_bracket.md | LSTM-3.2M @ 0.0108 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| T28 | 1.5825 | lr_bracket.md | Transformer (RoPE) @ 0.0027 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| T29 | 1.5684 | lr_bracket.md | Transformer (RoPE) @ 0.00405 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| T30 | 1.5980 | lr_bracket.md | Transformer (RoPE) @ 0.006075 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| T31 | 1.6325 | lr_bracket.md | LSTM-3.2M @ 0.0072 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| T32 | 1.6325 | lr_bracket.md | LSTM-3.2M @ 0.0162 @ 0 @ 0 @ RTX 3090† | Validation BPC |
| T33 | 48,500-49,999 | training_summary.md | notes | - |
| T34 | 39 runs | training_summary.md | notes | - |
| T35 | 0.065--0.072 | derived | training_summary.md "Gap at end", rows Transformer (RoPE), LSTM-3.2M, LSTM-2.1M and their lowest-BPC rows: 0.0649--0.0717 | - |
| T36 | 0.021--0.042 | derived | training_summary.md "Gap at 2k", same six rows: 0.0214--0.0415 | - |
| T37 | 0.38 passes | derived | 2,000 × 128 × 128 / 85,499,999 = 0.383 | - |
| T38 | 0.0664 | table_v_tie.md | LSTM-3.2M | s/step |
| T39 | 0.055 (validation, 1.8e-3 → 2.7e-3 at 10k) | derived | 1.63705 − 1.58254 = 0.0545 (S3, T28) | - |
| T40 | 0.012 | table_v_bestlr.md | notes | - |

## Positional encodings and dropout

| id | value | file | row | column |
|---|---|---|---|---|
| P1 | 1.4626 ± 0.0022 | pe_comparison.md | ALiBi | BPC |
| P2 | 1.4634 ± 0.0032 | pe_comparison.md | Sinusoidal | BPC |
| P3 | 1.4835 ± 0.0015 | pe_comparison.md | Hybrid | BPC |
| P4 | 1.4921 ± 0.0027 | pe_comparison.md | Learned | BPC |
| P5 | 1.5004 ± 0.0243 | pe_comparison.md | Relative bias | BPC |
| P6 | 1.6201 ± 0.0211 | pe_comparison.md | NoPE | BPC |
| P7 | +0.009 ± 0.006 | pe_comparison.md | ALiBi | ΔBPC vs RoPE |
| P8 | +0.010 ± 0.002 | pe_comparison.md | Sinusoidal | ΔBPC vs RoPE |
| P9 | +0.039 ± 0.008 | pe_comparison.md | Learned | ΔBPC vs RoPE |
| P10 | +0.047 ± 0.026 | pe_comparison.md | Relative bias | ΔBPC vs RoPE |
| P11 | +0.167 ± 0.016 | pe_comparison.md | NoPE | ΔBPC vs RoPE |
| P12 | +0.030 ± 0.004 | pe_comparison.md | Hybrid | ΔBPC vs RoPE |
| P13 | 0.024 (std) | pe_comparison.md | Relative bias | BPC |
| P14 | 0.120 | derived | 1.6201 − 1.5004 = 0.1197 (P6, P5) | - |
| P15 | 0.0018 | pe_comparison.md | Relative bias | ECE |
| P16 | 0.0039 | pe_comparison.md | Sinusoidal | ECE |
| P17 | 32,768 extra parameters | derived | 3,202,048 − 3,169,280 (pe_comparison.md, Learned / RoPE #Params) | - |
| P18 | 4,080 | derived | 3,173,360 − 3,169,280 | - |
| P19 | 0.009 (discussion) | pe_comparison.md | ALiBi | ΔBPC vs RoPE |
| P20 | 0.03--0.04 (discussion) | derived | P12, P9 | - |
| D1 | 1.500 | dropout_ablation.md | 0.1 | BPC |
| D2 | +0.047 ± 0.006 | dropout_ablation.md | notes | - |
| D3 | 0.0125 | dropout_ablation.md | 0.1 | ECE |
| D4 | 0.067 | dropout_ablation.md | 0.0 (chosen) | val − train gap (nats) |
| D5 | 0.012 | dropout_ablation.md | 0.1 | val − train gap (nats) |

## Length extrapolation

| id | value | file | row | column |
|---|---|---|---|---|
| E1 | 1.445 | extrapolation.md | ALiBi | W=256 |
| E2 | 1.420 | extrapolation.md | ALiBi | W=512 |
| E3 | 1.413 | extrapolation.md | ALiBi | W=1024 |
| E4 | 0.041 | derived | 1.4532 − 1.4127 = 0.0405 (T1, E3) | - |
| E5 | 0.100 | derived | 1.5122 − 1.4127 = 0.0995 (E9, E3) | - |
| E6 | 4.35 | extrapolation.md | RoPE | W=256 |
| E7 | 4.85 | extrapolation.md | RoPE | W=512 |
| E8 | 1.535 | extrapolation.md | Relative bias (clamped) | W=256 |
| E9 | 1.512 | extrapolation.md | LSTM-3.2M | W=1024 |
| E10 | 1.513 | extrapolation.md | LSTM-3.2M | W=256 |
| E11 | 1.453 | per_position_buckets_W1024.md | RoPE | 64-127 |
| E12 | 3.506 | per_position_buckets_W1024.md | RoPE | 128-191 |
| E13 | 1.654 | per_position_buckets_W1024.md | NoPE | 128-191 |
| E14 | 1.608 | per_position_buckets_W1024.md | NoPE | 64-127 |
| E15 | 2.326 | per_position_buckets_W1024.md | NoPE | 192-255 |
| E16 | 4.5 | per_position_buckets_W1024.md | NoPE | 960-1023 |
| E17 | 1.989 | extrapolation.md | NoPE | W=256 |
| E18 | 1.679 | extrapolation.md | NoPE | W=128 |
| E19 | 4.058 | extrapolation.md | RoPE + PI | W=1024 |
| E20 | 3.811 | extrapolation.md | Sinusoidal + PI | W=1024 |
| E21 | +2.645 ± 0.089 | extrapolation.md | notes | - |
| E22 | +2.399 ± 0.067 | extrapolation.md | notes | - |
| E23 | 3.964 | per_position_buckets_W1024.md | RoPE + PI | 0-63 |
| E24 | 27/27 | extrapolation.md | notes | - |
| E25 | 1.8e-09 | extrapolation.md | notes | - |
| E26 | position ~200 (relative bias starts to rise) | derived | per_position_buckets_W1024.md, Relative bias: 1.495 at 128-191, 1.564 at 192-255 | - |

## Attention

| id | value | file | row | column |
|---|---|---|---|---|
| X1 | 9.8 / 3.2 / 5.1 / 24.4 | attention_summary.md | RoPE @ 128 | Effective span (chars) |
| X2 | 0.91 / 0.97 / 0.92 / 0.68 | attention_summary.md | RoPE @ 128 | Mass on last 8 chars |
| X3 | 34.2 / 31.6 / 31.3 / 36.6 | attention_summary.md | ALiBi @ 128 | Effective span (chars) |
| X4 | 0.52 / 0.60 / 0.58 / 0.49 | attention_summary.md | ALiBi @ 128 | Mass on last 8 chars |
| X5 | 0.27--0.36 | derived | attention_summary.md, "Best word-boundary head" at W = 128 over the six explicit encodings: 0.272 (Hybrid) -- 0.356 (Relative bias) | - |
| X6 | 10--13× | derived | 0.272 / 0.028 = 9.7, 0.356 / 0.028 = 12.7 | - |
| X7 | 217--285 | derived | X8, X9: min 217.1, max 284.7 | - |
| X8 | 244.9 / 218.4 / 217.1 / 252.0 | attention_summary.md | RoPE @ 512 | Effective span (chars) |
| X9 | 284.7 / 222.5 / 221.4 / 251.6 | attention_summary.md | Sinusoidal @ 512 | Effective span (chars) |
| X10 | 0.07 / 0.08 / 0.11 / 0.10 | attention_summary.md | RoPE @ 512 | Mass on last 8 chars |
| X11 | 0.06 / 0.02 / 0.02 / 0.03 | attention_summary.md | Sinusoidal @ 512 | Mass on last 8 chars |
| X12 | 1.1--2.5 nats | derived | Entropy (nats), W = 512 minus W = 128 rows, RoPE: +1.91 / +1.16 / +1.10 / +1.64; Sinusoidal: +2.49 / +1.99 / +2.04 / +2.21 | - |
| X13 | 0.056 | attention_summary.md | RoPE @ 512 | Best word-boundary head |
| X14 | 0.010 | attention_summary.md | Sinusoidal @ 512 | Best word-boundary head |
| X15 | 0.41 / 0.50 / 0.48 / 0.42 | attention_summary.md | ALiBi @ 512 | Mass on last 8 chars |
| X16 | 106.6 / 103.2 / 98.8 / 113.4 | attention_summary.md | ALiBi @ 512 | Effective span (chars) |
| X17 | 0.287 | attention_summary.md | ALiBi @ 512 | Best word-boundary head |
| X18 | 0.43 / 0.22 / 0.56 / 0.54 | attention_summary.md | Relative bias @ 512 | Mass on last 8 chars |
| X19 | 55.8 / 63.2 / 3.4 / 9.1 | attention_summary.md | NoPE @ 128 | Effective span (chars) |
| X20 | 285.9 / 317.3 / 11.5 / 51.5 | attention_summary.md | NoPE @ 512 | Effective span (chars) |
| X21 | 0.282 | attention_summary.md | NoPE @ 128 | Best word-boundary head |
| X22 | 0.169 | attention_summary.md | NoPE @ 512 | Best word-boundary head |
| X23 | 51.6 / 10.5 / 14.8 / 141.1 | attention_summary.md | RoPE + PI | Effective span (chars) |
| X24 | 0.57 / 0.70 / 0.79 / 0.37 | attention_summary.md | RoPE + PI | Mass on last 8 chars |
| X25 | 0.334 | attention_summary.md | RoPE + PI | Best word-boundary head |
| X26 | 0.330 | attention_summary.md | RoPE @ 128 | Best word-boundary head |
| X27 | 1 / 1 / 1 / 1 | attention_short_range.md | RoPE, W=128 | Peak distance (chars) |
| X28 | 0 / 5 / 3 / 1 | attention_short_range.md | RoPE + PI, W=512 | Peak distance (chars) |
| X29 | 0.70 / 0.76 / 0.76 / 0.42 | attention_short_range.md | RoPE, W=128 | Mean weight on distances 0-2 |
| X30 | 0.25 / 0.19 / 0.31 / 0.16 | attention_short_range.md | RoPE + PI, W=512 | Mean weight on distances 0-2 |
| X31 | 0.028 / 0.004 (uniform) | derived | `space_mass_uniform` in `results/v3/raw/TR-final_s42/attention_W{128,512}/stats.json` (attention_summary.csv, column space_uniform) | - |

## Appendix

| id | value | file | row | column |
|---|---|---|---|---|
| Z1 | 1.9368, 2.0952, 2.1648, 2.0593, 1.9691, 1.9742 | pilot | Table "Preliminary experiments", "Best Loss" 1.3425, 1.4523, 1.5005, 1.4274, 1.3649, 1.3684 nats, divided by ln 2 | - |
| Z2 | 62.21, 59.08, 58.01, 60.94, 63.67, 61.91 / 57.90, 54.84, 53.96, 55.67, 57.60, 57.53 | pilot | same table, best Acc_last / best Acc_total | - |
| Z3 | 2.161, 2.128, 2.215 / 58.01, 58.89, 57.13 | pilot | Table "Ablation on context length" | - |
| Z4 | 2.0596, 2.0898, 2.0607, 1.9559, 1.9984 / 60.11, 58.98, 60.16, 62.40, 62.40 | pilot | Table "Ablation on different positional encoding methods" | - |
| Z5 | 2.0686 | pilot | Table "Ablation on pure relative positional embedding", E8 row | - |
| Z6 | generated sample (first 372 characters) | derived | `results/v4/raw/TR-final_s42/sample_best_loss.txt` | - |
