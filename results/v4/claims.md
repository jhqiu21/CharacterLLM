# Numbers behind the report

Every result number in the report (LaTeX source in the separate paper repository), with the file and row/field it comes
from. Tables in the report are `\input` from `results/v4/tables/*.tex` and are not repeated here; this list covers the numbers
in the prose, captions and hand-written appendix tables.

- `file` is a Markdown table under `results/v4/tables/`, or a path under `results/` for the follow-up rounds (`v5/tables/...`:
  repairs of RoPE beyond the training length; `v6/tables/...`: capacity and accuracy); `python results/v4/check_claims.py`
  checks every such row (exact match, or the table value rounded half-up to the decimals shown in the text; the sign is
  ignored when the text gives none).
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
| R7 | up to about four passes cost little; the value of further passes decays | external | Muennighoff et al. 2023, abstract (https://arxiv.org/abs/2305.16264) | - |
| R8 | 0.6 and 1.3 bits per character, 27 symbols, 100 characters of context | external | Shannon 1951, bounds table after Table I, column 100: upper 1.3, lower .6; spaces included as a 27th letter (p. 55) | - |
| R9 | errors occur most often at the beginning of words and syllables | external | Shannon 1951, p. 55: "The errors, as would be expected, occur most frequently at the beginning of words and syllables" | - |
| R10 | 80 of 100 trials at the first attempt, 99 preceding letters, dictionary and frequency tables | external | Shannon 1951, Table I, column 100, row 1 (= 80), and p. 56-57 (aids used by the subject) | - |

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
| X29 | 0.70 / 0.76 / 0.76 / 0.42 | attention_short_range.md | RoPE, W=128 | Profile summed over d = 0-2 |
| X30 | 0.25 / 0.19 / 0.31 / 0.16 | attention_short_range.md | RoPE + PI, W=512 | Profile summed over d = 0-2 |
| X31 | 0.028 / 0.004 (uniform) | derived | `space_mass_uniform` in `results/v3/raw/TR-final_s42/attention_W{128,512}/stats.json` (attention_summary.csv, column space_uniform) | - |

## Repairs of RoPE beyond the training length (v5)

| id | value | file | row | column |
|---|---|---|---|---|
| F1 | 0.023 (abstract, intro: SelfExtend gain) | v5/tables/differences.md | RoPE + SelfExtend floor, near 64 − RoPE at W=128, stride 1 | W=512 |
| F2 | 4, 8 and 16 (SelfExtend groups, n = 64) | derived | `utils/extrapolation_eval.self_extend_group(128, W, 64, "floor")` for W = 256, 512, 1024 | - |
| F3 | N = 128, n = 64 selected | derived | † rows of v5/tables/inference_variants_val.md (lowest validation BPC at W = 1024 per family) | - |
| F4 | 1.4532 | v5/tables/paper_repair.md | RoPE | W=128, stride 1 |
| F5 | 1.4535 | v5/tables/paper_repair.md | RoPE + local attention 128 | W=512 |
| F6 | +0.0003 ± 0.0006 | v5/tables/differences.md | RoPE + local attention 128 − RoPE at W=128, stride 1 | W=512 |
| F7 | 1.488 | v5/tables/inference_variants.md | RoPE + local attention 64 | W=256 |
| F8 | 1.536 | v5/tables/inference_variants.md | RoPE + local attention 32 | W=256 |
| F9 | 1.4304 | v5/tables/paper_repair.md | RoPE + SelfExtend (near 64) | W=512 |
| F10 | 0.0228 ± 0.0057 | v5/tables/differences.md | RoPE + SelfExtend floor, near 64 − RoPE at W=128, stride 1 | W=512 |
| F11 | better in every seed (SelfExtend vs truncation, W = 512) | derived | per seed 42 / 43 / 44: -0.0163 / -0.0266 / -0.0255 (`raw/TR-final_s*/extrapolation_sefloor64.json` minus the run's full-context BPC) | - |
| F12 | 2.55 | v5/tables/differences.md | RoPE + SelfExtend floor, near 64 − RoPE + PI | W=512 |
| F13 | +0.0051 ± 0.0072 | v5/tables/differences.md | RoPE + SelfExtend floor, near 64 − RoPE at W=128, stride 1 | W=1024 |
| F14 | 1.4337 | v5/tables/paper_repair.md | RoPE + 512-window fine-tuning @ 16.4M | W=512 |
| F15 | 1.4133 | v5/tables/paper_repair.md | RoPE + 512-window fine-tuning @ 65.5M | W=512 |
| F16 | 1.4588 | v5/tables/paper_repair.md | RoPE + PI + 512-window fine-tuning @ 16.4M | W=512 |
| F17 | 1.4201 | v5/tables/paper_repair.md | RoPE + PI + 512-window fine-tuning @ 65.5M | W=512 |
| F18 | 0.0252 ± 0.0022 | v5/tables/differences.md | PI fine-tuning − 512-window fine-tuning (does PI add anything after adaptation?) | W=512 |
| F19 | 0.0068 ± 0.0029 | v5/tables/budget_differences.md | PI fine-tuning − 512-window fine-tuning, 4,000 it. | W=512 |
| F20 | PI worse in every seed (1,000 and 4,000 it., W = 512) | derived | per seed 42 / 43 / 44: +0.0245 / +0.0234 / +0.0276 (1k), +0.0035 / +0.0085 / +0.0085 (4k) | - |
| F21 | 0.059 | v5/tables/differences.md | RoPE + PI + 512-window fine-tuning − its base model, W=128 stride 1 | W=128 |
| F22 | 0.017 | v5/tables/differences.md | RoPE + 512-window fine-tuning − its base model, W=128 stride 1 | W=128 |
| F23 | (it. 0) | v5/tables/ft_lr_grid.md | RoPE + 128-window fine-tuning | lr 3e-5 |
| F24 | 1.724 | v5/tables/paper_repair.md | RoPE + 512-window fine-tuning @ 65.5M | W=1024 |
| F25 | 5.400 | v5/tables/paper_repair.md | RoPE + PI + 512-window fine-tuning @ 65.5M | W=1024 |
| F26 | 1 / 1 / 1 / 1 | v5/tables/attention_short_range.md | RoPE + PI + 512-window fine-tuning, W=512 | Peak distance (chars) |
| F27 | 0.72 / 0.81 / 0.77 / 0.38 | v5/tables/attention_short_range.md | RoPE + PI + 512-window fine-tuning, W=512 | Profile summed over d = 0-2 |
| F28 | 0.70 / 0.76 / 0.76 / 0.42 | v5/tables/attention_short_range.md | RoPE, W=128 | Profile summed over d = 0-2 |
| F29 | -0.0063 ± 0.0083 | v5/tables/budget_differences.md | 512-window fine-tuning, 4,000 it. − ALiBi | W=512 |
| F30 | signs differing across seeds | derived | per seed 42 / 43 / 44: +0.0013 / -0.0151 / -0.0053 | - |
| F31 | 1.3969 | v5/tables/paper_repair.md | ALiBi + 512-window fine-tuning | W=1024 |
| F32 | 0.043 (discussion: PI deficit at W = 128, full context, 1,000 it.) | derived | v5/tables/finetune.md W=128 stride 1: 1.5127 − 1.4698 = 0.0429 (per seed +0.0415 / +0.0409 / +0.0462) | - |
| F33 | 0.025 (discussion) | v5/tables/differences.md | PI fine-tuning − 512-window fine-tuning (does PI add anything after adaptation?) | W=512 |
| F34 | iteration 49,999; warmup 50 / 200; lr 3e-4 / 1e-3 / 3e-5 (appendix table) | derived | `results/v5/raw/ft*/config.yaml` (run.init_from_step, scheduler.warmup_iters, training.learning_rate); * rows of v5/tables/ft_lr_grid.md | - |

## Capacity and next-character accuracy (v6)

| id | value | file | row | column |
|---|---|---|---|---|
| C1 | 70.2% (abstract) | v6/tables/paper_capacity.md | ALiBi-3.2M | Acc_all (%) |
| C2 | 73.2% (abstract, intro, analysis, conclusion) | v6/tables/paper_capacity.md | ALiBi-14.2M | Acc_all (%) |
| C3 | 14,204,928 | v6/tables/runs.md | v6-AL-14M_s42 | parameters |
| C4 | 819.2M characters | derived | 50,000 iterations × 32 × 512 = 819,200,000 (`timing.json` train_tokens) | - |
| C5 | 3.6e-3 * | v6/tables/paper_search.md | ALiBi-14.2M @ 3.6e-3 * | Peak lr |
| C38 | 3.6e-3 * | v6/tables/paper_search.md | ALiBi-3.2M @ 3.6e-3 * | Peak lr |
| C6 | 73.20 | v6/tables/paper_capacity.md | ALiBi-14.2M | Acc_all (%) |
| C7 | 70.23 | v6/tables/paper_capacity.md | ALiBi-3.2M | Acc_all (%) |
| C8 | +2.97 ± 0.04 | v6/tables/paper_capacity.md | ALiBi-14.2M − ALiBi-3.2M | Acc_all (%) |
| C9 | 0.136 (lower BPC) | v6/tables/paper_capacity.md | ALiBi-14.2M − ALiBi-3.2M | BPC |
| C10 | +2.76 | v6/tables/paired_test.md | 128 @ Acc_all (%) | ALiBi-14.2M − ALiBi-3.2M |
| C11 | +3.04 | v6/tables/paired_test.md | 256 @ Acc_all (%) | ALiBi-14.2M − ALiBi-3.2M |
| C12 | +2.95 ± 0.04 | v6/tables/formal_val.md | mean ± std | Δ Acc (pp) |
| C13 | 4.35 (rare characters) | v6/tables/paper_capacity.md | ALiBi-14.2M − ALiBi-3.2M | Acc_rare (%) |
| C14 | 0.63 | derived | 70.23 − 69.60 (v6/tables/paper_capacity.md, ALiBi-3.2M and Transformer (ALiBi) + 512-window fine-tuning) | - |
| C15 | 163.8M | derived | 10,000 × 16,384 = 163,840,000 | - |
| C16 | +1.92 | v6/tables/budget_scaling.md | ALiBi-3.2M | 5× characters: Δ Acc (pp) |
| C17 | +2.41 | v6/tables/budget_scaling.md | ALiBi-14.2M | 5× characters: Δ Acc (pp) |
| C18 | +2.50 | v6/tables/budget_scaling.md | ALiBi-14.2M − ALiBi-3.2M | val Acc_all, 164M chars (%) |
| C19 | +2.99 | v6/tables/budget_scaling.md | ALiBi-14.2M − ALiBi-3.2M | val Acc_all, 819M chars (%) |
| C20 | 70.64 | v6/tables/budget_scaling.md | ALiBi-14.2M | val Acc_all, 164M chars (%) |
| C21 | 70.06 | v6/tables/budget_scaling.md | ALiBi-3.2M | val Acc_all, 819M chars (%) |
| C22 | 17 (minutes) | v6/tables/runs.md | v6search-AL-14M_lr3.6e-3 | training min |
| C23 | 28 (minutes) | v6/tables/runs.md | v6-AL-3M_s42 | training min |
| C24 | 0.098 | v6/tables/runs.md | v6-AL-14M_s42 | s / update (median) |
| C25 | 0.031 | v6/tables/runs.md | v6-AL-3M_s42 | s / update (median) |
| C26 | 3.1 times | derived | 0.0982 / 0.0314 = 3.13 (C24, C25) | - |
| C27 | lowest validation loss in the last 1,000 iterations | derived | v6/tables/curves.md, "iteration of lowest val BPC": 49,000-49,999 in all 6 runs | - |
| C28 | 0.7--0.9 points | derived | v6/tables/curves.md, val Acc @ 50k − @ 40k: 0.74 / 0.77 / 0.81 (3.2M), 0.80 / 0.69 / 0.85 (14.2M) | - |
| C29 | 0.13--0.14 | derived | v6/tables/curves.md, "val − train BPC @ 50k", ALiBi-14.2M: 0.1371 / 0.1324 / 0.1355 | - |
| C30 | 0.08--0.09 | derived | same, ALiBi-3.2M: 0.0888 / 0.0815 / 0.0830 | - |
| C31 | about six of the 9.6 passes | derived | `figures/training_curves.*`: the ALiBi-14.2M gap steepens after about 500M characters = 5.8 passes of 85.5M | - |
| C32 | 5× (intro: capacity helped more than five times more characters) | derived | 819.2M / 163.84M = 5 (C16-C19) | - |
| C33 | 3.2M, 14.2M | derived | 3,169,280 and 14,204,928 (v6/tables/runs.md) | - |
| C34 | 0.08 points (limitations) | derived | 68.22 − 68.14 (v6/tables/paper_search.md, ALiBi-3.2M @ 7.2e-3 † and @ 3.6e-3 *); exact 68.2203 − 68.1393 = 0.081 | - |
| C35 | 0.1 point (possible overstatement) | derived | C34 | - |
| C36 | 4.5× | derived | 14,204,928 / 3,169,280 = 4.48 | - |
| C37 | q, j, x, z, k / space, e, t, a, n | derived | `frequency_groups` in `results/v6/raw/v6-AL-14M_s42/val_eval.json` (5 least / most frequent training characters) | - |

## Why accuracy is limited (v6 analysis)

| id | value | file | row | column |
|---|---|---|---|---|
| Y1 | 73.4 | v6/tables/test_by_W.md | ALiBi-14.2M (512, from scratch) @ Acc_all (%) | W=1024 |
| Y2 | 94.7 | v6/tables/paper_errors.md | space | Acc, 14.2M (%) |
| Y3 | 87.8 | v6/tables/paper_errors.md | 4th+ letter | Acc, 14.2M (%) |
| Y4 | 31.7 | v6/tables/paper_errors.md | 1st letter | Acc, 14.2M (%) |
| Y5 | 3.32 | v6/tables/paper_errors.md | 1st letter | BPC, 14.2M |
| Y6 | 17.1 | v6/tables/paper_errors.md | 1st letter | Targets (%) |
| Y7 | 43.6 | v6/tables/paper_errors.md | 1st letter | Errors, 14.2M (%) |
| Y8 | 62.8 | v6/tables/paper_errors.md | 2nd letter | Acc, 14.2M (%) |
| Y9 | two thirds | derived | 43.6 + 22.8 = 66.4 (v6/tables/paper_errors.md, Errors column) | - |
| Y10 | 82.8 | v6/tables/paper_capacity.md | ALiBi-14.2M | Top-2 (%) |
| Y11 | 91.9 | v6/tables/paper_capacity.md | ALiBi-14.2M | Top-5 (%) |
| Y12 | 26.8, 9.6 and 18.7 points | derived | 100 − 73.20; 82.77 − 73.20 = 9.57; 91.90 − 73.20 = 18.70 | - |
| Y13 | 3.1--4.1 points | derived | v6/tables/paper_errors.md Δ: +3.12, +3.83, +4.05, +3.15 | - |
| Y14 | 0.7 points (spaces) | v6/tables/paper_errors.md | space | Δ |
| Y15 | about 1.4 points per doubling | derived | 2.97 / log2(14,204,928 / 3,169,280) = 2.97 / 2.164 = 1.37 | - |
| Y16 | 37.8M, about 75% | derived | 12 layers, d_model 512, 8 heads: 37,832,704 parameters (`train.build_model`); 73.20 + 1.37 × log2(37.83 / 14.20) = 75.1 | - |
| Y17 | about five more doublings | derived | (80 − 73.20) / 1.37 = 5.0 | - |
| Y18 | 0.18 | derived | 73.38 − 73.20 (v6/tables/test_by_W.md, ALiBi-14.2M Acc_all, W=1024 and W=512) | - |
| Y19 | 0.30 | derived | 70.53 − 70.23 (same, ALiBi-3.2M) | - |
| Y20 | 1.234 | v6/tables/test_by_W.md | ALiBi-14.2M (512, from scratch) @ BPC | W=1024 |
| Y21 | 69.3% | v6/tables/test_W512.md | Transformer (ALiBi), 128 only | Acc_all (%) |
| Y22 | 1.420 | v6/tables/test_W512.md | Transformer (ALiBi), 128 only | BPC |
| Y23 | 1.244 | v6/tables/paper_capacity.md | ALiBi-14.2M | BPC |
| Y24 | about 22 points per BPC | derived | (73.20 − 69.31) / (1.4196 − 1.2440) = 22.2; least-squares slope over the five models of v6/tables/test_W512.md: 22.1 | - |
| Y25 | about 0.94 BPC | derived | least-squares line over v6/tables/test_W512.md: Acc = 100.75 − 22.14 × BPC, 80% at 0.937 | - |
| Y26 | twenty times larger | derived | 277M / 14.2M = 19.5 (R4) | - |
| Y27 | 17% / 44% (intro, abstract) | derived | Y6, Y7 rounded | - |

## Appendix

| id | value | file | row | column |
|---|---|---|---|---|
| Z1 | 1.9368, 2.0952, 2.1648, 2.0593, 1.9691, 1.9742 | pilot | Table "Preliminary experiments", "Best Loss" 1.3425, 1.4523, 1.5005, 1.4274, 1.3649, 1.3684 nats, divided by ln 2 | - |
| Z2 | 62.21, 59.08, 58.01, 60.94, 63.67, 61.91 / 57.90, 54.84, 53.96, 55.67, 57.60, 57.53 | pilot | same table, best Acc_last / best Acc_total | - |
| Z3 | 2.161, 2.128, 2.215 / 58.01, 58.89, 57.13 | pilot | Table "Ablation on context length" | - |
| Z4 | 2.0596, 2.0898, 2.0607, 1.9559, 1.9984 / 60.11, 58.98, 60.16, 62.40, 62.40 | pilot | Table "Ablation on different positional encoding methods" | - |
| Z5 | 2.0686 | pilot | Table "Ablation on pure relative positional embedding", E8 row | - |
| Z6 | generated sample (first 372 characters) | derived | `results/v4/raw/TR-final_s42/sample_best_loss.txt` | - |
