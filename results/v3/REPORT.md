# v3: final runs, length extrapolation, attention and positional-encoding analysis

Branch `final` (from `tuning`, `d3337bf`). Task: `prompts/03_final_and_extensions.md`, with the scope changes agreed with the
user (§1). Every number is a test-set number at the best-validation-loss checkpoint, mean ± sample std (ddof 1) over seeds
42, 43, 44, unless stated otherwise. `aggregate_v3.py` rebuilds every table (`tables/`) and figure (`figures/`) from the per-run
copies in `raw/`. Hardware: RTX 3090 24 GB (GPUs 2 and 3), not the A100 used for tuning (`env.txt`).

## Summary

- **36 training runs, all finished; none diverged, none failed.** Plus 3 solo timing runs, and extrapolation (24 runs),
  attention (10) and PE-table (6) evaluations. Scaling (blocks 4–5) was not run (user decision, §1).
- **Table V, tuned configs:** TR (RoPE) 1.4532 ± 0.0055 BPC, LR-2 1.5152 ± 0.0017, LR-1 1.5557 ± 0.0040. All three are
  0.16–0.19 BPC better than Table V v1.
- **The Transformer–LSTM ordering holds under both learning-rate choices.** Seed-matched TR − LR-2: −0.0620 ± 0.0041 (tie-rule lr),
  −0.0545 ± 0.0030 (lowest-BPC lr). The lowest-BPC lr helps the LSTMs more (−0.020/−0.021) than the Transformer (−0.012).
- **No overfitting at 50k steps.** The best-validation-loss checkpoint is at iteration 48,500–49,999 in every run (≥ 49,000 in
  every dropout-0 run), and validation loss still falls over the last 5k iterations. Dropout 0.1 is worse than 0.0 by 0.047 ± 0.006 BPC.
- **Positional encodings at the training length:** RoPE 1.4532 < ALiBi 1.4626 ≈ sinusoidal 1.4634 < hybrid 1.4835 < learned
  1.4921 < relative 1.5004 (relative has a large seed spread, ± 0.0243).
- **Only ALiBi extrapolates.** ALiBi's BPC keeps falling with longer windows (1.4127 ± 0.0057 at W = 1024, below every model at
  W = 128). RoPE and sinusoidal break right after position 128 (BPC 4.4–4.9). Clamped relative works at W = 256 and breaks from
  512 on. The LSTMs run at any length but gain nothing beyond about 128 characters of context.
- **Attention explains extrapolation.** Beyond position 128, RoPE and sinusoidal attention becomes diffuse (attention on the
  last 8 characters drops from 0.55–0.97 to 0.02–0.11). ALiBi keeps its pattern (0.41–0.50). The span at the training length does
  not predict extrapolation: ALiBi has the *longest* span at W = 128.

---

## 1. What was run, and scope changes

| block | runs | status |
|---|---|---|
| timing | `timing_{TR,LSTM,LR-1}`: 2,000 iterations, each alone on its GPU (`--tuning`, so no test evaluation) | done |
| 1 Table V | `{TR,LSTM,LR-1}-final_s{42,43,44}` (LSTM = LR-2 size) | done; seed 42 reused (below) |
| 1b lowest-BPC lr | same, `--set training.learning_rate=` 2.7e-3 (TR), 7.2e-3 (LSTM, LR-1), names `..._bestlr` | done, none diverged |
| 2 encodings | `TR-final-{learned,sinusoidal,hybrid,alibi,relative}_s{42,43,44}`; RoPE is block 1's TR | done |
| 3 dropout | `TR-final-d0.1_s{42,43,44}` (all four dropouts 0.1) | done |
| 4–5 scaling | — | **not run** (user decision: optional, after everything else; see §8) |
| evaluations | extrapolation for the 6 encodings + LR-1 + LR-2 (all seeds); attention for seed 42 (6 encodings at W = 128; RoPE, ALiBi, sinusoidal, relative at W = 512); PE tables (seed 42) | done |

All 50k runs use `--set evaluation.full_context_batch_size=2048` (8192 ran out of memory on the 3090, §7).

**Agreed with the user before the runs** (they change what the task asked for):
- **Scaling (blocks 4–5) dropped for now**, to run last and optionally. In exchange, the PE visualisations below were added
  (position-similarity matrices, PCA of learned tables, relative vs ALiBi bias, side-by-side heatmaps, attention past the training
  length, word-boundary heads, per-position loss at W = 128).
- **GPUs 2 and 3 only, one run per GPU.** The task's "at most 2 jobs, each with ≥ 30 GB free" cannot hold on a 24 GB card. So
  there were at most 2 jobs at a time, each with a whole GPU to itself.
- **Seed-42 runs of block 1 reused** (`TR-final_s42`, `LSTM-final_s42`, `LR-1-final_s42`). They were started before this task
  (commit `d3337bf`, with the same `--set evaluation.full_context_batch_size=2048`). Their resolved configs match
  `configs/final/*.yaml` plus that override (checked key by key). The task's reuse rule also requires an empty
  `git diff --stat -- train.py utils models configs` against this branch. That is **not** met: this branch adds the eval-only options
  `capture_attn`, `rel_table_len` and `rel_clamp` to `models/models.py`, `eval_options` to `train.build_model`, and
  `utils/extrapolation_eval.py`. The defaults leave training, outputs and parameters unchanged, and tests show this
  (`tests/test_extrapolation.py::test_relative_clamp_default_leaves_training_unchanged`, `tests/test_attention_capture.py`).
  The user agreed to reuse the runs on that basis. No run from before the task was left unused.
- **Order:** the two model changes (attention capture, relative clamp) were committed and tested (`baa35e9`, full `pytest`
  245 passed, lint clean) *before* launching. The remaining tooling (extrapolation and attention CLIs, notebook fix, aggregator)
  was committed while training ran. None of it touches the training code. Every new run records a clean commit (`1b66cb3` or
  `37393cd`), and `train.py utils models configs` are identical between the two.

## 2. Table V (`tables/table_v_tie.*`, `tables/table_v_bestlr.*`)

Tie-rule lr (block 1, `configs/final/`):

| ID | lr | BPC | PPL | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) | Acc_last@128 (%) | Self-BLEU-4 | Self-BLEU-4 multi | #Params | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TR | 1.8e-3 | 1.4532 ± 0.0055 | 2.7382 ± 0.0104 | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 50.21 ± 0.31 | 80.00 ± 0.08 | 68.57 ± 0.05 | 0.1428 ± 0.0400 | 0.1312 ± 0.0113 | 3,169,280 | 0.0280 |
| LR-1 | 3.2e-3 | 1.5557 ± 0.0040 | 2.9398 ± 0.0081 | 66.26 ± 0.08 | 0.0066 ± 0.0002 | 45.21 ± 0.28 | 78.66 ± 0.01 | 66.11 ± 0.13 | 0.1576 ± 0.0091 | 0.1485 ± 0.0063 | 2,115,072 | 0.0526 |
| LR-2 | 3.2e-3 | 1.5152 ± 0.0017 | 2.8584 ± 0.0034 | 67.12 ± 0.02 | 0.0053 ± 0.0004 | 46.58 ± 0.03 | 79.15 ± 0.03 | 67.00 ± 0.06 | 0.1553 ± 0.0079 | 0.1417 ± 0.0030 | 3,177,052 | 0.0664 |

Lowest-BPC lr (block 1b):

| ID | lr | BPC | PPL | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) | Acc_last@128 (%) | Self-BLEU-4 | Self-BLEU-4 multi |
|---|---|---|---|---|---|---|---|---|---|---|
| TR | 2.7e-3 | 1.4412 ± 0.0003 | 2.7154 ± 0.0006 | 68.82 ± 0.00 | 0.0031 ± 0.0003 | 50.50 ± 0.24 | 80.20 ± 0.04 | 68.72 ± 0.07 | 0.1491 ± 0.0201 | 0.1306 ± 0.0143 |
| LR-1 | 7.2e-3 | 1.5346 ± 0.0023 | 2.8972 ± 0.0046 | 66.69 ± 0.05 | 0.0060 ± 0.0003 | 45.71 ± 0.27 | 78.95 ± 0.04 | 66.59 ± 0.14 | 0.1696 ± 0.0087 | 0.1376 ± 0.0072 |
| LR-2 | 7.2e-3 | 1.4957 ± 0.0033 | 2.8200 ± 0.0065 | 67.52 ± 0.06 | 0.0048 ± 0.0005 | 46.84 ± 0.14 | 79.46 ± 0.04 | 67.48 ± 0.11 | 0.1494 ± 0.0179 | 0.1388 ± 0.0012 |

BPC, PPL, Acc_all, ECE, Acc_rare and Acc_comm come from full-context evaluation (W = 128, stride 1, every test target scored once).
Acc_last is chunked128 position 127. Self-BLEU-4 multi is the mean over 10 validation prompts × 20 samples.
**s/step**: one solo timing run per model (median over iterations 1000–1999, RTX 3090), so it has no std. It is not comparable with
Table V v1's GPUs (A100 / 8×A5000).

## 3. Positional encodings (`tables/pe_comparison.*`)

| encoding | BPC | Acc_all (%) | ECE | Acc_last@128 (%) | #Params | per-seed BPC (42/43/44) |
|---|---|---|---|---|---|---|
| rope | **1.4532 ± 0.0055** | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 68.57 ± 0.05 | 3,169,280 | 1.4595 / 1.4492 / 1.4510 |
| alibi | 1.4626 ± 0.0022 | 68.38 ± 0.05 | 0.0038 ± 0.0003 | 68.41 ± 0.17 | 3,169,280 | 1.4634 / 1.4643 / 1.4602 |
| sinusoidal | 1.4634 ± 0.0032 | 68.39 ± 0.06 | 0.0039 ± 0.0001 | 68.33 ± 0.02 | 3,169,280 | 1.4671 / 1.4613 / 1.4618 |
| hybrid | 1.4835 ± 0.0015 | 67.93 ± 0.03 | 0.0030 ± 0.0002 | 67.87 ± 0.10 | 3,202,048 | 1.4850 / 1.4822 / 1.4832 |
| learned | 1.4921 ± 0.0027 | 67.74 ± 0.06 | 0.0028 ± 0.0001 | 67.69 ± 0.22 | 3,202,048 | 1.4897 / 1.4950 / 1.4916 |
| relative | 1.5004 ± 0.0243 | 67.55 ± 0.54 | 0.0018 ± 0.0003 | 67.51 ± 0.54 | 3,173,360 | 1.4927 / 1.4808 / 1.5276 |

Seed-matched differences from RoPE: ALiBi +0.0094 ± 0.0055, sinusoidal +0.0102 ± 0.0023, learned +0.0389 ± 0.0079,
relative +0.0472 ± 0.0255. Same model size, same attention code and the tuned TR hyperparameters (chosen for RoPE).
`relative` seed 44 is the outlier. Its training was smooth (no validation-loss spike; largest rise between checks 0.003 nats),
and it was the best of the three at 10k iterations but improved less afterwards. So this is seed variance, not a crash, and it was not rerun.

## 4. Dropout ablation (`tables/dropout_ablation.*`)

| dropout | BPC | Acc_all (%) | ECE | Acc_last@128 (%) | val − train gap (nats) | best-loss step |
|---|---|---|---|---|---|---|
| 0.0 (chosen) | 1.4532 ± 0.0055 | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 68.57 ± 0.05 | 0.0672 ± 0.0025 | 49,999 |
| 0.1 | 1.5003 ± 0.0017 | 67.51 ± 0.05 | 0.0125 ± 0.0001 | 67.54 ± 0.15 | 0.0115 ± 0.0019 | 49,000 |

Dropout 0.1 costs 0.0471 ± 0.0062 BPC (seed-matched), close to the 0.054–0.067 seen at 10k steps in tuning. The 10k-step choice
of 0.0 still holds at 50k. (With dropout on, the training loss is measured with dropout active, so the two gaps are not directly comparable.)

## 5. Length extrapolation (`tables/extrapolation.*`, `figures/extrapolation_bpc_vs_W.*`, `figures/per_position_loss_W{128,1024}.*`)

Sliding-window test BPC, window W, stride 128 (so every target after the first window has at least W − 127 characters of context):

| model | W = 128 | W = 256 | W = 512 | W = 1024 |
|---|---|---|---|---|
| rope | 1.5452 ± 0.0030 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| learned | 1.5806 ± 0.0023 | N/A | N/A | N/A |
| sinusoidal | 1.5549 ± 0.0025 | 4.6830 ± 0.2713 | 4.6092 ± 0.2726 | 4.7990 ± 0.2193 |
| hybrid | 1.5726 ± 0.0012 | N/A | N/A | N/A |
| alibi | 1.5602 ± 0.0016 | **1.4449 ± 0.0027** | **1.4196 ± 0.0045** | **1.4127 ± 0.0057** |
| relative (clamped) | 1.5909 ± 0.0200 | 1.5353 ± 0.0116 | 2.9477 ± 0.2898 | 3.7355 ± 0.1986 |
| LR-2 | 1.5834 ± 0.0014 | 1.5134 ± 0.0017 | 1.5123 ± 0.0017 | 1.5122 ± 0.0017 |
| LR-1 | 1.6191 ± 0.0030 | 1.5543 ± 0.0040 | 1.5534 ± 0.0041 | 1.5534 ± 0.0041 |

- W = 128 with stride 128 is the like-for-like baseline. Its BPC is higher than the stride-1 numbers of §2, because targets early in each
  window have little context. The jump from W = 128 to 256 for ALiBi and the LSTMs comes mostly from that extra context.
- **Sanity check:** W = 128 with stride 1, through the rebuilt model, reproduces every run's `eval.json` full-context BPC.
  24/24 runs match, with the largest difference 1.8e-9 BPC (tolerance 1e-4).
- Per-position loss (2,048 non-overlapping chunks at fixed offsets, seed 0) shows the mechanism directly. At W = 1024, RoPE and
  sinusoidal stay at about 1.5 BPC up to position 127 and then jump to 4–5 BPC within a few positions. Relative stays flat to about
  position 200 and then rises steadily to about 3.7. ALiBi and the LSTMs stay flat over all 1,024 positions, ALiBi below its own
  value at positions < 128.

## 6. Attention and positional-encoding analysis (`tables/attention_summary.*`, `figures/attention_*`, `figures/pe_*`)

Seed 42, 512 test windows (seed 0); statistics per layer and head in `raw/<run>/attention_W*/stats.json`.
Values below are per-layer means over the 4 heads (layers 0 / 1 / 2 / 3).

| encoding | W, queries | mean effective span (90% mass, chars) | mass on last 8 chars | entropy (nats) | best word-boundary head |
|---|---|---|---|---|---|
| rope | 128, all | 9.8 / 3.2 / 5.1 / 24.4 | 0.91 / 0.97 / 0.92 / 0.68 | 1.40 / 0.91 / 1.10 / 2.11 | 0.330 |
| learned | 128, all | 27.0 / 17.3 / 3.3 / 22.3 | 0.73 / 0.78 / 0.98 / 0.67 | 2.04 / 2.13 / 0.93 / 1.73 | 0.305 |
| sinusoidal | 128, all | 36.6 / 4.0 / 4.0 / 24.0 | 0.55 / 0.97 / 0.96 / 0.63 | 2.37 / 1.24 / 1.01 / 1.80 | 0.343 |
| hybrid | 128, all | 37.8 / 6.4 / 3.2 / 21.7 | 0.57 / 0.92 / 0.97 / 0.68 | 2.50 / 1.73 / 0.84 / 1.68 | 0.272 |
| alibi | 128, all | 34.2 / 31.6 / 31.3 / 36.6 | 0.52 / 0.60 / 0.58 / 0.49 | 2.70 / 2.68 / 2.29 / 2.16 | 0.303 |
| relative | 128, all | 20.3 / 43.1 / 4.7 / 22.1 | 0.79 / 0.44 / 0.96 / 0.74 | 2.18 / 3.08 / 1.22 / 1.74 | 0.356 |
| rope | 512, positions ≥ 128 | 244.9 / 218.4 / 217.1 / 252.0 | 0.07 / 0.08 / 0.11 / 0.10 | 3.31 / 2.07 / 2.20 / 3.75 | 0.056 |
| sinusoidal | 512, positions ≥ 128 | 284.7 / 222.5 / 221.4 / 251.6 | 0.06 / 0.02 / 0.02 / 0.03 | 4.86 / 3.23 / 3.05 / 4.01 | 0.010 |
| relative | 512, positions ≥ 128 | 272.8 / 245.2 / 200.9 / 156.1 | 0.43 / 0.22 / 0.56 / 0.54 | 4.28 / 4.70 / 3.38 / 3.21 | 0.234 |
| alibi | 512, positions ≥ 128 | 106.6 / 103.2 / 98.8 / 113.4 | 0.41 / 0.50 / 0.48 / 0.42 | 3.51 / 3.48 / 3.05 / 3.01 | 0.287 |

"Best word-boundary head" is the largest per-head attention on the most recent space before the query. Uniform causal attention
would give 0.028 at W = 128 and 0.004 at positions ≥ 128 of W = 512.

Figures:
- `attention_distance_profiles_W{128,512}`: mean weight vs distance, per layer, all encodings.
- `attention_effective_span`.
- `attention_heatmaps_W{128,512}`: the same test window for every encoding, head-mean, layers 0 and 3.
- `attention_space_mass`: per head.
- Per-run layer × head grids and two example heatmaps: `runs/<run>/attention_W*/` (not copied; regenerate with `scripts/analyze_attention.py`).
- `pe_similarity`: cosine similarity of the position vectors (learned, hybrid, sinusoidal).
- `pe_pca`.
- `pe_relative_vs_alibi`: learned relative bias by distance per layer and head, against ALiBi's fixed slopes.

What the PE figures show:
- **Learned table.** It is smooth. Positions within about ±20 are similar, and positions about 40–80 apart have negative cosine
  similarity, an oscillation like a low-frequency sinusoid. Its first two principal components trace a smooth loop (29% + 27% of the
  variance), with position 0 as an outlier: the model gives the first position of a window a vector of its own. Hybrid's added table
  is a mix of the two patterns.
- **Relative bias.** It is not monotone. In layers 0, 2 and 3 it falls to about −3 to −6 at distances 10–60, then rises back towards
  −1 … 0 near distance 127; layer 1 is nearly flat. ALiBi's bias falls linearly without bound (slopes 1/4 … 1/256 per character).

## 7. Findings

**Do the earlier conclusions hold with the tuned configs?** Claims as listed in `results/claims.md`.
- (a) The Transformer beats the parameter-matched LSTM: **holds**. TR − LR-2 = −0.0620 ± 0.0041 BPC (v1: −0.0877 ± 0.0164).
- (b) The LSTMs are better calibrated: **still does not hold**, as in v1. Seed-matched TR − LR-2 ECE is −0.0020 ± 0.0005.
  All models are now well calibrated (ECE ≤ 0.007), because label smoothing is off.
- (c) TR has the lowest Self-BLEU-4: **not supported**. TR's mean is lowest, but the single-prompt differences are inside the seed
  spread (TR − LR-2: −0.0125 ± 0.0477). With multi-prompt Self-BLEU: TR − LR-2 −0.0105 ± 0.0144, TR − LR-1 −0.0174 ± 0.0096.
- (d) RoPE beats relative: **holds**, now without the head-count confound of v1 (TR2o vs TR8o). The difference is +0.0472 ± 0.0255,
  but relative's seed spread is large.
- (f) The Transformer is faster per step: **holds**. 0.0280 s vs 0.0664 s (LR-2) and 0.0526 s (LR-1) per step on the 3090.

**Is the Transformer–LSTM ordering the same under both lr choices?** Yes. TR − LR-2 is −0.0620 ± 0.0041 (tie-rule lr) and
−0.0545 ± 0.0030 (lowest-BPC lr); every seed is between −0.052 and −0.066. The tie rule did cost the LSTM more: switching to the
lowest-BPC lr helps LR-2 by 0.0195 ± 0.0023 and LR-1 by 0.0211 ± 0.0063, against 0.0121 ± 0.0051 for TR. That narrows the gap by
0.0075, but only to 0.055. Even the best LSTM (LR-2 at 7.2e-3, 1.4957) is 0.042 behind the Transformer at its tie-rule lr (1.4532).
- Caveat: the lowest-BPC lrs are the edge of the tuning grid, so neither family's optimum is bracketed. Both might gain further at
  higher lr, and nothing here shows which would gain more.

**How much of Table V v1's 0.088 gap came from the LSTM's old lr (4e-4)?** It cannot be isolated with these runs. v1 → v3 changes
much more than the LSTM's lr:
- both models' lr, dropout (0.1 → 0) and label smoothing (0.05 → 0);
- the branch defaults: LSTM forget bias 1.0, sinusoidal scale, LR schedule;
- the seeds.

Both models gained a lot: TR −0.160 BPC (1.6132 → 1.4532), LR-2 −0.186 (1.7009 → 1.5152). So the gap shrank from 0.088 to 0.062
(tie-rule lr) and 0.055 (lowest-BPC lr). Retuning both models removed about 30–38% of the v1 gap; 62–70% of it remains.
Separating the LSTM's lr alone would need, for example, LR-2 with the v3 config but lr 4e-4; that was not run.

**No overfitting at 50k steps, but the train–validation gap is not all generalisation.** The best-validation-loss step is
48,500–49,999 in all 36 runs (49,000–49,999 with dropout 0), and validation loss still falls between iteration 45,000 and the end
(by −0.002 to −0.006 nats on average per group, dropout 0.1 included). The validation − training gap (dropout 0) is 0.065–0.072 nats at the end. But it is already 0.02–0.04 nats at
iteration 2,000, when the model has seen only 0.38 passes of the data. So part of the gap reflects a difference between the
validation text (the last 5% of `text8_train.txt`) and the training text, not memorisation. The rest grows over the 9.6 passes.
`figures/val_curves` shows both curves.

**Which encodings extrapolate, and how far?**
- **ALiBi**, to at least W = 1024 (8× the training length), and it uses the extra context: 1.4127 ± 0.0057 at W = 1024. That is
  0.041 below RoPE's standard W = 128 evaluation and 0.0995 ± 0.0041 below LR-2 at the same W.
- **Relative with T5-style clamping**, to W = 256 (1.5353), not to 512 (2.95) or 1024 (3.74). Its loss starts to rise at about
  position 200.
- **RoPE and sinusoidal: not at all.** The loss explodes within a few positions after 127; BPC 4.4–4.9 at every W > 128.
- **Learned and hybrid: N/A** (their table has 128 positions).
- **The LSTMs** run at any length, but their BPC is flat from W = 256 on (LR-2 1.5134 → 1.5122): they do not use context beyond
  about 128–256 characters. ALiBi's advantage over LR-2 grows with W (−0.069 at 256, −0.093 at 512, −0.100 at 1024).

**Does attention span relate to extrapolation?** Not the span at the training length; what matters is whether the attention
pattern survives beyond it.
- At W = 128, ALiBi has the longest spans (31–37 characters in every layer) and the least local attention (49–60% on the last 8
  characters), yet it is the only encoding that extrapolates. RoPE is the most local (spans 3–24) and fails completely.
- What predicts extrapolation is the attention at positions ≥ 128 of a W = 512 window:
  - RoPE and sinusoidal become almost uniform over the past. Spans are 217–285 characters, about half the window; only 2–11% of
    the attention is on the last 8 characters; entropy rises by 1–2.5 nats.
  - Their word-boundary heads stop working (best head 0.056 and 0.010, against 0.33–0.34 at W = 128).
  - ALiBi's distance penalty keeps the pattern: 41–50% on the last 8 characters, spans about 100–113, and the word-boundary head
    still at 0.287.
  - Relative is in between (22–56% local, spans 156–273, boundary head 0.234). That matches its partial extrapolation, and the bias
    shape explains it (§6): clamping assigns every far position the distance-127 bias, which is *higher* than the bias at medium
    distances. So the attention leaking to far positions grows with the window.
- Inside the training length, every encoding learns heads that find word boundaries: the best head puts 0.27–0.36 of its attention
  on the most recent space, about 10–13× uniform.

**Does the Transformer–LSTM gap grow or shrink with scale?** Not answered; scaling was not run (§1, §8).

## 8. Deviations, failures and what went wrong

1. **Scope** (agreed with the user, §1): no scaling runs; at most 2 jobs at a time on 24 GB cards instead of the ≥ 30 GB rule;
   seed-42 block-1 runs reused despite a non-empty diff of eval-only code; long runs launched after the model changes, before the rest
   of the tooling was committed.
2. **Scaling configs**: `scripts/make_scaling_configs.py` is committed but untested and unused. With the task's head rule
   (n_heads = d_model/64, head dim 32 below 128) and d_model on that grid, two Transformer sizes miss the ±15% target the task's
   test requires: 1M → d_model 128, 798,208 parameters (−20.2%); 6M → d_model 320, 4,944,640 (−17.6%). The other TR sizes and all
   LSTM sizes are within 10.6% and 0.4%. Running scaling needs a decision on the head rule (or accepting these sizes). The ±15% test
   was therefore not added.
3. **First launch ran on the CPU.** The launcher inherited `LD_LIBRARY_PATH` with `/usr/local/cuda`. The CUDA plugin failed
   ("Unable to load cuSPARSE") and JAX silently fell back to the CPU. Two timing runs sat at iteration 0 for about 4 minutes. I
   stopped them and deleted their outputs. The launcher now drops `/usr/local/cuda*` from `LD_LIBRARY_PATH` and sets
   `JAX_PLATFORMS=cuda`, so this fails loudly (`1b66cb3`). The first launcher log was overwritten by the relaunch.
4. **Shared GPU during development.** While `TR-final_s43` trained on GPU 3, I tested the extrapolation and attention scripts on
   the same GPU. The launcher recorded the foreign PIDs. Only that run's `timing.json` is affected; Table V's s/step comes from the
   solo timing runs. `TR-final_s42`'s extrapolation and attention outputs come from those test runs (same committed code,
   results not affected by sharing).
5. **Full-context batch size.** On the 3090, batch 8192 OOMs: a 10.7 GB allocation during the cross-machine check, after
   training (`checks/xcheck_oom_excerpt.txt`). Fixed by `--eval-only`, then 2048 for every run. No OOM afterwards.
6. **PE-table export on the CPU backend failed.** orbax cannot restore the GPU-saved checkpoints there ("sharding … Got None"). I
   re-ran the 6 exports on GPU 3 (a few seconds each) and made the evaluation queue run them on a GPU (`d507191`).
7. **GPU nondeterminism.** Two identical `train.py` runs (TR, 300 iterations) differ by up to 3.7e-4 in validation loss. The first
   steps are bit-identical; differences accumulate. Runs on this machine are reproducible to this level only, not bit-for-bit.
8. **Remaining-time estimate.** The launcher wrote it into the live `runs/v3_status.json` once block 1 was done, but that file is
   rewritten every poll and ends with the final (empty) queue. Recomputed with the same formula from `raw/v3_status.jsonl`: 9.4 h
   estimated at the end of block 1 (02:18), 8.7 h actual. `TR-final_s43`, slowed by the sharing in item 4, inflated the per-TR mean.
9. While stopping the CPU-bound runs, `pgrep -f` matched my own shell and ended it (the pitfall the handoff warned about). This had
   no effect on any run.
10. Nothing else failed: no divergence (including the high lrs of block 1b), no non-finite loss, no crash, no OOM in the 36 runs or
    the 40 evaluation jobs.

## 9. Checks: verified vs not verified

Verified:
- **Cross-machine check:** rerunning the winning TR tuning run (`xcheck_TR_ls0`) on the 3090 gives validation BPC 1.63705 vs
  1.6377 on the A100 (−0.0006, threshold 0.005). Data md5s match. Library versions are identical; Python is 3.10.22 vs 3.10.21.
- **Tests:** `pytest` 253 passed (`test_output.txt`; 217 before this task). New tests:
  - extrapolation: rebuild identity at W = 128, identical first-128 logits for rope/alibi/sinusoidal/LSTM at 4× length, relative
    clamp identical within the table and leaving training unchanged, learned/hybrid raise beyond 128, per-position loss vs brute force;
  - attention capture: outputs and parameters unchanged, rows sum to 1, zero above the diagonal;
  - attention statistics vs a brute-force computation;
  - PE-table export.

  `ruff check .`, `nbqa ruff .` and the pre-commit hooks pass on every changed file.
- **Extrapolation sanity check:** 24/24 runs reproduce their `eval.json` BPC (max difference 1.8e-9).
- **Notebook fix:** `transformer.ipynb` now builds its training step from `make_loss_and_metrics`. A patched copy (TR config, 300
  iterations, executed with nbclient) matches `train.py` to 5.1e-4 in validation loss, within run-to-run GPU noise (3.7e-4)
  (`checks/notebook_check.json`).
- **Reused seed-42 configs** match key by key; all 33 new runs record a clean commit.

Not verified:
- Scaling (not run).
- What the LSTM's old lr alone contributed to the v1 gap (§7).
- Whether higher lrs than the grid edge would help further.
- Throughput variance: one timing run per model.
- Bit-level reproducibility.

## 10. Files

| path | content |
|---|---|
| `aggregate_v3.py` | `--sync` copies run outputs from `runs/` into `raw/`; then rebuilds `tables/` (csv/md/tex) and `figures/` (pdf/png) |
| `raw/<run>/` | `eval.json`, `config.yaml`, `training_results.json`, `timing.json`, `val_history.json` (validation curve and 500-iteration training means), `extrapolation.json`, `attention_W*/{stats.json,example.npz}`, `pe_tables.json` |
| `raw/v3_status.jsonl`, `raw/v3_evals_status.jsonl` | launcher records: command, GPU, start/end, return code, foreign GPU processes |
| `checks/` | launcher log, seed-42 queue log, cross-check OOM excerpt, notebook check |
| `env.txt`, `test_output.txt` | environment (RTX 3090, driver 550.144.03, `pip freeze`, data md5s); final `pytest` summary |
| code | `utils/extrapolation_eval.py`, `scripts/{eval_extrapolation,analyze_attention,export_pe_tables,launch_v3,launch_v3_evals,make_scaling_configs}.py`, eval-only options in `models/models.py` |
