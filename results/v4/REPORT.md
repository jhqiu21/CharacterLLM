# v4: NoPE, position interpolation, learning-rate bracket check

Branch `final`. Builds on `results/v3/` (unchanged). Every test number is at the best-validation-loss checkpoint, mean ± sample
std (ddof 1) over seeds 42, 43, 44; "seed-matched" differences are mean ± std of the three per-seed differences.
`aggregate_v4.py` rebuilds every table (`tables/`) and figure (`figures/`) of the report from `results/v3/raw/` and `raw/`.
Hardware: RTX 3090 24 GB, GPUs 2 and 3, one job per GPU, no other process on either GPU during any job (`raw/v4_status.jsonl`).

## Summary

- **21 GPU jobs, all finished; none failed, none diverged:** 3 NoPE training runs (50k iterations), 6 short lr runs
  (10k iterations, validation only), 6 position-interpolation evaluations, 3 NoPE extrapolation evaluations, 3 attention analyses.
- **NoPE is the worst encoding at the training length:** 1.6201 ± 0.0211 test BPC, +0.1668 ± 0.0157 behind RoPE (1.4532), and
  0.1197 behind the next-worst (relative bias, 1.5004). Same parameter count (3,169,280) and training budget.
- **NoPE extrapolates only a little:** its loss is still close to its in-range value up to about position 190 (1.654 BPC at
  positions 128–191 vs 1.608 at 64–127), then rises steadily (2.33 at 192–255, 4.47 at 960–1023). Sliding-window BPC 1.9893 at
  W = 256 and 4.4600 at W = 1024. Unlike RoPE and sinusoidal it degrades gradually rather than at once, but it never gains
  from the extra context.
- **Position interpolation without fine-tuning does not rescue RoPE or sinusoidal.** It lowers BPC beyond 128 (RoPE 4.4356 → 4.0577,
  sinusoidal 4.7990 → 3.8114 at W = 1024) but stays 2.4–2.6 BPC worse than ALiBi (1.4127), and it breaks the positions *inside*
  the training range too (≈ 4.0 BPC at positions 0–127).
- **PI restores the *coarse* attention pattern, but not the character-level one.** At W = 512, RoPE + PI puts 0.37–0.79 of the
  attention on the last 8 characters and its best word-boundary head reaches 0.334 (plain RoPE: 0.07–0.11 and 0.056; RoPE at
  W = 128: 0.68–0.97 and 0.330). But the peak of the distance profile moves from 1 character to 3–5 characters: interpolation
  maps 4 characters onto one trained position step, so the model no longer resolves which of the last few characters is which.
- **The learning-rate optimum is now bracketed for both families** (decision rule fixed in advance, §3). Transformer: 4.05e-3 is
  0.0141 better than 2.7e-3 (inside the 0.02 tie), 6.075e-3 is 0.0155 worse. LSTM-3.2M: 1.08e-2 is 0.0042 better than 7.2e-3,
  1.62e-2 equal (−0.0000). No stop condition fired; no 50k run was added.

---

## 1. What was run

| block | runs / jobs | status |
|---|---|---|
| 1a NoPE | `TR-final-nope_s{42,43,44}`: `configs/final/TR.yaml --set model.pos_encoding_type=none`, 50,000 iterations | done, 30.8–31.1 min each |
| 1b lr bracket | `bracket_TR_lr{2.7e-3,4.05e-3,6.075e-3}` (`configs/final/TR.yaml`), `bracket_LSTM_lr{7.2e-3,1.08e-2,1.62e-2}` (`configs/final/LSTM.yaml`): `--tuning`, seed 4212, 10,000 iterations, warmup 2,000, validation every 500 | done |
| 1c-1 NoPE extrapolation | `scripts/eval_extrapolation.py --lengths 128 256 512 1024 --sanity`, 3 seeds | done |
| 1c-2 PI | `scripts/eval_extrapolation.py --lengths 128 256 512 1024 --interpolate` on the v3 RoPE and sinusoidal runs, 3 seeds | done |
| 1c-3 attention | NoPE seed 42 at W = 128 and 512; RoPE + PI seed 42 at W = 512 | done |

Every training run used `--set evaluation.full_context_batch_size=2048`. The launcher (`scripts/launch_v4.py`) ran the jobs in
priority order; its log is `raw/launch_v4.txt`, one record per job in `raw/v4_status.jsonl`. The tooling was committed before launch (NoPE and `pos_scale` in `ae0d7a8`, launcher in `7a46fbc`).
`TR-final-nope_s42` and `_s43` record `7a46fbc`. Every later job (NoPE s44, all bracket runs, all evaluations) records
`3197e6a`, the Part 3 clean-up commit I made while the queue was running (§6, item 9).

## 2. Positional encodings at the training length (`tables/pe_comparison.*`)

| Encoding | BPC | Acc_all (%) | ECE | Acc_last (%) | ΔBPC vs RoPE | #Params |
|---|---|---|---|---|---|---|
| RoPE | 1.4532 ± 0.0055 | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 68.57 ± 0.05 | — | 3,169,280 |
| ALiBi | 1.4626 ± 0.0022 | 68.38 ± 0.05 | 0.0038 ± 0.0003 | 68.41 ± 0.17 | +0.0094 ± 0.0055 | 3,169,280 |
| Sinusoidal | 1.4634 ± 0.0032 | 68.39 ± 0.06 | 0.0039 ± 0.0001 | 68.33 ± 0.02 | +0.0102 ± 0.0023 | 3,169,280 |
| Hybrid | 1.4835 ± 0.0015 | 67.93 ± 0.03 | 0.0030 ± 0.0002 | 67.87 ± 0.10 | +0.0302 ± 0.0040 | 3,202,048 |
| Learned | 1.4921 ± 0.0027 | 67.74 ± 0.06 | 0.0028 ± 0.0001 | 67.69 ± 0.22 | +0.0389 ± 0.0079 | 3,202,048 |
| Relative bias | 1.5004 ± 0.0243 | 67.55 ± 0.54 | 0.0018 ± 0.0003 | 67.51 ± 0.54 | +0.0472 ± 0.0255 | 3,173,360 |
| **NoPE** | 1.6201 ± 0.0211 | 64.86 ± 0.50 | 0.0020 ± 0.0001 | 64.66 ± 0.48 | +0.1668 ± 0.0157 | 3,169,280 |

NoPE per seed: 1.6441 / 1.6044 / 1.6117 (`tables/pe_comparison.csv`). All three trained smoothly: no non-finite loss, the largest
rise of the fixed-set validation loss between two checks is 0.0038 nats, and the best checkpoint is the last iteration (49,999) in
every run. Validation loss was still falling at the end (seed 42: 1.736 → 1.730 bits between iterations 45,000 and 49,999).
So NoPE is not unstable; it learns more slowly. The hyperparameters were tuned for RoPE (§6).

**Where NoPE ranks:** last of seven, 0.1197 behind the next-worst (relative bias), with a seed spread (0.0211) similar to
relative bias's (0.0243). Kazemnejad et al. (2023) studied NoPE on reasoning and arithmetic tasks, mainly for length generalisation;
here, in character-level language modelling, the model has to infer exact character offsets from the causal mask alone, and
that costs a lot at this size and training budget.

## 3. Learning-rate bracket check (`tables/lr_bracket.*`, `figures/lr_bracket.*`)

**Protocol and rule (fixed before the runs).** Same protocol as `results/v3_tuning/`: `--tuning`, seed 4212, 10,000 iterations,
scored by full-context BPC on the whole validation split (stride 8, 4,499,999 targets). Two further steps of factor 1.5 above the
lowest-BPC lr of the v3 search, plus a rerun of that lr on the same hardware (RTX 3090; the v3 search ran on an A100). The new runs
use the final configurations (dropout 0, label smoothing ε = 0). Rule: a new lr that is worse than or within 0.02 BPC of the 3090
rerun, or diverges, means the optimum is bracketed (or close enough); a new lr more than 0.02 better is a stop condition.

| Model | lr | validation BPC (RTX 3090, ε 0, dropout 0) | Δ vs rerun (positive = better) |
|---|---|---|---|
| Transformer (RoPE) | 1.8e-3 (v3 cross-machine check, chosen lr) | 1.6370 | — |
| Transformer (RoPE) | 2.7e-3 (rerun of the v3 lowest-BPC lr) | 1.5825 | reference |
| Transformer (RoPE) | 4.05e-3 | 1.5684 | +0.0141 |
| Transformer (RoPE) | 6.075e-3 | 1.5980 | −0.0155 |
| LSTM-3.2M | 7.2e-3 (rerun of the v3 lowest-BPC lr) | 1.6325 | reference |
| LSTM-3.2M | 1.08e-2 | 1.6284 | +0.0042 |
| LSTM-3.2M | 1.62e-2 | 1.6325 | +0.0000 |

**Result: bracketed for both families.** No new lr is more than 0.02 better; neither diverged. For the Transformer the curve has a
clear minimum between 2.7e-3 and 6.075e-3 (BPC rises again at 6.075e-3). For the LSTM it is flat from 7.2e-3 to 1.62e-2 (within
0.0042). So the "lowest-BPC lr" of Table V (2.7e-3 / 7.2e-3) is within 0.015 validation BPC of the best lr found at 10k iterations
for both families, and the 50k runs were not repeated.

Notes:
- The v3 search's lowest points (A100, ε = 0.05: 1.6707 / 1.6677) are not comparable with these rows: label smoothing raises BPC by
  about 0.04 (v3 tuning §4). The rerun at the same lr with ε = 0 is the reference, as the rule requires.
- At 10k iterations the step from the tie-rule lr 1.8e-3 to 2.7e-3 is large for the Transformer (−0.0545 validation BPC). At 50k
  iterations the same change gives only −0.0121 ± 0.0051 test BPC (`tables/table_v_bestlr.md`): short runs favour high learning
  rates. The bracket check therefore says the 50k optimum is *at most* around these values; it does not say a higher lr would win at 50k.
- Hardware: the 3090 rerun of v3's chosen TR point gave 1.6370 vs 1.6377 on the A100 (v3 §9); the 3090 numbers here are compared only
  with each other.

## 4. Length extrapolation (`tables/extrapolation.*`, `tables/per_position_buckets_W1024.*`, `figures/extrapolation_bpc_vs_W.*`, `figures/per_position_loss_W1024.*`)

Sliding-window test BPC, stride 128, all models trained at 128:

| Model | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| RoPE | 1.5452 ± 0.0030 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| RoPE + PI | 1.5452 ± 0.0030 | 3.6156 ± 0.0614 | 3.9813 ± 0.0435 | 4.0577 ± 0.0937 |
| Sinusoidal | 1.5549 ± 0.0025 | 4.6830 ± 0.2713 | 4.6092 ± 0.2726 | 4.7990 ± 0.2193 |
| Sinusoidal + PI | 1.5549 ± 0.0025 | 2.9955 ± 0.0711 | 3.6540 ± 0.0326 | 3.8114 ± 0.0615 |
| Learned | 1.5806 ± 0.0023 | N/A | N/A | N/A |
| Hybrid | 1.5726 ± 0.0012 | N/A | N/A | N/A |
| Relative bias (clamped) | 1.5909 ± 0.0200 | 1.5353 ± 0.0116 | 2.9477 ± 0.2898 | 3.7355 ± 0.1986 |
| **NoPE** | 1.6794 ± 0.0120 | 1.9893 ± 0.0244 | 4.0939 ± 0.1676 | 4.4600 ± 0.3055 |
| ALiBi | 1.5602 ± 0.0016 | **1.4449 ± 0.0027** | **1.4196 ± 0.0045** | **1.4127 ± 0.0057** |
| LSTM-3.2M | 1.5834 ± 0.0014 | 1.5134 ± 0.0017 | 1.5123 ± 0.0017 | 1.5122 ± 0.0017 |
| LSTM-2.1M | 1.6191 ± 0.0030 | 1.5543 ± 0.0040 | 1.5534 ± 0.0041 | 1.5534 ± 0.0041 |

Per-position BPC at W = 1024 (64-position buckets, `tables/per_position_buckets_W1024.md`):

| Model | 0–63 | 64–127 | 128–191 | 192–255 | 448–511 | 960–1023 |
|---|---|---|---|---|---|---|
| RoPE | 1.617 | 1.453 | 3.506 | 5.183 | 4.884 | 4.440 |
| RoPE + PI | 3.964 | 4.041 | 4.049 | 4.050 | 4.043 | 4.053 |
| Sinusoidal + PI | 3.684 | 3.710 | 3.724 | 3.740 | 3.773 | 3.809 |
| Relative bias (clamped) | 1.657 | 1.500 | 1.495 | 1.564 | 3.093 | 3.747 |
| NoPE | 1.724 | 1.608 | 1.654 | 2.326 | 4.157 | 4.465 |
| ALiBi | 1.633 | 1.462 | 1.443 | 1.432 | 1.409 | 1.407 |

**Does NoPE extrapolate?** *Partly, over a short range, and without benefit.* Its per-position loss rises only slightly in the first
64 positions past the training length (1.654 vs 1.608) and then climbs steadily to about 4.5 BPC; the sliding-window BPC at W = 256
(1.9893) is already worse than at W = 128 (1.6794). The degradation is gradual, unlike RoPE and sinusoidal, whose loss jumps within a
few positions after 127, and similar in shape to clamped relative bias (which holds to about position 255). This agrees with Wang et al. (2024, "Length
Generalization of Causal Transformers without Position Encoding", Findings of ACL), who find that NoPE extends somewhat beyond the
training length (about 20% in their language-modelling setting) before failing, and that the failure coincides with diffuse
attention (§5). Kazemnejad et al. (2023) found NoPE to generalise better than explicit encodings on their reasoning tasks; for
character-level language modelling, ALiBi is far better at every W > 128.

**Does position interpolation rescue RoPE and sinusoidal?** *No.* PI removes the catastrophic jump after position 127 and lowers
the long-window BPC (seed-matched, W = 1024: RoPE + PI is 0.38 below RoPE, sinusoidal + PI 0.99 below sinusoidal), but the loss is
now high at *every* position, including 0–127, because the compressed positions are off-distribution everywhere. Compared with
ALiBi at W = 1024 (seed-matched, `tables/extrapolation.md` notes): RoPE + PI +2.6450 ± 0.0887, sinusoidal + PI +2.3987 ± 0.0667,
NoPE +3.0473 ± 0.3085. Chen et al. (2023) apply PI with a short fine-tuning phase; without it, our 3.2M-parameter character
models do not tolerate the compressed positions. Fine-tuning was not tried (§6).

**Checks.**
- W = 128 with stride 1 through the rebuilt model reproduces every run's `eval.json` full-context BPC: 27/27 runs (24 v3 + 3 NoPE),
  largest difference 1.8e-9 BPC (NoPE: 0.0).
- PI at W = 128 (pos_scale 1) vs plain W = 128: largest difference 3.4e-8 BPC over 6 runs. The model is identical (tested
  bit-for-bit on CPU, `tests/test_nope_and_interpolation.py`); the 3.4e-8 is GPU run-to-run noise between the v3 and v4 evaluations.

## 5. Attention analysis (`tables/attention_summary.*`, `tables/attention_short_range.*`, `figures/attention_*`)

Seed 42, 512 test windows; per-layer values (layers 0 / 1 / 2 / 3) are means over the 4 heads; W = 512 rows use queries ≥ 128.

| Encoding | W | Effective span (chars) | Mass on last 8 chars | Entropy (nats) | Best word-boundary head |
|---|---|---|---|---|---|
| RoPE | 128 | 9.8 / 3.2 / 5.1 / 24.4 | 0.91 / 0.97 / 0.92 / 0.68 | 1.40 / 0.91 / 1.10 / 2.11 | 0.330 |
| NoPE | 128 | 55.8 / 63.2 / 3.4 / 9.1 | 0.26 / 0.14 / 0.99 / 0.85 | 3.31 / 3.18 / 1.36 / 1.36 | 0.282 |
| RoPE | 512 | 244.9 / 218.4 / 217.1 / 252.0 | 0.07 / 0.08 / 0.11 / 0.10 | 3.31 / 2.07 / 2.20 / 3.75 | 0.056 |
| NoPE | 512 | 285.9 / 317.3 / 11.5 / 51.5 | 0.04 / 0.02 / 0.79 / 0.41 | 5.08 / 5.20 / 2.32 / 3.37 | 0.169 |
| ALiBi | 512 | 106.6 / 103.2 / 98.8 / 113.4 | 0.41 / 0.50 / 0.48 / 0.42 | 3.51 / 3.48 / 3.05 / 3.01 | 0.287 |
| RoPE + PI | 512 | 51.6 / 10.5 / 14.8 / 141.1 | 0.57 / 0.70 / 0.79 / 0.37 | 2.54 / 1.77 / 2.02 / 3.81 | 0.334 |

(The full table with all encodings is `tables/attention_summary.md`.)

Short-range structure (`tables/attention_short_range.md`, `figures/attention_short_range.*`): distance with the largest head-mean
weight, layers 0 / 1 / 2 / 3, and the weight on distances 0–2:

| Model | Peak distance | Weight on distances 0–2 |
|---|---|---|
| RoPE, W = 128 | 1 / 1 / 1 / 1 | 0.70 / 0.76 / 0.76 / 0.42 |
| RoPE + PI, W = 512 | 0 / 5 / 3 / 1 | 0.25 / 0.19 / 0.31 / 0.16 |
| RoPE, W = 512 (all queries) | 1 / 1 / 1 / 1 | 0.22 / 0.24 / 0.26 / 0.15 |

**Does PI restore local attention?** *At the scale of words, yes; at the scale of characters, no.* With PI, RoPE's attention past
position 128 is local again (spans 10–52 characters in layers 0–2 instead of 217–245; 0.57–0.79 of the mass on the last 8 characters)
and the word-boundary head is back to its W = 128 strength (0.334 vs 0.330). But the profile is stretched: in layers 1 and 2 the peak
moves from distance 1 to distances 5 and 3, and the weight on distances 0–2 drops from 0.70–0.76 to 0.19–0.31. One trained position
step now covers 4 characters, so the heads that at W = 128 single out the previous one or two characters now spread over a
neighbourhood of 3–7. Restoring a local pattern is therefore not sufficient: the model needs the exact short offsets, which PI
(without fine-tuning) destroys. This also explains why PI hurts positions 0–127.

**NoPE's attention.** At W = 128, NoPE's first two layers are diffuse (spans 56–63 characters, entropy 3.2–3.3 nats) and its last
two are local (spans 3–9). This fits the theoretical argument of Kazemnejad et al. (2023) that a causal model without positional encoding can
represent position (e.g. a first layer attending uniformly over the i+1 visible tokens recovers i); we did not test whether
these layers actually encode position. Beyond the
training length, the diffuse layers spread over the whole window (spans 286–317) while layer 2 stays local (span 11.5, 0.79 on the
last 8 characters); the word-boundary head weakens from 0.282 to 0.169. The diffuse early layers match the "attention distraction" that Wang et al.
(2024) link to NoPE's failure beyond its context limit. As with PI, local attention in some layers does not prevent failure.

## 6. Deviations, limitations and what went wrong

1. **The report source is not in this repository.** It lives in a separate Overleaf-connected repository (user decision); the
   compiled PDF is not copied here.
2. **Dataset was tracked in git.** `data/text8_train.txt` (90 MB) and `data/text8_test.txt` were committed in earlier history
   despite `.gitignore`. They are now untracked (`git rm --cached`); `scripts/prepare_data.py` recreates both from text8 and checks the
   md5s against `results/v3/env.txt` (verified). **They are still in the git history**; purging needs a history rewrite, left to the
   user when splitting off the new repository.
3. **Binaries outside `figures/`.** `results/v3/raw/*/attention_W*/example.npz` (13 MB in total) and the new
   `raw/*/attention_W*/example.npz` are tracked: the heatmap figures are drawn from them and `results/v3/REPORT.md` refers to them.
   `tests/data/pe_reference_logits.npz` (30 KB) is a test fixture.
4. **Bracket runs use ε = 0, the v3 search's grid used ε = 0.05.** The rule compares new lrs only with the 3090 rerun at the same
   ε, so the decision is not affected; the A100 rows in `tables/lr_bracket.*` are shown for context only.
5. **NoPE and PI are evaluated with RoPE-tuned hyperparameters** (as all encodings in v3). A NoPE-specific lr or longer training might
   narrow its gap; this was not tested.
6. **PI without fine-tuning only.** The method as proposed includes a short fine-tuning phase; that was out of scope.
7. **Attention analysis on one seed** (42), as in v3.
8. **My own mistakes, no effect on results:** the first `conda env create` test and the first two notebook checks failed for setup
   reasons (conda Terms-of-Service channel; a warmup longer than the 60-iteration check run; an invalid evaluation stride in my
   patch). Fixed and rerun; reported results are from the successful reruns.
9. **Clean-up committed during the queue.** `3197e6a` removed dead code (unused metric and sampling helpers, the never-used
   `CharLSTM.tie_weights` branch), hid an empty panel in `training_curves.pdf` and changed docstrings. It does not touch any
   training or evaluation code path (full `pytest` identical before and after, 290 passed; the bit-identity tests pass), so NoPE
   s42/s43 (`7a46fbc`) and s44 (`3197e6a`) ran the same computation. It still means the jobs do not all record one commit.
10. Nothing else failed: no divergence, no non-finite loss, no OOM, no foreign GPU process during any of the 21 jobs.

## 7. Checks

- `pytest`: 290 passed (`test_output.txt`; 253 in v3). New: `tests/test_nope_and_interpolation.py` (NoPE parameter count
  3,169,280 = RoPE, causality, identical first 128 logits at max_len 512, every older encoding bit-identical to logits saved from
  `fc314d4`, `pos_scale = 1` bit-identical, scaled positions < 128 at W = 256, PI at W = 128 equals plain, clear errors for other
  encodings and the LSTM); NoPE added to the causality, attention-capture, decay-mask, parameter-count and extrapolation tests.
- `environment.yml` creates a fresh environment (conda-forge only) in which the full suite passes: 290 passed.
- `transformer.ipynb` executes end to end with the current code (patched copy: final TR config with NoPE, 60 iterations, cheap
  evaluation settings, CPU, nbclient).
- `ruff check .`, `nbqa ruff .`, `pre-commit run --all-files`: pass.

## 8. Files

| path | content |
|---|---|
| `aggregate_v4.py` | `--sync` copies v4 outputs from `runs/` into `raw/`; builds `tables/` (csv/md/tex) and `figures/` (pdf/png) from `results/v3/raw/` + `raw/` |
| `raw/TR-final-nope_s*/` | `eval.json`, `config.yaml`, `training_results.json`, `timing.json`, `val_history.json`, `extrapolation.json`, `attention_W{128,512}/` (seed 42) |
| `raw/TR-final{,-sinusoidal}_s*/` | v4 outputs for v3 runs only: `extrapolation_pi.json`, `attention_W512_pi/` (RoPE seed 42) |
| `raw/bracket_*/` | `val_eval.json`, `config.yaml`, `val_history.json` |
| `raw/v4_status.jsonl`, `raw/launch_v4.txt` | launcher records and log |
| `claims.md`, `check_claims.py` | every number in the report and its source; the checker verifies the table-backed ones |
| `export_paper.py` | copies `tables/*.tex` and `figures/*.pdf` into the report's LaTeX source |
| `status.md` | task status and open questions |
| `test_output.txt` | final `pytest` summary |
