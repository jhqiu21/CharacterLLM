# v6: Does more capacity raise next-character accuracy?

Branch `final`. Question: can a larger model, trained on the same data and budget, move the overall top-1
next-character accuracy on the text8 test set (Acc_all) towards 80%? This round (1) measures the actual Acc_all of the
best v5 models, (2) trains a 3.2M and a 14.2M ALiBi Transformer from scratch under identical conditions, and (3) uses
the result to decide between more capacity, more training characters, and a 38M stage. The 38M stage was not started.

**Metrics.** All test and validation numbers are sliding-window evaluations (`utils/full_context_eval.py`) with
**stride 128**. Every target character of the split is scored exactly once: 4,999,999 test targets, 4,499,999 validation
targets. Every target after the first window sees at least W − 127 characters. **Main comparison: W = 512**; W = 128 and
1024 are auxiliary.
- **Acc_all**: argmax prediction == true next character, over all targets. It is read from the evaluation, never derived from BPC.
- **Acc_common / Acc_rare**: top-1 accuracy on the targets that are one of the 5 most frequent training characters
  (space, e, t, a, n: 47.8% of the training text) / the 5 least frequent (q, j, x, z, k: 1.4%).
- **Top-k**: the target is among the k highest logits (fewer than k characters score strictly higher).

Aggregates are mean ± sample std (ddof 1) over seeds 42, 43, 44. "Paired" differences are mean ± std of the per-seed
differences. `aggregate_v6.py` rebuilds every table (`tables/`) and figure (`figures/`) from `raw/` (+ `results/v5/raw`
and `results/v3/raw` for the reference rows).

## Summary

| test, W = 512, stride 128 | params | BPC | Acc_all (%) | Acc_common (%) | Top-2 (%) | Top-5 (%) |
|---|---|---|---|---|---|---|
| ALiBi + 512-window fine-tuning 4k (v5 best) | 3.2M | 1.4075 ± 0.0034 | 69.60 ± 0.07 | 80.68 ± 0.04 | 80.15 ± 0.05 | 90.61 ± 0.02 |
| RoPE + 512-window fine-tuning 4k, no PI (v5) | 3.2M | 1.4133 ± 0.0099 | 69.43 ± 0.21 | 80.57 ± 0.15 | 80.05 ± 0.15 | 90.57 ± 0.07 |
| **AL-3M**: ALiBi, 512 from scratch (v6) | 3,169,280 | 1.3795 ± 0.0008 | 70.23 ± 0.02 | 81.08 ± 0.02 | 80.58 ± 0.01 | 90.82 ± 0.00 |
| **AL-14M**: ALiBi, 512 from scratch (v6) | 14,204,928 | **1.2440 ± 0.0008** | **73.20 ± 0.03** | 82.93 ± 0.03 | 82.77 ± 0.01 | 91.90 ± 0.00 |
| AL-14M − AL-3M (paired) | | −0.1355 ± 0.0017 | **+2.97 ± 0.04** (3/3 seeds) | +1.85 ± 0.03 | +2.19 ± 0.03 | +1.08 ± 0.01 |

- **Current level (v5).** The best v5 model reaches Acc_all **69.60%** at W = 512 (69.82% at W = 1024): **10.4 points
  below 80%**. Its Top-2 accuracy (80.15%) and its accuracy on the 5 most common characters (80.68%) are about 80%; its
  top-1 accuracy on all characters is not.
- **Capacity raises accuracy, by a lot and very consistently.** Same data, objective, optimizer, schedule, 819,200,000
  training characters and 50,000 updates. AL-14M beats AL-3M by +2.97 ± 0.04 points at W = 512 on the test set (+2.95 ±
  0.04 on validation). The gain has the same sign in every seed, at every W (+2.76 to +3.04), and in every metric: rare
  characters +4.35, common characters +1.85, Top-5 +1.08. BPC −0.136.
- **Best result of this round: 73.20% at W = 512, 73.38% at W = 1024 (AL-14M). Still 6.80 / 6.62 points below 80%.**
- **Capacity is worth more than training characters here.** 5× the characters (164M → 819M) gives +1.92 (3.2M) and +2.41
  (14.2M) validation points. 4.5× the parameters gives +2.50 (at 164M) and +2.99 (at 819M). AL-14M after only 164M
  characters (17 GPU-minutes) is already better than AL-3M after 819M characters (28 GPU-minutes): 70.64% vs 70.06%.
  More characters also means more repetition: 819M characters are already 9.6 passes over the 85.5M-character training split.
- **No overfitting yet, but the larger model memorises faster.** Validation BPC reaches its minimum at the last 0–1,000
  iterations in every run. But the validation − training BPC gap grows from +0.051 (10k iterations) to +0.135 for
  AL-14M, against +0.044 → +0.084 for AL-3M (means over seeds).
- **80% is far away.** About 43.6% of AL-14M's errors are on the first letter of a word, where it is right only 31.7% of
  the time. Reaching 80% with first-letter accuracy unchanged would need 90.0% on all other characters, against 81.8%
  now. Extrapolating the measured gain per parameter doubling (+1.37 points), 80% would need about 440M parameters.
- **Verdicts (§8):** (1) yes, scaling up raised Acc_all; (2) capacity, not more characters, is the better investment now;
  (3) the 38M stage is worth running as the next measurement, expected to land near **75%**, not 80%. It should come
  with a regularisation control, because memorisation grows with size.

## 1. What was run

| phase | jobs | protocol | GPU h |
|---|---|---|---|
| `v5topk` | 9 evaluations | best v5 models (ALiBi / RoPE + 4,000 iterations of 512-window fine-tuning) and the 128-trained ALiBi, 3 seeds each, re-evaluated on the test set at W = 128–1024 to add top-k; written to `extrapolation_v6.json`, the v5 / v3 files are untouched | 0.29 |
| `search` | 6 training runs | lr 0.5× / 1× / 2× of `configs/final/TR.yaml`'s 1.8e-3 for each model; 10,000 iterations (163.8M characters), warmup 2,000, cosine to 0, seed 4212 (tuning seed), validation only (`--tuning`) | 1.29 |
| `formal` | 2 training runs | seed 42, 50,000 iterations at each model's chosen lr, validation only | 1.91 |
| `seeds` | 4 training runs | seeds 43, 44 of both models (rule below, met) | 3.82 |
| `bracket` | 2 training runs | post-hoc lr 4× (7.2e-3), same protocol as `search`; not used for any choice | 0.43 |
| `evals` | 24 evaluations | validation split, then test split at W = 128–1024 (stride 128); error breakdown by position in the word (W = 512) | 0.70 |
| **total** | **47 jobs, 47 complete** | wall clock 10:52–17:19 (6.4 h) | **8.42** |

GPU hours are the sum of the job wall times (`tables/gpu_time.md`): 6.05 h on GPU 3 and 2.37 h on GPU 2. Every job had
its GPU to itself; no foreign process was seen on it. Not included: a 300-iteration smoke test of AL-14M (about 1 min,
GPU 3) before the launch.

**Models** (`configs/AL-3M.yaml`, `configs/AL-14M.yaml`). Both are identical to `configs/final/TR.yaml` except for:
- ALiBi;
- 512-character windows × batch 32 (16,384 training characters per update, the same as TR.yaml's 128 × 128);
- selection by validation accuracy;
- a single stage checkpoint;
- the size: 4 layers / d_model 256 / 4 heads = **3,169,280** parameters vs 8 / 384 / 6 = **14,204,928** (printed by
  `train.py` and recorded in `tables/runs.md`).

The models share the data split, preprocessing, loss (cross-entropy with the linear position weights, label smoothing 0,
no entropy bonus), AdamW (b1 0.9, b2 0.95, weight decay 0.1 on matrices only, clip 1.0), warmup 2,000 + cosine to 0,
dropout 0 and the evaluation. Batch 32 × 512 fits without gradient accumulation (peak 7.9 GiB for AL-14M), so none is
used. No PI, SelfExtend or local attention.

**Budget.** Each formal run makes **50,000 updates on 819,200,000 training characters** (`timing.json`: `updates`,
`train_tokens`). Both models match exactly. Training took 27.5 min for AL-3M and 84.2 min for AL-14M (0.0314 / 0.0982 s
per update).

**Checkpoint selection** (new, `training.checkpoint_selection = acc`). The checkpoint is the validation check with the
highest accuracy on the fixed validation set (1,024 windows × 512 from the validation split, seed 0). Ties go to the
lower loss, i.e. the lower BPC on the same set; an exact tie keeps the earlier check. Checks happen every 500 iterations.
Every v6 number comes from this checkpoint, `checkpoints/<run>/best/best_acc_bpc_<step>` (`tables/runs.md`). It lies at
iteration 48,000–49,999. The best-loss checkpoint is within 0.02 points of it on the validation split
(`tables/selection.md`). The v5 / v3 rows keep their own rule (best validation loss).

**Learning rate rule.** For each model, the search lr with the highest validation Acc_all of the selected checkpoint on
the whole validation split (W = 512, stride 128). Ties are broken by the lower validation BPC.

**Rule for seeds 43 / 44** (fixed in `scripts/launch_v6.py` before the formal runs). Seed-42 AL-14M must beat AL-3M by
≥ 0.3 validation points with a lower validation BPC. Result: +2.99 points, BPC −0.138, so the rule was met.

**Test set.** It was opened only for the existing v5 / v3 models (`v5topk`) and in the final `evals` phase, after every
training run and the bracket check had finished. Nothing was chosen on it.

## 2. The current level: v5's best models (`tables/test_W*.md`, `tables/reuse_check.md`)

| test, stride 128 | W | BPC | Acc_all (%) | Acc_common (%) | Acc_rare (%) | Top-2 (%) | Top-3 (%) | Top-5 (%) |
|---|---|---|---|---|---|---|---|---|
| ALiBi + 512 fine-tuning 4k | 128 | 1.5622 ± 0.0018 | 66.12 ± 0.04 | 78.34 ± 0.02 | 47.12 ± 0.15 | 77.40 ± 0.03 | 82.95 ± 0.03 | 89.22 ± 0.01 |
| | **512** | 1.4075 ± 0.0034 | **69.60 ± 0.07** | 80.68 ± 0.04 | 52.62 ± 0.16 | 80.15 ± 0.05 | 85.12 ± 0.04 | 90.61 ± 0.02 |
| | 1024 | 1.3969 ± 0.0045 | 69.82 ± 0.09 | 80.85 ± 0.06 | 52.85 ± 0.36 | 80.34 ± 0.07 | 85.27 ± 0.06 | 90.70 ± 0.03 |
| RoPE + 512 fine-tuning 4k, no PI | 128 | 1.5551 ± 0.0040 | 66.26 ± 0.08 | 78.46 ± 0.05 | 47.33 ± 0.34 | 77.50 ± 0.06 | 83.04 ± 0.04 | 89.27 ± 0.03 |
| | **512** | 1.4133 ± 0.0099 | **69.43 ± 0.21** | 80.57 ± 0.15 | 52.27 ± 0.46 | 80.05 ± 0.15 | 85.07 ± 0.12 | 90.57 ± 0.07 |
| | 1024 | 1.7240 ± 0.1034 | 62.58 ± 2.20 | 76.13 ± 1.47 | 41.85 ± 4.27 | 75.49 ± 1.62 | 81.84 ± 1.20 | 88.75 ± 0.72 |

The v5 files already held Acc_all, Acc_common and Acc_rare at exactly this protocol (same targets, stride, checkpoint),
so they are reused; v5 only did not tabulate them. The re-evaluation adds top-k. It reproduces every BPC and
Acc_all of the v5 / v3 files exactly: largest difference 0 at all 36 (run, W) pairs. For reference, v5's own
"W = 128, stride 1" Acc_all (every target sees 127 characters) is 68.42% for ALiBi + fine-tuning, seed 42.

## 3. Capacity at equal training conditions (`tables/search.md`, `formal_val.md`, `test_by_W.md`, `paired_test.md`)

**Learning-rate search** (validation Acc_all, W = 512, after 10,000 iterations):

| model | 9e-4 (0.5×) | 1.8e-3 (1×) | 3.6e-3 (2×) | 7.2e-3 (4×, post-hoc) |
|---|---|---|---|---|
| AL-3M | 67.25 | 67.88 | **68.14** ← chosen | 68.22 |
| AL-14M | 70.16 | 70.58 | **70.64** ← chosen | 70.49 |

Both models picked 2×, the upper edge of the planned grid. The post-hoc 4× point, run after the formal runs, brackets
the optimum for AL-14M (4× is 0.16 points worse). For AL-3M it does not: 4× is 0.08 better at 10,000 iterations. Short
runs favour high learning rates (v4 §3), so whether 4× would also win at 50,000 is open. If it did, AL-3M is slightly
under-tuned and the capacity gain below is overstated by roughly 0.1 point, well inside its size.

**Formal runs, validation split** (W = 512, selected checkpoints):

| seed | AL-3M Acc_all | AL-3M BPC | AL-14M Acc_all | AL-14M BPC | Δ Acc (points) | Δ BPC |
|---|---|---|---|---|---|---|
| 42 | 70.06 | 1.3950 | 73.05 | 1.2574 | +2.99 | −0.1377 |
| 43 | 70.12 | 1.3915 | 73.03 | 1.2580 | +2.90 | −0.1335 |
| 44 | 70.09 | 1.3923 | 73.05 | 1.2573 | +2.96 | −0.1350 |
| mean ± std | 70.09 ± 0.03 | 1.3930 ± 0.0018 | 73.04 ± 0.01 | 1.2575 ± 0.0004 | **+2.95 ± 0.04** | −0.1354 ± 0.0021 |

**Test set, every window** (stride 128; validation in `tables/val_by_W.md` shows the same pattern):

| | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| AL-3M Acc_all (%) | 66.07 ± 0.02 | 69.31 ± 0.01 | 70.23 ± 0.02 | 70.53 ± 0.01 |
| AL-14M Acc_all (%) | 68.83 ± 0.00 | 72.35 ± 0.02 | 73.20 ± 0.03 | **73.38 ± 0.06** |
| Δ Acc_all, paired (3/3 seeds +) | +2.76 ± 0.02 | +3.04 ± 0.03 | **+2.97 ± 0.04** | +2.85 ± 0.07 |
| AL-3M BPC | 1.5645 ± 0.0006 | 1.4215 ± 0.0006 | 1.3795 ± 0.0008 | 1.3651 ± 0.0005 |
| AL-14M BPC | 1.4390 ± 0.0007 | 1.2839 ± 0.0009 | 1.2440 ± 0.0008 | 1.2341 ± 0.0025 |
| Δ BPC, paired (3/3 seeds −) | −0.1255 ± 0.0014 | −0.1376 ± 0.0015 | −0.1355 ± 0.0017 | −0.1310 ± 0.0029 |

Seed spreads are 0.01–0.06 points, the paired differences ±0.02–0.07. A gain of 2.8–3.0 points is some 40–150 times
the standard deviation of the paired differences, with no ambiguity across seeds. Both models trained on 512 still gain from 1024-character windows (ALiBi
extrapolates), by +0.18 (AL-14M) and +0.30 (AL-3M) points.

**AL-3M vs v5's best.** AL-3M (70.23%) also beats v5's best (69.60%) at the same size, by +0.63 points. That is not a
clean comparison:
- the lr differs (3.6e-3 here vs 1.8e-3);
- AL-3M trains on 512-character windows from the start, while v5 pre-trained on 128 and then fine-tuned for 65.5M
  characters at 512;
- the selection rule differs.

The capacity question is answered only by the AL-3M vs AL-14M pair, which differs in nothing but size.

## 4. Training curves (`figures/training_curves.*`, `tables/curves.md`)

![training curves](figures/training_curves.png)

| fixed validation set (all positions) | Acc @ 10k | @ 20k | @ 30k | @ 40k | @ 50k | lowest val BPC at | val − train BPC @ 10k | @ 30k | @ 50k |
|---|---|---|---|---|---|---|---|---|---|
| AL-3M (3 seeds) | 64.2–64.3 | 65.3–65.5 | 66.5–66.6 | 67.9–68.0 | 68.7 | 49,500–49,999 | +0.042–0.047 | +0.056–0.061 | +0.082–0.089 |
| AL-14M (3 seeds) | 66.5–66.7 | 67.9 | 69.2–69.3 | 70.8–70.9 | 71.6 | 49,000–49,999 | +0.049–0.053 | +0.070–0.073 | +0.132–0.137 |

Training BPC is the mean unweighted loss on the training batches over the last 1,000 iterations. Dropout is 0, so it is
the same model as at validation. The fixed set scores every position 0–511, including those with little context, so its
accuracy is lower than the W = 512 sliding-window values.

- **Both models were still improving when training ended.** Validation BPC is lowest at one of the last three checks.
  Accuracy rose 0.7–0.8 points over the last 10,000 iterations, part of which is the cosine schedule annealing to 0.
- **No overfitting** in the sense that matters here: validation never got worse.
- **The larger model memorises more of the repeated data.** The validation − training gap grows throughout. After about
  500M characters (6 passes) it grows much faster for AL-14M (+0.135 at the end, mean) than for AL-3M (+0.084). Training more,
  or larger, without regularisation will eventually turn this into overfitting.
- The three seeds of each model lie on top of each other; the capacity gap is the same in every seed at every point.

## 5. Capacity or more characters? (`tables/budget_scaling.md`, `figures/acc_vs_compute.*`)

Two budgets per model, each with a complete schedule: the search run at the chosen lr (10,000 iterations, 164M
characters, seed 4212) and the formal run (50,000 iterations, 819M characters, seed 42). Validation, W = 512:

| | 164M characters | 819M characters | 5× characters |
|---|---|---|---|
| AL-3M | 68.14% (1.4822 BPC), 5.8 GPU-min | 70.06% (1.3950), 27.5 GPU-min | +1.92 |
| AL-14M | 70.64% (1.3639), 17.3 GPU-min | 73.05% (1.2574), 84.2 GPU-min | +2.41 |
| 4.5× parameters | +2.50 | +2.99 | |

![accuracy vs compute](figures/acc_vs_compute.png)

- **Per doubling:** parameters give +1.16 (164M) / +1.38 (819M) points. Characters give +0.83 (3.2M) / +1.04 (14.2M).
  A 4.5× larger model costs only 3.1× the time per update here (attention and small matrices do not scale with d_model²),
  while 5× the characters cost 5×.
- **At equal time capacity wins.** AL-14M after 164M characters (17 min) beats AL-3M after 819M characters (28 min) by
  0.58 points.
- **The two help each other.** The capacity gain grows with characters (+2.50 → +2.99) and the gain from characters grows
  with capacity (+1.92 → +2.41). So a larger model should not be trained on fewer characters than now.
- **More characters means more repetition.** text8 has 90M characters; the training split has 85.5M. Already 819M
  characters are 9.6 passes, and §4 shows the larger model starting to memorise. Raising the character budget further
  mostly adds passes over the same text.

## 6. Where the errors are (`tables/errors_by_word_position.md`, `scripts/analyze_errors.py`)

Test, W = 512, stride 128, grouped by the target's position in its word (the same targets as Acc_all):

| target | share of targets | AL-3M acc | AL-14M acc | Δ | AL-14M share of all errors |
|---|---|---|---|---|---|
| space (end of a word) | 17.1% | 93.97 | 94.68 | +0.71 | 3.4% |
| 1st letter of a word | 17.1% | 28.57 | **31.69** | +3.12 | **43.6%** |
| 2nd letter | 16.5% | 58.97 | 62.80 | +3.83 | 22.8% |
| 3rd letter | 14.1% | 69.18 | 73.23 | +4.05 | 14.1% |
| 4th and later letters | 35.2% | 84.61 | 87.76 | +3.15 | 16.1% |
| all | 100% | 70.23 | 73.20 | +2.97 | 100% |

(std ≤ 0.06 over seeds; v5's two models, also in the table file, are within 1.2 points of AL-3M in every group.)

- The first letter of a word is the hardest target by far: 31.7% correct, 3.3 bits. With the next word unknown, it is
  close to a guess among plausible words. It accounts for 43.6% of AL-14M's errors.
- To reach 80% overall, 25.4% of the current errors (6.80 of 26.80 points) must go. If first letters stay at 31.7%,
  every other character must be right 90.0% of the time (now 81.8%). If the other characters stay as they are, first
  letters must reach 71.5% (now 31.7%). Even with every other character perfect, first letters alone cap Acc_all at 88.3%.
- Scaling 4.5× improved every position by 3–4 points except spaces (already 94%). That is broad progress, but far from
  the 8 points per non-initial character that 80% needs.

## 7. The distance to 80%

| | best Acc_all, W = 512 | W = 1024 | below 80% |
|---|---|---|---|
| v5 best (ALiBi + fine-tuning, 3.2M) | 69.60 ± 0.07 | 69.82 ± 0.09 | 10.40 / 10.18 points |
| **v6 AL-14M** | **73.20 ± 0.03** | **73.38 ± 0.06** | **6.80 / 6.62 points** |

This round closed 3.6 of the 10.4 points; **6.8 remain**. Two rough extrapolations, both outside the measured range and
neither a measured accuracy:
- **Parameters.** At the measured +1.37 test points per doubling (3.2M → 14.2M, 819M characters), 38M would give about 75.1%.
  80% would need about 5 more doublings (≈ 440M parameters), far beyond what 85.5M training characters can support
  without overfitting.
- **BPC.** Across this study's models, Acc_all falls by 22 points per BPC (W = 512: 69.3% at 1.420 … 73.2% at 1.244).
  On that line, 80% sits near 0.94 BPC. Published text8 results for far larger models are around 1.08 BPC
  (Transformer-XL, 277M parameters, Dai et al. 2019).

So 80% top-1 is not plausible for this model family at a feasible size on text8. If 80% is the target, it is already
met by other measures: Top-2 accuracy (82.77%) and accuracy on the 5 most common characters (82.93%).

## 8. Verdicts

1. **Did scaling up raise Acc_all? Yes.** 3.2M → 14.2M under identical conditions: test Acc_all 70.23% → 73.20%
   (+2.97 ± 0.04 points at W = 512), positive in 3/3 seeds and at every W (+2.76 to +3.04), and on validation as well
   (+2.95 ± 0.04). BPC improved by 0.136. Every character group gained 3–4 points except spaces.
2. **Capacity or training characters? Capacity, for now.**
   - Per doubling it gives more (+1.38 vs +1.04 points) at lower cost per gain, and at equal GPU time the larger model wins.
   - More characters can only come as more passes over the same 85.5M characters, and the larger model is already
     memorising (gap +0.135).
   - The two are complementary, so the next stage should keep at least the current 819M characters, not trade them for size.
3. **Go to 38M (12 layers, d_model 512, 8 heads, 37,832,704 parameters)? Yes, as the next measurement, with
   adjusted expectations.**
   - *Why run it:* the capacity gain is large, consistent and not yet limited by overfitting, so a third point of the
     scaling curve is the most informative next step. If the per-doubling gain holds, expect about **75%** (+1.9 points),
     still about 5 points below 80%.
   - *Cost (estimate, not measured):* about 2.5× the per-update time of AL-14M, i.e. about 3.5 h per 50k-iteration run
     and about 12–14 GPU-hours for an equal-budget lr search plus 3 seeds. Batch 32 × 512 should still fit in 24 GB, to
     be checked with a smoke run.
   - *Precautions:*
     - include a dropout or weight-decay control chosen on validation, because at 38M the gap from repeated data will
       grow further (2.3 unique training characters per parameter);
     - extend the lr grid below 1× as well, since the 2× choice was at the edge for AL-3M;
     - stop if the gain over AL-14M is below about 1 point, i.e. well under the per-doubling trend.

## 9. Deviations, limitations, what went wrong

1. **lr at the edge.** Both models chose 2× of the planned 0.5× / 1× / 2× grid. The 4× check was added after the
   search, and run after the formal runs, so it changed no choice (§3). It brackets AL-14M but not AL-3M, by 0.08 points.
2. **The lr was chosen on one seed (4212) and 10,000 iterations** and reused for 50,000. A longer run may prefer a lower lr.
3. **GPUs.** Until 14:43 everything ran on GPU 3 only (CLAUDE.md default). When GPU 2 was made available, the GPU-3-only
   launcher was stopped and the remaining phases restarted on GPUs 2 and 3 (`scripts/run_v6_rest.sh`,
   `raw/launch_v6_b.txt`).
   - The run that was then training, `v6-AL-3M_s43`, was not interrupted; the new launcher waited for it (it detects
     running jobs and never starts them twice).
   - Its status record was written by hand from the first launcher's log (start time) and the mtime of its
     `val_eval.json` (end time). Its return code and foreign GPU processes after 14:42 were not observed (`note` field
     in `raw/v6_status.jsonl`).
   - `v6-AL-14M_s43` and `v6-AL-3M_s44` trained on GPU 2, everything else on GPU 3. Both cards are RTX 3090s with the
     same software, so results can differ only by floating-point nondeterminism.
   - My first attempt to set up a waiter for the bracket phase used `pkill` with a pattern that also matched my own
     shell, so that command was killed. No job was affected; the waiter was later replaced by `run_v6_rest.sh`.
4. **Uncommitted code.** Every run records `73f222e-dirty`. Each launch saved HEAD plus the full uncommitted diff
   (untracked source files included) to `code/` (7 launch snapshots, plus `*_final_*` of the finished code).
   - The training and evaluation code (`train.py`, `models/`, `utils/`, `scripts/eval_extrapolation.py`) is identical in
     all 7 snapshots (same hash), so all 47 jobs ran the same computation.
   - What changed between launches: the launcher (new phases, job ordering, waiting for running jobs), the new
     `scripts/analyze_errors.py`, tests, documentation and this folder's aggregation.
   - The last launch snapshot and the final one apply cleanly to a clean export of HEAD and reproduce the current files.
   - The code was committed after the runs as `27afcb9`; its code files are identical to the final snapshot.
5. **One stage checkpoint per run** instead of one every 10,000 iterations: the disk had only 25 GB free. Training is
   unaffected; intermediate checkpoints were not needed.
6. **Two validation measurements.** Selection uses the fixed set (1,024 × 512 windows, all positions); reported
   validation numbers use the whole split at W = 512, stride 128. The v5 / v3 rows use their best-loss checkpoints; in v6
   the two selection rules differ by ≤ 0.02 points (`tables/selection.md`).
7. **The 2 × 2 of §5 mixes seeds** (search runs seed 4212, formal runs seed 42). The seed spread is 0.03 points, far
   below the effects.
8. **Not done:** W = 128 with stride 1 for the v6 models (stride 128 at every W keeps one protocol); attention analysis
   of the larger model; the 38M stage (by request).

## 10. Code changes and checks

- `utils/full_context_eval.py`: `acc_top2` / `acc_top3` / `acc_top5` (target among the k highest logits), counted on the
  same scored targets as `acc`.
- `train.py`, `utils/logger.py`, `utils/checkpoint_saver.py`: `training.checkpoint_selection`.
  - Default `loss`: unchanged behaviour, no extra checkpoint.
  - `acc` saves `best_acc_bpc_` (highest fixed-validation accuracy, ties → lower loss, exact tie → earlier).
  - `--tuning` scores the selected checkpoint (plus the best-loss one for reference) into `val_eval.json`, with a
    `selected_checkpoint` field. The full test evaluation runs every mode on the selected checkpoint.
  - `timing.json` records `updates` and `train_tokens`.
- `scripts/eval_extrapolation.py`: scores the run's selected checkpoint by default (unchanged for older runs), plus
  `--checkpoint PREFIX` and `--tag NAME`.
- `scripts/analyze_errors.py` (new): accuracy / BPC by position in the word. `scripts/launch_v6.py`,
  `scripts/run_v6_rest.sh` (new): this round's queue. `.pre-commit-config.yaml`: the whitespace hooks skip
  `results/v6/code/` (verbatim diffs).
- Tests:
  - `tests/test_checkpoint_selection.py`: tie-break order, the selection switch, its default and errors.
  - `tests/test_cli.py`, four new end-to-end runs:
    - `acc` saves and scores the checkpoint the history predicts;
    - `loss` saves no extra checkpoint;
    - the test evaluation runs every mode on the selected checkpoint;
    - the evaluation tools load it.
  - `tests/test_full_context_eval.py`: top-k against brute force.
  - `tests/test_long_context.py`: output names with `--tag`.
  - `tests/test_analyze_errors.py`: word positions, grouped accuracy / BPC against brute force at stride 1 and 7, and
    agreement with `full_context_eval`.
  - The existing causality, target-count (every target exactly once) and bit-identity tests are reused unchanged.
- **Results.**
  - `pytest`: **335 passed**, 0 failed (`test_output.txt`; 290 at v4). The 3 error-analysis tests were checked against
    two planted bugs (a target offset and a word-position off-by-one), and both are caught.
  - `ruff check .`, `nbqa ruff .` and `pre-commit run --all-files` pass. The same hooks also pass with `--files` on the 166
    new, untracked files (which `--all-files` does not see), and they changed no file.
- `transformer.ipynb` needs no change: it calls `train.evaluate` with the default selection (`loss`), whose behaviour is unchanged.

## 11. Files and reproduction

| path | content |
|---|---|
| `configs/AL-3M.yaml`, `configs/AL-14M.yaml` | the two models (lr set per run by `--set`) |
| `raw/v6-AL-*_s{42,43,44}/` | formal runs: `config.yaml` (seed, lr, overrides, commit), `timing.json`, `val_eval.json`, `val_history.json` (fixed-set curve; training loss as 100-iteration means), `extrapolation{,_val}.json`, `errors_W512.json` |
| `raw/v6search-AL-*_lr*/` | search and bracket runs (validation only) |
| `raw/<v5 / v3 run>/extrapolation_v6.json`, `errors_W512.json` | re-evaluation of the reference models |
| `raw/v6_status.jsonl`, `raw/launch_v6*.txt` | one record per job (GPU, seconds, code snapshot), launcher logs |
| `code/<time>_<phase>_<HEAD>_<sha>.{patch,txt}` | code at every launch: HEAD, `git status`, full diff |
| `tables/`, `figures/` | built by `aggregate_v6.py` |
| `test_output.txt` | final `pytest` summary |

Checkpoints of every number: `checkpoints/<run>/best/best_acc_bpc_<step>` (`tables/runs.md`), not tracked.

```bash
for p in v5topk search formal seeds bracket evals; do python scripts/launch_v6.py --gpus 3 --phase $p; done
python results/v6/aggregate_v6.py --sync
```
