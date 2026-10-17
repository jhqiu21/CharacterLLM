# v5: Fixing RoPE beyond its training length

Branch `final`. Follow-up to `results/v4/` §4–5, where the Transformer (RoPE), trained on 128-character windows, went from
1.55 BPC at W = 128 to 4.4–4.9 BPC at W = 256–1024, and position interpolation (PI) without fine-tuning did not repair it
(3.6–4.1 BPC). This round tests three proposed fixes, in the order proposed: **(1) local attention** as a quick diagnostic,
**(2) PI plus long-window fine-tuning**, with the four requested controls, and **(3) SelfExtend**, which keeps exact
distances for near characters and compresses only the far ones.

Every number is test BPC at the best-validation-loss checkpoint, mean ± sample std (ddof 1) over seeds 42, 43, 44.
"Seed-matched" differences are mean ± std of the three per-seed differences; negative means the first model is better.
"W" columns are sliding-window BPC with stride 128, as in v4 §4: every target after the first window sees at least W − 127
characters. "W = 128, stride 1" is the full-context metric of the main results (every target sees 127 characters). It is
also the **truncation baseline**: the original RoPE model applied to the last 128 characters only, 1.4532 ± 0.0055.
A long-context method helps only if it beats this number.
`aggregate_v5.py` rebuilds every table (`tables/`) and figure (`figures/`) from `raw/`, plus `results/v3/raw` and
`results/v4/raw` for the reference rows.

## Summary

| Method | Training | W=256 | W=512 | W=1024 | W=128, stride 1 |
|---|---|---|---|---|---|
| RoPE | — | 4.3522 | 4.8519 | 4.4356 | 1.4532 |
| RoPE + PI | — | 3.6156 | 3.9813 | 4.0577 | 1.4532 |
| (1) RoPE + local attention 128 | — | 1.4535 | 1.4535 | 1.4535 | 1.4532 |
| (3) RoPE + SelfExtend (near 64, floor) | — | 1.4389 | **1.4304** | 1.4583 | 1.4532 |
| (2) RoPE + PI + 512-window fine-tuning, 1k it. | 16.4M chars | 1.4917 | 1.4588 | 5.1769 | 1.5127 |
| RoPE + 512-window fine-tuning, 1k it. | 16.4M chars | 1.4530 | 1.4337 | 1.5189 | 1.4698 |
| RoPE + PI + 512-window fine-tuning, 4k it. | 65.5M chars | 1.4550 | 1.4201 | 5.4003 | 1.4769 |
| RoPE + 512-window fine-tuning, 4k it. | 65.5M chars | 1.4385 | **1.4133** | 1.7240 | 1.4570 |
| ALiBi | — | 1.4449 | 1.4196 | **1.4127** | 1.4626 |
| ALiBi + 512-window fine-tuning, 4k it. | 65.5M chars | 1.4400 | 1.4075 | **1.3969** | 1.4608 |

(std in `tables/inference_variants.md`, `tables/finetune.md`, `tables/budget.md`; pre-training was 819M characters.)

- **(1) Local attention shows exactly where the failure comes from.** If every layer may only look at the last 128
  characters, RoPE goes from 4.4–4.9 BPC back to 1.4535 at every W ≥ 256. That is the truncation baseline to within
  +0.0003 ± 0.0006. So the collapse is caused entirely by keys at distances ≥ 128, which the model never saw in training.
  Stacking 4 layers could in principle carry information from up to 508 characters back, but this brings nothing.
- **(3) SelfExtend is the best of the methods that need no training.** Near window 64, distances ≥ 64 grouped with the
  smallest group that keeps every relative position below 128. At W = 512 it reaches 1.4304, which is −0.0228 ± 0.0057
  against the truncation baseline (better in all 3 seeds) and −2.55 against uniform PI. It is even slightly better than
  1,000 iterations of long-window fine-tuning, by 0.003 in every seed. The gain needs the group to stay small: at W = 1024
  (group 16) it disappears (+0.0051 ± 0.0072). The floor (SelfExtend) and continuous ("linear") compression are within
  0.004 of each other. The near window is bracketed: 16 < 32 < 64 > 96 > 112.
- **(2) Long-window fine-tuning repairs RoPE, but PI makes it worse, not better.** With the same extra characters, PI +
  fine-tuning is worse than fine-tuning without PI in all 3 seeds: +0.0252 ± 0.0022 at W = 512 after 1,000 iterations and
  +0.0068 ± 0.0029 after 4,000. It also costs more short-context accuracy: +0.059 vs +0.017 at W = 128 with stride 1 after
  1,000 iterations, and +0.024 vs +0.004 after 4,000. Extra training at the old length (128 windows, same characters)
  gives nothing: no learning rate ever went below the starting validation loss. So the repair comes from training on long
  windows, not from training longer.
- **After fine-tuning, PI's attention looks fine but its predictions are still worse.** The PI-induced shift of the
  attention peak from distance 1 to 3–5 (v4 §5) is undone: every layer peaks at distance 1 again. Yet BPC stays behind
  fine-tuning without PI. The peak shift is therefore a symptom, not the whole cause.
- **Nothing beats ALiBi.** The best RoPE result, 4,000 iterations of 512-window fine-tuning (1.4133 at W = 512), ties
  ALiBi at W = 512: −0.0063 ± 0.0083, signs differ across seeds. It also matches ALiBi's 1.4127 at W = 1024. But it fails
  beyond the fine-tuning length (1.72 at W = 1024), whereas ALiBi keeps improving with more context. With the same
  fine-tuning, ALiBi reaches 1.3969 at W = 1024, the best number in this study.

## 1. What was run

All jobs ran on GPU 3 (RTX 3090), one at a time. `raw/v5_status.jsonl` holds 143 job records, all complete, about 6
GPU-hours; no other process used the GPU during any job. Two more jobs (`ft-short128_lr3e-5_s42`, `ft-pi512_lr1e-3_s43`)
kept running while their launcher was restarted to reorder the queue (§6), so they have no record line. Their outputs are
complete.

| block | jobs | protocol |
|---|---|---|
| evaluation-only variants | 3 seeds × (validation: plain, PI, local 32/64/128, SelfExtend floor/linear × near 16/32/64/96/112; test: local, SelfExtend) | `scripts/eval_extrapolation.py`, W = 128 (local), 256, 512, 1024; the setting of each family is chosen by **validation** BPC at W = 1024, fixed before the test numbers were read |
| fine-tuning lr grid | 4 conditions × lr {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, seed 42, validation only (`--tuning`) | start from the 50k best-loss checkpoint of the same seed (`train.py --init-from`), fresh AdamW, 1,000 iterations, warmup 50, cosine to 0, validation every 50; lr = lowest best fixed-validation loss |
| fine-tuning | seeds 43, 44 at the chosen lr (seed 42 = the grid run) | as above; test evaluation W = 128–1024 (stride 128) and W = 128 with stride 1 |
| budget | the continued conditions, 4,000 iterations, all seeds, same lr | warmup 200, validation every 200 |
| attention | PI and no-PI fine-tuning, seed 42, W = 512 | `scripts/analyze_attention.py`, 512 test windows |

**Fine-tuning conditions.** All use 16,384 characters per iteration (32 × 512 or 128 × 128), so every condition at a given
number of iterations sees the same number of extra training characters. 1,000 iterations are 2% of pre-training.

| condition | answers |
|---|---|
| RoPE (no fine-tuning; v3) | how bad is the extrapolation failure? |
| RoPE + PI (no fine-tuning; v4) | how much does scaling the positions alone help? |
| RoPE + 512-window fine-tuning (`long512`) | how much does adapting to long inputs alone help? |
| RoPE + PI + 512-window fine-tuning (`pi512`, pos_scale 0.25 in training and at every W) | does PI still add something after adaptation? |
| RoPE + 128-window fine-tuning (`short128`, added) | is the gain only from more training? |
| ALiBi + 512-window fine-tuning (`alibi512`, added) | the comparison with ALiBi at an equal training budget |

**Variants.** *Local attention N*: every layer masks keys at distance ≥ N (`attn_window`). *SelfExtend near n*: pairs at
distance < n keep their exact RoPE angles. Farther pairs rotate the query by ⌊i/G⌋ + n − ⌊n/G⌋ and the key by ⌊j/G⌋
('floor', Jin et al. 2024), or by i/G + n − n/G and j/G ('linear', a continuous compression as in leaky ReRoPE). G is the
smallest group (integer for floor) that keeps every relative position ≤ 127. Example, near 64: G = 4 / 8 / 16 at
W = 256 / 512 / 1024 for floor (3.0 / 7.1 / 15.2 for linear). Both variants are implemented in `models/models.py`. They are checked against an
independent numpy reference that rotates every (i, j) pair by its SelfExtend relative position
(`tests/test_long_context.py`).

## 2. Local attention: a diagnostic (`tables/inference_variants.md`)

| | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| local 32 | 1.5882 ± 0.0037 | 1.5359 ± 0.0050 | 1.5359 ± 0.0050 | 1.5359 ± 0.0050 |
| local 64 | 1.5561 ± 0.0022 | 1.4881 ± 0.0032 | 1.4881 ± 0.0032 | 1.4881 ± 0.0032 |
| local 128 | 1.5452 ± 0.0030 | 1.4535 ± 0.0054 | 1.4535 ± 0.0055 | 1.4535 ± 0.0055 |

Local 128 vs the truncation baseline, seed-matched: +0.0002 / +0.0003 / +0.0003 at W = 256 / 512 / 1024.
**Masking the far keys removes the whole failure, so the far keys are the failure.** The 4-layer stack could in principle
pass information from up to 4 × 127 characters back, but the result equals plain truncation to the fourth decimal. The
model has not learned to use such indirect paths, because in training a 128-character window never contained more.
Windows narrower than 128 are worse, so the model does use its whole training context. **Conclusion: restricting far
connections is enough to stop the collapse, but by itself it gains nothing.** A gain requires the model to read
distances ≥ 128 in a form it understands, which is what SelfExtend and fine-tuning try to provide.

## 3. SelfExtend: exact near distances, compressed far ones (`tables/inference_variants*.md`, `figures/bpc_vs_W.*`)

Validation BPC at W = 1024 (selection): near 16 / 32 / **64** / 96 / 112 = 1.5205 / 1.4875 / **1.4750** / 1.4993 / 1.6067
(floor) and 1.5192 / 1.4893 / **1.4785** / 1.4981 / 1.5540 (linear). 96 and 112 were added after the first three because
64 was the upper edge. Test BPC of the selected settings:

| | W=256 | W=512 | W=1024 |
|---|---|---|---|
| SelfExtend floor, near 64 | 1.4389 ± 0.0074 | 1.4304 ± 0.0111 | 1.4583 ± 0.0121 |
| SelfExtend linear, near 64 | 1.4387 ± 0.0075 | 1.4309 ± 0.0113 | 1.4619 ± 0.0126 |
| − truncation baseline (floor) | −0.0143 ± 0.0020 | **−0.0228 ± 0.0057** | +0.0051 ± 0.0072 |
| − RoPE + PI (floor) | −2.1767 ± 0.0541 | −2.5509 ± 0.0332 | −2.5994 ± 0.0816 |
| − ALiBi (floor) | | +0.0108 (3/3 seeds worse) | +0.0456 (3/3 seeds worse) |

**This answers the question of proposal (3): yes.** Keeping the near distances exact is what uniform PI lacked. With the
same weights, SelfExtend at W = 512 is 2.55 BPC better than PI. It is also the only training-free method that reads
context beyond 128 to a net gain: about 0.02 BPC, in every seed. This fits the v4 attention analysis, where PI moved the
attention peak from distance 1 to 3–5.

**Its limit is the far band's resolution.** With a fixed trained range of 128 relative positions, a longer window needs a
larger group: 8 at W = 512, 16 at W = 1024 (near 64). Per-position loss at W = 1024 (`tables/per_position_W1024.md`) is 1.423 at
positions 448–511, where the window holds ≤ 512 characters and the far band is still informative. It rises to 1.456 at
positions 960–1023. A larger near window leaves fewer positions for the far band: near 112 needs G = 26 at W = 512, and
its BPC rises again. Floor and linear compression behave the same (within 0.004); integer grouping is not what matters.

## 4. Long-window fine-tuning, with and without PI (`tables/finetune.md`, `tables/differences.md`, `tables/ft_lr_grid.md`)

**Learning rate** (seed 42, best fixed-validation BPC; 512-character windows except `short128`):

| condition | before | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|---|---|
| RoPE + PI + 512 | 4.0353 | 2.2687 | 1.8156 | 1.5808 | **1.5402** | 1.5998 |
| RoPE + 512 | 3.9224 | 1.9630 | 1.6694 | **1.5069** | 1.5193 | 1.5716 |
| RoPE + 128 | 1.5757 | 1.5757 (it. 0) | (it. 0) | (it. 0) | (it. 0) | (it. 0) |
| ALiBi + 512 | 1.4865 | **1.4819** | 1.4822 | 1.4860 | (it. 0) | (it. 0) |

3e-3 was added because PI's best was the upper edge, 1e-3. All grids are bracketed. `short128`'s best checkpoint is
iteration 0 at every lr: the 50k run ended with the lr annealed to 0, and more training at the same length only moves
the model away from that minimum. Its selected model is therefore the source model itself. It was not run on further
seeds (launcher rule, `scripts/launch_v5.py`), and its row is reported as "= RoPE".

**Test BPC (1,000 iterations):**

| | W=128, stride 1 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| RoPE + 512-window fine-tuning | 1.4698 ± 0.0073 | 1.4530 ± 0.0083 | 1.4337 ± 0.0112 | 1.5189 ± 0.0016 |
| RoPE + PI + 512-window fine-tuning | 1.5127 ± 0.0069 | 1.4917 ± 0.0082 | 1.4588 ± 0.0112 | 5.1769 ± 0.1942 |
| … + PI again at W=1024 (pos_scale 0.125) | | | | 2.6232 ± 0.0913 |
| ALiBi + 512-window fine-tuning | 1.4624 ± 0.0019 | 1.4424 ± 0.0023 | 1.4118 ± 0.0038 | 1.4021 ± 0.0049 |

Seed-matched, at W = 512:
- PI fine-tuning − no-PI fine-tuning: **+0.0252 ± 0.0022** (3/3 seeds worse). At W = 128 with stride 1: +0.0429.
- No-PI fine-tuning − truncation baseline: −0.0196 ± 0.0057. PI fine-tuning − truncation baseline: +0.0056 ± 0.0059, so
  the extra context gives PI no net gain after 1,000 iterations.
- No-PI fine-tuning − ALiBi: +0.0141 ± 0.0093. Against ALiBi + fine-tuning: +0.0219 ± 0.0095.

Answers to the four requested comparisons:

| comparison | answer |
|---|---|
| plain RoPE | 4.4–4.9 BPC at W ≥ 256 (catastrophic) |
| RoPE + PI (no training) | 3.6–4.1 BPC; damages positions 0–127 too (v4) |
| RoPE + long-window training | 1.4337 at W = 512 (1k it.), 1.4133 (4k it.): repairs the failure and gains from context up to the training length |
| RoPE + PI + long-window training | 1.4588 (1k), 1.4201 (4k): also repairs it, but worse than without PI at both budgets, and worse at short context |

Neither fine-tuned RoPE model extrapolates past its new training length. Without PI, BPC rises to 1.52 (1k) or 1.72 (4k)
at W = 1024: the longer it is fine-tuned on 512, the worse it gets beyond 512. With PI and pos_scale kept at 0.25,
positions beyond 512 are unseen again (5.2 BPC). Interpolating once more (0.125) gives 2.6.

## 5. Budget: 4,000 iterations (`tables/budget.md`, `tables/budget_differences.md`)

Is PI just slower to adapt? With 4× the fine-tuning (same lr), the gap shrinks but does not reverse:

| seed-matched, W=512 | 1,000 it. | 4,000 it. |
|---|---|---|
| PI fine-tuning − no-PI fine-tuning | +0.0252 ± 0.0022 | **+0.0068 ± 0.0029** (3/3 seeds worse) |
| no-PI fine-tuning − ALiBi | +0.0141 ± 0.0093 | −0.0063 ± 0.0083 (seeds: +0.0013 / −0.0151 / −0.0053) |
| no-PI fine-tuning − ALiBi + fine-tuning (same budget) | +0.0219 ± 0.0095 | +0.0058 ± 0.0086 |
| PI fine-tuning − ALiBi | +0.0392 ± 0.0102 | +0.0005 ± 0.0062 |
| short-context cost (W=128, stride 1; PI / no PI) | +0.0594 / +0.0166 | +0.0236 / +0.0038 |

PI improved more from the extra iterations than no-PI fine-tuning did (−0.0387 vs −0.0204 at W = 512). It started further
from a good model, because every position, including those below 128, has a new angle. Whether it would catch up with
far more training is open; at 4,000 iterations it has not. ALiBi gains little from fine-tuning (−0.0078 at 1k, −0.0121 at
4k, W = 512): it already used long contexts before. At W = 512, RoPE fine-tuned for 4,000 iterations is statistically
indistinguishable from ALiBi, with or without ALiBi's own fine-tuning. It does not beat ALiBi at any W where both
are defined, and only ALiBi keeps improving beyond 512.

## 6. Attention after fine-tuning (`tables/attention_short_range.md`)

Seed 42, W = 512, all queries; head-mean distance profile per layer (layers 0 / 1 / 2 / 3):

| model | peak distance | profile summed over d = 0–2 | effective span (chars) |
|---|---|---|---|
| RoPE, W = 128 (v4) | 1 / 1 / 1 / 1 | 0.70 / 0.76 / 0.76 / 0.42 | 9.8 / 3.2 / 5.1 / 24.4 |
| RoPE + PI, W = 512 (v4) | 0 / 5 / 3 / 1 | 0.25 / 0.19 / 0.31 / 0.16 | 43.2 / 10.2 / 13.6 / 114.3 |
| RoPE + PI + 512-window fine-tuning | 1 / 1 / 1 / 1 | 0.72 / 0.81 / 0.77 / 0.38 | 34.6 / 3.7 / 5.8 / 74.4 |
| RoPE + 512-window fine-tuning | 1 / 1 / 1 / 1 | 0.66 / 0.74 / 0.73 / 0.37 | 32.5 / 6.9 / 15.8 / 90.5 |

PI fine-tuning restores the character-level attention pattern completely: peak at distance 1 in every layer, and as
much weight near the query as RoPE at W = 128. Yet it is 0.025 BPC behind fine-tuning without PI. The v4 hypothesis
("PI moves the peak to 3–5 characters, so the model cannot tell the last few characters apart") explains PI's failure
*without* training. A plausible explanation for what remains after training (not tested here): with pos_scale 0.25 every
rotation frequency advances 4× more slowly per character, so neighbouring characters differ by a quarter of the trained
angle, and the model has to separate them with smaller differences in its attention scores. This fits PI's deficit
against no-PI fine-tuning being largest at short context: +0.043 at W = 128 with stride 1, +0.025 at W = 512 (1k it.).

## 7. Deviations, limitations

1. **Additions to the requested design.** (a) The 128-window and ALiBi fine-tuning conditions. (b) Validation-based
   selection of every lr and variant setting. (c) The 4,000-iteration budget check. (d) The 'linear' SelfExtend variant.
   (e) Near windows 96 and 112, and lr 3e-3, added after the first grids put the best value on an edge.
2. **The lr is chosen on seed 42 only**, and the 1,000-iteration lr is reused for 4,000. Longer runs often prefer a lower
   lr, so the 4k numbers may be slightly pessimistic for both RoPE conditions.
3. **Only one fine-tuning length (512) and one PI scale (0.25).** Fine-tuning at 1024, or PI with a scale chosen for
   a longer target, were not tried. The fine-tuned models are not evaluated with SelfExtend on top.
4. **Variant selection used W = 1024.** The fixed criterion picks near 64 / local 128. At W = 512 the ranking of the near
   windows is the same.
5. **Three seeds.** Differences of 0.005–0.01 with std ≈ 0.008 (RoPE fine-tuning vs ALiBi) are ties; differences quoted
   with "3/3 seeds" have the same sign in every seed.
6. **Uncommitted code.** Every run records `git_commit 7bbc227…-dirty`: the code changes of this round (below) were not
   committed before launch. They only add options with defaults; the pre-existing reference test
   (`tests/data/pe_reference_logits.npz`, bit-identical logits for every encoding) passes.
7. **Queue reordering.** The launcher was stopped and restarted twice: once to put the fine-tuning phase before the
   evaluation-only variants and add lr 3e-3, once to drop the `short128` seeds. Running jobs kept running and were
   skipped afterwards ("finished meanwhile", `raw/launch_v5_b.txt`). The first launcher's log is `raw/launch_v5.txt`.

## 8. Code changes (this round)

- `models/models.py`, `models/positional_encodings.py`: `attn_window` (local attention, any encoding) and
  `self_extend_window/_group/_mode` (RoPE only) as evaluation options. Defaults leave the model bit-identical.
- `train.py`: `model.pos_scale` config key (default 1.0; PI in training), `--init-from RUN` (fine-tuning from a run's
  best-loss checkpoint), and `apply_chunked` (the 1024 × 512 validation set does not fit in one pass; T = 128 is unchanged).
- `utils/extrapolation_eval.py`, `scripts/eval_extrapolation.py`: `--local-window`, `--self-extend`, `--self-extend-mode`,
  `--split val` (never opens the test file), `--stride1`. PI composes with a run's own `pos_scale`.
- `scripts/launch_v5.py` (phases grid / infer / ft / budget), `results/v5/aggregate_v5.py`.
- Tests: `tests/test_long_context.py` (local window = cropped input for one layer; receptive field; SelfExtend vs numpy
  reference; group sizes; pos_scale composition; chunked validation) and two `--init-from` tests in `tests/test_cli.py`.

## Reproduction

```bash
python scripts/launch_v5.py --gpus 3 --phase grid      # lr grid (validation only)
python scripts/launch_v5.py --gpus 3 --phase ft        # other seeds at the chosen lr + test evaluations + attention
python scripts/launch_v5.py --gpus 3 --phase infer     # local attention / SelfExtend, validation then test
python scripts/launch_v5.py --gpus 3 --phase budget    # 4,000-iteration fine-tuning
python results/v5/aggregate_v5.py --sync
```
