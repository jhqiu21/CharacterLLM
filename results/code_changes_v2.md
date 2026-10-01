# Code changes v2: fairness and correctness fixes before hyperparameter tuning

Branch `arch-fairness-fixes`, created from tag **`tableV-v1`** (= `67b9caf`, the tip of `fix-dropout-rng-full-context-eval`
that produced every number in `results/` outside `results/v2/` and this file). Not pushed. No hyperparameter values were changed:
model sizes, learning rates, dropout rates and label smoothing are exactly as in the `tableV-v1` configs. The new defaults
(`sinusoidal_scale: unit`, `forget_bias: 1.0`, the LR schedule, unified attention, generation) mean that the Table V numbers in
`results/REPORT.md` are **not** reproduced by this branch; that is expected.

Commits (each part separately):

| commit | content |
|---|---|
| `31fcc68` | Part 1: all code and config changes (items 1–8) and the notebook |
| `a14a281` | Part 2: tests |
| `5bc2d64` | Part 4: README |
| (this commit) | Part 3/4: this report and `results/v2/` (smoke and step-time records, test output) |

**Summary**
- All eight items are implemented. Every pre-existing test plus the new ones pass: **217 passed** (68 old + 149 new).
  `ruff check .`, `nbqa ruff .` and `pre-commit run --all-files` pass.
- Smoke: all 6 configs (300 iterations, end to end including test evaluation), the 2 `--tuning` runs and the 2 sinusoidal runs exit 0 with finite losses.
- **Item 5 did not deliver what it was meant to.** The old training step's extra metrics cost essentially no GPU time
  (0.08 ms/step). The lean step changes how XLA schedules the compiled program, and that makes **TR2o 9.4 % slower**
  (23.05 → 25.21 ms/step). LR-2 is unchanged (65.55 → 65.74 ms/step). So the LR-2 / TR2o step-time ratio *narrows*, from 2.84 to 2.61.
  I kept the lean step because the task asked for it and it is correct (same loss and gradients, tested). §3 has the
  diagnosis and a one-line revert, if you prefer the old step.

---

## 1. Changes

| # | what | why | files |
|---|---|---|---|
| 1 | `model.sinusoidal_scale: unit \| raw` (default `unit`). `unit` multiplies the fixed sinusoidal table by `1/sqrt(d_model/2)`, so every row has norm 1. It applies in `SinusoidalPositionalEncoding` and to the sinusoidal component of `HybridPositionalEncoding` (before the 0.5/0.5 mix). `raw` is the old table. `get_sinusoidal_embeddings` is unchanged. An unknown value raises. | Row norm was √128 ≈ 11.3 against ≈1 for token embeddings, so the residual stream started dominated by a fixed positional signal. | `models/positional_encodings.py` (`sinusoidal_scale_factor`), `models/models.py`, `train.py` (`MODEL_DEFAULTS`, `build_model`), transformer configs |
| 2 | Every `pos_encoding_type` uses `MultiHeadAttentionWithPE`; the `nn.SelfAttention` branch is gone. Parameter counts are unchanged. | Flax `nn.SelfAttention` defaults to `broadcast_dropout=True` (one T×T attention-dropout mask for all batch elements and heads), the custom module draws independent masks, so `learned`/`sinusoidal`/`hybrid` vs `rope`/`alibi`/`relative` mixed in an implementation difference. | `models/models.py` |
| 3 | `model.forget_bias` (LSTM configs, default `1.0`) initialises `b_i[H:2H]` (gates split `(i, f, g, o)`); the rest of `b_i` stays 0. `0.0` = old behaviour. Nothing else in the LSTM changed. | Standard forget-gate initialisation (Gers et al. 2000; Jozefowicz et al. 2015). | `models/LSTM.py`, `train.py`, `configs/LR-1.yaml`, `configs/LR-2.yaml` |
| 4 | LR schedule with `decay_steps = epochs` (warmup included, as optax defines it). Smoke-test override (warmup → epochs/10 when epochs ≤ warmup) kept, now also applied after `--set`. | The old `decay_steps = epochs − warmup` reached 0 at step 48,000; the last 2,000 steps did nothing. | `train.py` (`warmup_cosine_schedule`, `make_lr_schedule`, `build_run_config`) |
| 5 | Training step uses `make_train_loss(config)`: the optimised loss plus only `loss_train`, `loss_all`, `loss_last`. Validation keeps `make_loss_and_metrics`. Both share `make_training_loss`, so the objective is defined once. | The training step computed ECE, perplexity, entropy etc. every step only to discard them. (See §3: this did not speed anything up.) | `train.py`, `transformer.ipynb` |
| 6 | Generation: prompt at positions 0..S−1 of the fixed window; logits read at `cur_len − 1`; sample written at `cur_len`; window slides once full; prompts longer than W keep their last W tokens. Both the `lax.scan` path and the n-gram path. No token-0 padding: positions after the prompt hold a filler (last prompt token, overridable with `fill_token`) that is never read. Callers no longer truncate prompts to their *first* `max_len` characters. Multi-prompt Self-BLEU added (below). | Left-padding with token 0 (`'a'`) conditioned every sample on ~116 `'a'`s before `"hello my fri"`. | `utils/generation.py` (`initial_window`, `next_token_logits`), `utils/eval.py`, `train.py` |
| 7 | `--set key=value` (repeatable): dotted keys, int/float/bool/string parsing, unknown keys and sections raise, `training.seed` must come from `--seed`, recorded in `run.overrides` of `config.yaml`. `--tuning`: never opens the test file; selects checkpoints on the fixed validation set as before; scores the best-loss checkpoint with `full_context` on the **validation split**, stride `evaluation.tuning_val_stride` (default 8) → `val_eval.json`; no `eval.json`, no samples, no Self-BLEU. `--data-dir` (used by the tests). | Support for the bounded search. | `train.py`, configs |
| 8 | Docstring: untied head. `get_positional_encoding` deleted (grep: no importers). `ALiBiPositionalBias.self.slopes` (and its now-empty `setup`) removed. Notebook uses `make_train_loss` for the step. | Cleanup. | `models/models.py`, `models/positional_encodings.py`, `transformer.ipynb` |

**Multi-prompt Self-BLEU** (item 6): P = `evaluation.self_bleu_multi_n_prompts` = 10 prompts. Each prompt is a passage of
`self_bleu_multi_prompt_len` = 128 characters from the validation split, at sorted, distinct offsets from `np.random.default_rng(0)`
(`train.self_bleu_prompt_offsets`, independent of the training seed). For each prompt, `self_bleu_multi_n_samples` = 20 samples
are drawn as one batch, with key `fold_in(PRNGKey(evaluation.seed), p)` and the existing gen_len 1000, temperature 0.7 and n-gram
setting (none). Self-BLEU-4 is computed within each prompt's samples. `eval.json` → `self_bleu_4_multi`: `mean`, `std`
(ddof 1, over the 10 prompts), `per_prompt`, `prompt_offsets`, …. The single-prompt `self_bleu_4` (`"hello my fri"`, 20 sequential
samples) is kept. Its values change anyway because of the new conditioning.

**Things done differently from the letter of the task**
- **LR schedule (item 4):** passing `decay_steps=epochs` to `optax.warmup_cosine_decay_schedule` alone does **not** satisfy
  "lr(49,999) > 0": optax evaluates `0.5·(1 + cos(π·t/T))` in float32, which rounds to exactly 0 at t = 47,999 of 48,000 cosine steps
  (true value ≈ 4.3e-13). `train.warmup_cosine_schedule` is the same function with the same arguments (decay_steps = total length),
  computed as `end + (peak − end)·cos²(π·t/2T)` (the identity 0.5(1+cos x) = cos²(x/2)), clamped to `end_value` at t ≥ T. A test
  checks it against optax with rel 1e-4 at 56 steps. Effect on training: none beyond float32 rounding. The rates involved are below 1e-9 × peak.
- **Generation filler:** "fill the rest with any token" and "do not pad with token 0 anywhere" are both satisfied by repeating the last
  prompt token. Tests show that the filler value (default, 0 or 13) never changes the output.
- **`--set training.seed`** is rejected (use `--seed`); `--data-dir` is an extra flag.
- **Old checkpoints** of `learned`/`sinusoidal`/`hybrid` models (parameter tree `SelfAttention_0/{query,key,value,out}`) no longer load.
  No reported run used those encodings.

## 2. Tests

`pytest`: **217 passed** in 394 s (CPU; full output in `results/v2/test_output.txt`); 68 were there before (all still pass).

| file (new) | checks | tests |
|---|---|---|
| `test_sinusoidal_scale.py` | `unit` rows have norm 1 (1e-5) at d_model 256; `raw` = old table (norm √128); hybrid's sinusoidal half scaled the same way for both settings; unknown value rejected; config switch reaches sinusoidal and hybrid models | 7 |
| `test_attention_unified.py` | no `SelfAttention` in any parameter tree (6 PE types); counts learned 3,202,048 / sinusoidal 3,169,280 / hybrid 3,202,048 / rope 3,169,280 / alibi 3,169,280 / relative 3,173,360; old `nn.SelfAttention` weights (q/k/v (D,H,hd), out (H,hd,D)) reshaped to (D,D) give the same output to 1e-5 without dropout; same sequence repeated over the batch gives different per-element outputs in training mode (module and full model; the old module gives identical rows, so the test can tell them apart) | 19 |
| `test_lstm_forget_bias.py` | every layer: `b_i[H:2H] == forget_bias`, rest 0, for 1.0 and 0.0; default 1.0 in LR-1/LR-2; counts 2,115,072 and 3,177,052 for both values | 7 |
| `test_lr_schedule.py` | epochs 50,000, warmup 2,000: lr(0) = init, lr(2,000) = peak, lr(48,000) > end, lr(49,999) > end, lr(50,000) = lr(50,100) = end (two init/end settings); monotone decay; equals optax's warmup-cosine with total decay_steps; smoke override (300 iterations → warmup 30) works | 5 |
| `test_train_loss.py` | lean loss vs a **verbatim copy of the `tableV-v1` loss**: loss, logged metrics and all gradients allclose (rtol 1e-5), Transformer and LSTM, 3 loss variants; one whole jitted optimizer step gives the same parameters; validation metrics unchanged; `ls_eps: 0` + `LABEL_SMOOTHING: True` gives bit-identical loss and gradients to `LABEL_SMOOTHING: False` (two tail schemes), and `ls_eps > 0` differs | 10 |
| `test_cli.py` | `--set` parses int/float/bool/string; dotted overrides applied and returned; keys added by defaults reachable; unknown key, unknown section, sub-key of a value, section as value, missing `=`, `training.seed` raise; **`--tuning` end to end** through `train.main` on a tiny dataset with `open` monkeypatched to fail on `text8_test`, test evaluation / Self-BLEU / sampling monkeypatched to fail: the test file is never opened, `val_eval.json` exists and scores len(val) − 1 targets, no `eval.json` or `.txt` reports, overrides recorded in `config.yaml`; `load_data(include_test=False)` | 17 |
| `test_generation.py` | for Transformer (learned, rope) and LSTM, prompt lengths 1, 12, W, W + 5: first-step logits equal `model.apply` on the unpadded prompt (1e-5); greedy and sampled scan-path generation equal a brute-force loop on exactly the real tokens (also across the window slide); filler 0 / 13 / default give identical output and the default window has no token 0; batch 3 works (greedy rows equal batch-1 runs; repeated prompts sample differently); n-gram path (`ngram_size=1`) gives the same tokens as the scan path, greedy and sampled; n-gram blocking still blocks | 81 |
| `test_self_bleu.py` | prompts are validation slices at deterministic, sorted, distinct offsets (= `default_rng(0)`); same prompts for training seeds 4212 / 4213; config defaults P = 10, 20 samples, 128 chars; per-prompt score equals a by-hand batch generation | 3 |

## 3. Verification runs (GPU 0: NVIDIA A100 80GB PCIe)

### Smoke (`results/v2/smoke_status.jsonl`)

300 iterations, validation every 100, full-context stride 64, seed 4212, commit `5bc2d64`. They ran **one at a time**: the first attempt
ran three in parallel and `v2smoke_TR2o` hit a GPU OOM in full-context evaluation (batch 8192 needs a 10.7 GB buffer; three
concurrent runs held ~56 GB). That attempt was discarded and the whole queue rerun sequentially. All runs exit 0 and
every training loss is finite.

| run | val loss_all @ 0 / 100 / 200 / 299 | best step | full-context BPC (split, stride) | targets | Self-BLEU-4 single / multi (mean ± std) |
|---|---|---|---|---|---|
| baseline (learned) | 3.978 / 2.403 / 2.370 / 2.361 | 299 | 3.411 (test, 64) | 4,999,999 | 0.057 / 0.036 ± 0.012 |
| TR2o | 4.013 / 2.159 / 1.956 / 1.925 | 299 | 2.709 (test, 64) | 4,999,999 | 0.074 / 0.087 ± 0.009 |
| TR8o | 3.961 / 2.385 / 2.345 / 2.330 | 299 | 3.378 (test, 64) | 4,999,999 | 0.023 / 0.021 ± 0.001 |
| LR-1 | 3.300 / 2.872 / 2.730 / 2.692 | 299 | 3.863 (test, 64) | 4,999,999 | 0.008 / 0.007 ± 0.001 |
| LR-2 | 3.288 / 2.871 / 2.693 / 2.662 | 299 | 3.815 (test, 64) | 4,999,999 | 0.012 / 0.011 ± 0.001 |
| TR2o-nomask | 4.013 / 2.159 / 1.956 / 1.925 | 299 | 2.709 (test, 64) | 4,999,999 | 0.083 / 0.083 ± 0.008 |
| `--tuning` TR2o | 4.013 / 2.159 / 1.956 / 1.925 | 299 | 2.734 (**validation**, 8) | 4,499,999 | — (none written) |
| `--tuning` LR-2 | 3.288 / 2.871 / 2.693 / 2.662 | 299 | 3.823 (**validation**, 8) | 4,499,999 | — (none written) |

The tuning runs wrote `val_eval.json` and no `eval.json`, and their logs say `test NOT LOADED (tuning mode)`. Their validation losses
agree with the normal runs' to within 1e-4 (same seed and code). All smoke and step-time runs of the new code record the clean commit `5bc2d64`; the old step-time runs record `67b9caf` (`tableV-v1`). The patched notebook (epochs 20, validation every 10, warmup 2, stride 64) ran
start to finish via `jupyter-execute`, evaluation cell included (`eval.json` with both Self-BLEU metrics, 4,999,999 targets);
the committed notebook keeps its outputs cleared.

### Step time (`results/v2/step_time_status.jsonl`)

TR2o and LR-2, 2,000 iterations, seed 4212. Old step = `tableV-v1` checked out in the worktree `../CharacterLLM-tableV-v1`; new = this branch.
Median s/step over iterations 1,000–1,999 excluding validation iterations (997 steps), as `train.py` measures it. Runs were
sequential in ABBA order (two repeats each). No other GPU compute process was seen during any run (polled every 2 s). The **host was
shared** (16 logged-in users, load average ≈ 4–5 on 64 cores).

| model | old step (tableV-v1) | new lean step | change |
|---|---|---|---|
| TR2o | 23.15, 22.95 → **23.05 ms** | 25.18, 25.23 → **25.21 ms** | **+9.4 %** |
| LR-2 | 64.74, 66.35 → **65.55 ms** | 65.78, 65.69 → **65.74 ms** | +0.3 % |
| LR-2 / TR2o | 2.84 | 2.61 | |

**Why the lean step is not faster** (isolated with a step micro-benchmark and `nsys`, on TR2o):
- The schedule change has no effect: old loss with the optax or new schedule gives 22.7–23.4 ms either way; the lean loss gives 25.1–25.2 ms with either schedule.
- Returning only the loss (no metrics at all) is also ~25 ms. Adding back only the ECE computation restores ~22.9 ms.
  Adding perplexity, entropy or a softmax-max metric does not.
- GPU kernel time per step is the same: 11.93 ms old vs 11.85 ms lean (the lean step does *less* GPU work). So the old metrics cost ≈ 0.08 ms/step,
  and the "wasted work" in item 5 was negligible on this GPU. The GEMMs are identical (48 cuBLAS + 51 Triton fusions in both programs).
- The difference is host-side execution of the compiled program: the old program runs as 2 CUDA-graph launches per step, the lean one as 1, with
  more `cuLaunchHostFunc` callbacks (~100 vs ~83 per step) that account for most of the extra wall time. The old program also has one
  `triton_softmax` fusion (from the ECE path) that the lean one lacks. I did not trace exactly why XLA (jaxlib 0.6.2) partitions the two
  programs differently.
- To restore the old step, change one line in `train.train`: `make_train_step(model, make_train_loss(config), tx)` →
  `make_train_step(model, loss_and_metrics, tx)` (the logger only reads the three keys both versions return). I did not do this, because it
  contradicts item 5. **Before the final runs, decide which step to use.** Either way, s/step is a property of the XLA program, not just of
  the model, so compare throughput only between runs that use the same step code.

### Sinusoidal `unit` vs `raw` (a check that the switch works, not a result)

`configs/baseline.yaml` + `--set model.pos_encoding_type=sinusoidal --set model.sinusoidal_scale={unit,raw}`, `--tuning`, 300 iterations, seed 4212.
Validation loss (fixed 1024 × 128 set) at the last iteration (299): **unit 2.3290**, **raw 2.4343**. The two runs differ, so the
switch reaches the model. One seed and 300 steps support no conclusion about the encodings.

## 4. What is verified and what is not

**Verified**
- 217/217 tests, including old-vs-new equivalence of the attention module (no dropout) and of the training loss/gradients, the absence of
  test-file access in `--tuning` mode, exact generation conditioning for both model types, unchanged parameter counts.
- `ruff check .`, `nbqa ruff .`, `pre-commit run --all-files` pass.
- 10/10 smoke runs exit 0 with finite losses; full-context evaluation scored N − 1 targets on test (4,999,999) and validation (4,499,999).
- Notebook runs end to end against the new `train.py`.
- Step times before/after on a GPU with no other GPU jobs (host shared), and the cause of the TR2o difference down to the XLA runtime level.

**Not verified / not done**
- Whether any fix changes final model quality (no long runs, by design). In particular, the effect of `sinusoidal_scale: unit`, of
  `forget_bias: 1.0`, of independent attention-dropout masks for learned/sinusoidal/hybrid, and of the last 2,000 LR steps is unknown.
- Whether the new generation conditioning explains the paper's "early-step instability" (§V.A): not examined; it stays a hypothesis.
- Single-prompt Self-BLEU continuity: the metric is kept, but its values are not comparable to `tableV-v1` because conditioning changed.
- Why XLA compiles the lean step into a slower runtime schedule (beyond the CUDA-graph / host-callback evidence above), and whether this
  also happens on other GPUs or jaxlib versions.
- The old step's timing was taken from 2 runs per arm; run-to-run spread was ≤ 2.5 % here (vs ±20 % across the earlier Table V runs).
- Nothing pushed; the hyperparameter search was not started. `CLAUDE.md` was updated in place and not committed; `paper.pdf` untouched.
