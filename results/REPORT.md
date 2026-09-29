# Table V re-run: dropout-RNG fix, weight-decay mask, full-context evaluation

All numbers below are produced by `python results/aggregate.py` from `results/raw/<run>/eval.json`
(copies of `runs/<run>/eval.json`); the generated tables are `table_v.{csv,md,tex}`, `ablation.md`,
`sanity_gate.md`, `throughput.md`, `claims.md` and `per_run.csv`. Mean ± sample std (ddof = 1) over seeds
4212, 4213, 4214. Hardware: one NVIDIA A100 80GB PCIe, **shared with other jobs for part of the time** (§5).

**Summary**
- All 18 runs (4 models × 3 seeds + 2 ablation arms × 3 seeds) finished (exit code 0, no NaN); 68/68 tests pass.
- Sanity gate **passes**: TR2o-legacy, evaluated the paper's way (legacy32), gives BPC 1.8457 ± 0.0021 vs the paper's 1.8468.
  The other models' legacy32 numbers also land within 0.011 BPC of the paper.
- Evaluation context matters far more than either training fix: TR2o's test BPC is 1.843 (legacy32) → 1.684 (chunked128) → **1.613** (full context).
- Claims (a) TR2o < LR-2 in BPC, (d) RoPE < relative, and (f) Transformer faster per step still hold (a and d with larger margins).
  Claims (b) LSTMs better calibrated and (c) TR2o lowest Self-BLEU-4 **do not hold**; the paper's Transformer ECE and Self-BLEU values are not reproduced by any evaluation protocol I tried.
- The dropout-RNG fix has **no detectable effect** at dropout 0.1; the weight-decay mask gives a small, seed-consistent gain (−0.003 to −0.008 BPC).

---

## 1. Code changes

Branch `fix-dropout-rng-full-context-eval` (from `main`; 10 commits, not pushed). One line per file:

| file | change |
|---|---|
| `train.py` (new) | Canonical entry point replacing the notebook loop. Changes vs the notebook: dropout key split every step (`next_dropout_key`; `--legacy-dropout-rng` reuses one key), init/dropout keys from `split(PRNGKey(seed))`, batches from `np.random.default_rng(seed)`, `make_loss_and_metrics(config)` factory (no global config), `optimizer:` section + `decay_mask` (AdamW `mask=`: leaves with ndim ≥ 2, never `rel_embed`), one fixed 1024 × T validation set (seed 0), per-step `block_until_ready` timing, three test-evaluation modes written to `eval.json`, outputs in `runs/<run_name>/` and `checkpoints/<run_name>/{best,stage}`, GPU from the environment. Extra flags beyond the spec: `--fc-stride` (smoke tests) and `--eval-only` (re-evaluate a finished run; not used for any reported number). |
| `utils/full_context_eval.py` (new) | Sliding-window evaluation (W = `max_len`, stride S ∈ [1, W], rejected otherwise); every target 1..N−1 scored exactly once; windows built on device by index arithmetic; fixed batch shape with masked padding (one compile); ECE from accumulated per-bin counts / confidence sums / correct counts; per-class counts for rare/common accuracy by training-split frequency. |
| `models/LSTM.py` (new) | Ported verbatim from `origin/LSTM-baseline`; only the signature changed to `__call__(idx, *, deterministic)` with `train = not deterministic`. |
| `utils/test.py` | `test_checkpoint(..., sequence_length=32, batch_size=1024)` (defaults = old behaviour); `get_batch_test` stacks with `np.stack` and copies once instead of `jnp.stack` over 1024 slices (identical values; the old code took ~4 s per batch, ~10 min per legacy32 pass). |
| `utils/logger.py` | `print_metrics` prints `self.ece[-1]` (was `self.ece[0]`). |
| `environment.yml` | Adds `nltk`. |
| `configs/TR2o.yaml`, `TR8o.yaml`, `LR-1.yaml`, `LR-2.yaml`, `TR2o-nomask.yaml` (new) | Part 3 settings. `TR2o-nomask` = TR2o with `decay_mask: false`; used by the TR2o-legacy (`--legacy-dropout-rng`) and TR2o-rngfix arms. |
| `configs/baseline.yaml` | Adds `model.type`, the `optimizer:` section (`decay_mask: true`) and full-context settings; other values unchanged. |
| `transformer.ipynb` | Rebuilt on `train.py` (data, model, loss factory, optimizer with mask, train step, seeded batches, fixed 128-length val set, `next_dropout_key` in the loop, `train.evaluate`); no `CUDA_VISIBLE_DEVICES`; outputs cleared. |
| `tests/` (new) | pytest suite below; `conftest.py` sets `JAX_PLATFORMS=cpu` so tests never touch the GPU. |
| `pyproject.toml` | `[tool.pytest.ini_options] testpaths = ["tests"]`. |
| `scripts/launch_all.py` (new) | Queues the 18 runs, one run per free GPU; skips finished runs; `--smoke`, `--allow-shared`; writes `runs/launch_status.jsonl`. |
| `scripts/gpu_sampler.py` (new) | Logs GPU compute processes every 5 s, used to decide whether a run's timing window was shared. |
| `results/aggregate.py` (new) | Rebuilds every table from `results/raw/`. `results/checks/*.py`: the Self-BLEU and ECE side checks of §4. |
| `.gitignore` | Un-ignores `results/**/*.json`. |

**Tests** (`pytest`, full output in `results/test_output.txt`): **68 passed** in 133 s.

| file | what it checks | tests |
|---|---|---|
| `test_dropout_rng.py` | consecutive fixed-loop step keys give different training-mode outputs (Transformer and LSTM); `--legacy-dropout-rng` path gives identical outputs (and dropout is active); 200 fixed-loop keys never repeat; `train_step` output depends on the step key | 6 |
| `test_causality.py` | changing tokens after t leaves logits ≤ t unchanged, for all six `pos_encoding_type`s and the LSTM, t ∈ {0, 5, 14} (`rel_embed` randomised) | 21 |
| `test_full_context_eval.py` | exactly N−1 targets scored for strides 1 and 7 (two batch sizes with padded last batches; both model types); scored indices are exactly 1..N−1; stride 1 matches a brute-force per-target loop in total NLL, accuracy, ECE and rare/common accuracy; stride 7 matches a brute force of its window plan; strides outside [1, W] rejected | 17 |
| `test_decay_mask.py` | mask is True exactly for ndim ≥ 2 leaves except `rel_embed`, for every PE type and the LSTM; with zero gradients AdamW moves exactly the masked leaves (and all leaves with `decay_mask: false`) | 12 |
| `test_param_counts.py` | TR2o 3,169,280, LR-1 2,115,072, LR-2 3,177,052 (paper values); TR8o 3,173,360 with the current code (paper: 3,238,640 — the 65,280 = 255 × 256 gap is not reproduced, as expected); the five run configs | 9 |
| `test_train_pieces.py` | fixed val set deterministic and seed-independent; batches reproducible from the seed; vectorised `encode` equals the notebook's | 3 |

Lint: `ruff check .` and `nbqa ruff .` pass; `pre-commit run --all-files` (CI's pinned ruff v0.6.4 hooks) passes.
Smoke tests: all six run variants ran 300 iterations (val every 100, full-context stride 64) end to end including evaluation (`results/raw/smoke_status.jsonl`, all exit 0). Notebook: a patched copy (epochs 20, val every 10, warmup 2, stride 64) executed start to finish via nbclient, including the evaluation cell.

## 2. Sanity gate — PASS

TR2o-legacy (fixed dropout key + decay on all parameters, the paper's training) in `legacy32` mode:

| run | legacy32 BPC |
|---|---|
| TR2o-legacy_s4212 | 1.8465 |
| TR2o-legacy_s4213 | 1.8433 |
| TR2o-legacy_s4214 | 1.8473 |
| **mean** | **1.8457 ± 0.0021** (paper TR2o: 1.8468; \|diff\| = 0.0011 ≤ 0.05) |

The rest of the old evaluation reproduces too (legacy32, new runs vs paper Table V; rare/common here are the per-batch definition the old code used):

| ID | BPC new / paper | Acc_all | Acc_last@31 | Acc_rare* | Acc_comm* |
|---|---|---|---|---|---|
| TR2o-legacy | 1.8457 / 1.8468 | 60.31 / 60.28 | 64.17 / 64.08 | 39.77 / 39.27 | 72.13 / 72.10 |
| TR2o (fixed) | 1.8427 / 1.8468 | 60.37 / 60.28 | 64.28 / 64.08 | 39.60 / 39.27 | 72.23 / 72.10 |
| TR8o | 1.8886 / 1.8993 | 59.72 / 59.50 | 63.17 / 62.94 | 37.59 / 37.06 | 71.96 / 71.77 |
| LR-1 | 1.9200 / 1.9198 | 59.01 / 59.02 | 62.54 / 62.53 | 37.63 / 37.60 | 71.35 / 71.29 |
| LR-2 | 1.8855 / 1.8896 | 59.72 / 59.67 | 63.38 / 63.45 | 38.14 / 38.22 | 71.87 / 71.82 |

## 3. New Table V, ablation, throughput

### Table V (replaces the paper's)

Test set, checkpoint with minimum loss on the fixed 1024 × 128 validation set. BPC, PPL, Acc_all, ECE, Acc_rare, Acc_comm:
`full_context` (W = 128, **stride 1**: every one of the 4,999,999 targets scored once, each after the first window with a full 128-character context).
Acc_last: `chunked128` (position 127 of non-overlapping 128-character blocks). Rare = {q, j, x, z, k}, common = {n, a, t, e, space}
(bottom / top 20 % of the 27 characters by training-split frequency). s/step: solo runs only (see Throughput).

| ID | BPC ↓ | PPL ↓ | Acc_all (%) | ECE ↓ | Acc_rare (%) | Acc_comm (%) | Acc_last (%) | Self-BLEU-4 ↓ | #Params | s/step |
|---|---|---|---|---|---|---|---|---|---|---|
| TR2o | **1.6132 ± 0.0174** | 3.0595 ± 0.0371 | **65.51 ± 0.42** | **0.0098 ± 0.0007** | **45.89 ± 1.01** | **77.85 ± 0.22** | **65.42 ± 0.41** | 0.1577 ± 0.0200 | 3,169,280 | 0.0207 ± 0.0032 (n=2) |
| TR8o | 1.7063 ± 0.0054 | 3.2633 ± 0.0122 | 63.79 ± 0.14 | 0.0318 ± 0.0002 | 41.92 ± 0.27 | 77.16 ± 0.08 | 63.61 ± 0.20 | **0.1165 ± 0.0191** | 3,173,360 | 0.0185 ± 0.0016 (n=3) |
| LR-1 | 1.7460 ± 0.0018 | 3.3544 ± 0.0042 | 62.76 ± 0.05 | 0.0254 ± 0.0003 | 41.42 ± 0.24 | 76.35 ± 0.08 | 62.55 ± 0.07 | 0.1482 ± 0.0098 | 2,115,072 | 0.0517 ± 0.0002 (n=2) |
| LR-2 | 1.7009 ± 0.0011 | 3.2510 ± 0.0025 | 63.75 ± 0.03 | 0.0262 ± 0.0000 | 42.26 ± 0.13 | 76.97 ± 0.03 | 63.49 ± 0.12 | 0.1697 ± 0.0151 | 3,177,052 | 0.0654 ± 0.0007 (n=2) |

LaTeX (booktabs): `results/table_v.tex`; CSV: `results/table_v.csv`.

### Ablation (TR2o config, 3 seeds per arm) — full tables in `results/ablation.md`

| arm | dropout key | decay | legacy32 BPC | chunked128 BPC | chunked128 Acc_last@127 | full_context BPC | full_context Acc_all |
|---|---|---|---|---|---|---|---|
| TR2o-legacy | fixed (bug) | all params | 1.8457 ± 0.0021 | 1.6924 ± 0.0072 | 65.07 ± 0.47 | 1.6278 ± 0.0160 | 65.19 ± 0.39 |
| TR2o-rngfix | split per step | all params | 1.8460 ± 0.0029 | 1.6891 ± 0.0058 | 65.28 ± 0.32 | 1.6208 ± 0.0155 | 65.33 ± 0.37 |
| TR2o | split per step | mask | 1.8427 ± 0.0031 | 1.6835 ± 0.0068 | 65.42 ± 0.41 | 1.6132 ± 0.0174 | 65.51 ± 0.42 |

Seed-matched differences (mean ± std over the 3 seed pairs):

| difference | legacy32 | chunked128 | full_context |
|---|---|---|---|
| RNG fix: rngfix − legacy | +0.0003 ± 0.0012 | −0.0033 ± 0.0080 | −0.0070 ± 0.0151 (per seed −0.024, +0.003, +0.001) |
| decay mask: TR2o − rngfix | −0.0034 ± 0.0003 | −0.0056 ± 0.0010 | −0.0076 ± 0.0020 (all seeds negative) |
| both: TR2o − legacy | −0.0030 ± 0.0015 | −0.0089 ± 0.0084 | −0.0146 ± 0.0160 |

### Throughput (B = 128, T = 128; `results/throughput.md`)

s/step = median wall time of one iteration (batch sampling + jitted step + `block_until_ready`) over iterations 1000–1999
excluding validation iterations; tokens/s = 16,384 / (s/step). Only runs whose timing window had the GPU to themselves
(checked with `scripts/gpu_sampler.py`, or by hand for the first three runs) are used; per-run evidence is in `throughput.md`.

| model | solo runs | s/step | tokens/s | paper s/step (8×A5000) |
|---|---|---|---|---|
| TR2o | 2 (s4213, s4214) | 0.0207 ± 0.0032 | 8.0 × 10⁵ | 0.027 |
| TR8o | 3 | 0.0185 ± 0.0016 | 8.9 × 10⁵ | 0.023 |
| LR-1 | 2 (s4212, s4214) | 0.0517 ± 0.0002 | 3.2 × 10⁵ | 0.044 |
| LR-2 | 2 (s4212, s4214) | 0.0654 ± 0.0007 | 2.5 × 10⁵ | 0.069 |

GPU: NVIDIA A100 80GB PCIe. Treat these as indicative: identical-architecture solo runs (the five solo TR2o-config runs across
the three TR2o arms) range from 0.0165 to 0.0232 s/step, i.e. ±20 %; the cause was not investigated (the host is shared with other users,
and host-side load is invisible to the GPU sampler).

## 4. The paper's claims

**(a) TR2o beats the parameter-matched LR-2 on BPC — holds, and the margin doubles with full context.**
Full context: TR2o − LR-2 = −0.088 ± 0.016 BPC (per seed −0.095, −0.069, −0.100). Legacy32: −0.043 ± 0.003 (paper: −0.043).
TR2o is also ahead on every other Table V accuracy column (Acc_all +1.8, Acc_rare +3.6, Acc_comm +0.9, Acc_last@127 +1.9 points).

**(b) The LSTMs are better calibrated (lower ECE) — does not hold for TR2o; holds narrowly for TR8o.**
Test ECE (full context): TR2o 0.0098 ± 0.0007 vs LR-1 0.0254 ± 0.0003 and LR-2 0.0262 ± 0.0000 — TR2o is *lower* by 0.016 in every seed.
TR8o 0.0318 ± 0.0002 is higher than the LSTMs by 0.006 (every seed). The paper's Transformer ECEs (0.083, 0.132) are not reproduced by
any protocol I tried: test ECE (above), validation ECE on the fixed 1024 × 128 set at the checkpoint (TR2o 0.010, TR8o 0.030), or the old
notebook's protocol of random 1024 × 32 validation batches (TR2o-legacy 0.013, TR8o 0.027, LR-2 0.021; `results/checks/ece_check.py`).
The paper's LSTM ECEs (0.023, 0.026) do match our trained LSTMs (validation 0.022). One possible explanation, **not verified**: the old
`Logger.print_metrics` printed `self.ece[0]`, i.e. the iteration-0 (untrained) ECE, and our iteration-0 validation ECE is 0.10–0.18 for the
Transformers vs 0.002–0.055 for the LSTMs (`claims.md`, last table), so reading the Transformer values off that log line would give numbers
in the paper's range.

**(c) TR2o has the lowest Self-BLEU-4 — does not hold.**
TR8o is lowest (0.117 ± 0.019); TR2o is 0.158 ± 0.020, higher than TR8o in every seed (+0.041 ± 0.011) and indistinguishable from the
LSTMs (TR2o − LR-1 = +0.010 ± 0.013, TR2o − LR-2 = −0.012 ± 0.013). All values are 0.12–0.17 against the paper's 0.016–0.154; e.g. TR2o-legacy
seed 4212 (the paper's configuration) gives 0.192 vs the paper's 0.0159. Checks: the value is identical under nltk 3.8.1, 3.9.1, 3.9.2
and 3.10.3 on the same samples, and 20 real 1000-character test passages score 0.048–0.049, so the paper's TR2o value is below that of
natural text. The metric is noisy (20 samples of one prompt; seed std 0.01–0.02) and the cause of the paper's value is unknown.

**(d) RoPE (TR2o) beats relative encoding (TR8o) — holds, but is confounded.**
Full context: −0.093 ± 0.020 BPC (per seed −0.107, −0.070, −0.102); legacy32: −0.046 ± 0.002 (paper −0.053). TR2o also wins on every
accuracy column and on ECE. TR2o and TR8o differ in learning rate (4e-4 vs 3e-4) and dropout (0.1 vs 0.0) as well as position encoding,
so this does not isolate the encoding. TR8o has 3,173,360 parameters with the current code, not the paper's 3,238,640.

**(e) Effect of the dropout-RNG fix and of evaluation context.**
- *Dropout-RNG fix*: no detectable effect at dropout 0.1 — +0.0003 ± 0.0012 (legacy32), −0.003 ± 0.008 (chunked128), −0.007 ± 0.015
  (full context, driven by one seed; the full-context seed std of each arm is ~0.016). The fixed-key bug therefore does not explain the
  Transformer's advantage over the LSTM (whose branch already split keys correctly).
- *Weight-decay mask*: small but consistent: −0.0034 ± 0.0003 / −0.0056 ± 0.0010 / −0.0076 ± 0.0020 BPC, negative in all 9 seed-mode pairs.
- *Evaluation context* is the dominant effect: going from legacy32 to full context lowers BPC by 0.229 (TR2o), 0.182 (TR8o),
  0.174 (LR-1), 0.185 (LR-2) — more than 10× the combined effect of both training fixes (≤ 0.015 BPC). Model ranking is unchanged (TR2o < LR-2 < TR8o < LR-1; LR-2 vs TR8o is
  close: TR8o − LR-2 = +0.003 ± 0.002 legacy32, +0.005 ± 0.006 full context), but TR2o's lead grows because it gains most from long context.
  Moving Acc_last from position 31 to 127 gains TR2o +1.1 points and TR8o +0.4, the LSTMs ≤ 0.1 (62.54 → 62.55, 63.38 → 63.49).

**(f) The Transformer is 1.6–2.6× faster per step — holds directionally; the ratio is larger on this GPU.**
Solo runs: LR-1 / TR2o = 2.50×, LR-2 / TR2o = 3.17×, LR-1 / TR8o = 2.80×, LR-2 / TR8o = 3.54× (paper: 1.6–2.6×). Indicative only
(different hardware, 2–3 solo runs per model, ±20 % run-to-run variation).

## 5. Deviations, failures, and what was / was not verified

**Deviations from the task**
1. Branch name: `fix-dropout-rng-full-context-eval` instead of `fix/dropout-rng-full-context-eval` — a local branch named `fix` already
   exists, so git cannot create `fix/...`; I did not rename or delete it.
2. GPU sharing: the only GPU was shared with other jobs during parts of the queue: another project's `scripts/rgo_reeval.py`
   (5 processes, ~33–41 GB, 100 % utilisation) until shortly before 13:07, `scripts/rgo_diag.py` 13:09–14:51, and intermittent
   `python -`, `/tests/verify.py` and `pt_main_thread` processes between 18:32 and 21:07 (`runs/gpu_usage.jsonl`). I ran strictly one of my runs at a time
   (`--allow-shared`: start only if ≥ 30 GB free). My peak device memory was 10.1 GiB; no OOM occurred. Contended runs are excluded from
   the throughput numbers (runs TR2o-legacy_s4212/s4214, TR2o-rngfix_s4212, TR2o_s4212, LR-1_s4213, LR-2_s4213).
3. Added `--fc-stride` and `--eval-only` CLI flags; when `--epochs` ≤ warmup (smoke tests only) the warmup is shrunk to epochs/10 because the
   notebook's schedule formula fails otherwise; recorded in each run's `config.yaml` overrides. No reported run uses any override.
4. `get_batch_test` performance change (§1) — same values.
5. The TR2o-legacy arm *approximates* the paper's training: it also uses the new fixed 1024 × 128 validation set for checkpoint selection
   and seeded batch sampling.
6. Full-context stride: 1 for every model (1–5 min per checkpoint at batch 8192), so no stride compromise was needed.
7. Rare/common accuracy in `legacy32` / `chunked128` still uses the old per-batch definition (`token_frequency_analysis`, unchanged);
   only `full_context` (used in Table V) uses training-split frequency.
8. Environment: Python 3.10.21, JAX 0.6.2 (last release supporting Python 3.10) with the CUDA 12 plugin, flax 0.10.7, optax 0.2.8,
   nltk 3.10.3 (`results/env.txt`). `flax.training.checkpoints` works on this flax (orbax backend), so no pin was needed.
9. `git_commit` recorded by runs: `b5b308e` (clean or `-dirty`) or `357807f`. The `-dirty` flag comes only from an uncommitted `.gitignore`
   edit; `git diff b5b308e 357807f -- train.py utils models configs` is empty, so all 18 runs used identical code.
10. The other best checkpoints (`best_acc_total_`, `best_acc_last_`) were also evaluated in legacy32 and chunked128 (in each `eval.json`).

**Existing behaviour kept although it looks wrong (not in the fix list; not changed)**
- Stage checkpoints are saved only at the last iteration: the stage condition is checked only on validation iterations (every 500),
  and `(it + 1) % 10000 == 0` never holds on those.
- `warmup_cosine_decay_schedule(decay_steps = epochs − warmup)`: optax's `decay_steps` is the *total* schedule length, so the learning
  rate reaches 0 at step 48,000 and the last 2,000 steps change nothing (checked: lr(47,999) = lr(49,999) = 0). Three runs report best
  step 49,999 — ties with the frozen step-48,000 parameters.
- The docstring mentions weight tying; the Transformer has an untied head. Generation left-pads prompts with token 0 (`'a'`).
- The positional-encoding factory's `relative` branch (unused) still passes the wrong arguments.

**Verified**
- All 68 tests; lint (ruff, nbqa, pre-commit); smoke runs of all six variants; notebook execution at 20 iterations.
- Every full-context evaluation scored exactly 4,999,999 = N − 1 targets (asserted in `train.py` for all 18 runs); stride-1 results equal
  a brute-force per-target loop on tiny models (tests).
- Sanity gate; 18/18 runs exit 0 with finite losses; parameter counts; code identity across run commits.
- Self-BLEU independent of nltk version (4 versions); ECE under the old validation protocol (checks in `results/checks/`).

**Not verified / not done**
- Why the paper's Transformer ECE and all its Self-BLEU values differ from ours (hypothesis for ECE above, unverified).
- The source of the ±20 % run-to-run throughput variation on solo runs; no dedicated timing runs were made.
- The paper's TR8o parameter count (3,238,640) — not reproducible with the current code, as expected.
- `README.md` was not updated, and the untracked `CLAUDE.md` (which says there is no train script and no `models/LSTM.py` on this branch)
  is now stale; neither was committed or edited, nor was `paper.pdf`.
- Nothing was pushed.
