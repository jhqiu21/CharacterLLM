# v3: bounded hyperparameter search (validation only)

Branch `tuning` (from `arch-fairness-fixes`). Task: `prompts/02_tuning.md`. Every number below is the **full-context BPC on the
validation split** (stride 8, all 4,499,999 targets) of the best-loss checkpoint, from `raw/<run>/val_eval.json`. `aggregate_tuning.py`
regenerates `tuning.csv` and `tables.md` from `raw/`. No run opened the test file (all `--tuning`), and nothing was evaluated on test.

**Summary**
- Part 0: the training step is built on `make_loss_and_metrics` again (`bdb600c`). `pytest`: 217 passed. 300-iteration `--tuning`
  smoke runs of TR2o and LR-2 exit 0, and their validation losses match the v2 smoke runs to 1e-5.
- 22 runs, all finished, **none diverged**: 16 grid + 2 edge extensions per family + 1 label-smoothing-off run per family.
  Every run recorded the clean commit `10b7ecc`.
- **Chosen values:**

  | family | config | learning_rate | dropout (all) | ls_eps | rule |
  |---|---|---|---|---|---|
  | Transformer (RoPE, TR2o size) | `configs/final/TR.yaml` | **1.8e-3** (was 4e-4) | **0.0** (was 0.1) | **0.0** (was 0.05) | edge extension, then tie rule; eps: eps 0 better by 0.041 |
  | LSTM (LR-2 size) | `configs/final/LSTM.yaml` | **3.2e-3** (was 4e-4) | **0.0** (was 0.1) | **0.0** (was 0.05) | edge extension, then tie rule; eps: eps 0 better by 0.037 |
  | LSTM (LR-1 size) | `configs/final/LR-1.yaml` | 3.2e-3 | 0.0 | 0.0 | copied from the LSTM (not tuned at this size) |

- For both families the lowest-BPC lr is the **largest lr tried, even after the edge extension**. The tie rule then picked a
  smaller lr. For the LSTM this was by a margin of 0.0197 against the 0.02 threshold. See §5 before starting the 50k runs.

---

## 1. Protocol

| | |
|---|---|
| base configs | `configs/TR2o.yaml` (RoPE Transformer, 3,169,280 params), `configs/LR-2.yaml` (LSTM hidden 314, 3,177,052 params) |
| per run | `--tuning`, seed 4212, 10,000 iterations, warmup 2,000 (cosine to 0 at iteration 10,000), validation every 500, batch 128 × 128 |
| varied | `training.learning_rate`; dropout d = every dropout parameter (`emb/attn/resid/mlp_dropout`; LSTM `dropout`); `loss.ls_eps` (eps 0 runs only) |
| unchanged | model sizes, architecture, objective (label smoothing ε aside), batch size, sequence length, optimizer, decay mask |
| checkpoint | the iteration with the lowest loss on the fixed 1024 × 128 validation set (`best_loss_all_`); it is iteration 9,999 in 21 of 22 runs (`tune_LSTM_lr4e-4_d0.1`: 9,500) |
| score | full-context BPC of that checkpoint on the whole validation split (4.5 M characters), stride 8 |
| GPU | one A100 80GB, at most 2 of these runs at a time, a run starts only with ≥ 30 GiB free (`scripts/launch_tuning.py`) |

Each run is `train.py --config <base> --seed 4212 --run-name <name> --tuning --set training.epochs=10000 --set scheduler.warmup_iters=2000
--set training.validation_interval=500 --set training.learning_rate=<lr> --set <each dropout key>=<d> [--set loss.ls_eps=0.0]`.
The exact command for each run is in `raw/tuning_status.jsonl`, and each resolved config is in `raw/<run>/config.yaml`.

### Decision rules and how they were made concrete

The rules in Part 2 of the task left four details open. I fixed them in `aggregate_tuning.py` and committed it (`10b7ecc`)
before the first grid run started. The launcher applies the same code that builds the tables.

1. **Edge rule trigger.** The edge rule fires when the *lowest-BPC* grid run is at the smallest or largest grid lr. The two new
   runs use that run's dropout. The trigger is not the run the tie rule picks: the edge rule asks whether the optimum may lie
   outside the grid. The rule runs once. The edge runs are done *before* the label-smoothing run, so the ε comparison uses the final (lr, dropout).
2. **Tie rule.** The tied set is every ε = 0.05 run (grid + edge) whose BPC is **less than 0.02** above the lowest. From it, pick
   the run closest to the current config (lr 4e-4, dropout 0.1). Distance is the number of lr steps plus the number of dropout
   steps, counted as index distance in the sorted list of lr values tried (grid + edge) and in (0.0, 0.1). At equal distance, the lower BPC wins.
3. **Label smoothing.** Choose ε = 0 only if its BPC is at least 0.02 below the ε = 0.05 run at the chosen (lr, dropout). A tie or a worse result keeps 0.05.
4. **Diverged runs** (non-finite loss) score +∞. None occurred.

## 2. Part 0: training step

`train.train` builds its step with `make_train_step(model, loss_and_metrics, tx)` again (commit `bdb600c`). `make_train_loss` and
its tests are kept, and the docstrings say why it is unused.

- `pytest`: **217 passed** in 399 s (`test_output.txt`).
- Smoke: `--tuning`, 300 iterations, validation every 100 (`smoke_status.jsonl`). Both runs exit 0 and all training losses are finite.

  | run | val loss_all @ 0 / 100 / 200 / 299 | full-context val BPC (stride 8) | v2 lean-step smoke |
  |---|---|---|---|
  | TR2o | 4.0132 / 2.1591 / 1.9558 / 1.9247 | 2.7343 | 2.7343 |
  | LR-2 | 3.2883 / 2.8709 / 2.6930 / 2.6618 | 3.8230 | 3.8229 |

  The two training steps compute the same loss and gradients. The remaining ≤ 1e-5 differences are float32 rounding between the two compiled programs.

## 3. Grid (validation BPC, lower is better)

**bold** = chosen, † = lowest BPC, ~ = tied with the lowest (< 0.02), — = not run. No run diverged.

### Transformer (`configs/TR2o.yaml`, ε = 0.05)

| lr | dropout 0.0 | dropout 0.1 (current) |
|---|---|---|
| 4e-4 (current) | 1.7753 | 1.8423 |
| 6e-4 | 1.7452 | 1.8041 |
| 8e-4 | 1.7256 | 1.7832 |
| 1.2e-3 | 1.7019 | 1.7557 |
| 1.8e-3 (edge ext.) | **1.6788** ~ | — |
| 2.7e-3 (edge ext.) | 1.6707 † | — |

- Lowest grid BPC: lr 1.2e-3, d 0.0 (1.7019). This is the high edge, so the edge rule added 1.8e-3 and 2.7e-3 at d 0.0.
- Lowest overall: 2.7e-3 (1.6707). Tied set: 2.7e-3 (distance 6) and 1.8e-3 (+0.0081, distance 5). 1.2e-3 is +0.0312, so not tied.
- **Chosen: lr 1.8e-3, dropout 0.0**, by the tie rule. The chosen value comes from the edge extension.
- Dropout 0.1 is worse than 0.0 at every lr, by 0.054–0.067 BPC. This is the clearest effect in the search.

### LSTM (`configs/LR-2.yaml`, ε = 0.05)

| lr | dropout 0.0 | dropout 0.1 (current) |
|---|---|---|
| 4e-4 (current) | 1.9218 | 1.9003 |
| 8e-4 | 1.8182 | 1.8162 |
| 1.6e-3 | 1.7424 | 1.7526 |
| 3.2e-3 | **1.6874** ~ | 1.7163 |
| 4.8e-3 (edge ext.) | 1.6760 ~ | — |
| 7.2e-3 (edge ext.) | 1.6677 † | — |

- Lowest grid BPC: lr 3.2e-3, d 0.0 (1.6874). This is the high edge, so the edge rule added 4.8e-3 and 7.2e-3 at d 0.0.
- Lowest overall: 7.2e-3 (1.6677). Tied set: 7.2e-3 (distance 6), 4.8e-3 (+0.0083, distance 5) and 3.2e-3 (**+0.0197**, distance 4).
- **Chosen: lr 3.2e-3, dropout 0.0**, by the tie rule. The margin is 0.0003 below the threshold.
- Dropout matters less for the LSTM than for the Transformer, and its sign flips with lr. At lr ≤ 8e-4, 0.1 is better by 0.002–0.022.
  At lr ≥ 1.6e-3, 0.0 is better by 0.010–0.029.

## 4. Label smoothing

One run with `loss.ls_eps=0.0` at each family's chosen (lr, dropout). `LABEL_SMOOTHING` stays `True`; ε = 0 gives the same loss
and gradients as switching it off (tested in v2).

| family | lr | dropout | BPC ε 0.05 | BPC ε 0 | ε 0 − ε 0.05 | val acc ε 0.05 / ε 0 (%) | val ECE ε 0.05 / ε 0 | chosen ε |
|---|---|---|---|---|---|---|---|---|
| Transformer | 1.8e-3 | 0.0 | 1.6788 | 1.6377 | **−0.0411** | 64.45 / 64.47 | 0.0287 / 0.0026 | **0.0** |
| LSTM | 3.2e-3 | 0.0 | 1.6874 | 1.6503 | **−0.0370** | 64.27 / 64.14 | 0.0350 / 0.0039 | **0.0** |

Both differences are clear wins (≥ 0.02), so ε = 0 is chosen for both families. Accuracy barely moves. Label smoothing raises the
cross-entropy that BPC measures and leaves the models under-confident: validation ECE is about 10× higher with ε = 0.05.

## 5. Read this before the 50k runs

These points do not change the rule-based choice, but they matter for how well it transfers to the final runs.

- **The lr optimum was not bracketed.** In both families BPC still falls at the largest lr tried (TR 2.7e-3, LSTM 7.2e-3). The
  edge rule allows one extension. The chosen values are the smaller ones that the tie rule allows.
- **10k-step proxy vs 50k final runs.** 10,000 iterations are 1.9 passes over the 85.5 M training characters; 50,000 are 9.6 passes.
  Every dropout-0.0 run already has a train−validation gap of 0.049–0.060 nats: training loss averaged over the last 500 iterations
  vs validation loss on the fixed set. For dropout 0.1 the gap is −0.010 to +0.026, but those training losses are measured with
  dropout on, so the gaps are not directly comparable. Validation loss on the fixed set had not started to rise in any run: it was
  still falling at iteration 9,999 in 21 runs and flat over the last 500 iterations in `tune_LSTM_lr4e-4_d0.1`. So the search shows
  no overfitting. It cannot show whether dropout 0.0 and these learning rates are still best at 5× the
  training length, or whether the higher lrs stay stable for 50k steps (none diverged in 10k).
- **The tie threshold comes from TR2o.** The 0.02 BPC threshold is TR2o's seed-to-seed std. LR-2's seed std in Table V is 0.0011
  (test, 50k steps). Measured against that noise, LSTM lr 7.2e-3 and 4.8e-3 beat 3.2e-3 clearly. The fixed rule still makes them ties.
- **One seed per setting.** Differences below 0.02 are ties by construction. The dropout effect for the Transformer (≥ 0.054) and
  the label-smoothing effect (≥ 0.037) are well above that threshold.
- **LR-1** (`configs/final/LR-1.yaml`) uses the LSTM values found at the LR-2 size; LR-1 was not tuned itself.
- **Not comparable with Table V.** The splits differ (validation vs test), as do stride (8 vs 1) and length (10k vs 50k).
  As an observation only: at the chosen values with ε = 0, the Transformer is 0.013 BPC ahead of the LSTM (1.6377 vs 1.6503;
  one seed, 10k steps). In Table V (old hyperparameters) the gap was 0.088.

## 6. What went wrong

1. **The session ended during Part 0.** This killed the first TR2o smoke run (at batch 60/69 of its final evaluation) and the
   first `pytest` run (at about 33 %). Both were deleted and rerun from scratch, and the reruns are what is reported. Everything
   after that ran under `nohup setsid`.
2. **The GPU was shared.** From about 16:15, other jobs used the same GPU. First came a `paperacc_launch.py` job from another
   project (3 processes, ~49 GB). Later came a `train_agent.py` job (~21 GB) and other short-lived processes. The PIDs seen during
   each run are in `raw/tuning_status.jsonl`. With less than 30 GiB free, the launcher ran one job at a time and waited twice:
   17:08–17:41 and 18:54–19:28. The search took 15:17–20:15. No throughput is reported (the task does not use it).
3. **One evaluation OOM.** `tune_TR_lr6e-4_d0.1` started with 32.6 GB free. It finished training (best-loss checkpoint at
   iteration 9,999 saved). Then its full-context evaluation hit `RESOURCE_EXHAUSTED` (a 4.0 GiB allocation) because another
   process had taken memory in the meantime (`raw/tune_TR_lr6e-4_d0.1/oom_attempt_log.txt`). I did not retrain it. I scored the
   saved checkpoint with `train.py --config configs/TR2o.yaml --seed 4212 --run-name tune_TR_lr6e-4_d0.1 --tuning --eval-only`
   (same commit `10b7ecc`, the run's own saved config): BPC 1.8041 (`eval_only_log.txt`). This is the same computation the
   original run would have done. The launcher had recorded the run as failed and stopped, because the TR edge rule needed the
   full grid. After the re-evaluation I restarted it (`raw/launch_tuning_log.txt`).
4. Nothing else failed: no divergence, no other OOM, and every run scored all 4,499,999 validation targets.

## 7. Files

| path | content |
|---|---|
| `configs/final/TR.yaml`, `LSTM.yaml`, `LR-1.yaml` | chosen values, 50,000 epochs, warmup 2,000. `model_name` `TR-final` / `LSTM-final` / `LR-1-final`, so default run names do not collide with the Table V runs. Apart from `model_name` and `training.epochs`, `TR.yaml` and `LSTM.yaml` resolve to exactly the config of `tune_TR_lr1.8e-3_d0.0_ls0` and `tune_LSTM_lr3.2e-3_d0.0_ls0` (checked) |
| `results/v3_tuning/raw/<run>/` | `val_eval.json`, `config.yaml`, `val_history.json` (validation curve on the fixed set) for each of the 22 runs |
| `results/v3_tuning/raw/tuning_status.jsonl`, `launch_tuning_log.txt` | launcher records: command, start/end, outcome, free GPU memory at start, foreign GPU PIDs |
| `results/v3_tuning/tuning.csv` | one row per run (full-context val BPC, loss, perplexity, acc, ECE, rare/common acc, best step, fixed-set val loss, commit) |
| `results/v3_tuning/tables.md` | generated tables and decisions |
| `results/v3_tuning/aggregate_tuning.py` | grid, rules, tables: `python results/v3_tuning/aggregate_tuning.py [--sync] [--next]` |
| `scripts/launch_tuning.py` | runs whatever the rules call for next (≤ 2 jobs, ≥ 30 GiB free) |
| `results/v3_tuning/test_output.txt`, `smoke_status.jsonl` | Part 0 checks |

**Not done:** no 50k-step run (waiting for review). `transformer.ipynb` still builds its step from `make_train_loss`; the task named
only `train.train`. `CLAUDE.md` (untracked) was updated in place and not committed. `paper.pdf` and `prompts/` are untouched.
