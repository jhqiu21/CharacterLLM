# v4 status

Branch `final`. Nothing pushed. Findings and deviations: `REPORT.md`.

## Timeline

| time (2026-10-14) | event |
|---|---|
| 14:39 | Part 0 committed (`ae0d7a8`, `5f7302e`): NoPE, eval-time position interpolation, tests (290 passed, lint clean) |
| 14:41 | `scripts/launch_v4.py` (`7a46fbc`) started on GPUs 2 and 3 (both idle, no other process): NoPE runs, PI evals, lr bracket, NoPE evals |
| 14:59 | Part 3 code clean-up committed (`3197e6a`) while the queue ran (jobs started after this record `3197e6a`; REPORT §6 item 9) |
| 15:11 | NoPE s42, s43 done (1.6441, 1.6044 test BPC) |
| 15:24 | all 6 PI evaluations and the RoPE + PI attention analysis done |
| 15:44 | Transformer bracket runs done: bracketed |
| 16:14 | LSTM bracket runs done (16:07): bracketed; NoPE s44 and all NoPE evaluations done; queue empty, 21/21 jobs complete |

## Part 1b decision (lr bracket)

Rule fixed before the runs: a new lr more than 0.02 BPC better than the 3090 rerun of the lowest-BPC lr is a stop condition.
**Not triggered.** Transformer: +0.0141 (4.05e-3), −0.0155 (6.075e-3); LSTM-3.2M: +0.0042 (1.08e-2), +0.0000 (1.62e-2).
Both optima are bracketed; no 50k run was added (`tables/lr_bracket.md`).

## Done

- Part 0: NoPE (`pos_encoding_type: none`) and `pos_scale` / `--interpolate`, with tests.
- Part 1: 3 NoPE runs, 6 bracket runs, all evaluations (21 jobs, none failed).
- Part 2: `raw/`, `aggregate_v4.py`, `tables/`, `figures/`, `REPORT.md`, `claims.md` + `check_claims.py`.
- Part 3 (README rewrite pending): `results/README.md`, code clean-up, `environment.yml` (verified: fresh env, 290 passed),
  `scripts/prepare_data.py` (verified md5s), dataset untracked, notebook re-checked, local `CLAUDE.md` updated.

## Not done / open questions for the user

- **`report/` was missing from this repository.** The report source is the separate Overleaf-connected repository
  `/home/jinhang/charllm_paper/` (user instruction: pull first, commit there locally, never push; no PDF copied into this repository).
- **Dataset in git history.** `data/text8_*.txt` (95 MB) are untracked now but remain in earlier commits. Purging them needs a
  history rewrite (e.g. `git filter-repo --path data/ --invert-paths`) when the new repository is split off.
- `results/v3/raw/*/attention_W*/example.npz` (13 MB) stay tracked because the heatmap figures and `results/v3/REPORT.md` use them.
- NoPE and PI use RoPE-tuned hyperparameters; PI was evaluated without fine-tuning (as specified).
