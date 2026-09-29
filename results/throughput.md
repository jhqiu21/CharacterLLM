# Throughput (B=128, T=128)

s/step = median wall time of one training iteration (batch sampling + jitted step + block_until_ready) over iterations 1000-1999, excluding validation iterations; tokens/s = B*T / (s/step). 'solo' = no other process on the GPU during that timing window (scripts/gpu_sampler.py log, or hand-recorded evidence for the first runs; see the per-run list below).

| ID | GPU | solo runs | s/step (solo) | tokens/s (solo) | all runs | s/step (all) | tokens/s (all) | runs that shared the GPU | paper s/step (8xA5000) |
|---|---|---|---|---|---|---|---|---|---|
| TR2o | NVIDIA A100 80GB PCIe | 2 | 0.0207 ± 0.0032 | 8.03e+05 | 3 | 0.0230 ± 0.0046 | 7.33e+05 | TR2o_s4212 | 0.027 |
| TR8o | NVIDIA A100 80GB PCIe | 3 | 0.0185 ± 0.0016 | 8.93e+05 | 3 | 0.0185 ± 0.0016 | 8.93e+05 | - | 0.023 |
| LR-1 | NVIDIA A100 80GB PCIe | 2 | 0.0517 ± 0.0002 | 3.17e+05 | 3 | 0.0520 ± 0.0005 | 3.15e+05 | LR-1_s4213 | 0.044 |
| LR-2 | NVIDIA A100 80GB PCIe | 2 | 0.0654 ± 0.0007 | 2.5e+05 | 3 | 0.0646 ± 0.0015 | 2.54e+05 | LR-2_s4213 | 0.069 |
| TR2o-legacy | NVIDIA A100 80GB PCIe | 1 | 0.0165 | 9.9e+05 | 3 | 0.0245 ± 0.0085 | 7.26e+05 | TR2o-legacy_s4212, TR2o-legacy_s4214 |  |
| TR2o-rngfix | NVIDIA A100 80GB PCIe | 2 | 0.0232 ± 0.0000 | 7.06e+05 | 3 | 0.0254 ± 0.0038 | 6.53e+05 | TR2o-rngfix_s4212 |  |

## Per run

| run | s/step | shared GPU in timing window | evidence |
|---|---|---|---|
| LR-1_s4212 | 0.0515 | False | gpu_sampler: 28 samples in window, foreign: none |
| LR-1_s4213 | 0.0525 | True | gpu_sampler: 30 samples in window, foreign: python /tests/verify.py |
| LR-1_s4214 | 0.0519 | False | gpu_sampler: 28 samples in window, foreign: none |
| LR-2_s4212 | 0.0659 | False | gpu_sampler: 34 samples in window, foreign: none |
| LR-2_s4213 | 0.0630 | True | gpu_sampler: 34 samples in window, foreign: [pt_main_thread]; python /tests/verify.py |
| LR-2_s4214 | 0.0649 | False | gpu_sampler: 34 samples in window, foreign: none |
| TR2o-legacy_s4212 | 0.0334 | True | manual: another project's scripts/rgo_reeval.py (5 processes, ~33-41 GB, 100% util) ran for the whole run; the launcher saw them at every poll |
| TR2o-legacy_s4213 | 0.0165 | False | manual: config.yaml written 13:07:35, iteration 2000 reached 65.4 s after the training loop started, so the window ended by ~13:09:10 (allowing <=30 s setup); rgo_reeval.py had exited; the next foreign job (scripts/rgo_diag.py, 3 processes) started 13:09:37 (ps lstart); measured 0.0165 s/step vs 0.0334 under contention |
| TR2o-legacy_s4214 | 0.0235 | True | manual: started 13:36:05 while scripts/rgo_diag.py (started 13:09:37) was running |
| TR2o-rngfix_s4212 | 0.0299 | True | gpu_sampler: 26 samples in window, foreign: python scripts/rgo_diag.py --out results/router_gonogo_20260928T014247Z --router F1m=runs/lr2e-7/rou |
| TR2o-rngfix_s4213 | 0.0232 | False | gpu_sampler: 22 samples in window, foreign: none |
| TR2o-rngfix_s4214 | 0.0232 | False | gpu_sampler: 22 samples in window, foreign: none |
| TR2o_s4212 | 0.0276 | True | gpu_sampler: 24 samples in window, foreign: python scripts/rgo_diag.py --out results/router_gonogo_20260928T014247Z --router F1m=runs/lr2e-7/rou |
| TR2o_s4213 | 0.0184 | False | gpu_sampler: 21 samples in window, foreign: none |
| TR2o_s4214 | 0.0229 | False | gpu_sampler: 22 samples in window, foreign: none |
| TR8o_s4212 | 0.0168 | False | gpu_sampler: 17 samples in window, foreign: none |
| TR8o_s4213 | 0.0185 | False | gpu_sampler: 17 samples in window, foreign: none |
| TR8o_s4214 | 0.0201 | False | gpu_sampler: 19 samples in window, foreign: none |
