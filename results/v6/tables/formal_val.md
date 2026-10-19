Formal runs (50,000 iterations, 819,200,000 training characters): whole validation split, W = 512, stride 128, at the selected checkpoint.

| seed | ALiBi-3.2M val Acc_all (%) | ALiBi-3.2M val BPC | step | ALiBi-14.2M val Acc_all (%) | ALiBi-14.2M val BPC | step | Δ Acc (pp) | Δ BPC |
|---|---|---|---|---|---|---|---|---|
| 42 | 70.06 | 1.3950 | 48000 | 73.05 | 1.2574 | 49000 | +2.99 | -0.1377 |
| 43 | 70.12 | 1.3915 | 49999 | 73.03 | 1.2580 | 49999 | +2.90 | -0.1335 |
| 44 | 70.09 | 1.3923 | 49000 | 73.05 | 1.2573 | 49500 | +2.96 | -0.1350 |
| mean ± std | 70.09 ± 0.03 | 1.3930 ± 0.0018 |  | 73.04 ± 0.01 | 1.2575 ± 0.0004 |  | +2.95 ± 0.04 | -0.1354 ± 0.0021 |

Rule for seeds 43 / 44 (fixed before the formal runs): seed-42 AL-14M − AL-3M validation Acc_all ≥ 0.3 pp and lower validation BPC. Seed 42: met.
