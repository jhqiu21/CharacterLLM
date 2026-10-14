Dropout ablation, Transformer (RoPE): test set, full-context W=128 stride 1, mean ± std over seeds 42-44. Gap: final validation loss minus the mean training loss of the last 500 iterations (measured with dropout on).

| Dropout | BPC | Acc_all (%) | ECE | Acc_last (%) | val − train gap (nats) | best-loss step |
|---|---|---|---|---|---|---|
| 0.0 (chosen) | 1.4532 ± 0.0055 | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 68.57 ± 0.05 | 0.0672 ± 0.0025 | 49,999 |
| 0.1 | 1.5003 ± 0.0017 | 67.51 ± 0.05 | 0.0125 ± 0.0001 | 67.54 ± 0.15 | 0.0115 ± 0.0019 | 49,000 |

Seed-matched BPC difference (0.1 - 0.0): +0.0471 ± 0.0062.
