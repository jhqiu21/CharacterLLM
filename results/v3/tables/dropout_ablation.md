Dropout ablation (TR, RoPE): test set, full-context W=128 stride 1. Gap: final fixed-set validation loss minus the mean training loss of the last 500 iterations (training loss measured with dropout on).

| dropout | seeds | BPC | Acc_all (%) | ECE | Acc_last@128 (%) | val - train gap (nats) | best-loss step |
|---|---|---|---|---|---|---|---|
| 0.0 (chosen) | 3 | 1.4532 ± 0.0055 | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 68.57 ± 0.05 | 0.0672 ± 0.0025 | 49,999 |
| 0.1 | 3 | 1.5003 ± 0.0017 | 67.51 ± 0.05 | 0.0125 ± 0.0001 | 67.54 ± 0.15 | 0.0115 ± 0.0019 | 49,000 |
