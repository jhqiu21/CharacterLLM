Table V, tie-rule lr (block 1): test set, checkpoint with minimum validation loss, mean ± std over seeds 42-44. BPC/PPL/Acc/ECE: full-context (W=128, stride 1); Acc_last: chunked128 position 127; s/step: separate solo timing run (2,000 iterations, median over iterations 1000-1999, RTX 3090).

| ID | lr | seeds | BPC | PPL | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) | Acc_last@128 (%) | Self-BLEU-4 | Self-BLEU-4 multi | #Params | s/step (solo) | best-loss step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TR | 0.0018 | 3 | 1.4532 ± 0.0055 | 2.7382 ± 0.0104 | 68.55 ± 0.11 | 0.0033 ± 0.0001 | 50.21 ± 0.31 | 80.00 ± 0.08 | 68.57 ± 0.05 | 0.1428 ± 0.0400 | 0.1312 ± 0.0113 | 3,169,280 | 0.0280 | 49,999 |
| LR-1 | 0.0032 | 3 | 1.5557 ± 0.0040 | 2.9398 ± 0.0081 | 66.26 ± 0.08 | 0.0066 ± 0.0002 | 45.21 ± 0.28 | 78.66 ± 0.01 | 66.11 ± 0.13 | 0.1576 ± 0.0091 | 0.1485 ± 0.0063 | 2,115,072 | 0.0526 | 49,666 |
| LR-2 | 0.0032 | 3 | 1.5152 ± 0.0017 | 2.8584 ± 0.0034 | 67.12 ± 0.02 | 0.0053 ± 0.0004 | 46.58 ± 0.03 | 79.15 ± 0.03 | 67.00 ± 0.06 | 0.1553 ± 0.0079 | 0.1417 ± 0.0030 | 3,177,052 | 0.0664 | 49,666 |
