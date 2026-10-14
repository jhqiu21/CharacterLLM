Table V, lowest-BPC lr (block 1b): test set, checkpoint with minimum validation loss, mean ± std over seeds 42-44. BPC/PPL/Acc/ECE: full-context (W=128, stride 1); Acc_last: chunked128 position 127; s/step: separate solo timing run (2,000 iterations, median over iterations 1000-1999, RTX 3090).

| ID | lr | seeds | BPC | PPL | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) | Acc_last@128 (%) | Self-BLEU-4 | Self-BLEU-4 multi | #Params | s/step (solo) | best-loss step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TR | 0.0027 | 3 | 1.4412 ± 0.0003 | 2.7154 ± 0.0006 | 68.82 ± 0.00 | 0.0031 ± 0.0003 | 50.50 ± 0.24 | 80.20 ± 0.04 | 68.72 ± 0.07 | 0.1491 ± 0.0201 | 0.1306 ± 0.0143 | 3,169,280 | 0.0280 | 49,833 |
| LR-1 | 0.0072 | 3 | 1.5346 ± 0.0023 | 2.8972 ± 0.0046 | 66.69 ± 0.05 | 0.0060 ± 0.0003 | 45.71 ± 0.27 | 78.95 ± 0.04 | 66.59 ± 0.14 | 0.1696 ± 0.0087 | 0.1376 ± 0.0072 | 2,115,072 | 0.0526 | 49,999 |
| LR-2 | 0.0072 | 3 | 1.4957 ± 0.0033 | 2.8200 ± 0.0065 | 67.52 ± 0.06 | 0.0048 ± 0.0005 | 46.84 ± 0.14 | 79.46 ± 0.04 | 67.48 ± 0.11 | 0.1494 ± 0.0179 | 0.1388 ± 0.0012 | 3,177,052 | 0.0664 | 49,999 |
