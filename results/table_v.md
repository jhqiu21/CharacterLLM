# Table V (new code, re-run)

Test set, checkpoint with minimum validation loss (fixed 1024x128 validation set). Mean ± sample std over seeds.
BPC, PPL, Acc_all, ECE, Acc_rare, Acc_comm: `full_context` evaluation. Acc_last: `chunked128` (non-overlapping 128-char blocks, position 127). s/step: median over iterations 1000-1999, averaged over the runs that had the GPU to themselves during that window ('solo runs'; see throughput.md).

full_context: sliding window W=128, stride S=1 over the 5M-character test set (every target scored once; >= 128 chars of context after the first window). Rare = {'q', 'j', 'x', 'z', 'k'}, common = {'n', 'a', 't', 'e', ' '} (bottom / top 20% of the 27 characters by training-split frequency).

| ID | seeds | BPC | PPL | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) | Acc_last@128 (%) | Self-BLEU-4 | #Params | s/step (solo) | solo runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TR2o | 3 | 1.6132 ± 0.0174 | 3.0595 ± 0.0371 | 65.51 ± 0.42 | 0.0098 ± 0.0007 | 45.89 ± 1.01 | 77.85 ± 0.22 | 65.42 ± 0.41 | 0.1577 ± 0.0200 | 3,169,280 | 0.0207 ± 0.0032 | 2 |
| TR8o | 3 | 1.7063 ± 0.0054 | 3.2633 ± 0.0122 | 63.79 ± 0.14 | 0.0318 ± 0.0002 | 41.92 ± 0.27 | 77.16 ± 0.08 | 63.61 ± 0.20 | 0.1165 ± 0.0191 | 3,173,360 | 0.0185 ± 0.0016 | 3 |
| LR-1 | 3 | 1.7460 ± 0.0018 | 3.3544 ± 0.0042 | 62.76 ± 0.05 | 0.0254 ± 0.0003 | 41.42 ± 0.24 | 76.35 ± 0.08 | 62.55 ± 0.07 | 0.1482 ± 0.0098 | 2,115,072 | 0.0517 ± 0.0002 | 2 |
| LR-2 | 3 | 1.7009 ± 0.0011 | 3.2510 ± 0.0025 | 63.75 ± 0.03 | 0.0262 ± 0.0000 | 42.26 ± 0.13 | 76.97 ± 0.03 | 63.49 ± 0.12 | 0.1697 ± 0.0151 | 3,177,052 | 0.0654 ± 0.0007 | 2 |

## Paper Table V (old code, legacy32-style evaluation), for reference

| ID | BPC | Acc_all | Acc_last | Acc_rare | Acc_comm | ECE | Self-BLEU-4 | #Params | s/step (8xA5000) |
|---|---|---|---|---|---|---|---|---|---|
| TR2o | 1.8468 | 60.28 | 64.08 | 39.27 | 72.1 | 0.083 | 0.0159 | 3,169,280 | 0.027 |
| TR8o | 1.8993 | 59.5 | 62.94 | 37.06 | 71.77 | 0.1322 | 0.0983 | 3,238,640 | 0.023 |
| LR-1 | 1.9198 | 59.02 | 62.53 | 37.6 | 71.29 | 0.0227 | 0.1541 | 2,115,072 | 0.044 |
| LR-2 | 1.8896 | 59.67 | 63.45 | 38.22 | 71.82 | 0.026 | 0.1212 | 3,177,052 | 0.069 |
