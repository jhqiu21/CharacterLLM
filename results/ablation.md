# Ablation: dropout-RNG fix and weight-decay mask (TR2o config, 3 seeds each)

All arms: TR2o config (RoPE, lr 4e-4, dropout 0.1), fixed 1024x128 validation set for checkpoint selection, best-validation-loss checkpoint. Mean ± sample std over seeds.

## legacy32: non-overlapping 32-char blocks (the paper's evaluation; *rare/common by per-batch frequency)

| arm | training | seeds | BPC | Acc_all (%) | Acc_last@31 (%) | Acc_rare* (%) | Acc_comm* (%) |
|---|---|---|---|---|---|---|---|
| TR2o-legacy | fixed dropout key (legacy), decay on all params | 3 | 1.8457 ± 0.0021 | 60.31 ± 0.05 | 64.17 ± 0.07 | 39.77 ± 0.42 | 72.13 ± 0.03 |
| TR2o-rngfix | dropout key split every step, decay on all params | 3 | 1.8460 ± 0.0029 | 60.30 ± 0.07 | 64.17 ± 0.09 | 39.60 ± 0.09 | 72.12 ± 0.05 |
| TR2o | dropout key split every step, decay mask | 3 | 1.8427 ± 0.0031 | 60.37 ± 0.07 | 64.28 ± 0.08 | 39.60 ± 0.24 | 72.23 ± 0.06 |

## chunked128: non-overlapping 128-char blocks

| arm | training | seeds | BPC | Acc_all (%) | Acc_last@127 (%) |
|---|---|---|---|---|---|
| TR2o-legacy | fixed dropout key (legacy), decay on all params | 3 | 1.6924 ± 0.0072 | 63.73 ± 0.18 | 65.07 ± 0.47 |
| TR2o-rngfix | dropout key split every step, decay on all params | 3 | 1.6891 ± 0.0058 | 63.80 ± 0.15 | 65.28 ± 0.32 |
| TR2o | dropout key split every step, decay mask | 3 | 1.6835 ± 0.0068 | 63.92 ± 0.17 | 65.42 ± 0.41 |

## full_context: sliding window W=128

| arm | training | seeds | BPC | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) |
|---|---|---|---|---|---|---|---|
| TR2o-legacy | fixed dropout key (legacy), decay on all params | 3 | 1.6278 ± 0.0160 | 65.19 ± 0.39 | 0.0095 ± 0.0005 | 45.28 ± 0.85 | 77.62 ± 0.20 |
| TR2o-rngfix | dropout key split every step, decay on all params | 3 | 1.6208 ± 0.0155 | 65.33 ± 0.37 | 0.0095 ± 0.0007 | 45.69 ± 0.82 | 77.74 ± 0.15 |
| TR2o | dropout key split every step, decay mask | 3 | 1.6132 ± 0.0174 | 65.51 ± 0.42 | 0.0098 ± 0.0007 | 45.89 ± 1.01 | 77.85 ± 0.22 |

## All Table V models in all three evaluation modes

### legacy32

| ID |  | seeds | BPC | Acc_all (%) | Acc_last@31 (%) | Acc_rare* (%) | Acc_comm* (%) |
|---|---|---|---|---|---|---|---|
| TR2o |  | 3 | 1.8427 ± 0.0031 | 60.37 ± 0.07 | 64.28 ± 0.08 | 39.60 ± 0.24 | 72.23 ± 0.06 |
| TR8o |  | 3 | 1.8886 ± 0.0016 | 59.72 ± 0.02 | 63.17 ± 0.08 | 37.59 ± 0.45 | 71.96 ± 0.03 |
| LR-1 |  | 3 | 1.9200 ± 0.0022 | 59.01 ± 0.05 | 62.54 ± 0.07 | 37.63 ± 0.50 | 71.35 ± 0.05 |
| LR-2 |  | 3 | 1.8855 ± 0.0007 | 59.72 ± 0.02 | 63.38 ± 0.07 | 38.14 ± 0.15 | 71.87 ± 0.03 |

### chunked128

| ID |  | seeds | BPC | Acc_all (%) | Acc_last@127 (%) |
|---|---|---|---|---|---|
| TR2o |  | 3 | 1.6835 ± 0.0068 | 63.92 ± 0.17 | 65.42 ± 0.41 |
| TR8o |  | 3 | 1.7563 ± 0.0029 | 62.68 ± 0.07 | 63.61 ± 0.20 |
| LR-1 |  | 3 | 1.7905 ± 0.0021 | 61.80 ± 0.06 | 62.55 ± 0.07 |
| LR-2 |  | 3 | 1.7487 ± 0.0010 | 62.70 ± 0.03 | 63.49 ± 0.12 |

### full_context

| ID |  | seeds | BPC | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) |
|---|---|---|---|---|---|---|---|
| TR2o |  | 3 | 1.6132 ± 0.0174 | 65.51 ± 0.42 | 0.0098 ± 0.0007 | 45.89 ± 1.01 | 77.85 ± 0.22 |
| TR8o |  | 3 | 1.7063 ± 0.0054 | 63.79 ± 0.14 | 0.0318 ± 0.0002 | 41.92 ± 0.27 | 77.16 ± 0.08 |
| LR-1 |  | 3 | 1.7460 ± 0.0018 | 62.76 ± 0.05 | 0.0254 ± 0.0003 | 41.42 ± 0.24 | 76.35 ± 0.08 |
| LR-2 |  | 3 | 1.7009 ± 0.0011 | 63.75 ± 0.03 | 0.0262 ± 0.0000 | 42.26 ± 0.13 | 76.97 ± 0.03 |
