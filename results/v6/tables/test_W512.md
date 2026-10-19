Test set, W = 512, stride 128 (4,999,999 targets, each scored once); mean ± std over seeds. v5 / v3 rows: best-validation-loss checkpoints (their selection rule); v6 rows: best-validation-accuracy checkpoints (BPC tie-break).

| Model | Seeds | BPC | Acc_all (%) | Acc_common (%) | Acc_rare (%) | Top-2 (%) | Top-3 (%) | Top-5 (%) |
|---|---|---|---|---|---|---|---|---|
| ALiBi-3.2M (512, from scratch) | 3 | 1.3795 ± 0.0008 | 70.23 ± 0.02 | 81.08 ± 0.02 | 53.78 ± 0.16 | 80.58 ± 0.01 | 85.45 ± 0.01 | 90.82 ± 0.00 |
| ALiBi-14.2M (512, from scratch) | 3 | 1.2440 ± 0.0008 | 73.20 ± 0.03 | 82.93 ± 0.03 | 58.14 ± 0.14 | 82.77 ± 0.01 | 87.13 ± 0.01 | 91.90 ± 0.00 |
| Transformer (ALiBi) + 512-window fine-tuning | 3 | 1.4075 ± 0.0034 | 69.60 ± 0.07 | 80.68 ± 0.04 | 52.62 ± 0.16 | 80.15 ± 0.05 | 85.12 ± 0.04 | 90.61 ± 0.02 |
| Transformer (RoPE) + 512-window fine-tuning | 3 | 1.4133 ± 0.0099 | 69.43 ± 0.21 | 80.57 ± 0.15 | 52.27 ± 0.46 | 80.05 ± 0.15 | 85.07 ± 0.12 | 90.57 ± 0.07 |
| Transformer (ALiBi), 128 only | 3 | 1.4196 ± 0.0045 | 69.31 ± 0.09 | 80.57 ± 0.09 | 51.83 ± 0.22 | 79.93 ± 0.07 | 84.95 ± 0.06 | 90.50 ± 0.04 |
