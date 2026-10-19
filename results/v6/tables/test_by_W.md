Test BPC and Acc_all at each window (stride 128), mean ± std over seeds.

| Model | Metric | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|---|
| ALiBi-3.2M (512, from scratch) | BPC | 1.5645 ± 0.0006 | 1.4215 ± 0.0006 | 1.3795 ± 0.0008 | 1.3651 ± 0.0005 |
| ALiBi-3.2M (512, from scratch) | Acc_all (%) | 66.07 ± 0.02 | 69.31 ± 0.01 | 70.23 ± 0.02 | 70.53 ± 0.01 |
| ALiBi-14.2M (512, from scratch) | BPC | 1.4390 ± 0.0007 | 1.2839 ± 0.0009 | 1.2440 ± 0.0008 | 1.2341 ± 0.0025 |
| ALiBi-14.2M (512, from scratch) | Acc_all (%) | 68.83 ± 0.00 | 72.35 ± 0.02 | 73.20 ± 0.03 | 73.38 ± 0.06 |
| Transformer (ALiBi) + 512-window fine-tuning | BPC | 1.5622 ± 0.0018 | 1.4400 ± 0.0022 | 1.4075 ± 0.0034 | 1.3969 ± 0.0045 |
| Transformer (ALiBi) + 512-window fine-tuning | Acc_all (%) | 66.12 ± 0.04 | 68.90 ± 0.05 | 69.60 ± 0.07 | 69.82 ± 0.09 |
| Transformer (RoPE) + 512-window fine-tuning | BPC | 1.5551 ± 0.0040 | 1.4385 ± 0.0070 | 1.4133 ± 0.0099 | 1.7240 ± 0.1034 |
| Transformer (RoPE) + 512-window fine-tuning | Acc_all (%) | 66.26 ± 0.08 | 68.90 ± 0.15 | 69.43 ± 0.21 | 62.58 ± 2.20 |
| Transformer (ALiBi), 128 only | BPC | 1.5602 ± 0.0016 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |
| Transformer (ALiBi), 128 only | Acc_all (%) | 66.17 ± 0.03 | 68.77 ± 0.06 | 69.31 ± 0.09 | 69.45 ± 0.11 |
