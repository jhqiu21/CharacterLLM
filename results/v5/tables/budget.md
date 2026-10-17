Fine-tuning budget: 1,000 vs 4,000 iterations (16,384 characters each; warmup and validation interval 5% of the run; the lr chosen for 1,000), test BPC, mean ± std over 3 seeds; sliding window, stride 128.

| Model | Iterations | W=128, stride 1 | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|---|---|
| RoPE + PI + 512-window fine-tuning | 1,000 | 1.5127 ± 0.0069 | 1.6110 ± 0.0053 | 1.4917 ± 0.0082 | 1.4588 ± 0.0112 | 5.1769 ± 0.1942 |
| RoPE + PI + 512-window fine-tuning | 4,000 | 1.4769 ± 0.0047 | 1.5794 ± 0.0041 | 1.4550 ± 0.0054 | 1.4201 ± 0.0071 | 5.4003 ± 0.2813 |
| RoPE + 512-window fine-tuning | 1,000 | 1.4698 ± 0.0073 | 1.5648 ± 0.0047 | 1.4530 ± 0.0083 | 1.4337 ± 0.0112 | 1.5189 ± 0.0016 |
| RoPE + 512-window fine-tuning | 4,000 | 1.4570 ± 0.0059 | 1.5551 ± 0.0040 | 1.4385 ± 0.0070 | 1.4133 ± 0.0099 | 1.7240 ± 0.1034 |
| ALiBi + 512-window fine-tuning | 1,000 | 1.4624 ± 0.0019 | 1.5622 ± 0.0016 | 1.4424 ± 0.0023 | 1.4118 ± 0.0038 | 1.4021 ± 0.0049 |
| ALiBi + 512-window fine-tuning | 4,000 | 1.4608 ± 0.0020 | 1.5622 ± 0.0018 | 1.4400 ± 0.0022 | 1.4075 ± 0.0034 | 1.3969 ± 0.0045 |
| ALiBi (no fine-tuning) | 0 | 1.4626 ± 0.0022 | 1.5602 ± 0.0016 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |
| RoPE (no fine-tuning) | 0 | 1.4532 ± 0.0055 |  |  |  |  |
