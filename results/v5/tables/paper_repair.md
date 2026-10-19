Repairs of RoPE beyond its training length, test BPC, mean ± std over 3 seeds. W=128, stride 1: full-context evaluation at the training length (= RoPE: the same model there). Other columns: sliding window, stride 128. Extra training: characters of 512-window fine-tuning after the 819.2M characters of pre-training.

| Model | Extra training | W=128, stride 1 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|---|
| RoPE | -- | 1.4532 ± 0.0055 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| RoPE + PI | -- | = RoPE | 3.6156 ± 0.0614 | 3.9813 ± 0.0435 | 4.0577 ± 0.0937 |
| RoPE + local attention 128 | -- | = RoPE | 1.4535 ± 0.0054 | 1.4535 ± 0.0055 | 1.4535 ± 0.0055 |
| RoPE + SelfExtend (near 64) | -- | = RoPE | 1.4389 ± 0.0074 | 1.4304 ± 0.0111 | 1.4583 ± 0.0121 |
| RoPE + 512-window fine-tuning | 16.4M | 1.4698 ± 0.0073 | 1.4530 ± 0.0083 | 1.4337 ± 0.0112 | 1.5189 ± 0.0016 |
| RoPE + PI + 512-window fine-tuning | 16.4M | 1.5127 ± 0.0069 | 1.4917 ± 0.0082 | 1.4588 ± 0.0112 | 5.1769 ± 0.1942 |
| RoPE + 512-window fine-tuning | 65.5M | 1.4570 ± 0.0059 | 1.4385 ± 0.0070 | 1.4133 ± 0.0099 | 1.7240 ± 0.1034 |
| RoPE + PI + 512-window fine-tuning | 65.5M | 1.4769 ± 0.0047 | 1.4550 ± 0.0054 | 1.4201 ± 0.0071 | 5.4003 ± 0.2813 |
| ALiBi | -- | 1.4626 ± 0.0022 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |
| ALiBi + 512-window fine-tuning | 65.5M | 1.4608 ± 0.0020 | 1.4400 ± 0.0022 | 1.4075 ± 0.0034 | 1.3969 ± 0.0045 |
