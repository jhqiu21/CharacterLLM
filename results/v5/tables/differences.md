Seed-matched test-BPC differences (mean ± std over 3 seeds; negative = the first model is better). Sliding window, stride 128, except the last block (W=128, stride 1).

| Comparison (seed-matched) | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| PI fine-tuning − 512-window fine-tuning (does PI add anything after adaptation?) | +0.0463 ± 0.0024 | +0.0387 ± 0.0026 | +0.0252 ± 0.0022 | +3.6580 ± 0.1942 |
| 512-window fine-tuning − RoPE | +0.0195 ± 0.0017 | -2.8992 ± 0.2494 | -3.4183 ± 0.3331 | -2.9167 ± 0.2656 |
| PI fine-tuning − RoPE + PI | +0.0658 ± 0.0028 | -2.1239 ± 0.0532 | -2.5225 ± 0.0325 | +1.1192 ± 0.2205 |
| 512-window fine-tuning − RoPE at W=128, stride 1 | +0.1115 ± 0.0007 | -0.0002 ± 0.0029 | -0.0196 ± 0.0057 | +0.0657 ± 0.0070 |
| PI fine-tuning − RoPE at W=128, stride 1 | +0.1578 ± 0.0023 | +0.0385 ± 0.0033 | +0.0056 ± 0.0059 | +3.7237 ± 0.1967 |
| PI fine-tuning − ALiBi | +0.0508 ± 0.0061 | +0.0469 ± 0.0086 | +0.0392 ± 0.0102 | +3.7642 ± 0.1921 |
| 512-window fine-tuning − ALiBi | +0.0045 ± 0.0049 | +0.0082 ± 0.0079 | +0.0141 ± 0.0093 | +0.1062 ± 0.0072 |
| 512-window fine-tuning − ALiBi + 512-window fine-tuning | +0.0025 ± 0.0049 | +0.0106 ± 0.0078 | +0.0219 ± 0.0095 | +0.1168 ± 0.0065 |
| PI fine-tuning − ALiBi + 512-window fine-tuning | +0.0488 ± 0.0061 | +0.0493 ± 0.0084 | +0.0470 ± 0.0102 | +3.7748 ± 0.1926 |
| ALiBi + 512-window fine-tuning − ALiBi | +0.0020 ± 0.0000 | -0.0024 ± 0.0004 | -0.0078 ± 0.0007 | -0.0106 ± 0.0008 |
| RoPE + local attention 128 − RoPE + PI | +0.0000 ± 0.0000 | -2.1621 ± 0.0560 | -2.5278 ± 0.0381 | -2.6042 ± 0.0889 |
| RoPE + local attention 128 − RoPE at W=128, stride 1 | +0.0920 ± 0.0024 | +0.0002 ± 0.0005 | +0.0003 ± 0.0006 | +0.0003 ± 0.0006 |
| RoPE + SelfExtend floor, near 64 − RoPE + PI | n/a | -2.1767 ± 0.0541 | -2.5509 ± 0.0332 | -2.5994 ± 0.0816 |
| RoPE + SelfExtend floor, near 64 − RoPE at W=128, stride 1 | n/a | -0.0143 ± 0.0020 | -0.0228 ± 0.0057 | +0.0051 ± 0.0072 |
| RoPE + SelfExtend linear, near 64 − RoPE + PI | n/a | -2.1769 ± 0.0540 | -2.5504 ± 0.0331 | -2.5957 ± 0.0812 |
| RoPE + SelfExtend linear, near 64 − RoPE at W=128, stride 1 | n/a | -0.0145 ± 0.0020 | -0.0224 ± 0.0058 | +0.0087 ± 0.0076 |
| RoPE + PI + 512-window fine-tuning − its base model, W=128 stride 1 | +0.0594 ± 0.0025 |  |  |  |
| RoPE + 512-window fine-tuning − its base model, W=128 stride 1 | +0.0166 ± 0.0019 |  |  |  |
| ALiBi + 512-window fine-tuning − its base model, W=128 stride 1 | -0.0002 ± 0.0003 |  |  |  |
