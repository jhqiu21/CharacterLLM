Evaluation-only long-context variants of the RoPE model (no fine-tuning), test BPC, sliding window with stride 128, mean ± std over 3 seeds. † = the setting of each family with the lowest validation BPC at W=1024 (chosen on the validation split). Local attention: every layer sees the last N characters. SelfExtend: distances below the near window keep their exact RoPE positions, farther ones are compressed (floor: grouped positions as in SelfExtend; linear: continuous compression) with the smallest group size that keeps every relative position below 128.

| Model | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| RoPE | 1.5452 ± 0.0030 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| RoPE + PI | 1.5452 ± 0.0030 | 3.6156 ± 0.0614 | 3.9813 ± 0.0435 | 4.0577 ± 0.0937 |
| RoPE + local attention 32 | 1.5882 ± 0.0037 | 1.5359 ± 0.0050 | 1.5359 ± 0.0050 | 1.5359 ± 0.0050 |
| RoPE + local attention 64 | 1.5561 ± 0.0022 | 1.4881 ± 0.0032 | 1.4881 ± 0.0032 | 1.4881 ± 0.0032 |
| RoPE + local attention 128 † | 1.5452 ± 0.0030 | 1.4535 ± 0.0054 | 1.4535 ± 0.0055 | 1.4535 ± 0.0055 |
| RoPE + SelfExtend floor, near 16 | n/a | 1.4504 ± 0.0052 | 1.4560 ± 0.0056 | 1.5071 ± 0.0131 |
| RoPE + SelfExtend floor, near 32 | n/a | 1.4414 ± 0.0070 | 1.4388 ± 0.0091 | 1.4716 ± 0.0112 |
| RoPE + SelfExtend floor, near 64 † | n/a | 1.4389 ± 0.0074 | 1.4304 ± 0.0111 | 1.4583 ± 0.0121 |
| RoPE + SelfExtend floor, near 96 | n/a | 1.4392 ± 0.0074 | 1.4348 ± 0.0118 | 1.4832 ± 0.0164 |
| RoPE + SelfExtend floor, near 112 | n/a | 1.4411 ± 0.0075 | 1.4547 ± 0.0132 | 1.5930 ± 0.0355 |
| RoPE + SelfExtend linear, near 16 | n/a | 1.4444 ± 0.0063 | 1.4526 ± 0.0059 | 1.5061 ± 0.0122 |
| RoPE + SelfExtend linear, near 32 | n/a | 1.4403 ± 0.0072 | 1.4369 ± 0.0096 | 1.4735 ± 0.0115 |
| RoPE + SelfExtend linear, near 64 † | n/a | 1.4387 ± 0.0075 | 1.4309 ± 0.0113 | 1.4619 ± 0.0126 |
| RoPE + SelfExtend linear, near 96 | n/a | 1.4395 ± 0.0074 | 1.4351 ± 0.0118 | 1.4819 ± 0.0164 |
| RoPE + SelfExtend linear, near 112 | n/a | 1.4412 ± 0.0075 | 1.4485 ± 0.0127 | 1.5392 ± 0.0255 |
| ALiBi (reference) | 1.5602 ± 0.0016 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |

Reference: RoPE at W=128 with stride 1 (every target sees 127 characters; test): 1.4532 ± 0.0055.
