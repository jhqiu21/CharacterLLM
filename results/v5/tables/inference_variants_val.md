Evaluation-only long-context variants of the RoPE model (no fine-tuning), validation BPC, sliding window with stride 128, mean ± std over 3 seeds. † = the setting of each family with the lowest validation BPC at W=1024 (chosen on the validation split). Local attention: every layer sees the last N characters. SelfExtend: distances below the near window keep their exact RoPE positions, farther ones are compressed (floor: grouped positions as in SelfExtend; linear: continuous compression) with the smallest group size that keeps every relative position below 128.

| Model | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| RoPE | 1.5705 ± 0.0037 | 4.3567 ± 0.2383 | 4.8645 ± 0.3263 | 4.4358 ± 0.2599 |
| RoPE + PI | n/a | 3.6164 ± 0.0613 | 3.9857 ± 0.0431 | 4.0635 ± 0.0952 |
| RoPE + local attention 32 | 1.6175 ± 0.0030 | 1.5653 ± 0.0040 | 1.5653 ± 0.0040 | 1.5653 ± 0.0040 |
| RoPE + local attention 64 | 1.5818 ± 0.0026 | 1.5112 ± 0.0036 | 1.5112 ± 0.0036 | 1.5112 ± 0.0036 |
| RoPE + local attention 128 † | 1.5705 ± 0.0037 | 1.4738 ± 0.0066 | 1.4738 ± 0.0066 | 1.4738 ± 0.0066 |
| RoPE + SelfExtend floor, near 16 | n/a | 1.4686 ± 0.0070 | 1.4711 ± 0.0082 | 1.5205 ± 0.0138 |
| RoPE + SelfExtend floor, near 32 | n/a | 1.4611 ± 0.0080 | 1.4561 ± 0.0101 | 1.4875 ± 0.0112 |
| RoPE + SelfExtend floor, near 64 † | n/a | 1.4589 ± 0.0084 | 1.4485 ± 0.0118 | 1.4750 ± 0.0124 |
| RoPE + SelfExtend floor, near 96 | n/a | 1.4592 ± 0.0084 | 1.4529 ± 0.0125 | 1.4993 ± 0.0165 |
| RoPE + SelfExtend floor, near 112 | n/a | 1.4610 ± 0.0086 | 1.4723 ± 0.0139 | 1.6067 ± 0.0351 |
| RoPE + SelfExtend linear, near 16 | n/a | 1.4633 ± 0.0077 | 1.4677 ± 0.0085 | 1.5192 ± 0.0134 |
| RoPE + SelfExtend linear, near 32 | n/a | 1.4601 ± 0.0082 | 1.4543 ± 0.0105 | 1.4893 ± 0.0115 |
| RoPE + SelfExtend linear, near 64 † | n/a | 1.4587 ± 0.0084 | 1.4489 ± 0.0121 | 1.4785 ± 0.0129 |
| RoPE + SelfExtend linear, near 96 | n/a | 1.4595 ± 0.0084 | 1.4531 ± 0.0126 | 1.4981 ± 0.0165 |
| RoPE + SelfExtend linear, near 112 | n/a | 1.4611 ± 0.0086 | 1.4661 ± 0.0135 | 1.5540 ± 0.0255 |
