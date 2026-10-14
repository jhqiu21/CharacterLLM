Length extrapolation: test-set sliding-window BPC with window W and stride 128 (all models trained at 128), mean ± std over seeds 42-44. PI: position interpolation at evaluation time (positions scaled by 128/W), no fine-tuning. N/A: learned table of 128 positions.

| Model | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| RoPE | 1.5452 ± 0.0030 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| RoPE + PI | 1.5452 ± 0.0030 | 3.6156 ± 0.0614 | 3.9813 ± 0.0435 | 4.0577 ± 0.0937 |
| Sinusoidal | 1.5549 ± 0.0025 | 4.6830 ± 0.2713 | 4.6092 ± 0.2726 | 4.7990 ± 0.2193 |
| Sinusoidal + PI | 1.5549 ± 0.0025 | 2.9955 ± 0.0711 | 3.6540 ± 0.0326 | 3.8114 ± 0.0615 |
| Learned | 1.5806 ± 0.0023 | N/A | N/A | N/A |
| Hybrid | 1.5726 ± 0.0012 | N/A | N/A | N/A |
| Relative bias (clamped) | 1.5909 ± 0.0200 | 1.5353 ± 0.0116 | 2.9477 ± 0.2898 | 3.7355 ± 0.1986 |
| NoPE | 1.6794 ± 0.0120 | 1.9893 ± 0.0244 | 4.0939 ± 0.1676 | 4.4600 ± 0.3055 |
| ALiBi | 1.5602 ± 0.0016 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |
| LSTM-3.2M | 1.5834 ± 0.0014 | 1.5134 ± 0.0017 | 1.5123 ± 0.0017 | 1.5122 ± 0.0017 |
| LSTM-2.1M | 1.6191 ± 0.0030 | 1.5543 ± 0.0040 | 1.5534 ± 0.0041 | 1.5534 ± 0.0041 |

Sanity check (W=128, stride 1 vs eval.json full_context): 27/27 runs within 1e-4, largest |difference| 1.8e-09 BPC. PI at W=128 (pos_scale 1) vs plain W=128: largest |difference| 3.4e-08 BPC over 6 runs. Seed-matched difference from ALiBi at W=1024: RoPE + PI +2.6450 ± 0.0887; Sinusoidal + PI +2.3987 ± 0.0667; NoPE +3.0473 ± 0.3085.
