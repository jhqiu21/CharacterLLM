Length extrapolation: test-set sliding-window BPC with window W and stride 128 (trained at 128), mean ± std over seeds. N/A: learned table of 128 positions.

| model | seeds | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|---|
| rope | 3 | 1.5452 ± 0.0030 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| learned | 3 | 1.5806 ± 0.0023 | N/A | N/A | N/A |
| sinusoidal | 3 | 1.5549 ± 0.0025 | 4.6830 ± 0.2713 | 4.6092 ± 0.2726 | 4.7990 ± 0.2193 |
| hybrid | 3 | 1.5726 ± 0.0012 | N/A | N/A | N/A |
| alibi | 3 | 1.5602 ± 0.0016 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |
| relative | 3 | 1.5909 ± 0.0200 | 1.5353 ± 0.0116 | 2.9477 ± 0.2898 | 3.7355 ± 0.1986 |
| LR-2 | 3 | 1.5834 ± 0.0014 | 1.5134 ± 0.0017 | 1.5123 ± 0.0017 | 1.5122 ± 0.0017 |
| LR-1 | 3 | 1.6191 ± 0.0030 | 1.5543 ± 0.0040 | 1.5534 ± 0.0041 | 1.5534 ± 0.0041 |

Sanity check (W=128, stride 1 vs eval.json full_context): 24/24 runs within 1e-4, largest |difference| 1.8e-09 BPC.
