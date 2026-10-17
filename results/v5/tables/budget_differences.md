Seed-matched test-BPC differences for the fine-tuning budget (mean ± std over 3 seeds; negative = the first is better). Sliding window, stride 128, except the last block (W=128, stride 1).

| Comparison (seed-matched) | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|
| PI fine-tuning − 512-window fine-tuning, 4,000 it. | +0.0243 ± 0.0004 | +0.0165 ± 0.0017 | +0.0068 ± 0.0029 | +3.6762 ± 0.2022 |
| 512-window fine-tuning: 4,000 − 1,000 it. | -0.0097 ± 0.0008 | -0.0146 ± 0.0014 | -0.0204 ± 0.0012 | +0.2052 ± 0.1022 |
| PI fine-tuning: 4,000 − 1,000 it. | -0.0316 ± 0.0019 | -0.0367 ± 0.0031 | -0.0387 ± 0.0042 | +0.2234 ± 0.0991 |
| ALiBi fine-tuning: 4,000 − 1,000 it. | -0.0001 ± 0.0001 | -0.0025 ± 0.0001 | -0.0043 ± 0.0004 | -0.0052 ± 0.0004 |
| 512-window fine-tuning, 4,000 it. − ALiBi | -0.0052 ± 0.0043 | -0.0064 ± 0.0067 | -0.0063 ± 0.0083 | +0.3113 ± 0.1056 |
| 512-window fine-tuning, 4,000 it. − ALiBi fine-tuning, 4,000 it. | -0.0071 ± 0.0044 | -0.0015 ± 0.0067 | +0.0058 ± 0.0086 | +0.3271 ± 0.1053 |
| PI fine-tuning, 4,000 it. − ALiBi | +0.0192 ± 0.0045 | +0.0102 ± 0.0055 | +0.0005 ± 0.0062 | +3.9875 ± 0.2802 |
| 512-window fine-tuning, 4,000 it. − RoPE at W=128, stride 1 | +0.1019 ± 0.0015 | -0.0148 ± 0.0015 | -0.0400 ± 0.0045 | +0.2708 ± 0.1085 |
| PI fine-tuning, 4,000 it. − RoPE at W=128, stride 1 | +0.1262 ± 0.0015 | +0.0018 ± 0.0005 | -0.0331 ± 0.0017 | +3.9470 ± 0.2846 |
| RoPE + PI + 512-window fine-tuning, 4,000 it. − its base model, W=128 stride 1 | +0.0236 ± 0.0009 |  |  |  |
| RoPE + 512-window fine-tuning, 4,000 it. − its base model, W=128 stride 1 | +0.0038 ± 0.0005 |  |  |  |
| ALiBi + 512-window fine-tuning, 4,000 it. − its base model, W=128 stride 1 | -0.0019 ± 0.0002 |  |  |  |
