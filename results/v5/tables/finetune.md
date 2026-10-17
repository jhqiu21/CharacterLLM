Long-window fine-tuning, test BPC, mean ± std over 3 seeds. Every fine-tuned run starts from the 50k checkpoint of the same seed and trains 1,000 more iterations of 16,384 characters (32 × 512 or 128 × 128; 2% of the pre-training characters); lr chosen on the validation split (seed 42). W columns: sliding window, stride 128. PI fine-tuning trains with pos_scale 0.25 (512 positions onto the 128 trained ones) and keeps it at every W.

| Model | lr | W=128, stride 1 | W=128 | W=256 | W=512 | W=1024 |
|---|---|---|---|---|---|---|
| RoPE (no fine-tuning) | — | 1.4532 ± 0.0055 | 1.5452 ± 0.0030 | 4.3522 ± 0.2446 | 4.8519 ± 0.3319 | 4.4356 ± 0.2642 |
| RoPE + PI (no fine-tuning) | — | 1.4532 ± 0.0055 | 1.5452 ± 0.0030 | 3.6156 ± 0.0614 | 3.9813 ± 0.0435 | 4.0577 ± 0.0937 |
| RoPE + 512-window fine-tuning | 3e-4 | 1.4698 ± 0.0073 | 1.5648 ± 0.0047 | 1.4530 ± 0.0083 | 1.4337 ± 0.0112 | 1.5189 ± 0.0016 |
| RoPE + PI + 512-window fine-tuning | 1e-3 | 1.5127 ± 0.0069 | 1.6110 ± 0.0053 | 1.4917 ± 0.0082 | 1.4588 ± 0.0112 | 5.1769 ± 0.1942 |
|   … + PI again at W=1024 (pos_scale 0.125) | 1e-3 |  |  |  |  | 2.6232 ± 0.0913 |
| RoPE + 128-window fine-tuning | any | = RoPE (no fine-tuning): no lr improved the validation loss |  |  |  |  |
| ALiBi (no fine-tuning) | — | 1.4626 ± 0.0022 | 1.5602 ± 0.0016 | 1.4449 ± 0.0027 | 1.4196 ± 0.0045 | 1.4127 ± 0.0057 |
| ALiBi + 512-window fine-tuning | 3e-5 | 1.4624 ± 0.0019 | 1.5622 ± 0.0016 | 1.4424 ± 0.0023 | 1.4118 ± 0.0038 | 1.4021 ± 0.0049 |
