Test set, W = 512, stride 128, mean ± std over 3 seeds; every model trained on 819.2M characters (+ 65.5M of fine-tuning for the first two). First two rows: best-validation-loss checkpoints; ALiBi rows: best-validation-accuracy checkpoints.

| Model | Params | Training windows | BPC | Acc_all (%) | Acc_comm (%) | Acc_rare (%) | Top-2 (%) | Top-5 (%) |
|---|---|---|---|---|---|---|---|---|
| Transformer (ALiBi) + 512-window fine-tuning | 3.2M | 128, then 512 (+65.5M) | 1.4075 ± 0.0034 | 69.60 ± 0.07 | 80.68 ± 0.04 | 52.62 ± 0.16 | 80.15 ± 0.05 | 90.61 ± 0.02 |
| Transformer (RoPE) + 512-window fine-tuning | 3.2M | 128, then 512 (+65.5M) | 1.4133 ± 0.0099 | 69.43 ± 0.21 | 80.57 ± 0.15 | 52.27 ± 0.46 | 80.05 ± 0.15 | 90.57 ± 0.07 |
| ALiBi-3.2M | 3.2M | 512 from scratch | 1.3795 ± 0.0008 | 70.23 ± 0.02 | 81.08 ± 0.02 | 53.78 ± 0.16 | 80.58 ± 0.01 | 90.82 ± 0.00 |
| ALiBi-14.2M | 14.2M | 512 from scratch | 1.2440 ± 0.0008 | 73.20 ± 0.03 | 82.93 ± 0.03 | 58.14 ± 0.14 | 82.77 ± 0.01 | 91.90 ± 0.00 |
| ALiBi-14.2M − ALiBi-3.2M |  | seed-matched | -0.1355 ± 0.0017 | +2.97 ± 0.04 | +1.85 ± 0.03 | +4.35 ± 0.05 | +2.19 ± 0.03 | +1.08 ± 0.01 |
