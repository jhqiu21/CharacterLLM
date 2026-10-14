Transformer vs LSTMs, lowest-BPC lr: test set, best-validation-loss checkpoint, mean ± std over seeds 42-44. BPC/PPL/Acc/ECE: full-context evaluation (W=128, stride 1); Acc_last: last position of non-overlapping 128-character blocks; s/step: separate solo timing run (median over iterations 1000-1999, RTX 3090).

| Model | lr | BPC | PPL | Acc_all (%) | ECE | Acc_rare (%) | Acc_comm (%) | Acc_last (%) | Self-BLEU-4 | Self-BLEU-4 multi | #Params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer (RoPE) | 0.0027 | 1.4412 ± 0.0003 | 2.7154 ± 0.0006 | 68.82 ± 0.00 | 0.0031 ± 0.0003 | 50.50 ± 0.24 | 80.20 ± 0.04 | 68.72 ± 0.07 | 0.1491 ± 0.0201 | 0.1306 ± 0.0143 | 3,169,280 |
| LSTM-3.2M | 0.0072 | 1.4957 ± 0.0033 | 2.8200 ± 0.0065 | 67.52 ± 0.06 | 0.0048 ± 0.0005 | 46.84 ± 0.14 | 79.46 ± 0.04 | 67.48 ± 0.11 | 0.1494 ± 0.0179 | 0.1388 ± 0.0012 | 3,177,052 |
| LSTM-2.1M | 0.0072 | 1.5346 ± 0.0023 | 2.8972 ± 0.0046 | 66.69 ± 0.05 | 0.0060 ± 0.0003 | 45.71 ± 0.27 | 78.95 ± 0.04 | 66.59 ± 0.14 | 0.1696 ± 0.0087 | 0.1376 ± 0.0072 | 2,115,072 |

Seed-matched BPC difference Transformer (RoPE) - LSTM-3.2M: -0.0545 ± 0.0030. Seed-matched BPC difference Transformer (RoPE) - LSTM-2.1M: -0.0935 ± 0.0026. Transformer (RoPE): lowest-BPC lr - tie-rule lr: -0.0121 ± 0.0051 BPC (seed-matched). LSTM-3.2M: lowest-BPC lr - tie-rule lr: -0.0195 ± 0.0023 BPC (seed-matched). LSTM-2.1M: lowest-BPC lr - tie-rule lr: -0.0211 ± 0.0063 BPC (seed-matched).
