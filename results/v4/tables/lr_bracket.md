Learning-rate search and bracket check: full-context BPC on the validation split (stride 8), seed 4212, 10,000 iterations, best-validation-loss checkpoint. A100 rows: the original search (label smoothing 0.05 except the last row per family); † RTX 3090 runs with the final configuration (dropout 0, ε = 0): the rerun of the lowest-BPC lr and two further steps of factor 1.5.

| Model | lr | Dropout | Label smoothing ε | GPU | Validation BPC |
|---|---|---|---|---|---|
| Transformer (RoPE) | 0.0004 | 0 | 0.05 | A100 | 1.7753 |
| Transformer (RoPE) | 0.0006 | 0 | 0.05 | A100 | 1.7452 |
| Transformer (RoPE) | 0.0008 | 0 | 0.05 | A100 | 1.7256 |
| Transformer (RoPE) | 0.0012 | 0 | 0.05 | A100 | 1.7019 |
| Transformer (RoPE) | 0.0018 | 0 | 0.05 | A100 | 1.6788 |
| Transformer (RoPE) | 0.0027 | 0 | 0.05 | A100 | 1.6707 (lowest of search) |
| Transformer (RoPE) | 0.0004 | 0.1 | 0.05 | A100 | 1.8423 |
| Transformer (RoPE) | 0.0006 | 0.1 | 0.05 | A100 | 1.8041 |
| Transformer (RoPE) | 0.0008 | 0.1 | 0.05 | A100 | 1.7832 |
| Transformer (RoPE) | 0.0012 | 0.1 | 0.05 | A100 | 1.7557 |
| Transformer (RoPE) | 0.0018 | 0 | 0 | A100 | 1.6377 |
| Transformer (RoPE) | 0.0018 | 0 | 0 | RTX 3090† | 1.6370 |
| Transformer (RoPE) | 0.0027 | 0 | 0 | RTX 3090† | 1.5825 |
| Transformer (RoPE) | 0.00405 | 0 | 0 | RTX 3090† | 1.5684 |
| Transformer (RoPE) | 0.006075 | 0 | 0 | RTX 3090† | 1.5980 |
| LSTM-3.2M | 0.0004 | 0 | 0.05 | A100 | 1.9218 |
| LSTM-3.2M | 0.0008 | 0 | 0.05 | A100 | 1.8182 |
| LSTM-3.2M | 0.0016 | 0 | 0.05 | A100 | 1.7424 |
| LSTM-3.2M | 0.0032 | 0 | 0.05 | A100 | 1.6874 |
| LSTM-3.2M | 0.0048 | 0 | 0.05 | A100 | 1.6760 |
| LSTM-3.2M | 0.0072 | 0 | 0.05 | A100 | 1.6677 (lowest of search) |
| LSTM-3.2M | 0.0004 | 0.1 | 0.05 | A100 | 1.9003 |
| LSTM-3.2M | 0.0008 | 0.1 | 0.05 | A100 | 1.8162 |
| LSTM-3.2M | 0.0016 | 0.1 | 0.05 | A100 | 1.7526 |
| LSTM-3.2M | 0.0032 | 0.1 | 0.05 | A100 | 1.7163 |
| LSTM-3.2M | 0.0032 | 0 | 0 | A100 | 1.6503 |
| LSTM-3.2M | 0.0072 | 0 | 0 | RTX 3090† | 1.6325 |
| LSTM-3.2M | 0.0108 | 0 | 0 | RTX 3090† | 1.6284 |
| LSTM-3.2M | 0.0162 | 0 | 0 | RTX 3090† | 1.6325 |

Transformer (RoPE): improvement over the 3090 rerun at lr 0.0027 (BPC 1.5825): lr 0.00405: +0.0141, lr 0.006075: -0.0155. Bracketed (no new lr is better by more than 0.02).
LSTM-3.2M: improvement over the 3090 rerun at lr 0.0072 (BPC 1.6325): lr 0.0108: +0.0042, lr 0.0162: +0.0000. Bracketed (no new lr is better by more than 0.02).
