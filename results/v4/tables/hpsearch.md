Hyperparameter search (A100): validation BPC (full context, stride 8) after 10,000 iterations, seed 4212, label smoothing ε = 0.05. † lowest BPC; --: not run. Left: Transformer (RoPE); right: LSTM-3.2M.

| lr | dropout 0 | dropout 0.1 | lr  | dropout 0  | dropout 0.1  |
|---|---|---|---|---|---|
| 4.0e-4 | 1.7753 | 1.8423 | 4.0e-4 | 1.9218 | 1.9003 |
| 6.0e-4 | 1.7452 | 1.8041 | 8.0e-4 | 1.8182 | 1.8162 |
| 8.0e-4 | 1.7256 | 1.7832 | 1.6e-3 | 1.7424 | 1.7526 |
| 1.2e-3 | 1.7019 | 1.7557 | 3.2e-3 | 1.6874 | 1.7163 |
| 1.8e-3 | 1.6788 | -- | 4.8e-3 | 1.6760 | -- |
| 2.7e-3 | 1.6707† | -- | 7.2e-3 | 1.6677† | -- |

selected lr 1.8e-3, dropout 0, ε = 0: 1.6377; selected lr 3.2e-3, dropout 0, ε = 0: 1.6503
