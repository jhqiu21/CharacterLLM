Training dynamics of the 50,000-iteration runs (seeds 42-44). Best-loss iteration: range over seeds of the checkpoint with the lowest fixed-set validation loss. Val. loss change: from iteration 45,000 to the end. Gap: fixed-set validation loss minus the mean training loss of the preceding 500 iterations (dropout runs: training loss measured with dropout on).

| Group | Best-loss iteration | Val. loss change 45k-end (nats) | Gap at 2k (nats) | Gap at end (nats) |
|---|---|---|---|---|
| Transformer (RoPE) | 49,999-49,999 | -0.0041 ± 0.0006 | 0.0367 ± 0.0016 | 0.0672 ± 0.0025 |
| LSTM-3.2M | 49,000-49,999 | -0.0026 ± 0.0002 | 0.0280 ± 0.0029 | 0.0717 ± 0.0016 |
| LSTM-2.1M | 49,000-49,999 | -0.0022 ± 0.0004 | 0.0214 ± 0.0032 | 0.0669 ± 0.0014 |
| Transformer (RoPE), lowest-BPC lr | 49,500-49,999 | -0.0055 ± 0.0005 | 0.0386 ± 0.0019 | 0.0649 ± 0.0017 |
| LSTM-3.2M, lowest-BPC lr | 49,999-49,999 | -0.0037 ± 0.0002 | 0.0415 ± 0.0022 | 0.0710 ± 0.0012 |
| LSTM-2.1M, lowest-BPC lr | 49,999-49,999 | -0.0038 ± 0.0001 | 0.0383 ± 0.0030 | 0.0658 ± 0.0011 |
| Transformer (ALiBi) | 49,999-49,999 | -0.0038 ± 0.0004 | 0.0205 ± 0.0011 | 0.0683 ± 0.0010 |
| Transformer (Sinusoidal) | 49,999-49,999 | -0.0041 ± 0.0004 | 0.0258 ± 0.0014 | 0.0640 ± 0.0004 |
| Transformer (Hybrid) | 49,500-49,999 | -0.0042 ± 0.0004 | 0.0273 ± 0.0021 | 0.0627 ± 0.0006 |
| Transformer (Learned) | 49,999-49,999 | -0.0043 ± 0.0006 | 0.0272 ± 0.0046 | 0.0602 ± 0.0008 |
| Transformer (Relative bias) | 49,500-49,999 | -0.0041 ± 0.0005 | 0.0178 ± 0.0073 | 0.0575 ± 0.0015 |
| Transformer (NoPE) | 49,999-49,999 | -0.0043 ± 0.0005 | 0.0227 ± 0.0184 | 0.0585 ± 0.0045 |
| Transformer (RoPE), dropout 0.1 | 48,500-49,999 | -0.0027 ± 0.0005 | -0.0235 ± 0.0020 | 0.0115 ± 0.0019 |

All 39 runs: best-loss iteration 48,500-49,999.
