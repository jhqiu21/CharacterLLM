Model size × training characters, whole validation split (W = 512, stride 128), selected checkpoints. 164M = the search run at the chosen lr (seed 4212, complete 10,000-iteration schedule); 819M = the formal run, seed 42.

| model | lr | val Acc_all, 164M chars (%) | val BPC | val Acc_all, 819M chars (%) | val BPC | 5× characters: Δ Acc (pp) |
|---|---|---|---|---|---|---|
| AL-3M (3.2M) | 3.6e-3 | 68.14 | 1.4822 | 70.06 | 1.3950 | +1.92 |
| AL-14M (14.2M) | 3.6e-3 | 70.64 | 1.3639 | 73.05 | 1.2574 | +2.41 |
| AL-14M − AL-3M |  | +2.50 | -0.1183 | +2.99 | -0.1377 |  |

The training split holds 85,500,000 characters, so 164M characters are 1.9 passes and 819M are 9.6 passes over it.
