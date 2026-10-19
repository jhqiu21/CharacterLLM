Fixed validation set (1,024 windows × 512, all positions) during the formal runs. Train BPC = mean unweighted loss on the training batches over the last 1,000 iterations (no dropout, so the same model as at validation). Passes over the 85.5M-character training split: 0.96 at 5k iterations, 1.9 at 10k, 9.6 at 50k.

| model | seed | val Acc @ 5k (%) | val Acc @ 10k (%) | val Acc @ 20k (%) | val Acc @ 30k (%) | val Acc @ 40k (%) | val Acc @ 50k (%) | val BPC @ 50k | iteration of lowest val BPC | val − train BPC @ 5k | val − train BPC @ 10k | val − train BPC @ 30k | val − train BPC @ 50k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AL-3M | 42 | 63.18 | 64.32 | 65.46 | 66.55 | 67.97 | 68.71 | 1.4612 | 49999 | +0.0329 | +0.0440 | +0.0609 | +0.0888 |
| AL-3M | 43 | 63.05 | 64.22 | 65.31 | 66.60 | 67.88 | 68.65 | 1.4579 | 49999 | +0.0323 | +0.0419 | +0.0564 | +0.0815 |
| AL-3M | 44 | 63.10 | 64.20 | 65.32 | 66.54 | 67.86 | 68.67 | 1.4570 | 49500 | +0.0327 | +0.0468 | +0.0565 | +0.0830 |
| AL-14M | 42 | 65.36 | 66.65 | 67.89 | 69.32 | 70.82 | 71.62 | 1.3229 | 49999 | +0.0347 | +0.0503 | +0.0726 | +0.1371 |
| AL-14M | 43 | 65.49 | 66.51 | 67.90 | 69.29 | 70.90 | 71.59 | 1.3236 | 49000 | +0.0314 | +0.0492 | +0.0697 | +0.1324 |
| AL-14M | 44 | 65.42 | 66.47 | 67.90 | 69.19 | 70.78 | 71.63 | 1.3223 | 49500 | +0.0340 | +0.0528 | +0.0723 | +0.1355 |
