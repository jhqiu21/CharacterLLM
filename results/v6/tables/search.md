Learning-rate search: 10,000 iterations (163.84M training characters), seed 4212, warmup 2,000, cosine to 0. Whole validation split, W = 512, stride 128, at the checkpoint selected by fixed-validation accuracy (BPC tie-break). Chosen: highest val Acc_all, ties -> lower BPC.

| model | lr | val Acc_all (%) | val BPC | selected step | Acc_all of the best-loss checkpoint (%) |
|---|---|---|---|---|---|
| AL-3M | 9e-4 | 67.252 | 1.5237 | 9999 | 67.252 |
| AL-3M | 1.8e-3 | 67.883 | 1.4951 | 9999 | 67.883 |
| AL-3M | 3.6e-3 ← chosen | 68.139 | 1.4822 | 9999 | 68.139 |
| AL-3M | 7.2e-3 (post-hoc bracket check) | 68.220 | 1.4789 | 9999 | 68.220 |
| AL-14M | 9e-4 | 70.155 | 1.3866 | 9999 | 70.155 |
| AL-14M | 1.8e-3 | 70.577 | 1.3679 | 9999 | 70.577 |
| AL-14M | 3.6e-3 ← chosen | 70.642 | 1.3639 | 9999 | 70.642 |
| AL-14M | 7.2e-3 (post-hoc bracket check) | 70.485 | 1.3709 | 9999 | 70.485 |
