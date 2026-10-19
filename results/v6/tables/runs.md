Every v6 training run: seed, actual updates and training characters (timing.json), the checkpoint that every reported number of the run comes from, speed and memory. Code: the commit recorded by train.py; -dirty = with the uncommitted changes of results/v6/code/ (snapshot of each launch).

| run | seed | peak lr | parameters | updates | training characters | selected checkpoint (checkpoints/<run>/best/) | s / update (median) | training min | job min (incl. validation-split eval) | GPU | peak GiB | code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6search-AL-3M_lr9e-4 | 4212 | 0.0009 | 3,169,280 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0314 | 5.8 | 6.7 | 3 | 2.52 | 73f222e-dirty |
| v6search-AL-3M_lr1.8e-3 | 4212 | 0.0018 | 3,169,280 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0314 | 5.8 | 6.7 | 3 | 2.52 | 73f222e-dirty |
| v6search-AL-3M_lr3.6e-3 | 4212 | 0.0036 | 3,169,280 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0314 | 5.8 | 6.7 | 3 | 2.53 | 73f222e-dirty |
| v6search-AL-3M_lr7.2e-3 | 4212 | 0.0072 | 3,169,280 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0314 | 5.8 | 6.7 | 2 | 2.5 | 73f222e-dirty |
| v6-AL-3M_s42 | 42 | 0.0036 | 3,169,280 | 50000 | 819,200,000 | best_acc_bpc_48000 | 0.0314 | 27.5 | 28.3 | 3 | 2.55 | 73f222e-dirty |
| v6-AL-3M_s43 | 43 | 0.0036 | 3,169,280 | 50000 | 819,200,000 | best_acc_bpc_49999 | 0.0315 | 27.5 | 28.3 | 3 | 2.57 | 73f222e-dirty |
| v6-AL-3M_s44 | 44 | 0.0036 | 3,169,280 | 50000 | 819,200,000 | best_acc_bpc_49000 | 0.0313 | 27.4 | 28.2 | 2 | 2.56 | 73f222e-dirty |
| v6search-AL-14M_lr9e-4 | 4212 | 0.0009 | 14,204,928 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0981 | 17.3 | 19.0 | 3 | 7.75 | 73f222e-dirty |
| v6search-AL-14M_lr1.8e-3 | 4212 | 0.0018 | 14,204,928 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0981 | 17.3 | 19.0 | 3 | 7.76 | 73f222e-dirty |
| v6search-AL-14M_lr3.6e-3 | 4212 | 0.0036 | 14,204,928 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0986 | 17.4 | 19.2 | 3 | 7.75 | 73f222e-dirty |
| v6search-AL-14M_lr7.2e-3 | 4212 | 0.0072 | 14,204,928 | 10000 | 163,840,000 | best_acc_bpc_9999 | 0.0979 | 17.2 | 19.1 | 3 | 7.72 | 73f222e-dirty |
| v6-AL-14M_s42 | 42 | 0.0036 | 14,204,928 | 50000 | 819,200,000 | best_acc_bpc_49000 | 0.0982 | 84.2 | 86.0 | 3 | 7.92 | 73f222e-dirty |
| v6-AL-14M_s43 | 43 | 0.0036 | 14,204,928 | 50000 | 819,200,000 | best_acc_bpc_49999 | 0.0985 | 84.7 | 86.4 | 2 | 7.93 | 73f222e-dirty |
| v6-AL-14M_s44 | 44 | 0.0036 | 14,204,928 | 50000 | 819,200,000 | best_acc_bpc_49500 | 0.0980 | 84.2 | 86.1 | 3 | 7.92 | 73f222e-dirty |
