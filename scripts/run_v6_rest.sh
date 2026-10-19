#!/bin/bash
# Remaining v6 phases on GPUs 2 and 3 (restarted after GPU 2 became available; see results/v6/REPORT.md):
# the other seeds, the post-hoc lr bracket check, then every evaluation (the test set is opened only in the last phase).
source /home/jinhang/miniconda3/etc/profile.d/conda.sh && conda activate charllm-env
cd "$(dirname "$0")/.." || exit 1
for p in seeds bracket evals; do
    python scripts/launch_v6.py --gpus 2,3 --phase "$p" || exit 1
done
