"""ECE the way the old notebook measured it: loss_and_metrics on random 1024x32 validation batches."""
import os
import sys

sys.path.insert(0, "/home/kaiwen/VLM/CharacterLLM")
os.chdir("/home/kaiwen/VLM/CharacterLLM")
import numpy as np

import train
from utils import load_checkpoint

_, val, _ = train.load_data()
for run in sys.argv[1:]:
    cfg = train.load_config(f"runs/{run}/config.yaml")
    model = train.build_model(cfg)
    _, stage_dir, best_dir = train.run_paths(cfg, run)
    params = load_checkpoint("best_loss_all_", "", best_dir, stage_dir)["params"]
    lm = train.make_loss_and_metrics(cfg)
    apply_fn = train.make_apply_fn(model)
    rng = np.random.default_rng(123)
    out = {}
    for T in (32, 128):
        eces = []
        for _ in range(5):
            x, y = train.get_batch(val, 1024, T, rng)
            _, m = lm(apply_fn(params, x), y)
            eces.append(float(m["ece"][0]))
        out[T] = (np.mean(eces), np.std(eces))
    print(f"{run}: val ECE 1024x32 {out[32][0]:.4f} ± {out[32][1]:.4f} | 1024x128 {out[128][0]:.4f} ± {out[128][1]:.4f}", flush=True)
