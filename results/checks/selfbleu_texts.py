"""Regenerate the 20 Self-BLEU continuations for a run's best_loss_all_ checkpoint (same code path as
utils.eval.self_bleu) and save them, so the BLEU computation can be repeated under other nltk versions."""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/kaiwen/VLM/CharacterLLM")
os.chdir("/home/kaiwen/VLM/CharacterLLM")
import jax
import jax.numpy as jnp

import train
from utils import load_checkpoint
from utils.generation import generate_tokens

run = sys.argv[1]
cfg = train.load_config(f"runs/{run}/config.yaml")
model = train.build_model(cfg)
_, stage_dir, best_dir = train.run_paths(cfg, run)
params = load_checkpoint("best_loss_all_", "", best_dir, stage_dir)["params"]
ev = cfg.evaluation
prompt_int = jnp.array([[train.CHAR_TO_INT.get(c, 27) for c in ev.prompt.lower()[:cfg.model.max_len]]], dtype=jnp.int32)
keys = jax.random.split(jax.random.PRNGKey(ev.seed), ev.self_belu_n_samples)
texts = []
for k in keys:
    out = generate_tokens(model, params, k, prompt_int, ev.gen_len, block_size=cfg.model.max_len,
                          temperature=ev.temperature, sample=ev.sample, ngram_size=None)
    texts.append("".join(train.INT_TO_CHAR.get(int(x), "?") for x in list(out[0])).strip())
json.dump(texts, open(sys.argv[2], "w"))
print(len(texts), [len(t.split()) for t in texts][:5])
