"""
Write configs/scaling/{TR,LSTM}_<size>.yaml: configs/final/TR.yaml and configs/final/LSTM.yaml with only the size
changed (and model_name, so run names stay distinct). Exact parameter counts come from initialising the real model.

    python scripts/make_scaling_configs.py [--dry-run]

Transformer: d_model varies; 4 layers, n_heads = d_model / 64 (head dim 32 below d_model 128), MLP ratio 4.
LSTM: hidden_size varies; 4 layers. For each target the dimension with the closest exact count is chosen.
The 3.2M sizes are the final configs themselves (d_model 256, hidden 314), so no file is written for them.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import yaml  # noqa: E402

import train  # noqa: E402
from utils.config import Config  # noqa: E402

TARGETS = {"0.5M": 0.5e6, "1M": 1e6, "2M": 2e6, "3.2M": 3.2e6, "6M": 6e6, "10M": 10e6}
REUSED = "3.2M"
BASE = {"TR": ROOT / "configs/final/TR.yaml", "LSTM": ROOT / "configs/final/LSTM.yaml"}
TR_D_MODELS = [64, 96] + list(range(128, 769, 64))


def tr_heads(d_model):
    return d_model // 32 if d_model < 128 else d_model // 64


def size_keys(family, dim):
    return {"d_model": dim, "n_heads": tr_heads(dim)} if family == "TR" else {"hidden_size": dim}


def sized(base, family, dim, size_name):
    d = copy.deepcopy(base)
    d["model"].update(size_keys(family, dim))
    d["model_name"] = f"{family}-scaling-{size_name}"
    return d


def param_count(d):
    config = train.resolve_config(Config(copy.deepcopy(d)))
    model = train.build_model(config)
    return int(train.count_params(train.init_params(model, config, jax.random.PRNGKey(0))))


def candidates(family, base, target):
    if family == "TR":
        return TR_D_MODELS
    # LSTM: about 4 * 4h^2 * n_layers... narrow with the exact count of two sizes, then check the neighbourhood
    lo, hi = 32, 1024
    while hi - lo > 4:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if param_count(sized(base, family, mid, "x")) < target else (lo, mid)
    return list(range(max(8, lo - 3), hi + 4))


def choose(family, base):
    rows = []
    for name, target in TARGETS.items():
        counts = {dim: param_count(sized(base, family, dim, name)) for dim in candidates(family, base, target)}
        dim = min(counts, key=lambda k: abs(counts[k] - target))
        rows.append({"family": family, "size": name, "target": int(target), "dim": dim, "params": counts[dim],
                     "rel_err": counts[dim] / target - 1, **size_keys(family, dim)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "configs/scaling"))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    rows = []
    for family, path in BASE.items():
        base = yaml.safe_load(open(path))
        for r in choose(family, base):
            rows.append(r)
            if r["size"] == REUSED or args.dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            header = (f"# Scaling: {path.relative_to(ROOT)} with {size_keys(family, r['dim'])} "
                      f"({r['params']:,} parameters, target {r['target']:,}); written by scripts/make_scaling_configs.py\n")
            with open(out_dir / f"{family}_{r['size']}.yaml", "w") as f:
                f.write(header + yaml.safe_dump(sized(base, family, r["dim"], r["size"]), sort_keys=False))
    print("| family | size | dimension | exact params | target | error |\n|---|---|---|---|---|---|")
    for r in rows:
        dim = f"d_model {r['dim']} ({r['n_heads']} heads)" if r["family"] == "TR" else f"hidden {r['dim']}"
        note = " (final config, reused)" if r["size"] == REUSED else ""
        print(f"| {r['family']} | {r['size']} | {dim}{note} | {r['params']:,} | {r['target']:,} | {100 * r['rel_err']:+.1f}% |")
    return rows


if __name__ == "__main__":
    main()
