"""
Where the top-1 errors are: sliding-window accuracy and BPC grouped by the target's position in its word.

    python scripts/analyze_errors.py --run v6-AL-14M_s42 --length 512        # -> runs/<run>/errors_W512.json

Same windows and scoring as utils/full_context_eval.py (window W, stride S, every target scored exactly once), so the
overall accuracy equals extrapolation.json's. Each target character t is put in one group: 'space' (the target is a
space, i.e. the end of a word), or the position of the letter in its word (1st = the previous character is a space or
t starts the file, ..., 4th+). The checkpoint is the run's selected one (train.selected_checkpoint) unless --checkpoint.
--split val scores the validation split and never opens the test file (_val).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import train  # noqa: E402
from eval_extrapolation import checkpoint_prefix, load_run  # noqa: E402
from utils.extrapolation_eval import auto_batch_size, rebuild_for_length  # noqa: E402
from utils.full_context_eval import num_windows, window_bounds  # noqa: E402

MAX_POS = 4
GROUP_NAMES = ["space", "1st letter", "2nd letter", "3rd letter", "4th+ letter"]


def word_position_groups(data, max_pos=MAX_POS):
    """Group of every character: 0 for a space, else its position in the word, capped at max_pos."""
    data = np.asarray(data)
    is_space = data == train.CHAR_TO_INT[" "]
    idx = np.arange(len(data))
    last = np.maximum.accumulate(np.where(is_space, idx, -1))
    last_before = np.concatenate([[-1], last[:-1]])  # index of the last space before each character (-1: none)
    return np.where(is_space, 0, np.minimum(idx - last_before, max_pos)).astype(np.int32)


def grouped_eval(model, params, data_int, window, stride=128, batch_size=256):
    n_chars = len(data_int)
    n_win = num_windows(n_chars, window, stride)
    n_groups = len(GROUP_NAMES)
    labels = jnp.asarray(word_position_groups(data_int))
    data = jnp.asarray(np.asarray(data_int), dtype=jnp.int32)
    positions = jnp.arange(window)

    @jax.jit
    def eval_batch(params, k0):
        k = k0 + jnp.arange(batch_size)
        end, n_new = window_bounds(k, n_chars, window, stride, xp=jnp)
        idx = (end - window)[:, None] + positions[None, :]
        x, y, g = data[idx], data[idx + 1], labels[idx + 1].reshape(-1)
        logits = model.apply({"params": params}, x, deterministic=True)
        scored = (positions[None, :] >= (window - n_new)[:, None]).reshape(-1)
        nll = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), y[..., None], axis=-1).reshape(-1)
        correct = (jnp.argmax(logits, axis=-1) == y).reshape(-1)
        return (jnp.zeros(n_groups, jnp.int32).at[g].add(scored.astype(jnp.int32)),
                jnp.zeros(n_groups, jnp.int32).at[g].add((correct & scored).astype(jnp.int32)),
                jnp.zeros(n_groups, jnp.float32).at[g].add(jnp.where(scored, nll, 0.0)))

    count, n_correct, nll = (np.zeros(n_groups, np.float64) for _ in range(3))
    t0 = time.time()
    for b in range(math.ceil(n_win / batch_size)):
        c, r, s = jax.device_get(eval_batch(params, jnp.int32(b * batch_size)))
        count, n_correct, nll = count + c, n_correct + r, nll + s
    n, n_err = count.sum(), (count - n_correct).sum()
    groups = {name: {"count": int(count[g]), "share_of_targets": float(count[g] / n),
                     "acc": float(n_correct[g] / count[g]) if count[g] else None,
                     "bpc": float(nll[g] / count[g] / np.log(2)) if count[g] else None,
                     "share_of_errors": float((count[g] - n_correct[g]) / n_err) if n_err else 0.0}
              for g, name in enumerate(GROUP_NAMES)}
    return {"window": int(window), "stride": int(stride), "n_scored": int(n), "acc": float(n_correct.sum() / n),
            "bpc": float(nll.sum() / n / np.log(2)), "groups": groups, "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--checkpoint", default=None, choices=[p for p, _ in train.CHECKPOINT_TYPES])
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)

    run_dir, config, ckpt = load_run(args.run, args.checkpoint)
    _, val_int, test_int = train.load_data(args.data_dir, include_test=args.split == "test")
    data = test_int if args.split == "test" else val_int
    model = rebuild_for_length(config, args.length, train.build_model)
    bs = auto_batch_size(config, args.length, 2048)
    out = grouped_eval(model, ckpt["params"], data, args.length, args.stride, bs)
    assert out["n_scored"] == len(data) - 1
    out.update({"run": args.run, "checkpoint": checkpoint_prefix(config, args.checkpoint), "step": int(ckpt["step"]),
                "split": args.split, "batch_size": bs, "git_commit": train.git_commit(), "device": jax.devices()[0].device_kind})
    name = f"errors_W{args.length}" + ("_val" if args.split == "val" else "") + (f"_{args.tag}" if args.tag else "") + ".json"
    json.dump(out, open(run_dir / name, "w"), indent=1)
    print(f"{args.run} W={args.length}: acc {100 * out['acc']:.2f}%, BPC {out['bpc']:.4f}; "
          + "; ".join(f"{k} {100 * v['share_of_targets']:.1f}% of targets, acc {100 * v['acc']:.1f}%" for k, v in out["groups"].items()))
    print(f"Done: {run_dir / name}")


if __name__ == "__main__":
    main()
