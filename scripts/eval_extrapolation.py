"""
Length-extrapolation evaluation of a trained run (utils/extrapolation_eval.py) on the test set.

    python scripts/eval_extrapolation.py --run TR-final_s42 --lengths 128 256 512 1024

For each W: sliding-window BPC (window W, stride --stride) and the per-position loss of --n-chunks
non-overlapping chunks (seed 0). learned / hybrid beyond their training length are recorded as N/A.
--sanity also scores W = training length with stride 1 and compares it with the run's eval.json full_context BPC.
--interpolate uses position interpolation (rope / sinusoidal only, no fine-tuning): pos_scale = L / W for each W,
and writes runs/<run>/extrapolation_pi.json instead (the plain results are never overwritten).
The evaluation batch size starts at --batch-size and is lowered for large W so the estimate stays below --budget-gib.
Writes runs/<run>/extrapolation.json (checkpoint best_loss_all_).

    python scripts/eval_extrapolation.py --run TR-final_s42 --lengths 256 512 1024 --interpolate

Evaluation-only variants (no fine-tuning), each written to its own file (the suffixes combine, see output_name):
--local-window N: local attention, every layer sees only the last N positions (_local<N>);
--self-extend N [--self-extend-mode floor|linear]: RoPE SelfExtend, distances < N keep their exact positions, farther
ones are compressed with the smallest group size that keeps every relative position below the training length
(_se<mode><N>). --stride1 W ...: also sliding-window BPC with stride 1 at these W (every target sees >= W-1 characters).
--split val scores the validation split instead and never opens the test file (_val), e.g. to choose a variant.

    python scripts/eval_extrapolation.py --run TR-final_s42 --lengths 256 512 1024 --self-extend 32 --split val
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

import train  # noqa: E402
from utils import load_checkpoint, load_config  # noqa: E402
from utils.extrapolation_eval import (  # noqa: E402
    ExtrapolationNotSupported,
    auto_batch_size,
    interpolation_scale,
    per_position_loss,
    rebuild_for_length,
    self_extend_group,
    sliding_window_eval,
    train_len,
)
from utils.full_context_eval import frequency_groups  # noqa: E402

DROP = ("ece_bins", "class_count", "class_correct")


def load_run(run):
    run_dir = ROOT / "runs" / run
    config = train.resolve_config(load_config(run_dir / "config.yaml"))
    _, stage_dir, best_dir = train.run_paths(config, run)
    ckpt = load_checkpoint("best_loss_all_", "Best Loss", best_dir, stage_dir)
    if ckpt is None:
        raise FileNotFoundError(f"no best_loss_all_ checkpoint for {run}")
    return run_dir, config, ckpt


def output_name(args):
    """extrapolation[_pi][_local<N>][_se<mode><N>][_val].json"""
    parts = ["extrapolation"]
    if args.interpolate:
        parts.append("pi")
    if args.local_window:
        parts.append(f"local{args.local_window}")
    if args.self_extend:
        parts.append(f"se{args.self_extend_mode}{args.self_extend}")
    if args.split == "val":
        parts.append("val")
    return "_".join(parts) + ".json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2048, help="windows per batch at the training length (lowered for large W)")
    ap.add_argument("--budget-gib", type=float, default=20.0)
    ap.add_argument("--n-chunks", type=int, default=2048)
    ap.add_argument("--sanity", action="store_true", help="also W = training length, stride 1, vs eval.json")
    ap.add_argument("--interpolate", action="store_true", help="position interpolation, pos_scale = L / W (rope, sinusoidal)")
    ap.add_argument("--local-window", type=int, default=0, help="local attention over the last N positions (0 = off)")
    ap.add_argument("--self-extend", type=int, default=0, help="RoPE SelfExtend near window N (0 = off)")
    ap.add_argument("--self-extend-mode", choices=["floor", "linear"], default="floor")
    ap.add_argument("--stride1", type=int, nargs="*", default=[], help="also sliding-window BPC with stride 1 at these W")
    ap.add_argument("--split", choices=["test", "val"], default="test", help="val: score the validation split (test never opened)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)
    if args.sanity and args.split != "test":
        raise SystemExit("--sanity compares with eval.json, which is a test-set result: use --split test")

    run_dir, config, ckpt = load_run(args.run)
    out_file = run_dir / output_name(args)
    params = ckpt["params"]
    train_int, val_int, test_int = train.load_data(args.data_dir, include_test=args.split == "test")
    eval_int = test_int if args.split == "test" else val_int
    groups = frequency_groups(train_int, config.model.vocab_size, train.INT_TO_CHAR)
    L = train_len(config)
    self_extend = (args.self_extend, args.self_extend_mode) if args.self_extend else None
    variant = {"interpolate": bool(args.interpolate), "attn_window": args.local_window,
               "self_extend": list(self_extend) if self_extend else None}
    out = {"run": args.run, "checkpoint": "best_loss_all_", "step": int(ckpt["step"]), "train_len": L,
           "model_type": config.model.type, "pos_encoding_type": getattr(config.model, "pos_encoding_type", None),
           "model_pos_scale": float(getattr(config.model, "pos_scale", 1.0)), "split": args.split,
           "n_chars": len(eval_int), "stride": args.stride, "n_chunks": args.n_chunks, "device": jax.devices()[0].device_kind,
           "git_commit": train.git_commit(), "interpolate": bool(args.interpolate), "variant": variant, "lengths": {}}

    def rebuild(W):
        return rebuild_for_length(config, W, train.build_model, interpolate=args.interpolate, attn_window=args.local_window,
                                  self_extend=self_extend)

    def fc_eval(model, W, stride):
        bs = auto_batch_size(config, W, args.batch_size, args.budget_gib)
        t0 = time.time()
        fc = sliding_window_eval(model, params, eval_int, W, stride=stride, batch_size=bs,
                                 vocab_size=config.model.vocab_size, rare_ids=groups["rare_ids"], common_ids=groups["common_ids"])
        print(f"  W={W} S={stride} batch {bs}: BPC {fc['bpc']:.4f} acc {100 * fc['acc']:.2f}% ({time.time() - t0:.0f}s)", flush=True)
        return {k: v for k, v in fc.items() if k not in DROP}

    if args.sanity:
        ref = json.load(open(run_dir / "eval.json"))["checkpoints"]["best_loss_all_"]["full_context"]
        fc = fc_eval(rebuild_for_length(config, L, train.build_model), L, 1)
        diff = fc["bpc"] - ref["bpc"]
        out["sanity"] = {"window": L, "stride": 1, "bpc": fc["bpc"], "eval_json_bpc": ref["bpc"], "diff": diff,
                         "eval_json_batch_size": ref.get("batch_size"), "batch_size": fc["batch_size"], "pass_1e-4": abs(diff) <= 1e-4}
        print(f"  sanity: {fc['bpc']:.6f} vs eval.json {ref['bpc']:.6f} (diff {diff:+.2e})", flush=True)

    if args.stride1:
        out["stride1"] = {}
    for W in args.stride1:
        out["stride1"][str(W)] = fc_eval(rebuild(W), W, 1)
        json.dump(out, open(out_file, "w"), indent=1)

    for W in args.lengths:
        try:
            model = rebuild(W)
        except ExtrapolationNotSupported as e:
            out["lengths"][str(W)] = {"na": str(e)}
            print(f"  W={W}: N/A ({e})", flush=True)
            continue
        entry = {"pos_scale": float(getattr(model, "pos_scale", 1.0)),
                 "attn_window": int(getattr(model, "attn_window", 0)),
                 "self_extend_group": float(getattr(model, "self_extend_group", 1.0)),
                 "sliding": fc_eval(model, W, min(args.stride, W))}
        if args.interpolate:
            assert entry["pos_scale"] == interpolation_scale(config, W) or W == L
        if self_extend:
            assert entry["self_extend_group"] == self_extend_group(L, W, *self_extend)
        bs = auto_batch_size(config, W, 256, args.budget_gib)
        entry["per_position"] = per_position_loss(model, params, eval_int, W, n_chunks=args.n_chunks, seed=0, batch_size=bs)
        print(f"  W={W}: per-position BPC mean {entry['per_position']['bpc_mean']:.4f}, "
              f"buckets {[round(b, 3) for b in entry['per_position']['bpc_buckets']]}", flush=True)
        out["lengths"][str(W)] = entry
        json.dump(out, open(out_file, "w"), indent=1)
    out["peak_device_memory_gib"] = train.peak_memory_gib()
    json.dump(out, open(out_file, "w"), indent=1)
    print(f"Done: {out_file}")


if __name__ == "__main__":
    main()
