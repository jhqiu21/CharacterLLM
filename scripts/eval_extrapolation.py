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
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)

    run_dir, config, ckpt = load_run(args.run)
    out_file = run_dir / ("extrapolation_pi.json" if args.interpolate else "extrapolation.json")
    params = ckpt["params"]
    train_int, _, test_int = train.load_data(args.data_dir)
    groups = frequency_groups(train_int, config.model.vocab_size, train.INT_TO_CHAR)
    L = train_len(config)
    out = {"run": args.run, "checkpoint": "best_loss_all_", "step": int(ckpt["step"]), "train_len": L,
           "model_type": config.model.type, "pos_encoding_type": getattr(config.model, "pos_encoding_type", None),
           "stride": args.stride, "n_chunks": args.n_chunks, "device": jax.devices()[0].device_kind,
           "git_commit": train.git_commit(), "interpolate": bool(args.interpolate), "lengths": {}}

    def fc_eval(model, W, stride):
        bs = auto_batch_size(config, W, args.batch_size, args.budget_gib)
        t0 = time.time()
        fc = sliding_window_eval(model, params, test_int, W, stride=stride, batch_size=bs,
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

    for W in args.lengths:
        try:
            model = rebuild_for_length(config, W, train.build_model, interpolate=args.interpolate)
        except ExtrapolationNotSupported as e:
            out["lengths"][str(W)] = {"na": str(e)}
            print(f"  W={W}: N/A ({e})", flush=True)
            continue
        entry = {"pos_scale": interpolation_scale(config, W) if args.interpolate else 1.0,
                 "sliding": fc_eval(model, W, min(args.stride, W))}
        bs = auto_batch_size(config, W, 256, args.budget_gib)
        entry["per_position"] = per_position_loss(model, params, test_int, W, n_chunks=args.n_chunks, seed=0, batch_size=bs)
        print(f"  W={W}: per-position BPC mean {entry['per_position']['bpc_mean']:.4f}, "
              f"buckets {[round(b, 3) for b in entry['per_position']['bpc_buckets']]}", flush=True)
        out["lengths"][str(W)] = entry
        json.dump(out, open(out_file, "w"), indent=1)
    out["peak_device_memory_gib"] = train.peak_memory_gib()
    json.dump(out, open(out_file, "w"), indent=1)
    print(f"Done: {out_file}")


if __name__ == "__main__":
    main()
