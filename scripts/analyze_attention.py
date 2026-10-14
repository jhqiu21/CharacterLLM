"""
Attention analysis of a trained Transformer run on --n-windows random test windows of length W (seed 0).

    python scripts/analyze_attention.py --run TR-final_s42 --length 128

Post-softmax attention weights are captured with the model's capture_attn flag (sow into 'intermediates');
W > training length uses utils/extrapolation_eval.rebuild_for_length (relative: clamped offsets; learned/hybrid: N/A).
For each layer and head, with distance d = i - j (query i, key j <= i):
- distance profile: mean weight at distance d, averaged over windows and queries i >= d;
- entropy (nats) of each query's attention row, mean over queries;
- local mass: attention on the last 8 / 32 characters (d < 8 / d < 32), mean over queries;
- effective span: smallest d whose cumulative mass over distances 0..d reaches 90%; mean and median over queries;
- space mass: weight on the most recent space strictly before the query (word boundary), mean over queries that
  have one, with the uniform-attention value 1/(i+1) as a reference.
Query-level statistics are also given for queries at positions >= training length (past_train_len) when W is larger.
--interpolate applies position interpolation (rope / sinusoidal, pos_scale = L / W) and writes to attention_W<W>_pi/.
Writes runs/<run>/attention_W<W>/stats.json, example.npz (all heads for window 0, float16) and plots
(distance profiles per layer x head, two example heatmaps).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import train  # noqa: E402
from eval_extrapolation import load_run  # noqa: E402
from utils.extrapolation_eval import rebuild_for_length, train_len  # noqa: E402

SPACE = train.CHAR_TO_INT[" "]
SPAN_MASS = 0.9


def window_starts(n_chars, W, n, seed=0):
    return np.sort(np.random.default_rng(seed).integers(0, n_chars - W - 1, size=n))


def make_stats_fn(model, W):
    i = jnp.arange(W)[:, None]
    d = jnp.arange(W)[None, :]
    src = jnp.clip(i - d, 0, W - 1)  # key index for (query i, distance d)
    valid = d <= i

    @jax.jit
    def stats(params, x):
        _, inter = model.apply({"params": params}, x, deterministic=True, mutable=["intermediates"])
        blocks = inter["intermediates"]
        w = jnp.stack([blocks[f"blocks_{k}"]["MultiHeadAttentionWithPE_0"]["attn_weights"][0]
                       for k in range(len(blocks))])  # (Lyr, B, H, W, W)
        D = jnp.where(valid, jnp.take_along_axis(w, jnp.broadcast_to(src, w.shape), axis=-1), 0.0)  # (Lyr,B,H,W_q,W_d)
        cum = jnp.cumsum(D, axis=-1)
        span = jnp.argmax(cum >= SPAN_MASS - 1e-6, axis=-1)  # (Lyr,B,H,W_q)
        ent = -jnp.sum(jnp.where(w > 0, w * jnp.log(jnp.maximum(w, 1e-30)), 0.0), axis=-1)
        # most recent space strictly before each query
        is_space = x == SPACE  # (B, W)
        pos = jnp.where(is_space, jnp.arange(W)[None], -1)
        last_space_upto = jax.lax.cummax(pos, axis=1)  # last space at or before j
        prev_space = jnp.concatenate([jnp.full((x.shape[0], 1), -1), last_space_upto[:, :-1]], axis=1)  # strictly before i
        has = prev_space >= 0
        sp_w = jnp.take_along_axis(w, jnp.maximum(prev_space, 0)[None, :, None, :, None], axis=-1)[..., 0]  # (Lyr,B,H,W_q)
        span_hist = jax.nn.one_hot(span, W, dtype=jnp.int32).sum(1)  # (Lyr,H,W_q,W_span)
        return {
            "profile_sum": D.sum((1, 3)),             # (Lyr,H,W_d)
            "ent_q": ent.sum(1),                      # (Lyr,H,W_q)
            "local8_q": D[..., :8].sum(-1).sum(1),
            "local32_q": D[..., :32].sum(-1).sum(1),
            "span_q": span.sum(1).astype(jnp.float32),
            "span_hist": span_hist,                   # (Lyr,H,W_q,W_span)
            "space_q": jnp.where(has[None, :, None, :], sp_w, 0.0).sum(1),
            "space_n_q": has.sum(0),                  # (W_q,)
            "uniform_space_q": jnp.where(has, 1.0 / (jnp.arange(W) + 1.0)[None], 0.0).sum(0),
        }

    return stats


def summarize(acc, n_windows, W, q_lo):
    """Per (layer, head) summary over queries q >= q_lo."""
    sl = slice(q_lo, W)
    nq = n_windows * (W - q_lo)
    hist = acc["span_hist"][:, :, sl].sum(2)  # (Lyr,H,W_span)
    cdf = np.cumsum(hist, -1) / hist.sum(-1, keepdims=True)
    n_sp = acc["space_n_q"][sl].sum()
    return {
        "queries_from": int(q_lo),
        "entropy": (acc["ent_q"][:, :, sl].sum(-1) / nq).tolist(),
        "local8": (acc["local8_q"][:, :, sl].sum(-1) / nq).tolist(),
        "local32": (acc["local32_q"][:, :, sl].sum(-1) / nq).tolist(),
        "span_mean": (acc["span_q"][:, :, sl].sum(-1) / nq).tolist(),
        "span_median": np.argmax(cdf >= 0.5, axis=-1).tolist(),
        "space_mass": (acc["space_q"][:, :, sl].sum(-1) / max(n_sp, 1)).tolist(),
        "space_mass_uniform": float(acc["uniform_space_q"][sl].sum() / max(n_sp, 1)),
    }


def plot_profiles(profile, path, title):
    n_l, n_h, W = profile.shape
    fig, axes = plt.subplots(n_l, n_h, figsize=(2.6 * n_h, 2.0 * n_l), sharex=True, sharey=True, squeeze=False)
    for li in range(n_l):
        for h in range(n_h):
            ax = axes[li, h]
            ax.plot(np.arange(W), np.maximum(profile[li, h], 1e-8), lw=1)
            ax.set_yscale("log")
            if li == 0:
                ax.set_title(f"head {h}", fontsize=9)
            if h == 0:
                ax.set_ylabel(f"layer {li}\nmean weight", fontsize=8)
            if li == n_l - 1:
                ax.set_xlabel("distance i - j", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=130)
    plt.close(fig)


def plot_heatmaps(example, picks, path, title):
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.4), squeeze=False)
    for ax, (li, h, label) in zip(axes[0], picks):
        a = example[li, h].astype(np.float32)
        im = ax.imshow(np.log10(np.maximum(a, 1e-6)), cmap="viridis", vmin=-6, vmax=0, interpolation="nearest")
        ax.set_title(f"layer {li} head {h} ({label})", fontsize=9)
        ax.set_xlabel("key j")
        ax.set_ylabel("query i")
    fig.colorbar(im, ax=axes[0].tolist(), label="log10 weight", shrink=0.85)
    fig.suptitle(title, fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--n-windows", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=0, help="windows per batch (default: 128 * (128/W)^2, at least 4)")
    ap.add_argument("--interpolate", action="store_true", help="position interpolation, pos_scale = L / W (rope, sinusoidal)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)

    run_dir, config, ckpt = load_run(args.run)
    if config.model.type != "transformer":
        sys.exit("attention analysis needs a Transformer run")
    params, W, L = ckpt["params"], args.length, train_len(config)
    model = rebuild_for_length(config, W, train.build_model, capture_attn=True, interpolate=args.interpolate)
    _, _, test_int = train.load_data(args.data_dir)
    starts = window_starts(len(test_int), W, args.n_windows)
    bs = args.batch_size or max(4, int(128 * (128 / W) ** 2))
    while args.n_windows % bs:
        bs //= 2
    stats_fn = make_stats_fn(model, W)
    data = np.asarray(test_int, np.int32)
    acc = None
    for b in range(0, args.n_windows, bs):
        x = jnp.asarray(np.stack([data[s:s + W] for s in starts[b:b + bs]]))
        out = jax.device_get(stats_fn(params, x))
        out = {k: np.asarray(v, np.float64) for k, v in out.items()}
        acc = out if acc is None else {k: acc[k] + out[k] for k in acc}
    n_valid = args.n_windows * (W - np.arange(W))
    profile = acc["profile_sum"] / n_valid  # (Lyr,H,W)

    res = {"run": args.run, "pos_encoding_type": config.model.pos_encoding_type, "length": W, "train_len": L,
           "interpolate": bool(args.interpolate), "pos_scale": L / W if args.interpolate else 1.0,
           "n_windows": args.n_windows, "seed": 0, "step": int(ckpt["step"]), "span_mass": SPAN_MASS,
           "git_commit": train.git_commit(), "device": jax.devices()[0].device_kind,
           "n_layers": int(profile.shape[0]), "n_heads": int(profile.shape[1]),
           "distance_profile": profile.tolist(), "all_queries": summarize(acc, args.n_windows, W, 0)}
    if W > L:
        res["past_train_len"] = summarize(acc, args.n_windows, W, L)
    out_dir = run_dir / (f"attention_W{W}_pi" if args.interpolate else f"attention_W{W}")
    out_dir.mkdir(exist_ok=True)
    json.dump(res, open(out_dir / "stats.json", "w"), indent=1)

    # window 0, every layer and head
    _, inter = model.apply({"params": params}, jnp.asarray(data[starts[0]:starts[0] + W])[None], deterministic=True,
                           mutable=["intermediates"])
    ex = np.stack([np.asarray(inter["intermediates"][f"blocks_{k}"]["MultiHeadAttentionWithPE_0"]["attn_weights"][0][0])
                   for k in range(res["n_layers"])])  # (Lyr,H,W,W)
    np.savez_compressed(out_dir / "example.npz", attn=ex.astype(np.float16), start=starts[0],
                        text="".join(train.INT_TO_CHAR[int(c)] for c in data[starts[0]:starts[0] + W]))

    tag = f"{args.run} ({res['pos_encoding_type']}{' + PI' if args.interpolate else ''}), W={W}"
    plot_profiles(profile, out_dir / "distance_profiles", f"Distance profiles: {tag}")
    span = np.asarray(res["all_queries"]["span_mean"])
    lo, hi = np.unravel_index(span.argmin(), span.shape), np.unravel_index(span.argmax(), span.shape)
    plot_heatmaps(ex, [(lo[0], lo[1], f"shortest span {span[lo]:.1f}"), (hi[0], hi[1], f"longest span {span[hi]:.1f}")],
                  out_dir / "heatmaps", f"Attention, test window 0: {tag}")
    print(f"{tag}: mean span per layer {np.round(span.mean(1), 1).tolist()}, "
          f"local8 per layer {np.round(np.asarray(res['all_queries']['local8']).mean(1), 3).tolist()} -> {out_dir}")


if __name__ == "__main__":
    main()
