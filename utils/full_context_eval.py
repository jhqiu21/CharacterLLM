"""
Sliding-window ("full context") test-set evaluation.

With window W (= model.max_len) and stride S, every target character 1..N-1 of the
N-character test stream is scored exactly once:

- window 0 has inputs data[0:W] and scores all W of its targets (1..W);
- window k >= 1 ends at target e_k = min(W + k*S, N-1), has inputs data[e_k-W:e_k]
  and scores only the targets it adds, (e_{k-1}, e_k] -- its last S positions
  (fewer for the final window).

So every target after the first W sees at least W-S+1 characters of context. Each window
is an independent forward pass, so the LSTM also starts from a zero state in every window
and both model types see the same context.

Windows are built on device from index arithmetic, batch by batch; the last batch is
padded to the fixed shape (padded windows score nothing), so the forward pass compiles once.
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np


def num_windows(n_chars, window, stride):
    if n_chars - 1 < window:
        raise ValueError(f"need at least window+1={window + 1} characters, got {n_chars}")
    return 1 + math.ceil((n_chars - 1 - window) / stride)


def window_bounds(k, n_chars, window, stride, xp=np):
    """For window indices k, return (end, n_new).

    end:   index of the last target the window predicts (its inputs are data[end-window:end]).
    n_new: how many of its last targets are scored. Windows past the end get n_new == 0.
    """
    end = xp.minimum(window + k * stride, n_chars - 1)
    prev_end = xp.where(k == 0, 0, xp.minimum(window + (k - 1) * stride, n_chars - 1))
    return end, end - prev_end


def scored_target_indices(n_chars, window, stride):
    """Target indices in scoring order (for tests: must be exactly 1..N-1, each once)."""
    k = np.arange(num_windows(n_chars, window, stride))
    end, n_new = window_bounds(k, n_chars, window, stride)
    return np.concatenate([np.arange(e - n + 1, e + 1) for e, n in zip(end, n_new)])


def frequency_groups(train_int, vocab_size, int_to_char, top_percent=0.2):
    """Rare / common characters: bottom / top `top_percent` of the vocabulary by training-corpus frequency.

    Uses the same group size as utils.eval.token_frequency_analysis: max(1, int(V * top_percent)).
    """
    counts = np.bincount(np.asarray(train_int), minlength=vocab_size)
    n_top = max(1, int(vocab_size * top_percent))
    order = np.argsort(counts, kind="stable")  # ascending: rare -> common
    rare, common = order[:n_top], order[-n_top:]
    total = counts.sum()

    def describe(ids):
        return [{"id": int(i), "char": int_to_char[int(i)], "train_freq": float(counts[i] / total)} for i in ids]

    return {
        "rare_ids": [int(i) for i in rare],
        "common_ids": [int(i) for i in common],
        "rare": describe(rare),
        "common": describe(common),
        "top_percent": top_percent,
        "n_per_group": n_top,
    }


def make_eval_batch_fn(model, n_chars, window, stride, batch_size, vocab_size, n_bins=20):
    positions = jnp.arange(window)
    # Same binning as utils.eval.expected_calibration_error.
    inner_edges = jnp.linspace(0, 1, n_bins + 1)[1:-1]

    @jax.jit
    def eval_batch(params, data, k0):
        k = k0 + jnp.arange(batch_size)
        end, n_new = window_bounds(k, n_chars, window, stride, xp=jnp)
        idx = (end - window)[:, None] + positions[None, :]  # (B, W)
        x, y = data[idx], data[idx + 1]
        logits = model.apply({"params": params}, x, deterministic=True)

        scored = positions[None, :] >= (window - n_new)[:, None]  # (B, W)
        m = scored.astype(jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        conf = jnp.exp(jnp.max(logp, axis=-1))
        correct = (jnp.argmax(logits, axis=-1) == y).astype(jnp.float32)

        bins = jnp.digitize(conf, inner_edges).reshape(-1)
        mf, cf, conf_f, yf = m.reshape(-1), correct.reshape(-1), conf.reshape(-1), y.reshape(-1)
        return {
            "nll_sum": jnp.sum(nll * m),
            "n_scored": jnp.sum(scored.astype(jnp.int32)),
            "n_correct": jnp.sum((correct * m).astype(jnp.int32)),
            "bin_count": jnp.zeros(n_bins, jnp.int32).at[bins].add(scored.reshape(-1).astype(jnp.int32)),
            "bin_conf_sum": jnp.zeros(n_bins, jnp.float32).at[bins].add(conf_f * mf),
            "bin_correct_sum": jnp.zeros(n_bins, jnp.int32).at[bins].add((cf * mf).astype(jnp.int32)),
            "class_count": jnp.zeros(vocab_size, jnp.int32).at[yf].add(scored.reshape(-1).astype(jnp.int32)),
            "class_correct": jnp.zeros(vocab_size, jnp.int32).at[yf].add((cf * mf).astype(jnp.int32)),
        }

    return eval_batch


def full_context_eval(model, params, data_int, window, stride=1, batch_size=4096, vocab_size=27,
                      rare_ids=None, common_ids=None, n_bins=20, max_windows=None, verbose=True):
    """Score every target of `data_int` once with sliding windows; see the module docstring.

    `max_windows` stops after the batches holding the first `max_windows` windows; the metrics
    are then partial, so use it for timing benchmarks only. Returns a dict of plain Python numbers.
    """
    n_chars = len(data_int)
    n_win = num_windows(n_chars, window, stride)
    if max_windows is not None:
        n_win = min(n_win, max_windows)
    n_batches = math.ceil(n_win / batch_size)
    eval_batch = make_eval_batch_fn(model, n_chars, window, stride, batch_size, vocab_size, n_bins)
    data = jnp.asarray(np.asarray(data_int), dtype=jnp.int32)

    t0 = time.time()
    totals = None
    pending = []

    def drain():
        nonlocal totals
        for out in jax.device_get(pending):
            out = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
            totals = out if totals is None else {k: totals[k] + out[k] for k in totals}
        pending.clear()

    for b in range(n_batches):
        pending.append(eval_batch(params, data, jnp.int32(b * batch_size)))
        if len(pending) >= 64:
            drain()
        if verbose and (b + 1) % max(1, n_batches // 10) == 0:
            print(f"\t full-context eval: batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s)", flush=True)
    drain()
    elapsed = time.time() - t0

    n = totals["n_scored"]
    mean_nll = totals["nll_sum"] / n
    ece = float(np.sum(np.abs(totals["bin_conf_sum"] - totals["bin_correct_sum"])) / n)

    def group_acc(ids):
        if ids is None:
            return None
        ids = list(ids)
        return float(totals["class_correct"][ids].sum() / totals["class_count"][ids].sum())

    return {
        "bpc": float(mean_nll / np.log(2)),
        "loss": float(mean_nll),
        "perplexity": float(np.exp(mean_nll)),
        "acc": float(totals["n_correct"] / n),
        "ece": ece,
        "acc_rare": group_acc(rare_ids),
        "acc_common": group_acc(common_ids),
        "n_scored": int(n),
        "n_targets_expected": int(n_chars - 1) if max_windows is None else None,
        "window": int(window),
        "stride": int(stride),
        "min_context_after_first_window": int(window - stride + 1),
        "n_windows": int(n_win),
        "batch_size": int(batch_size),
        "seconds": float(elapsed),
        "ece_bins": {
            "n_bins": n_bins,
            "count": [int(c) for c in totals["bin_count"]],
            "conf_sum": [float(c) for c in totals["bin_conf_sum"]],
            "correct_sum": [int(c) for c in totals["bin_correct_sum"]],
        },
        "class_count": [int(c) for c in totals["class_count"]],
        "class_correct": [int(c) for c in totals["class_correct"]],
    }
