"""
Length extrapolation: evaluate a model trained on windows of length L (= model.max_len) at context lengths W > L.

The model is rebuilt with max_len = W and the trained parameters are loaded unchanged:
- 'rope', 'alibi', 'sinusoidal', 'none' (NoPE): no parameter depends on max_len, so this works as it is;
- 'relative': the trained table rel_embed (H, 2L-1) is kept and offsets above L-1 are clamped to L-1
  (T5-style, MultiHeadAttentionWithPE.rel_clamp);
- 'learned', 'hybrid': a learned table of L rows, so W > L is impossible (ExtrapolationNotSupported, reported as N/A);
- LSTM: runs on any length (no positional encoding).

Position interpolation (interpolate=True, 'rope' and 'sinusoidal' only, no fine-tuning): positions p are replaced by
p * L / W (DecoderOnlyTransformer.pos_scale), so a window of W characters maps onto the trained range [0, L).

Two metrics per (run, W):
1. sliding_window_eval: full_context_eval with window W and stride S (default 128): every target after the first
   window sees at least W - S + 1 characters of context.
2. per_position_loss: mean NLL at every position of n_chunks non-overlapping test chunks of W+1 characters at fixed
   offsets (seed 0), plus the mean over buckets of 64 positions.
"""

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np

from models.models import POS_SCALE_TYPES
from utils.config import Config
from utils.full_context_eval import full_context_eval

BUCKET = 64
MEMORY_BUDGET_GIB = 20.0


class ExtrapolationNotSupported(ValueError):
    """The model cannot be evaluated at this length (learned / hybrid tables beyond their training length)."""


def train_len(config):
    return int(config.model.max_len)


def length_config(config, window):
    """Copy of `config` with model.max_len = window."""
    c = Config(copy.deepcopy(config.to_dict()))
    c.model.max_len = int(window)
    return c


def interpolation_scale(config, window):
    """pos_scale that maps a window of `window` positions onto the training range: L / W (1.0 at W = L)."""
    return train_len(config) / int(window)


def rebuild_for_length(config, window, build_model, capture_attn=False, interpolate=False):
    """Model for windows of length `window` that loads the parameters trained under `config`.

    build_model is train.build_model (passed in to keep utils free of a train.py import).
    interpolate: position interpolation with pos_scale = L / window ('rope' and 'sinusoidal' only).
    """
    L = train_len(config)
    m = config.model
    if interpolate and (m.type != "transformer" or m.pos_encoding_type not in POS_SCALE_TYPES):
        raise ValueError(f"position interpolation applies to {POS_SCALE_TYPES} only, not to "
                         f"{m.type if m.type != 'transformer' else repr(m.pos_encoding_type)}")
    if m.type == "lstm":
        if capture_attn:
            raise ValueError("the LSTM has no attention to capture")
        return build_model(length_config(config, window))
    pe = m.pos_encoding_type
    if window > L and pe in ("learned", "hybrid"):
        raise ExtrapolationNotSupported(f"'{pe}' has a learned table of {L} positions; cannot evaluate at W={window}")
    opts = {"capture_attn": True} if capture_attn else {}
    if pe == "relative":
        opts.update(rel_table_len=L, rel_clamp=True)
    if interpolate and window != L:
        opts["pos_scale"] = interpolation_scale(config, window)
    return build_model(length_config(config, window), **opts)


def bytes_per_window(config, window):
    """Rough float32 activation bytes of one window in an inference forward pass (peak, with temporaries)."""
    m = config.model
    if m.type == "lstm":
        return 4 * window * (8 * m.hidden_size + 4 * m.vocab_size) * 2
    attn = m.n_heads * window * window * 3      # scores, mask select, softmax
    mlp = window * m.d_model * m.mlp_ratio * 2  # hidden + gelu
    rest = window * (m.d_model * 6 + m.vocab_size * 4)
    return 4 * (attn + mlp + rest) * 2


def auto_batch_size(config, window, requested, budget_gib=MEMORY_BUDGET_GIB):
    """`requested` windows per batch, lowered (power of two) until the estimate fits `budget_gib`."""
    b = int(requested)
    while b > 1 and b * bytes_per_window(config, window) > budget_gib * 2**30:
        b //= 2
    return b


def sliding_window_eval(model, params, test_int, window, stride=128, batch_size=2048, vocab_size=27,
                        rare_ids=None, common_ids=None, verbose=False):
    fc = full_context_eval(model, params, test_int, window=window, stride=stride, batch_size=batch_size,
                           vocab_size=vocab_size, rare_ids=rare_ids, common_ids=common_ids, verbose=verbose)
    assert fc["n_scored"] == len(test_int) - 1, (fc["n_scored"], len(test_int) - 1)
    return fc


def chunk_offsets(n_chars, window, n_chunks, seed=0):
    """Start indices of n_chunks non-overlapping chunks of window+1 characters (inputs and shifted targets)."""
    n_slots = n_chars // (window + 1)
    if n_slots < n_chunks:
        raise ValueError(f"only {n_slots} non-overlapping chunks of {window + 1} characters, need {n_chunks}")
    slots = np.sort(np.random.default_rng(seed).choice(n_slots, size=n_chunks, replace=False))
    return slots * (window + 1)


def per_position_loss(model, params, test_int, window, n_chunks=2048, seed=0, batch_size=256, bucket=BUCKET):
    offsets = chunk_offsets(len(test_int), window, n_chunks, seed)
    data = jnp.asarray(np.asarray(test_int), dtype=jnp.int32)
    pos = jnp.arange(window)

    @jax.jit
    def nll_sum(params, starts):
        idx = starts[:, None] + pos[None, :]
        x, y = data[idx], data[idx + 1]
        logp = jax.nn.log_softmax(model.apply({"params": params}, x, deterministic=True), axis=-1)
        return -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0].sum(0)  # (W,)

    batch_size = math.gcd(int(batch_size), int(n_chunks))  # equal batches, so the jitted function compiles once
    total = np.zeros(window, np.float64)
    for b in range(0, n_chunks, batch_size):
        total += np.asarray(nll_sum(params, jnp.asarray(offsets[b:b + batch_size], jnp.int32)), np.float64)
    mean = total / n_chunks
    n_b = int(np.ceil(window / bucket))
    buckets = [float(mean[i * bucket:(i + 1) * bucket].mean()) for i in range(n_b)]
    return {
        "window": int(window),
        "n_chunks": int(n_chunks),
        "seed": int(seed),
        "bucket": int(bucket),
        "nll_per_position": [float(v) for v in mean],
        "nll_buckets": buckets,
        "bpc_buckets": [v / np.log(2) for v in buckets],
        "bpc_mean": float(mean.mean() / np.log(2)),
    }
