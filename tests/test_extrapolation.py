import jax
import numpy as np
import pytest

import train
from helpers import ALL_PE_TYPES, make_config, model_and_params
from utils.extrapolation_eval import (
    ExtrapolationNotSupported,
    auto_batch_size,
    chunk_offsets,
    per_position_loss,
    rebuild_for_length,
    sliding_window_eval,
)
from utils.full_context_eval import full_context_eval

L = 16  # training length of the tiny models (stands for 128)
EXTRAPOLATING = ["rope", "alibi", "sinusoidal", "relative", "none"]


def _tokens(n, seed=0):
    return np.asarray(jax.random.randint(jax.random.PRNGKey(seed), (1, n), 0, 27), dtype=np.int32)


def _setup(kind):
    config = make_config("lstm", dropout=0.0, max_len=L) if kind == "lstm" else make_config(pos_encoding_type=kind, dropout=0.0, max_len=L)
    model, params = model_and_params(config)
    return config, model, params


@pytest.mark.parametrize("kind", ALL_PE_TYPES + ["lstm"])
def test_rebuild_at_training_length_is_identical(kind):
    config, model, params = _setup(kind)
    x = _tokens(L)
    rebuilt = rebuild_for_length(config, L, train.build_model)
    np.testing.assert_array_equal(np.asarray(model.apply({"params": params}, x, deterministic=True)),
                                  np.asarray(rebuilt.apply({"params": params}, x, deterministic=True)))


@pytest.mark.parametrize("kind", EXTRAPOLATING + ["lstm"])
def test_rebuild_longer_keeps_first_positions(kind):
    config, model, params = _setup(kind)
    rebuilt = rebuild_for_length(config, 4 * L, train.build_model)
    # same parameter tree: the trained parameters load unchanged
    init_long = train.init_params(rebuilt, train.resolve_config(config), jax.random.PRNGKey(1))
    assert jax.tree_util.tree_map(np.shape, init_long) == jax.tree_util.tree_map(np.shape, params)
    x = _tokens(4 * L)
    long_logits = np.asarray(rebuilt.apply({"params": params}, x, deterministic=True))
    short_logits = np.asarray(model.apply({"params": params}, x[:, :L], deterministic=True))
    assert long_logits.shape == (1, 4 * L, 27) and np.isfinite(long_logits).all()
    np.testing.assert_allclose(long_logits[:, :L], short_logits, rtol=1e-5, atol=1e-5)


def test_relative_clamp_identical_within_training_length():
    config, model, params = _setup("relative")
    clamped = train.build_model(config, rel_table_len=L, rel_clamp=True)
    for T in (5, L):
        x = _tokens(T)
        np.testing.assert_array_equal(np.asarray(model.apply({"params": params}, x, deterministic=True)),
                                      np.asarray(clamped.apply({"params": params}, x, deterministic=True)))


def test_relative_clamp_default_leaves_training_unchanged():
    config, model, params = _setup("relative")
    assert model.rel_clamp is False and model.rel_table_len == 0 and model.capture_attn is False
    explicit = train.build_model(config, rel_table_len=L, rel_clamp=False)
    x, y = _tokens(L), _tokens(L, seed=1)
    loss_fn = train.make_loss_and_metrics(config)

    def loss(m, p):
        return loss_fn(m.apply({"params": p}, x, deterministic=True), y)[0]

    g0, g1 = jax.grad(lambda p: loss(model, p))(params), jax.grad(lambda p: loss(explicit, p))(params)
    np.testing.assert_array_equal(np.asarray(loss(model, params)), np.asarray(loss(explicit, params)))
    jax.tree_util.tree_map(np.testing.assert_array_equal, g0, g1)


def test_relative_without_clamp_refuses_long_sequences():
    config, model, params = _setup("relative")
    with pytest.raises(ValueError, match="rel_clamp"):
        train.build_model(config, rel_table_len=L).apply({"params": params}, _tokens(2 * L), deterministic=True)


def test_relative_clamp_uses_last_offset_beyond_table():
    """Past the table every offset >= L-1 gets the bias of offset L-1. With q = 0 the scores are the bias alone,
    so in the last query row every key at offset >= L-1 gets the same weight and nearer keys differ."""
    config, model, params = _setup("relative")
    attn = dict(params["blocks_0"]["MultiHeadAttentionWithPE_0"])
    rel = np.asarray(attn["rel_bias"]["rel_embed"])
    attn["rel_bias"] = {"rel_embed": rel + np.arange(rel.shape[1])[None] * 0.01}
    attn["q_proj"] = {"kernel": np.zeros_like(attn["q_proj"]["kernel"])}
    params = dict(params)
    params["blocks_0"] = dict(params["blocks_0"], MultiHeadAttentionWithPE_0=attn)
    T = 3 * L
    rebuilt = rebuild_for_length(config, T, train.build_model, capture_attn=True)
    _, inter = rebuilt.apply({"params": params}, _tokens(T), deterministic=True, mutable=["intermediates"])
    (w,) = inter["intermediates"]["blocks_0"]["MultiHeadAttentionWithPE_0"]["attn_weights"]
    last = np.asarray(w)[0, :, T - 1]            # (H, T): keys j, offset T-1-j
    far = last[:, : T - (L - 1)]                 # offsets >= L-1
    np.testing.assert_allclose(far, far[:, :1].repeat(far.shape[1], 1), rtol=1e-6)
    assert not np.allclose(last[:, T - L + 1], last[:, T - 1])


@pytest.mark.parametrize("kind", ["learned", "hybrid"])
def test_learned_tables_refuse_longer_windows(kind):
    config, model, params = _setup(kind)
    with pytest.raises(ExtrapolationNotSupported, match="learned table"):
        rebuild_for_length(config, 2 * L, train.build_model)
    with pytest.raises(ValueError, match="learned table"):
        model.apply({"params": params}, _tokens(2 * L), deterministic=True)


@pytest.mark.parametrize("kind", ["rope", "lstm"])
def test_sliding_window_at_training_length_matches_full_context(kind):
    config, model, params = _setup(kind)
    data = np.asarray(jax.random.randint(jax.random.PRNGKey(3), (200,), 0, 27), dtype=np.uint8)
    ref = full_context_eval(model, params, data, window=L, stride=1, batch_size=64, verbose=False)
    rebuilt = rebuild_for_length(config, L, train.build_model)
    got = sliding_window_eval(rebuilt, params, data, window=L, stride=1, batch_size=32)
    assert got["bpc"] == pytest.approx(ref["bpc"], abs=1e-6)
    long = sliding_window_eval(rebuild_for_length(config, 2 * L, train.build_model), params, data, window=2 * L, stride=L)
    assert long["n_scored"] == len(data) - 1 and long["min_context_after_first_window"] == L + 1


def test_per_position_loss_matches_brute_force():
    config, model, params = _setup("alibi")
    data = np.asarray(jax.random.randint(jax.random.PRNGKey(4), (400,), 0, 27), dtype=np.uint8)
    W, n = 2 * L, 6
    model_w = rebuild_for_length(config, W, train.build_model)
    out = per_position_loss(model_w, params, data, W, n_chunks=n, seed=0, batch_size=4, bucket=8)
    offs = chunk_offsets(len(data), W, n, seed=0)
    assert len(set(offs)) == n and np.all(np.diff(offs) >= W + 1)
    x = np.stack([data[o:o + W] for o in offs]).astype(np.int32)
    y = np.stack([data[o + 1:o + W + 1] for o in offs]).astype(np.int32)
    logp = jax.nn.log_softmax(model_w.apply({"params": params}, x, deterministic=True), axis=-1)
    nll = -np.take_along_axis(np.asarray(logp), y[..., None], axis=-1)[..., 0].mean(0)
    np.testing.assert_allclose(out["nll_per_position"], nll, rtol=1e-5)
    assert len(out["nll_buckets"]) == W // 8
    np.testing.assert_allclose(out["nll_buckets"][0], nll[:8].mean(), rtol=1e-5)


def test_auto_batch_size_shrinks_for_long_windows():
    config = make_config(d_model=256, n_heads=4, max_len=128)
    b128 = auto_batch_size(config, 128, 2048)
    b1024 = auto_batch_size(config, 1024, 2048)
    assert b128 == 2048 and b1024 < b128 and b1024 >= 1
