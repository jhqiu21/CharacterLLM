"""NoPE (pos_encoding_type 'none') and evaluation-time position interpolation (pos_scale)."""

from pathlib import Path

import jax
import numpy as np
import pytest

import train
from helpers import PE_TYPES, make_config, model_and_params
from models.positional_encodings import get_sinusoidal_embeddings, precompute_freqs_cis
from utils.extrapolation_eval import interpolation_scale, rebuild_for_length, sliding_window_eval

PAPER = dict(d_model=256, n_heads=4, n_layers=4, max_len=128)
L = 16  # training length of the tiny models (stands for 128)
REFERENCE = Path(__file__).resolve().parent / "data" / "pe_reference_logits.npz"


def _tokens(n, seed=0):
    return np.asarray(jax.random.randint(jax.random.PRNGKey(seed), (1, n), 0, 27), dtype=np.int32)


def _logits(model, params, x):
    return np.asarray(model.apply({"params": params}, x, deterministic=True))


# ---------------------------------------------------------------------------------------------- NoPE

def test_nope_parameter_count_equals_rope():
    _, nope = model_and_params(make_config(pos_encoding_type="none", **PAPER))
    _, rope = model_and_params(make_config(pos_encoding_type="rope", **PAPER))
    assert train.count_params(nope) == train.count_params(rope) == 3_169_280
    assert jax.tree_util.tree_map(np.shape, nope) == jax.tree_util.tree_map(np.shape, rope)


def test_final_config_with_nope_has_rope_parameter_count():
    config = train.resolve_config(train.load_config(train.REPO_ROOT / "configs" / "final" / "TR.yaml"))
    config.model.pos_encoding_type = "none"
    _, params = model_and_params(config)
    assert train.count_params(params) == 3_169_280


def test_nope_is_causal():
    model, params = model_and_params(make_config(pos_encoding_type="none", dropout=0.0, max_len=L))
    x = _tokens(L)
    y = x.copy()
    y[0, 10] = (y[0, 10] + 1) % 27
    a, b = _logits(model, params, x), _logits(model, params, y)
    np.testing.assert_array_equal(a[:, :10], b[:, :10])
    assert not np.allclose(a[:, 10:], b[:, 10:])


def test_nope_longer_max_len_keeps_first_positions():
    config = make_config(pos_encoding_type="none", dropout=0.0, max_len=128, d_model=32, n_heads=2)
    model, params = model_and_params(config)
    long = rebuild_for_length(config, 512, train.build_model)
    x = _tokens(512)
    np.testing.assert_allclose(_logits(long, params, x)[:, :128], _logits(model, params, x[:, :128]), rtol=1e-5, atol=1e-5)


def test_nope_attention_has_no_position_signal():
    """Without positional information the first layer cannot tell repeated tokens apart: a constant input gives the same
    logits at every position."""
    model, params = model_and_params(make_config(pos_encoding_type="none", dropout=0.0, max_len=L))
    out = _logits(model, params, np.full((1, L), 3, np.int32))
    np.testing.assert_allclose(out[0], np.repeat(out[0, :1], L, 0), rtol=1e-5, atol=1e-6)


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="Unknown pos_encoding_type"):
        model_and_params(make_config(pos_encoding_type="nope"))


# ------------------------------------------------------------------- existing encodings are unchanged

@pytest.mark.parametrize("key", ["learned", "sinusoidal_unit", "sinusoidal_raw", "hybrid_unit", "hybrid_raw", "rope", "alibi", "relative"])
def test_existing_encodings_bit_identical_to_reference(key):
    """tests/data/pe_reference_logits.npz was written by the code before NoPE and pos_scale were added (commit fc314d4)."""
    ref = np.load(REFERENCE)
    pe, _, scale = key.partition("_")
    kw = {"sinusoidal_scale": scale} if scale else {}
    model, params = model_and_params(make_config(pos_encoding_type=pe, dropout=0.0, max_len=16, **kw))
    np.testing.assert_array_equal(_logits(model, params, ref["tokens"]), ref[key])


@pytest.mark.parametrize("pe", ["rope", "sinusoidal"])
def test_pos_scale_one_is_bit_identical(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=L)
    model, params = model_and_params(config)
    explicit = train.build_model(config, pos_scale=1.0)
    x = _tokens(L)
    np.testing.assert_array_equal(_logits(model, params, x), _logits(explicit, params, x))
    assert model.pos_scale == 1.0


# ---------------------------------------------------------------------------------- position interpolation

def test_scaled_rope_angles_stay_in_trained_range():
    W = 256
    s = interpolation_scale(make_config(max_len=128), W)
    assert s == 0.5
    positions = np.arange(W) * s
    assert positions.max() < 128
    freqs = np.asarray(precompute_freqs_cis(64, W, pos_scale=s))
    trained = np.asarray(precompute_freqs_cis(64, 128))
    np.testing.assert_allclose(freqs[::2], trained, rtol=1e-6, atol=1e-6)  # even positions = trained integer positions
    assert np.all(np.abs(np.angle(freqs[:, 0])) <= np.pi)


def test_scaled_sinusoidal_evaluates_fractional_positions():
    W = 256
    table = np.asarray(get_sinusoidal_embeddings(W, 32, pos_scale=128 / W))
    trained = np.asarray(get_sinusoidal_embeddings(128, 32))
    np.testing.assert_allclose(table[::2], trained, rtol=1e-6, atol=1e-6)
    p = 0.5  # position 1 maps to 0.5
    div = np.exp(np.arange(0, 32, 2) * -(np.log(10000.0) / 32))
    np.testing.assert_allclose(table[1, 0::2], np.sin(p * div), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("pe", ["rope", "sinusoidal"])
def test_interpolated_model_runs_and_differs_beyond_training_length(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=L)
    _, params = model_and_params(config)
    W = 4 * L
    plain = rebuild_for_length(config, W, train.build_model)
    pi = rebuild_for_length(config, W, train.build_model, interpolate=True)
    assert pi.pos_scale == L / W and plain.pos_scale == 1.0
    x = _tokens(W)
    a, b = _logits(plain, params, x), _logits(pi, params, x)
    assert np.isfinite(b).all() and not np.allclose(a, b)


@pytest.mark.parametrize("pe", ["rope", "sinusoidal"])
def test_interpolate_at_training_length_equals_plain(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=L)
    _, params = model_and_params(config)
    data = np.asarray(jax.random.randint(jax.random.PRNGKey(5), (300,), 0, 27), dtype=np.uint8)
    plain = sliding_window_eval(rebuild_for_length(config, L, train.build_model), params, data, window=L, stride=L, batch_size=8)
    pi = sliding_window_eval(rebuild_for_length(config, L, train.build_model, interpolate=True), params, data, window=L,
                             stride=L, batch_size=8)
    assert pi["bpc"] == plain["bpc"]


@pytest.mark.parametrize("pe", [p for p in PE_TYPES + ["none"] if p not in ("rope", "sinusoidal")])
def test_interpolation_rejected_for_other_encodings(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=L)
    with pytest.raises(ValueError, match="position interpolation"):
        rebuild_for_length(config, 2 * L, train.build_model, interpolate=True)
    with pytest.raises(ValueError, match="position interpolation"):
        model_and_params_with(config, pos_scale=0.5)


def test_interpolation_rejected_for_lstm():
    with pytest.raises(ValueError, match="position interpolation"):
        rebuild_for_length(make_config("lstm", max_len=L), 2 * L, train.build_model, interpolate=True)


def model_and_params_with(config, **eval_options):
    model = train.build_model(config, **eval_options)
    return model, train.init_params(model, config, jax.random.PRNGKey(0))
