"""Long-context options: local attention (attn_window), RoPE SelfExtend, model.pos_scale in training."""

import argparse
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

import train
from helpers import make_config, model_and_params
from models.models import MultiHeadAttentionWithPE
from models.positional_encodings import self_extend_positions
from utils.extrapolation_eval import interpolation_scale, rebuild_for_length, self_extend_group

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_extrapolation import output_name  # noqa: E402

L = 16


def _tokens(n, seed=0):
    return np.asarray(jax.random.randint(jax.random.PRNGKey(seed), (1, n), 0, 27), dtype=np.int32)


def _logits(model, params, x):
    return np.asarray(model.apply({"params": params}, x, deterministic=True))


# ---------------------------------------------------------------------------------------------- local attention

@pytest.mark.parametrize("pe", ["rope", "alibi", "none", "sinusoidal"])
def test_local_window_covering_the_sequence_is_identical(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=L)
    model, params = model_and_params(config)
    x = _tokens(L)
    np.testing.assert_array_equal(_logits(model, params, x), _logits(train.build_model(config, attn_window=L), params, x))


def test_local_window_one_layer_rope_equals_cropped_input():
    """With one layer and relative positions only (RoPE), position t with a local window w sees exactly x[t-w+1..t]."""
    w, T = 5, 3 * L
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=T, n_layers=1)
    model, params = model_and_params(config)
    local = train.build_model(config, attn_window=w)
    x = _tokens(T, seed=1)
    out = _logits(local, params, x)
    for t in (w - 1, 17, T - 1):
        ref = _logits(model, params, x[:, t - w + 1:t + 1])[:, -1]
        np.testing.assert_allclose(out[:, t], ref, rtol=1e-5, atol=1e-5)


def test_local_window_receptive_field_grows_with_layers():
    w, n_layers, T = 4, 2, 3 * L
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=T, n_layers=n_layers)
    _, params = model_and_params(config)
    local = train.build_model(config, attn_window=w)
    x = _tokens(T, seed=2)
    t, reach = 30, n_layers * (w - 1)  # position t depends on positions t - reach .. t
    far, near = x.copy(), x.copy()
    far[0, t - reach - 1] = (far[0, t - reach - 1] + 1) % 27
    near[0, t - reach] = (near[0, t - reach] + 1) % 27
    base = _logits(local, params, x)
    np.testing.assert_array_equal(_logits(local, params, far)[:, t], base[:, t])
    assert not np.allclose(_logits(local, params, near)[:, t], base[:, t])


# ---------------------------------------------------------------------------------------------- SelfExtend

def _relative_positions(T, window, group, mode):
    """Reference: SelfExtend's relative position of query i and key j, written from the formula directly."""
    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    d = i - j
    if mode == "floor":
        far = (i // group + window - window // group) - j // group
    else:
        far = window + (d - window) / group
    return np.where(d < window, d, far).astype(np.float64)


@pytest.mark.parametrize("mode,group", [("floor", 3), ("linear", 2.5)])
def test_self_extend_positions_give_the_selfextend_relative_positions(mode, group):
    T, window = 40, 6
    q_pos, k_pos = map(np.asarray, self_extend_positions(T, window, group, mode))
    rel = _relative_positions(T, window, group, mode)
    d = np.arange(T)[:, None] - np.arange(T)[None, :]
    far = d >= window
    np.testing.assert_allclose((q_pos[:, None] - k_pos[None, :])[far], rel[far], rtol=0, atol=1e-5)


def test_self_extend_group_keeps_positions_in_trained_range():
    assert self_extend_group(128, 128, 32, "floor") == 1 and self_extend_group(128, 64, 32, "linear") == 1
    assert self_extend_group(128, 512, 32, "floor") == 6
    assert self_extend_group(128, 1024, 32, "floor") == 11
    assert self_extend_group(128, 512, 16, "floor") == 5
    for W in (256, 512, 1024):
        for near in (16, 32, 64):
            for mode in ("floor", "linear"):
                g = self_extend_group(128, W, near, mode)
                rel = _relative_positions(W, near, g, mode)
                assert rel.max() <= 127 + 1e-9, (W, near, mode, g, rel.max())
                if mode == "floor" and g > 1:  # the smallest such group
                    assert _relative_positions(W, near, g - 1, mode).max() > 127
    with pytest.raises(ValueError):
        self_extend_group(128, 512, 127, "floor")


@pytest.mark.parametrize("mode,group", [("floor", 3), ("linear", 2.5)])
def test_self_extend_attention_matches_reference(mode, group):
    """Attention weights of the module against a numpy reference that rotates every (i, j) pair by its SelfExtend
    relative position: RoPE's score is Re(sum_m q_m conj(k_m) exp(1j * rel * f_m)) / sqrt(d) for complex pairs."""
    T, D, H, window = 24, 16, 2, 5
    mha = MultiHeadAttentionWithPE(d_model=D, n_heads=H, attn_dropout=0.0, pos_encoding_type="rope", max_len=T,
                                   capture_attn=True, self_extend_window=window, self_extend_group=group, self_extend_mode=mode)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, T, D))
    mask = np.tril(np.ones((T, T), bool))[None, None]
    variables = mha.init(jax.random.PRNGKey(1), x, mask=mask, deterministic=True)
    _, inter = mha.apply(variables, x, mask=mask, deterministic=True, mutable=["intermediates"])
    weights = np.asarray(inter["intermediates"]["attn_weights"][0])[0]  # (H, T, T)

    p = variables["params"]
    hd = D // H
    q = (np.asarray(x[0]) @ np.asarray(p["q_proj"]["kernel"])).reshape(T, H, hd)
    k = (np.asarray(x[0]) @ np.asarray(p["k_proj"]["kernel"])).reshape(T, H, hd)
    qc, kc = q[..., ::2] + 1j * q[..., 1::2], k[..., ::2] + 1j * k[..., 1::2]  # (T, H, hd/2)
    freqs = 1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))
    rel = _relative_positions(T, window, group, mode)
    scores = np.real(np.einsum("ihm,jhm,ijm->hij", qc, np.conj(kc), np.exp(1j * rel[..., None] * freqs))) / np.sqrt(hd)
    scores = np.where(mask[0], scores, -np.inf)
    ref = np.exp(scores - scores.max(-1, keepdims=True))
    ref /= ref.sum(-1, keepdims=True)
    np.testing.assert_allclose(weights, ref, rtol=1e-4, atol=1e-5)


def test_self_extend_window_covering_the_sequence_is_identical():
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=L)
    model, params = model_and_params(config)
    x = _tokens(L)
    se = train.build_model(config, self_extend_window=L, self_extend_group=4, self_extend_mode="floor")
    np.testing.assert_allclose(_logits(se, params, x), _logits(model, params, x), rtol=1e-6, atol=1e-6)


def test_self_extend_rebuild():
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=L)
    _, params = model_and_params(config)
    at_L = rebuild_for_length(config, L, train.build_model, self_extend=(4, "floor"))
    assert at_L.self_extend_window == 0  # group 1: the plain model
    long = rebuild_for_length(config, 4 * L, train.build_model, self_extend=(4, "floor"))
    assert long.self_extend_window == 4 and long.self_extend_group == self_extend_group(L, 4 * L, 4, "floor") > 1
    plain = rebuild_for_length(config, 4 * L, train.build_model)
    x = _tokens(4 * L)
    a, b = _logits(plain, params, x), _logits(long, params, x)
    assert np.isfinite(b).all() and not np.allclose(a, b)
    np.testing.assert_allclose(b[:, :4], a[:, :4], rtol=1e-5, atol=1e-5)  # every pair is in the near band


@pytest.mark.parametrize("pe", ["alibi", "sinusoidal", "none"])
def test_self_extend_rejected_for_other_encodings(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=L)
    with pytest.raises(ValueError, match="self"):
        rebuild_for_length(config, 2 * L, train.build_model, self_extend=(4, "floor"))
    with pytest.raises(ValueError, match="self_extend"):
        model_and_params_with(config, self_extend_window=4, self_extend_group=2)


def test_self_extend_rejected_with_interpolation():
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=L)
    with pytest.raises(ValueError, match="self-extend"):
        rebuild_for_length(config, 2 * L, train.build_model, interpolate=True, self_extend=(4, "floor"))


def model_and_params_with(config, **eval_options):
    model = train.build_model(config, **eval_options)
    return model, train.init_params(model, config, jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------------------------- model.pos_scale

def test_model_pos_scale_defaults_to_one_and_is_used_by_build_model():
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=L)
    assert config.model.pos_scale == 1.0 and train.build_model(config).pos_scale == 1.0
    config.model.pos_scale = 0.25
    assert train.build_model(config).pos_scale == 0.25
    assert train.build_model(config, pos_scale=0.5).pos_scale == 0.5  # an eval option overrides it


def test_pi_trained_run_keeps_its_scale_and_composes_interpolation():
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=4 * L, pos_scale=0.25)
    _, params = model_and_params(config)
    assert rebuild_for_length(config, L, train.build_model).pos_scale == 0.25
    assert interpolation_scale(config, 8 * L) == 0.125
    assert rebuild_for_length(config, 8 * L, train.build_model, interpolate=True).pos_scale == 0.125
    # the first L positions of a PI-trained model are the same in a window of L and in its training window
    x = _tokens(4 * L)
    short = rebuild_for_length(config, L, train.build_model)
    full = rebuild_for_length(config, 4 * L, train.build_model)
    np.testing.assert_allclose(_logits(short, params, x[:, :L]), _logits(full, params, x)[:, :L], rtol=1e-5, atol=1e-5)


def test_set_model_pos_scale():
    d, overrides = train.build_run_config(train.parse_args(
        ["--config", str(train.REPO_ROOT / "configs" / "final" / "TR.yaml"), "--seed", "1", "--set", "model.pos_scale=0.25"]))
    assert d["model"]["pos_scale"] == 0.25 and overrides == {"model.pos_scale": 0.25}


def test_older_configs_get_pos_scale_one():
    config = train.resolve_config(train.load_config(train.REPO_ROOT / "configs" / "final" / "TR.yaml"))
    assert config.model.pos_scale == 1.0


# ---------------------------------------------------------------------------------------------- output names

@pytest.mark.parametrize("kw,name", [
    ({}, "extrapolation.json"),
    ({"interpolate": True}, "extrapolation_pi.json"),
    ({"local_window": 64}, "extrapolation_local64.json"),
    ({"self_extend": 32, "self_extend_mode": "linear", "split": "val"}, "extrapolation_selinear32_val.json"),
    ({"tag": "v6"}, "extrapolation_v6.json"),
    ({"split": "val", "tag": "v6"}, "extrapolation_val_v6.json"),
])
def test_output_names(kw, name):
    args = argparse.Namespace(**{"interpolate": False, "local_window": 0, "self_extend": 0, "self_extend_mode": "floor",
                                 "split": "test", "tag": "", **kw})
    assert output_name(args) == name


def test_apply_chunked_matches_one_pass():
    config = make_config(pos_encoding_type="rope", dropout=0.0, max_len=32)
    model, params = model_and_params(config)
    apply_fn = train.make_apply_fn(model)
    x = np.asarray(jax.random.randint(jax.random.PRNGKey(3), (12, 32), 0, 27), dtype=np.int32)
    one = np.asarray(apply_fn(params, x))
    np.testing.assert_array_equal(np.asarray(train.apply_chunked(apply_fn, params, x, budget=12 * 32 * 32)), one)
    np.testing.assert_allclose(np.asarray(train.apply_chunked(apply_fn, params, x, budget=4 * 32 * 32)), one, rtol=1e-5, atol=1e-5)
