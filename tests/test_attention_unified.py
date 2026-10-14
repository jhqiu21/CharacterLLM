import flax.linen as nn
import jax
import numpy as np
import pytest
from flax.linen import attention as attn

import train
from helpers import ALL_PE_TYPES, make_config, model_and_params
from models.models import MultiHeadAttentionWithPE

PAPER = dict(d_model=256, n_heads=4, n_layers=4, max_len=128)
D, H, T, B = 32, 4, 12, 3


def _path_strings(params):
    return ["/".join(train._path_names(path)) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]


@pytest.mark.parametrize("pe", ALL_PE_TYPES)
def test_no_flax_self_attention_in_any_parameter_tree(pe):
    _, params = model_and_params(make_config("transformer", pos_encoding_type=pe))
    paths = _path_strings(params)
    assert not any("SelfAttention" in p for p in paths), paths
    assert sum("MultiHeadAttentionWithPE" in p and p.endswith("q_proj/kernel") for p in paths) == 2  # one per layer


@pytest.mark.parametrize("pe,expected", [
    ("learned", 3_202_048), ("sinusoidal", 3_169_280), ("hybrid", 3_202_048),
    ("rope", 3_169_280), ("alibi", 3_169_280), ("relative", 3_173_360), ("none", 3_169_280),
])
def test_parameter_counts_unchanged(pe, expected):
    _, params = model_and_params(make_config("transformer", pos_encoding_type=pe, **PAPER))
    assert train.count_params(params) == expected


def _old_attention(dropout=0.0, **kw):
    """The module DecoderBlock used before for learned / sinusoidal / hybrid."""
    return nn.SelfAttention(num_heads=H, use_bias=False, dropout_rate=dropout, **kw)


def _copy_old_into_new(old_params):
    """SelfAttention query/key/value kernels (D, H, hd) and out kernel (H, hd, D) -> Dense kernels (D, D)."""
    return {
        "q_proj": {"kernel": old_params["query"]["kernel"].reshape(D, D)},
        "k_proj": {"kernel": old_params["key"]["kernel"].reshape(D, D)},
        "v_proj": {"kernel": old_params["value"]["kernel"].reshape(D, D)},
        "out_proj": {"kernel": old_params["out"]["kernel"].reshape(D, D)},
    }


@pytest.mark.parametrize("pe", ["learned", "sinusoidal", "hybrid"])
def test_equivalent_to_flax_self_attention_without_dropout(pe):
    x = jax.random.normal(jax.random.PRNGKey(0), (B, T, D))
    mask = attn.make_causal_mask(np.ones((B, T), dtype=bool))
    old = _old_attention()
    old_params = old.init(jax.random.PRNGKey(1), x, mask=mask, deterministic=True)["params"]
    new = MultiHeadAttentionWithPE(d_model=D, n_heads=H, attn_dropout=0.0, pos_encoding_type=pe, max_len=T)
    new_params = new.init(jax.random.PRNGKey(2), x, mask=mask, deterministic=True)["params"]
    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(_copy_old_into_new(old_params))

    y_old = old.apply({"params": old_params}, x, mask=mask, deterministic=True)
    y_new = new.apply({"params": _copy_old_into_new(old_params)}, x, mask=mask, deterministic=True)
    np.testing.assert_allclose(np.asarray(y_new), np.asarray(y_old), atol=1e-5, rtol=1e-5)


def _repeated_batch():
    seq = jax.random.normal(jax.random.PRNGKey(0), (1, T, D))
    return np.repeat(np.asarray(seq), B, axis=0)


def test_new_attention_draws_independent_dropout_masks_per_batch_element():
    x = _repeated_batch()
    mask = attn.make_causal_mask(np.ones((B, T), dtype=bool))
    new = MultiHeadAttentionWithPE(d_model=D, n_heads=H, attn_dropout=0.3, pos_encoding_type="learned", max_len=T)
    params = new.init(jax.random.PRNGKey(2), x, mask=mask, deterministic=True)["params"]
    y = np.asarray(new.apply({"params": params}, x, mask=mask, deterministic=False, rngs={"dropout": jax.random.PRNGKey(5)}))
    for i in range(1, B):
        assert not np.allclose(y[0], y[i], atol=1e-6)
    # sanity: the old module's default broadcast_dropout=True gives the same output for every element
    old = _old_attention(dropout=0.3)
    old_params = old.init(jax.random.PRNGKey(1), x, mask=mask, deterministic=True)["params"]
    y_old = np.asarray(old.apply({"params": old_params}, x, mask=mask, deterministic=False, rngs={"dropout": jax.random.PRNGKey(5)}))
    for i in range(1, B):
        np.testing.assert_allclose(y_old[0], y_old[i], atol=1e-6)


@pytest.mark.parametrize("pe", ["learned", "sinusoidal", "hybrid"])
def test_full_model_attention_dropout_differs_per_batch_element(pe):
    """Only attention dropout active; the same sequence in every batch row gives different training outputs."""
    config = make_config("transformer", pos_encoding_type=pe, dropout=0.0, attn_dropout=0.3)
    model, params = model_and_params(config)
    x = np.repeat(np.asarray(jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, 27)), B, axis=0)
    y = np.asarray(model.apply({"params": params}, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(3)}))
    for i in range(1, B):
        assert not np.allclose(y[0], y[i], atol=1e-6)
    y_eval = np.asarray(model.apply({"params": params}, x, deterministic=True))
    np.testing.assert_allclose(y_eval[0], y_eval[1], atol=1e-6)  # without dropout the rows agree
