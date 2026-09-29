import jax
import jax.numpy as jnp
import numpy as np
import pytest

import train
from helpers import PE_TYPES, make_config, model_and_params

DECAYED = {"kernel", "embedding", "positional_embed", "learned_embed", "W_out"}
NOT_DECAYED = {"bias", "scale", "rel_embed"}


def _leaves(params, mask):
    flat_p = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_m = jax.tree_util.tree_leaves(mask)
    return [(train._path_names(path), leaf, m) for (path, leaf), m in zip(flat_p, flat_m)]


@pytest.mark.parametrize("model_type,pe", [("transformer", pe) for pe in PE_TYPES] + [("lstm", None)])
def test_mask_is_ndim_ge_2_except_rel_embed(model_type, pe):
    config = make_config(model_type, pos_encoding_type=pe or "rope")
    _, params = model_and_params(config)
    mask = train.decay_mask_fn(params)
    assert jax.tree_util.tree_structure(mask) == jax.tree_util.tree_structure(params)
    for names, leaf, m in _leaves(params, mask):
        name = names[-1]
        assert m == (leaf.ndim >= 2 and name != "rel_embed"), (names, leaf.shape, m)
        if name in DECAYED or name.startswith(("Wx_", "Wh_")):
            assert m, names
        elif name in NOT_DECAYED or name.startswith("b_"):
            assert not m, names
        else:
            raise AssertionError(f"unexpected parameter {names}")


def test_relative_model_has_rel_embed_excluded():
    _, params = model_and_params(make_config("transformer", pos_encoding_type="relative"))
    rel = [(n, m) for n, _, m in _leaves(params, train.decay_mask_fn(params)) if n[-1] == "rel_embed"]
    assert len(rel) == 2 and not any(m for _, m in rel)  # one per layer, ndim 2, never decayed


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
@pytest.mark.parametrize("decay_mask", [True, False])
def test_optimizer_applies_decay_only_where_masked(model_type, decay_mask):
    """With zero gradients AdamW's update is pure weight decay, so undecayed leaves must not move."""
    config = make_config(model_type, pos_encoding_type="relative", decay_mask=decay_mask)
    _, params = model_and_params(config)
    params = jax.tree_util.tree_map(lambda p: p + 0.5, params)  # make biases / rel_embed non-zero
    tx = train.make_optimizer(config)
    updates, _ = tx.update(jax.tree_util.tree_map(jnp.zeros_like, params), tx.init(params), params)
    mask = train.decay_mask_fn(params)
    for (names, u, m) in _leaves(updates, mask):
        moved = bool(np.any(np.asarray(u) != 0))
        assert moved == (m or not decay_mask), (names, moved)
