import jax
import numpy as np
import pytest

from helpers import PE_TYPES, make_config, model_and_params

T = 16


def _randomize_rel_embed(params):
    """rel_embed is zero-initialised; give it random values so the relative bias is exercised."""
    def f(path, leaf):
        name = str(getattr(path[-1], "key", path[-1]))
        if name == "rel_embed":
            return jax.random.normal(jax.random.PRNGKey(7), leaf.shape)
        return leaf
    return jax.tree_util.tree_map_with_path(f, params)


@pytest.mark.parametrize("model_type,pe", [("transformer", pe) for pe in PE_TYPES] + [("lstm", None)])
@pytest.mark.parametrize("t", [0, 5, T - 2])
def test_future_tokens_do_not_change_past_logits(model_type, pe, t):
    config = make_config(model_type, pos_encoding_type=pe or "rope", dropout=0.0, max_len=T)
    model, params = model_and_params(config)
    params = _randomize_rel_embed(params)
    x = jax.random.randint(jax.random.PRNGKey(3), (3, T), 0, 27)
    noise = jax.random.randint(jax.random.PRNGKey(4), (3, T), 1, 27)
    x2 = x.at[:, t + 1:].set((x[:, t + 1:] + noise[:, t + 1:]) % 27)  # every later token changes

    l1 = np.asarray(model.apply({"params": params}, x, deterministic=True))
    l2 = np.asarray(model.apply({"params": params}, x2, deterministic=True))
    np.testing.assert_allclose(l1[:, :t + 1], l2[:, :t + 1], atol=1e-5, rtol=1e-5)
    assert not np.allclose(l1[:, t + 1:], l2[:, t + 1:], atol=1e-5)  # the test can detect a change
