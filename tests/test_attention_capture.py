import jax
import numpy as np
import pytest

import train
from helpers import PE_TYPES, make_config, model_and_params

T = 16


@pytest.mark.parametrize("pe", PE_TYPES)
def test_capture_flag_changes_nothing_and_weights_are_causal_distributions(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=T, n_layers=2, n_heads=2)
    model, params = model_and_params(config)
    capturing = train.build_model(config, capture_attn=True)
    # same parameters
    p2 = train.init_params(capturing, config, jax.random.PRNGKey(0))
    jax.tree_util.tree_map(np.testing.assert_array_equal, params, p2)

    x = np.asarray(jax.random.randint(jax.random.PRNGKey(2), (3, T), 0, 27), dtype=np.int32)
    plain = model.apply({"params": params}, x, deterministic=True)
    out, inter = capturing.apply({"params": params}, x, deterministic=True, mutable=["intermediates"])
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(out))
    # without the flag nothing is sown
    _, none = model.apply({"params": params}, x, deterministic=True, mutable=["intermediates"])
    assert not none

    for layer in range(2):
        (w,) = inter["intermediates"][f"blocks_{layer}"]["MultiHeadAttentionWithPE_0"]["attn_weights"]
        w = np.asarray(w)
        assert w.shape == (3, 2, T, T)
        np.testing.assert_allclose(w.sum(-1), 1.0, rtol=1e-5)
        assert np.all(w[..., np.triu_indices(T, k=1)[0], np.triu_indices(T, k=1)[1]] == 0)
