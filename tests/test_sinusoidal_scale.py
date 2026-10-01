import jax
import jax.numpy as jnp
import numpy as np
import pytest

import train
from helpers import make_config, model_and_params
from models.positional_encodings import (
    HybridPositionalEncoding,
    SinusoidalPositionalEncoding,
    get_sinusoidal_embeddings,
    sinusoidal_scale_factor,
)

D, L = 256, 128


def _added_table(module, params=None):
    """What the module adds to a zero input: its positional table (L, D)."""
    x = jnp.zeros((1, L, D))
    variables = module.init(jax.random.PRNGKey(0), x) if params is None else params
    return np.asarray(module.apply(variables, x)[0]), variables


def test_unit_scale_rows_have_norm_1():
    table, _ = _added_table(SinusoidalPositionalEncoding(d_model=D, max_len=L, scale="unit"))
    np.testing.assert_allclose(np.linalg.norm(table, axis=-1), 1.0, atol=1e-5)


def test_raw_scale_is_the_old_table():
    raw = np.asarray(get_sinusoidal_embeddings(L, D))
    table, _ = _added_table(SinusoidalPositionalEncoding(d_model=D, max_len=L, scale="raw"))
    np.testing.assert_array_equal(table, raw)
    np.testing.assert_allclose(np.linalg.norm(table, axis=-1), np.sqrt(D / 2), rtol=1e-5)  # ~11.3: the old imbalance


@pytest.mark.parametrize("scale", ["unit", "raw"])
def test_hybrid_sinusoidal_half_is_scaled_the_same_way(scale):
    module = HybridPositionalEncoding(d_model=D, max_len=L, scale=scale)
    table, variables = _added_table(module)
    learned = np.asarray(variables["params"]["learned_embed"])
    w = module.learned_weight
    sinusoidal_half = (table - w * learned) / (1 - w)
    expected = np.asarray(get_sinusoidal_embeddings(L, D)) * sinusoidal_scale_factor(scale, D)
    np.testing.assert_allclose(sinusoidal_half, expected, atol=1e-5)
    if scale == "unit":
        np.testing.assert_allclose(np.linalg.norm(sinusoidal_half, axis=-1), 1.0, atol=1e-5)


def test_unknown_scale_is_rejected():
    with pytest.raises(ValueError):
        sinusoidal_scale_factor("sqrt", D)


@pytest.mark.parametrize("pe", ["sinusoidal", "hybrid"])
def test_config_switch_reaches_the_model(pe):
    """model.sinusoidal_scale defaults to 'unit'; 'raw' changes the forward pass of sinusoidal / hybrid models."""
    unit_cfg = make_config("transformer", pos_encoding_type=pe, dropout=0.0)
    assert unit_cfg.model.sinusoidal_scale == "unit"
    raw_cfg = make_config("transformer", pos_encoding_type=pe, dropout=0.0, sinusoidal_scale="raw")
    m_unit, params = model_and_params(unit_cfg)
    m_raw = train.build_model(raw_cfg)
    assert m_unit.sinusoidal_scale == "unit" and m_raw.sinusoidal_scale == "raw"
    x = jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 27)
    out_unit = m_unit.apply({"params": params}, x, deterministic=True)
    out_raw = m_raw.apply({"params": params}, x, deterministic=True)  # same parameters, different fixed table
    assert not np.allclose(np.asarray(out_unit), np.asarray(out_raw), atol=1e-4)
