import jax
import numpy as np
import pytest

import train
from helpers import make_config, model_and_params


def _outputs(model, params, x, key):
    return np.asarray(model.apply({"params": params}, x, deterministic=False, rngs={"dropout": key}))


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_consecutive_step_keys_give_different_dropout(model_type):
    config = make_config(model_type, dropout=0.1)
    model, params = model_and_params(config)
    x = jax.random.randint(jax.random.PRNGKey(1), (4, 16), 0, 27)

    _, dropout_key = jax.random.split(jax.random.PRNGKey(4212))
    dropout_key, k1 = train.next_dropout_key(dropout_key, legacy=False)
    dropout_key, k2 = train.next_dropout_key(dropout_key, legacy=False)
    assert not np.array_equal(np.asarray(k1), np.asarray(k2))
    assert not np.allclose(_outputs(model, params, x, k1), _outputs(model, params, x, k2))


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_legacy_rng_reuses_the_same_dropout_mask(model_type):
    config = make_config(model_type, dropout=0.1)
    model, params = model_and_params(config)
    x = jax.random.randint(jax.random.PRNGKey(1), (4, 16), 0, 27)

    _, dropout_key = jax.random.split(jax.random.PRNGKey(4212))
    dropout_key, k1 = train.next_dropout_key(dropout_key, legacy=True)
    dropout_key, k2 = train.next_dropout_key(dropout_key, legacy=True)
    np.testing.assert_array_equal(np.asarray(k1), np.asarray(k2))
    np.testing.assert_array_equal(_outputs(model, params, x, k1), _outputs(model, params, x, k2))
    # dropout is actually active: training-mode output differs from eval-mode output
    assert not np.allclose(_outputs(model, params, x, k1), np.asarray(model.apply({"params": params}, x, deterministic=True)))


def test_fixed_loop_never_repeats_a_step_key():
    _, dropout_key = jax.random.split(jax.random.PRNGKey(4212))
    keys = []
    for _ in range(200):
        dropout_key, k = train.next_dropout_key(dropout_key, legacy=False)
        keys.append(tuple(np.asarray(k).tolist()))
    assert len(set(keys)) == len(keys)


def test_train_step_uses_the_step_key():
    """Two train steps from the same state with different step keys give different updates."""
    config = make_config("transformer", dropout=0.1)
    model, params = model_and_params(config)
    tx = train.make_optimizer(config)
    step = train.make_train_step(model, train.make_loss_and_metrics(config), tx)
    opt_state = tx.init(params)
    x = jax.random.randint(jax.random.PRNGKey(1), (4, 16), 0, 27)
    y = jax.random.randint(jax.random.PRNGKey(2), (4, 16), 0, 27)
    _, dropout_key = jax.random.split(jax.random.PRNGKey(0))
    dropout_key, k1 = train.next_dropout_key(dropout_key)
    dropout_key, k2 = train.next_dropout_key(dropout_key)
    _, _, m1 = step(params, opt_state, x, y, k1)
    _, _, m1b = step(params, opt_state, x, y, k1)
    _, _, m2 = step(params, opt_state, x, y, k2)
    assert float(m1["loss_train"]) == float(m1b["loss_train"])
    assert float(m1["loss_train"]) != float(m2["loss_train"])
