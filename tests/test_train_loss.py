import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import train
import utils.eval as eval_utils
from helpers import make_config, model_and_params
from utils.config import Config


def _old_loss_and_metrics(config):
    """Verbatim copy of train.make_loss_and_metrics at tag tableV-v1 (the full-metrics training loss)."""
    label_smoothing = bool(config.loss.LABEL_SMOOTHING) and float(config.loss.ls_eps) > 0.0
    ls_eps = float(config.loss.ls_eps)
    tail_scheme = config.loss.tail_scheme
    entropy_lambda = float(config.loss.entropy_lambda)

    @jax.jit
    def loss_and_metrics(logits, targets):
        B, T, V = logits.shape

        if label_smoothing:
            tgt = jax.nn.one_hot(targets, V)
            tgt = (1.0 - ls_eps) * tgt + (ls_eps / V)
            per_pos_ce = optax.softmax_cross_entropy(logits, tgt)
        else:
            per_pos_ce = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, V), targets.reshape(-1)
            ).reshape(B, T)

        if tail_scheme == "linear":
            w = jnp.linspace(0.3, 1.0, T)
        elif tail_scheme == "sqrt":
            w = 0.1 + 0.9 * jnp.sqrt(jnp.linspace(0.0, 1.0, T))
        else:
            w = jnp.ones((T,))
        w = w[None, :]

        loss_weighted = (per_pos_ce * w).sum() / (B * w.sum())

        if entropy_lambda > 0.0:
            logp = jax.nn.log_softmax(logits, axis=-1)
            p = jnp.exp(logp)
            entropy = -(p * logp).sum(-1).mean()
            train_loss = loss_weighted - entropy_lambda * entropy
        else:
            train_loss = loss_weighted

        loss_all = eval_utils.loss_all(logits, targets)
        loss_last = eval_utils.cross_entropy_last_token_only(logits, targets)
        acc_all, acc_last = eval_utils.accuracy(logits, targets)
        perplexity, avg_loss = eval_utils.perplexity(logits, targets)
        bpc = eval_utils.bits_per_character(avg_loss)
        avg_entropy, _ = eval_utils.prediction_entropy(logits)
        ece, calibration_data = eval_utils.expected_calibration_error(logits, targets)

        metrics = {
            "loss_train": train_loss, "loss_all": loss_all, "loss_last": loss_last, "acc": acc_all, "acc_last": acc_last,
            "perplexity": perplexity, "bpc": bpc, "pred_entropy": avg_entropy, "ece": [ece, calibration_data],
        }
        return train_loss, metrics

    return loss_and_metrics


def _with_loss(config, **loss):
    d = config.to_dict()
    d["loss"].update(loss)
    return Config(d)


LOSS_VARIANTS = [
    {},                                                                      # LS 0.05, linear tail (run configs)
    {"LABEL_SMOOTHING": False, "tail_scheme": "sqrt", "entropy_lambda": 0.1},
    {"ls_eps": 0.1, "tail_scheme": "none"},
]


def _batch(T=16, B=4):
    x = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, 27)
    y = jax.random.randint(jax.random.PRNGKey(2), (B, T), 0, 27)
    return x, y


def _loss_and_grads(model, params, loss_fn, x, y, key):
    def f(p):
        logits = model.apply({"params": p}, x, deterministic=False, rngs={"dropout": key})
        return loss_fn(logits, y)
    return jax.value_and_grad(f, has_aux=True)(params)


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
@pytest.mark.parametrize("loss", LOSS_VARIANTS)
def test_lean_training_loss_matches_old_full_metrics_version(model_type, loss):
    config = _with_loss(make_config(model_type, dropout=0.1), **loss)
    model, params = model_and_params(config)
    x, y = _batch()
    key = jax.random.PRNGKey(9)
    (l_old, m_old), g_old = _loss_and_grads(model, params, _old_loss_and_metrics(config), x, y, key)
    (l_new, m_new), g_new = _loss_and_grads(model, params, train.make_train_loss(config), x, y, key)

    assert set(m_new) == {"loss_train", "loss_all", "loss_last"}  # exactly what Logger.log_train reads
    np.testing.assert_allclose(float(l_new), float(l_old), rtol=1e-6)
    for k in m_new:
        np.testing.assert_allclose(float(m_new[k]), float(m_old[k]), rtol=1e-6)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7), g_new, g_old)


def test_train_step_with_lean_loss_matches_old_step():
    """Whole jitted step (grads + clipping + AdamW) from the same state: same parameters afterwards."""
    config = make_config("transformer", dropout=0.1)
    model, params = model_and_params(config)
    tx = train.make_optimizer(config)
    x, y = _batch()
    key = jax.random.PRNGKey(9)
    p_old, _, m_old = train.make_train_step(model, _old_loss_and_metrics(config), tx)(params, tx.init(params), x, y, key)
    p_new, _, m_new = train.make_train_step(model, train.make_train_loss(config), tx)(params, tx.init(params), x, y, key)
    np.testing.assert_allclose(float(m_new["loss_train"]), float(m_old["loss_train"]), rtol=1e-6)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7), p_new, p_old)


def test_validation_metrics_unchanged():
    """make_loss_and_metrics (validation) still returns the old full metric set with the same values."""
    config = make_config("transformer", dropout=0.0)
    model, params = model_and_params(config)
    x, y = _batch()
    logits = model.apply({"params": params}, x, deterministic=True)
    _, old = _old_loss_and_metrics(config)(logits, y)
    _, new = train.make_loss_and_metrics(config)(logits, y)
    assert set(new) == set(old)
    for k in old:
        jax.tree_util.tree_map(lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6), new[k], old[k])


@pytest.mark.parametrize("tail_scheme", ["linear", "none"])
def test_ls_eps_zero_equals_label_smoothing_off(tail_scheme):
    base = make_config("transformer", dropout=0.0)
    on_zero = _with_loss(base, LABEL_SMOOTHING=True, ls_eps=0.0, tail_scheme=tail_scheme)
    off = _with_loss(base, LABEL_SMOOTHING=False, ls_eps=0.05, tail_scheme=tail_scheme)
    model, params = model_and_params(base)
    x, y = _batch()
    key = jax.random.PRNGKey(0)
    (l1, _), g1 = _loss_and_grads(model, params, train.make_train_loss(on_zero), x, y, key)
    (l2, _), g2 = _loss_and_grads(model, params, train.make_train_loss(off), x, y, key)
    assert float(l1) == float(l2)
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), g1, g2)
    # and label smoothing really is active for ls_eps > 0
    (l3, _), _ = _loss_and_grads(model, params, train.make_train_loss(_with_loss(base, ls_eps=0.05)), x, y, key)
    assert float(l3) != float(l1)
