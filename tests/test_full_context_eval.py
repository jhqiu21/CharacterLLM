import jax
import numpy as np
import pytest

import utils.eval as eval_utils
from helpers import make_config, model_and_params
from utils.full_context_eval import full_context_eval, num_windows, scored_target_indices, window_bounds

W = 16
N = 101  # 100 targets; (N-1-W) is not a multiple of 7, so the final stride-7 window is partial
RARE, COMMON = [0, 1, 2], [3, 4, 26]


@pytest.fixture(scope="module", params=["transformer", "lstm"])
def setup(request):
    config = make_config(request.param, dropout=0.0, max_len=W)
    model, params = model_and_params(config)
    data = np.asarray(jax.random.randint(jax.random.PRNGKey(11), (N,), 0, 27), dtype=np.uint8)
    return model, params, data


def _brute_force(model, params, data, contexts):
    """contexts: list of (target index, start index). Score each target from data[start:target]."""
    nll, logits_all, targets = [], [], []
    for t, s in contexts:
        logits = np.asarray(model.apply({"params": params}, np.asarray(data[s:t], np.int32)[None], deterministic=True))[0, -1]
        logp = jax.nn.log_softmax(logits)
        nll.append(-float(logp[data[t]]))
        logits_all.append(logits)
        targets.append(int(data[t]))
    return np.array(nll), np.stack(logits_all), np.array(targets)


@pytest.mark.parametrize("stride", [1, 7, W])
def test_every_target_scored_exactly_once(stride):
    idx = scored_target_indices(N, W, stride)
    assert len(idx) == N - 1
    np.testing.assert_array_equal(np.sort(idx), np.arange(1, N))
    # every target after the first W has at least W - stride + 1 characters of context
    k = np.arange(num_windows(N, W, stride))
    end, n_new = window_bounds(k, N, W, stride)
    assert n_new[0] == W and (n_new[1:] <= stride).all() and (n_new > 0).all()


@pytest.mark.parametrize("stride", [0, W + 1])
def test_stride_outside_1_to_window_is_rejected(stride):
    with pytest.raises(ValueError):
        num_windows(N, W, stride)


@pytest.mark.parametrize("stride", [1, 7])
@pytest.mark.parametrize("batch_size", [8, 5])  # both leave a padded final batch
def test_scored_count_is_n_minus_1(setup, stride, batch_size):
    model, params, data = setup
    out = full_context_eval(model, params, data, window=W, stride=stride, batch_size=batch_size, verbose=False)
    assert out["n_scored"] == N - 1
    assert sum(out["class_count"]) == N - 1
    assert sum(out["ece_bins"]["count"]) == N - 1


def test_stride1_matches_brute_force(setup):
    model, params, data = setup
    out = full_context_eval(model, params, data, window=W, stride=1, batch_size=8,
                            rare_ids=RARE, common_ids=COMMON, verbose=False)
    contexts = [(t, max(0, t - W)) for t in range(1, N)]  # each target from its own window
    nll, logits, targets = _brute_force(model, params, data, contexts)

    np.testing.assert_allclose(out["loss"] * out["n_scored"], nll.sum(), rtol=1e-5)
    np.testing.assert_allclose(out["bpc"], nll.mean() / np.log(2), rtol=1e-5)
    correct = logits.argmax(-1) == targets
    assert out["acc"] == pytest.approx(correct.mean(), abs=1e-12)
    ece_ref, _ = eval_utils.expected_calibration_error(logits[None], targets[None])
    assert out["ece"] == pytest.approx(float(ece_ref), abs=1e-5)
    for ids, key in ((RARE, "acc_rare"), (COMMON, "acc_common")):
        sel = np.isin(targets, ids)
        assert out[key] == pytest.approx(correct[sel].mean(), abs=1e-12)


def test_top_k_accuracy_matches_brute_force(setup):
    model, params, data = setup
    out = full_context_eval(model, params, data, window=W, stride=1, batch_size=8, verbose=False)
    contexts = [(t, max(0, t - W)) for t in range(1, N)]
    _, logits, targets = _brute_force(model, params, data, contexts)
    rank = (logits > logits[np.arange(len(targets)), targets][:, None]).sum(-1)  # characters scored above the target
    for k in (2, 3, 5):
        assert out[f"acc_top{k}"] == pytest.approx((rank < k).mean(), abs=1e-12)
    assert out["acc"] <= out["acc_top2"] <= out["acc_top3"] <= out["acc_top5"]


def test_stride7_matches_its_window_plan(setup):
    model, params, data = setup
    stride = 7
    out = full_context_eval(model, params, data, window=W, stride=stride, batch_size=5, verbose=False)
    k = np.arange(num_windows(N, W, stride))
    end, n_new = window_bounds(k, N, W, stride)
    contexts = [(t, e - W) for e, n in zip(end, n_new) for t in range(e - n + 1, e + 1)]
    nll, _, _ = _brute_force(model, params, data, contexts)
    np.testing.assert_allclose(out["loss"] * out["n_scored"], nll.sum(), rtol=1e-5)
