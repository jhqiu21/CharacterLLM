"""scripts/analyze_errors.py: sliding-window accuracy and loss grouped by the target's position in its word."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

import train
from helpers import make_config, model_and_params
from utils.full_context_eval import full_context_eval, num_windows, window_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_errors import GROUP_NAMES, grouped_eval, word_position_groups  # noqa: E402

SPACE = train.CHAR_TO_INT[" "]
W, N = 16, 121


def test_word_position_groups():
    data = train.encode("ab c  def")
    #       target:      a  b  ' ' c  ' ' ' ' d  e  f
    #       position:    1  2  sp  1  sp  sp  1  2  3   (the first character has no previous one: start of a word)
    np.testing.assert_array_equal(word_position_groups(data, max_pos=3), [1, 2, 0, 1, 0, 0, 1, 2, 3])
    assert word_position_groups(train.encode("abcdef"), max_pos=3).tolist() == [1, 2, 3, 3, 3, 3]


@pytest.mark.parametrize("stride", [1, 7])
def test_grouped_eval_matches_brute_force(stride):
    config = make_config(dropout=0.0, max_len=W)
    model, params = model_and_params(config)
    rng = np.random.default_rng(5)
    data = np.where(rng.random(N) < 0.25, SPACE, rng.integers(0, 26, N)).astype(np.uint8)  # words and spaces
    out = grouped_eval(model, params, data, window=W, stride=stride, batch_size=5)

    k = np.arange(num_windows(N, W, stride))
    end, n_new = window_bounds(k, N, W, stride)
    groups = word_position_groups(data)
    count, correct, nll = np.zeros(len(GROUP_NAMES)), np.zeros(len(GROUP_NAMES)), np.zeros(len(GROUP_NAMES))
    for e, n in zip(end, n_new):
        logits = np.asarray(model.apply({"params": params}, np.asarray(data[e - W:e], np.int32)[None], deterministic=True))[0]
        logp = np.asarray(jax.nn.log_softmax(logits))
        for t in range(e - n + 1, e + 1):
            i = t - (e - W) - 1  # position of target t in this window
            g = groups[t]
            count[g] += 1
            correct[g] += logits[i].argmax() == data[t]
            nll[g] -= logp[i, data[t]]
    assert out["n_scored"] == N - 1 == count.sum()
    for g, name in enumerate(GROUP_NAMES):
        r = out["groups"][name]
        assert r["count"] == count[g]
        if count[g]:
            assert r["acc"] == pytest.approx(correct[g] / count[g], abs=1e-12)
            assert r["bpc"] == pytest.approx(nll[g] / count[g] / np.log(2), rel=1e-5)
    fc = full_context_eval(model, params, data, window=W, stride=stride, batch_size=5, verbose=False)
    assert out["acc"] == pytest.approx(fc["acc"], abs=1e-12) and out["bpc"] == pytest.approx(fc["bpc"], rel=1e-6)
    assert sum(r["share_of_errors"] for r in out["groups"].values()) == pytest.approx(1.0)
