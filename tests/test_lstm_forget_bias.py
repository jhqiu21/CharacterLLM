import numpy as np
import pytest

import train
from helpers import make_config, model_and_params


@pytest.mark.parametrize("forget_bias", [1.0, 0.0])
def test_forget_gate_slice_of_every_bias(forget_bias):
    H, n_layers = 16, 3
    config = make_config("lstm", hidden_size=H, n_layers=n_layers, forget_bias=forget_bias)
    assert train.build_model(config).forget_bias == forget_bias
    _, params = model_and_params(config)
    for i in range(n_layers):
        b = np.asarray(params[f"b_{i}"])
        assert b.shape == (4 * H,)
        np.testing.assert_array_equal(b[H:2 * H], forget_bias)  # gates split as (i, f, g, o)
        np.testing.assert_array_equal(b[:H], 0.0)
        np.testing.assert_array_equal(b[2 * H:], 0.0)


def test_forget_bias_defaults_to_1():
    config = make_config("lstm")
    assert config.model.forget_bias == 1.0
    for name in ("LR-1", "LR-2"):
        assert train.resolve_config(train.load_config(train.REPO_ROOT / "configs" / f"{name}.yaml")).model.forget_bias == 1.0


@pytest.mark.parametrize("forget_bias", [1.0, 0.0])
@pytest.mark.parametrize("hidden,expected", [(256, 2_115_072), (314, 3_177_052)])
def test_lstm_parameter_counts_unchanged(forget_bias, hidden, expected):
    _, params = model_and_params(make_config("lstm", hidden_size=hidden, n_layers=4, max_len=128, forget_bias=forget_bias))
    assert train.count_params(params) == expected
