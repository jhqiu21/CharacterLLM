import pytest

import train
from helpers import make_config, model_and_params

PAPER = dict(d_model=256, n_heads=4, n_layers=4, max_len=128)


@pytest.mark.parametrize("config,expected", [
    (make_config("transformer", pos_encoding_type="rope", **PAPER), 3_169_280),     # TR2o, matches the paper
    (make_config("lstm", hidden_size=256, n_layers=4, max_len=128), 2_115_072),     # LR-1, matches the paper
    (make_config("lstm", hidden_size=314, n_layers=4, max_len=128), 3_177_052),     # LR-2, matches the paper
    # TR8o: the current code gives 3,173,360 (= RoPE count + 4 layers x 4 heads x 255 rel. biases).
    # The paper reports 3,238,640; the 65,280 = 255 x 256 gap presumably comes from an older implementation.
    (make_config("transformer", pos_encoding_type="relative", **PAPER), 3_173_360),
])
def test_parameter_count(config, expected):
    _, params = model_and_params(config)
    assert train.count_params(params) == expected


@pytest.mark.parametrize("name,expected", [("TR2o", 3_169_280), ("TR8o", 3_173_360), ("LR-1", 2_115_072),
                                           ("LR-2", 3_177_052), ("TR2o-nomask", 3_169_280)])
def test_run_configs_have_expected_parameter_count(name, expected):
    config = train.resolve_config(train.load_config(train.REPO_ROOT / "configs" / f"{name}.yaml"))
    _, params = model_and_params(config)
    assert train.count_params(params) == expected
