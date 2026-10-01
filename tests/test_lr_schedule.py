import numpy as np
import optax
import pytest

import train
from utils.config import Config

EPOCHS, WARMUP, PEAK = 50_000, 2_000, 4e-4


def _schedule(init_value, end_value, epochs=EPOCHS, warmup=WARMUP):
    config = Config({"scheduler": {"warmup_iters": warmup, "init_value": init_value, "end_value": end_value},
                     "training": {"learning_rate": PEAK, "epochs": epochs}})
    return train.make_lr_schedule(config)


@pytest.mark.parametrize("init_value,end_value", [(0.0, 0.0), (1e-5, 1e-6)])
def test_schedule_endpoints(init_value, end_value):
    lr = _schedule(init_value, end_value)
    assert float(lr(0)) == pytest.approx(init_value, rel=1e-5, abs=0)  # float32: optax computes (init - peak) + peak
    assert float(lr(WARMUP)) == pytest.approx(PEAK, rel=1e-6)
    assert float(lr(48_000)) > end_value  # the old schedule was already at 0 here
    assert float(lr(49_999)) > end_value  # the last step still trains
    assert float(lr(EPOCHS)) == pytest.approx(end_value, rel=1e-6, abs=0)
    assert float(lr(EPOCHS + 100)) == pytest.approx(end_value, rel=1e-6, abs=0)
    assert float(lr(49_999)) > float(lr(EPOCHS))


def test_schedule_is_monotone_after_warmup():
    lr = _schedule(0.0, 0.0)
    values = np.array([float(lr(t)) for t in range(WARMUP, EPOCHS + 1, 500)] + [float(lr(EPOCHS - 1)), float(lr(EPOCHS))])
    assert np.all(np.diff(values[:-2]) <= 0) and values[-2] > values[-1] == 0.0


def test_matches_optax_warmup_cosine_with_total_decay_steps():
    """Same function as optax.warmup_cosine_decay_schedule(decay_steps=epochs), up to float32 rounding."""
    lr = _schedule(1e-5, 1e-6)
    ref = optax.warmup_cosine_decay_schedule(1e-5, PEAK, WARMUP, EPOCHS, 1e-6)
    for t in list(range(0, EPOCHS + 1, 997)) + [WARMUP - 1, WARMUP + 1, 48_000, EPOCHS - 1, EPOCHS]:
        assert float(lr(t)) == pytest.approx(float(ref(t)), rel=1e-4, abs=1e-10), t


def test_smoke_override_for_short_runs_still_works():
    args = train.parse_args(["--config", str(train.REPO_ROOT / "configs" / "TR2o.yaml"), "--seed", "1", "--epochs", "300"])
    d, overrides = train.build_run_config(args)
    assert d["scheduler"]["warmup_iters"] == 30 and overrides["scheduler.warmup_iters"] == 30
    lr = train.make_lr_schedule(Config(d))
    assert float(lr(30)) == pytest.approx(d["training"]["learning_rate"], rel=1e-6)
    assert float(lr(299)) > 0.0 and float(lr(300)) == 0.0
