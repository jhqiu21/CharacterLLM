import builtins
import io
import json

import numpy as np
import pytest
import yaml

import train
import utils.eval as eval_utils

TR2O = str(train.REPO_ROOT / "configs" / "TR2o.yaml")


@pytest.mark.parametrize("text,expected", [
    ("6e-4", 6e-4), ("0.1", 0.1), ("128", 128), ("-3", -3), ("true", True), ("False", False), ("rope", "rope"),
])
def test_override_values_are_parsed(text, expected):
    value = train.parse_override_value(text)
    assert value == expected and type(value) is type(expected)


def test_set_overrides_dotted_keys_and_records_them():
    args = train.parse_args(["--config", TR2O, "--seed", "1", "--set", "training.learning_rate=6e-4",
                             "--set", "model.attn_dropout=0.0", "--set", "optimizer.decay_mask=false",
                             "--set", "model.pos_encoding_type=alibi", "--set", "training.batch_size=64"])
    d, overrides = train.build_run_config(args)
    assert d["training"]["learning_rate"] == 6e-4
    assert d["model"]["attn_dropout"] == 0.0
    assert d["optimizer"]["decay_mask"] is False
    assert d["model"]["pos_encoding_type"] == "alibi"
    assert d["training"]["batch_size"] == 64 and isinstance(d["training"]["batch_size"], int)
    assert overrides == {"training.learning_rate": 6e-4, "model.attn_dropout": 0.0, "optimizer.decay_mask": False,
                         "model.pos_encoding_type": "alibi", "training.batch_size": 64}


def test_set_can_reach_keys_added_by_defaults():
    d, _ = train.build_run_config(train.parse_args(["--config", TR2O, "--seed", "1", "--set", "model.sinusoidal_scale=raw"]))
    assert d["model"]["sinusoidal_scale"] == "raw"


@pytest.mark.parametrize("item", ["training.learnig_rate=1e-3", "nosection.x=1", "model.d_model.x=1", "model=3",
                                  "training.learning_rate", "training.seed=3"])
def test_unknown_or_invalid_keys_raise(item):
    with pytest.raises((KeyError, ValueError)):
        train.build_run_config(train.parse_args(["--config", TR2O, "--seed", "1", "--set", item]))


# ----------------------------------------------------------------------------------------------
# --tuning: a complete tiny run through train.main
# ----------------------------------------------------------------------------------------------
TINY = {
    "model_name": "tiny",
    "model": {"type": "transformer", "vocab_size": 27, "d_model": 16, "n_heads": 2, "n_layers": 1, "max_len": 16,
              "mlp_ratio": 2, "emb_dropout": 0.1, "mlp_dropout": 0.1, "attn_dropout": 0.1, "resid_dropout": 0.1,
              "pos_encoding_type": "rope"},
    "loss": {"LABEL_SMOOTHING": True, "ls_eps": 0.05, "entropy_lambda": 0.0, "tail_scheme": "linear"},
    "scheduler": {"warmup_iters": 2, "init_value": 0.0, "end_value": 0.0},
    "training": {"batch_size": 4, "sequence_length": 16, "learning_rate": 1e-3, "epochs": 6,
                 "validation_interval": 2, "stage_checkpoint_interval": 100, "seed": 0},
    "evaluation": {"sample": True, "seed": 0, "prompt": "hello", "gen_len": 5, "temperature": 0.7,
                   "self_belu_n_grams": 4, "self_belu_n_samples": 2, "full_context_stride": 1,
                   "full_context_batch_size": 32},
}


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    text = "".join(np.array(train.CHAR_SET)[rng.integers(0, 27, size=6000)])
    (data_dir / "text8_train.txt").write_text(text)
    (data_dir / "text8_test.txt").write_text(text[:1000])
    cfg = dict(TINY, output={"checkpoint_dir": str(tmp_path / "ckpt"), "results_dir": str(tmp_path / "runs")})
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)  # train.main chdirs to the repo; restore afterwards
    return tmp_path, data_dir, cfg_path


def test_tuning_never_opens_the_test_file(tiny_run, monkeypatch):
    tmp_path, data_dir, cfg_path = tiny_run
    opened = []
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        opened.append(str(file))
        if "text8_test" in str(file):
            raise AssertionError(f"tuning mode opened the test file: {file}")
        return real_open(file, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("tuning mode must not run test evaluation or Self-BLEU")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(train, "test_checkpoint", forbidden)
    monkeypatch.setattr(eval_utils, "self_bleu", forbidden)
    monkeypatch.setattr(eval_utils, "self_bleu_multi_prompt", forbidden)
    monkeypatch.setattr(train, "generate_sample", forbidden)

    train.main(["--config", str(cfg_path), "--seed", "3", "--tuning", "--data-dir", str(data_dir),
                "--set", "evaluation.tuning_val_stride=4", "--set", "training.learning_rate=2e-3"])

    assert any("text8_train" in f for f in opened) and not any("text8_test" in f for f in opened)
    run_dir = tmp_path / "runs" / "tiny_s3"
    assert (run_dir / "val_eval.json").exists()
    assert not (run_dir / "eval.json").exists()
    assert not list(run_dir.glob("*.txt"))  # no sample / test report files

    val_len = len(train.load_data(data_dir, include_test=False)[1])
    out = json.load(real_open(run_dir / "val_eval.json"))
    fc = out["checkpoints"]["best_loss_all_"]["full_context"]
    assert out["split"] == "validation"
    assert fc["n_scored"] == val_len - 1 and fc["stride"] == 4 and fc["window"] == 16
    assert np.isfinite(fc["bpc"])
    assert "self_bleu_4" not in json.dumps(out)

    saved = yaml.safe_load(real_open(run_dir / "config.yaml"))
    assert saved["run"]["tuning"] is True
    assert saved["run"]["overrides"] == {"evaluation.tuning_val_stride": 4, "training.learning_rate": 2e-3}
    assert saved["training"]["learning_rate"] == 2e-3


def test_load_data_without_test_returns_none(tiny_run):
    _, data_dir, _ = tiny_run
    train_int, val_int, test_int = train.load_data(data_dir, include_test=False)
    assert test_int is None and len(train_int) == 5700 and len(val_int) == 300
