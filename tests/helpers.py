"""Tiny model configs shared by the tests."""

import train
from utils.config import Config

PE_TYPES = ["learned", "sinusoidal", "hybrid", "rope", "alibi", "relative"]
ALL_PE_TYPES = PE_TYPES + ["none"]  # with NoPE (no positional signal), which has no PE table to export


def make_config(model_type="transformer", pos_encoding_type="rope", dropout=0.1, max_len=16,
                d_model=32, n_heads=2, n_layers=2, hidden_size=16, decay_mask=True, **model_overrides):
    if model_type == "transformer":
        model = {
            "type": "transformer", "vocab_size": 27, "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "max_len": max_len, "mlp_ratio": 4, "emb_dropout": dropout, "mlp_dropout": dropout,
            "attn_dropout": dropout, "resid_dropout": dropout, "pos_encoding_type": pos_encoding_type,
        }
    else:
        model = {"type": "lstm", "vocab_size": 27, "hidden_size": hidden_size, "n_layers": n_layers,
                 "max_len": max_len, "dropout": dropout}
    model.update(model_overrides)
    return train.resolve_config(Config({
        "model_name": "test",
        "model": model,
        "loss": {"LABEL_SMOOTHING": True, "ls_eps": 0.05, "entropy_lambda": 0.0, "tail_scheme": "linear"},
        "scheduler": {"warmup_iters": 1, "init_value": 1e-2, "end_value": 1e-2},
        "optimizer": {"decay_mask": decay_mask},
        "training": {"batch_size": 4, "sequence_length": max_len, "learning_rate": 1e-2, "epochs": 10},
    }))


def model_and_params(config, seed=0):
    model = train.build_model(config)
    params = train.init_params(model, config, __import__("jax").random.PRNGKey(seed))
    return model, params
