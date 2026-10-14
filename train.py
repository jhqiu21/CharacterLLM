"""
Canonical training + evaluation entry point (transformer.ipynb is an interactive front end built on these functions).

    python train.py --config configs/final/TR.yaml --seed 42 [--run-name NAME] [--legacy-dropout-rng]
                    [--set section.key=value ...] [--tuning] [--data-dir DIR]
                    [--epochs N] [--val-interval N] [--fc-stride S] [--eval-only]

--set overrides a dotted config key (repeatable; values parsed as int, float, bool or string; unknown keys
are an error); overrides are recorded under run.overrides in the run's config.yaml.
--tuning never opens the test file: checkpoints are selected on the fixed validation set as usual, and the
best-loss checkpoint is then scored with full-context evaluation on the validation split
(stride evaluation.tuning_val_stride) into val_eval.json; no test metrics, samples or Self-BLEU.
--epochs / --val-interval / --fc-stride are for smoke tests only. --eval-only re-runs the final
evaluation from the checkpoints of an existing run. GPU selection comes from CUDA_VISIBLE_DEVICES.

Intentional differences from the original course notebook (results/code_changes_v2.md has the details):
  * the dropout key advances every step (--legacy-dropout-rng restores the old fixed key);
  * init / dropout keys and batch sampling are seeded from --seed;
  * optional weight-decay mask (optimizer.decay_mask) that skips biases, LayerNorm and rel_embed;
  * validation uses one fixed set of 1024 windows of length training.sequence_length (seed 0);
  * the loss is built by make_loss_and_metrics(config) instead of closing over a global config; the training
    step uses it too (the lean make_train_loss(config) has the same loss and gradients but compiles to a
    slower step, results/code_changes_v2.md section 3);
  * the warmup-cosine schedule reaches end_value at the last iteration (the notebook's reached 0 at epochs - warmup);
  * model defaults: one attention implementation for every positional encoding (no nn.SelfAttention), sinusoidal
    tables scaled to unit row norm (model.sinusoidal_scale), LSTM forget-gate bias 1.0 (model.forget_bias);
  * generation places the prompt at the start of the window (no left padding);
  * test evaluation in three modes (legacy32, chunked128, full_context), written to eval.json, plus
    single-prompt and multi-prompt Self-BLEU.
Additions not in the notebook: pos_encoding_type 'none' (NoPE) and the evaluation-only model options used by
utils/extrapolation_eval.py and scripts/analyze_attention.py (capture_attn, rel_clamp, pos_scale).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
import optax

import models.LSTM as lstm_models
import models.models as transformer_models
import utils.eval as eval_utils
import utils.generation as generation
from utils import (
    analyze_training_performance,
    load_checkpoint,
    load_config,
    logger,
    plot_training_curves,
    save_collected_checkpoints,
    save_config,
    test_checkpoint,
)
from utils.config import Config
from utils.full_context_eval import frequency_groups, full_context_eval

REPO_ROOT = Path(__file__).resolve().parent

# Vocabulary: a-z plus space (27 tokens), as in the notebook.
CHAR_SET = list("abcdefghijklmnopqrstuvwxyz ")
CHAR_TO_INT = {ch: i for i, ch in enumerate(CHAR_SET)}
INT_TO_CHAR = {i: ch for ch, i in CHAR_TO_INT.items()}
TRAIN_RATIO = 0.95
VAL_SET_SIZE = 1024
VAL_SET_SEED = 0
TIMING_WINDOW = (1000, 2000)

# Evaluated checkpoints: (on-disk prefix, name, run every evaluation mode?)
CHECKPOINT_TYPES = [
    ("best_loss_all_", "Best Loss", True),
    ("best_acc_total_", "Best Accuracy", False),
    ("best_acc_last_", "Best Last-Char Accuracy", False),
]

DEFAULTS = {
    "model": {"type": "transformer"},
    # Values previously hard-coded in the notebook, plus the decay mask switch.
    "optimizer": {"b1": 0.9, "b2": 0.95, "eps": 1e-8, "weight_decay": 0.1, "grad_clip_norm": 1.0, "decay_mask": True},
    "evaluation": {
        "full_context_stride": 1,
        "full_context_batch_size": 8192,
        # multi-prompt Self-BLEU: n_prompts validation passages of prompt_len chars, n_samples continuations each
        "self_bleu_multi_n_prompts": 10,
        "self_bleu_multi_n_samples": 20,
        "self_bleu_multi_prompt_len": 128,
        # --tuning: full-context evaluation of the validation split with this stride
        "tuning_val_stride": 8,
    },
}
# Per model.type defaults (only added for that type).
MODEL_DEFAULTS = {
    "transformer": {"sinusoidal_scale": "unit"},  # 'unit': sinusoidal rows have norm 1; 'raw': norm sqrt(d_model/2)
    "lstm": {"forget_bias": 1.0},                 # initial forget-gate bias; 0.0 = original all-zero bias
}
SELF_BLEU_PROMPT_SEED = 0


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def encode(s):
    """Encode a string to uint8 token ids (same mapping as the notebook's encode, vectorised)."""
    lut = np.full(256, 255, dtype=np.uint8)
    for ch, i in CHAR_TO_INT.items():
        lut[ord(ch)] = i
    ids = lut[np.frombuffer(s.encode("latin-1"), dtype=np.uint8)]
    if (ids == 255).any():
        bad = sorted(set(np.asarray(list(s))[ids == 255][:10]))
        raise KeyError(f"characters outside the vocabulary: {bad}")
    return ids


def load_data(data_dir=REPO_ROOT / "data", include_test=True):
    """First 95% of the train file is train, the rest validation; the test file is test.

    With include_test=False the test file is never opened and None is returned in its place.
    """
    with open(Path(data_dir) / "text8_train.txt", "r") as f:
        train_text_full = f.read()
    test_int = None
    if include_test:
        with open(Path(data_dir) / "text8_test.txt", "r") as f:
            test_int = encode(f.read())
    cut = int(len(train_text_full) * TRAIN_RATIO)
    return encode(train_text_full[:cut]), encode(train_text_full[cut:]), test_int


def get_batch(text_int, B, T, np_rng):
    """Random batch of B windows of length T (targets shifted by one), sampled from `np_rng`."""
    ix = np_rng.integers(0, len(text_int) - T, size=B)
    idx = ix[:, None] + np.arange(T)[None, :]
    return jnp.array(text_int[idx], dtype=jnp.int32), jnp.array(text_int[idx + 1], dtype=jnp.int32)


def make_fixed_val_set(val_int, T, n=VAL_SET_SIZE, seed=VAL_SET_SEED):
    """One fixed validation set, independent of the training seed, reused at every evaluation."""
    return get_batch(val_int, n, T, np.random.default_rng(seed))


def self_bleu_prompt_offsets(n_val, n_prompts, prompt_len, seed=SELF_BLEU_PROMPT_SEED):
    """Sorted, distinct start offsets of the multi-prompt Self-BLEU passages in the validation split.
    Depend only on the validation length and these constants, never on the training seed."""
    return np.sort(np.random.default_rng(seed).choice(n_val - prompt_len + 1, size=n_prompts, replace=False))


def make_self_bleu_prompts(val_int, n_prompts, prompt_len, seed=SELF_BLEU_PROMPT_SEED):
    """(offsets, prompts): n_prompts validation passages of prompt_len characters, shape (n_prompts, prompt_len)."""
    offsets = self_bleu_prompt_offsets(len(val_int), n_prompts, prompt_len, seed)
    return offsets, np.stack([np.asarray(val_int[o:o + prompt_len], dtype=np.int32) for o in offsets])


# ----------------------------------------------------------------------------------------------
# Config, model, loss, optimizer
# ----------------------------------------------------------------------------------------------
def resolve_config(config):
    """Fill in defaults for keys that older configs do not have."""
    d = config.to_dict() if isinstance(config, Config) else dict(config)
    for section, values in DEFAULTS.items():
        d.setdefault(section, {})
        for k, v in values.items():
            d[section].setdefault(k, v)
    for k, v in MODEL_DEFAULTS.get(d["model"]["type"], {}).items():
        d["model"].setdefault(k, v)
    return Config(d)


def parse_override_value(text):
    """'true'/'false' -> bool, then int, then float, else the string itself."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def apply_overrides(d, assignments):
    """Apply 'a.b.c=value' assignments to the config dict `d` in place; return {key: parsed value}.

    Only existing keys (after defaults are resolved) can be set, and only to non-section values.
    """
    applied = {}
    for item in assignments or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--set expects key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = d
        for part in parents:
            if not isinstance(node, dict) or not isinstance(node.get(part), dict):
                raise KeyError(f"--set {key}: unknown config section {part!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"--set {key}: unknown config key (known in this section: {sorted(node)})")
        if isinstance(node[leaf], dict):
            raise KeyError(f"--set {key}: is a section, not a value")
        node[leaf] = applied[key] = parse_override_value(raw.strip())
    return applied


def build_model(config, **eval_options):
    """eval_options: evaluation-only keyword arguments of DecoderOnlyTransformer (capture_attn, rel_table_len,
    rel_clamp, pos_scale); the defaults build the training model."""
    m = config.model
    if eval_options and m.type != "transformer":
        raise ValueError(f"eval_options {sorted(eval_options)} apply to transformers only")
    if m.type == "transformer":
        return transformer_models.DecoderOnlyTransformer(
            vocab_size=m.vocab_size,
            d_model=m.d_model,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            max_len=m.max_len,
            mlp_ratio=m.mlp_ratio,
            emb_dropout=m.emb_dropout,
            mlp_dropout=m.mlp_dropout,
            attn_dropout=m.attn_dropout,
            resid_dropout=m.resid_dropout,
            pos_encoding_type=m.pos_encoding_type,
            sinusoidal_scale=getattr(m, "sinusoidal_scale", MODEL_DEFAULTS["transformer"]["sinusoidal_scale"]),
            **eval_options,
        )
    if m.type == "lstm":
        return lstm_models.CharLSTM(
            vocab_size=m.vocab_size,
            hidden_size=m.hidden_size,
            n_layers=m.n_layers,
            dropout=m.dropout,
            max_len=m.max_len,
            forget_bias=float(getattr(m, "forget_bias", MODEL_DEFAULTS["lstm"]["forget_bias"])),
        )
    raise ValueError(f"Unknown model.type: {m.type}")


def init_params(model, config, key):
    dummy = jnp.zeros((1, min(16, config.model.max_len)), dtype=jnp.int32)
    return model.init({"params": key}, dummy, deterministic=True)["params"]


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def make_training_loss(config):
    """The optimised loss (label smoothing, tail position weights, entropy bonus) as a pure function
    (logits, targets) -> scalar. Shared by make_train_loss and make_loss_and_metrics, so both optimise
    exactly the same objective. Label smoothing is off when LABEL_SMOOTHING is false or ls_eps is 0."""
    label_smoothing = bool(config.loss.LABEL_SMOOTHING) and float(config.loss.ls_eps) > 0.0
    ls_eps = float(config.loss.ls_eps)
    tail_scheme = config.loss.tail_scheme
    entropy_lambda = float(config.loss.entropy_lambda)

    def training_loss(logits, targets):
        B, T, V = logits.shape

        if label_smoothing:
            tgt = jax.nn.one_hot(targets, V)
            tgt = (1.0 - ls_eps) * tgt + (ls_eps / V)
            per_pos_ce = optax.softmax_cross_entropy(logits, tgt)
        else:
            per_pos_ce = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, V), targets.reshape(-1)
            ).reshape(B, T)

        # Position weights: emphasize later positions, but don't discard early ones
        if tail_scheme == "linear":
            w = jnp.linspace(0.3, 1.0, T)
        elif tail_scheme == "sqrt":
            w = 0.1 + 0.9 * jnp.sqrt(jnp.linspace(0.0, 1.0, T))
        else:
            w = jnp.ones((T,))
        w = w[None, :]

        loss_weighted = (per_pos_ce * w).sum() / (B * w.sum())

        # Entropy bonus to encourage less-peaked distributions
        if entropy_lambda > 0.0:
            logp = jax.nn.log_softmax(logits, axis=-1)
            p = jnp.exp(logp)
            entropy = -(p * logp).sum(-1).mean()
            train_loss = loss_weighted - entropy_lambda * entropy
        else:
            train_loss = loss_weighted
        return train_loss

    return training_loss


def make_train_loss(config):
    """Lean loss: the optimised loss plus only what Logger.log_train records (loss_train, loss_all,
    loss_last). Same loss and gradients as make_loss_and_metrics. Not used by train(): its jitted step is
    ~9% slower for TR2o than the one built on make_loss_and_metrics (results/code_changes_v2.md section 3)."""
    training_loss = make_training_loss(config)

    @jax.jit
    def train_loss(logits, targets):
        loss = training_loss(logits, targets)
        return loss, {
            "loss_train": loss,
            "loss_all": eval_utils.loss_all(logits, targets),
            "loss_last": eval_utils.cross_entropy_last_token_only(logits, targets),
        }

    return train_loss


def make_loss_and_metrics(config):
    """Training loss (label smoothing, tail position weights, entropy bonus) plus unweighted metrics
    (accuracy, perplexity, BPC, entropy, ECE). Used for the training step and for validation.

    Config values are read once here, so the returned jitted function never depends on a global.
    """
    training_loss = make_training_loss(config)

    @jax.jit
    def loss_and_metrics(logits, targets):
        train_loss = training_loss(logits, targets)

        # Unweighted metrics for comparison (not used for backprop)
        loss_all = eval_utils.loss_all(logits, targets)
        loss_last = eval_utils.cross_entropy_last_token_only(logits, targets)
        acc_all, acc_last = eval_utils.accuracy(logits, targets)
        perplexity, avg_loss = eval_utils.perplexity(logits, targets)
        bpc = eval_utils.bits_per_character(avg_loss)
        avg_entropy, _ = eval_utils.prediction_entropy(logits)
        ece, calibration_data = eval_utils.expected_calibration_error(logits, targets)

        metrics = {
            "loss_train": train_loss,
            "loss_all": loss_all,
            "loss_last": loss_last,
            "acc": acc_all,
            "acc_last": acc_last,
            "perplexity": perplexity,
            "bpc": bpc,
            "pred_entropy": avg_entropy,
            "ece": [ece, calibration_data],
        }
        return train_loss, metrics

    return loss_and_metrics


def _path_names(path):
    return [str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p)))) for p in path]


def decay_mask_fn(params):
    """True for leaves that get weight decay: every leaf with ndim >= 2 (kernels, embeddings,
    learned position tables, LSTM Wx_*/Wh_*/W_out) except the relative-position bias table
    `rel_embed`. Biases and LayerNorm scale/bias (ndim 1) are never decayed."""
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: bool(leaf.ndim >= 2 and "rel_embed" not in _path_names(path)), params
    )


def warmup_cosine_schedule(init_value, peak_value, warmup_steps, decay_steps, end_value):
    """optax.warmup_cosine_decay_schedule with the same arguments (decay_steps = total length incl. warmup),
    computed in a float32-safe form.

    optax evaluates 0.5 * (1 + cos(pi * t / T)), which cancels to exactly 0 in float32 for the last few
    steps (e.g. step 49,999 of 50,000). This uses the identity 0.5 * (1 + cos x) = cos(x / 2)^2, so the rate
    stays > 0 until it equals end_value at step decay_steps.
    """
    warmup_steps = int(warmup_steps)
    cosine_steps = int(decay_steps) - warmup_steps
    if cosine_steps <= 0:
        raise ValueError(f"decay_steps ({decay_steps}) must exceed warmup_steps ({warmup_steps})")
    init_value, peak_value, end_value = float(init_value), float(peak_value), float(end_value)

    def cosine(count):
        frac = jnp.clip(count / cosine_steps, 0.0, 1.0)
        value = end_value + (peak_value - end_value) * jnp.square(jnp.cos(0.5 * math.pi * frac))
        return jnp.where(count >= cosine_steps, end_value, value)

    return optax.join_schedules([optax.linear_schedule(init_value, peak_value, warmup_steps), cosine], [warmup_steps])


def make_lr_schedule(config):
    """Linear warmup over scheduler.warmup_iters, then cosine decay that reaches end_value at
    training.epochs (optax's decay_steps is the total length, warmup included)."""
    return warmup_cosine_schedule(
        init_value=config.scheduler.init_value,
        peak_value=config.training.learning_rate,
        warmup_steps=config.scheduler.warmup_iters,
        decay_steps=config.training.epochs,
        end_value=config.scheduler.end_value,
    )


def make_optimizer(config):
    opt = config.optimizer
    return optax.chain(
        optax.clip_by_global_norm(float(opt.grad_clip_norm)),
        optax.adamw(
            learning_rate=make_lr_schedule(config),
            b1=float(opt.b1),
            b2=float(opt.b2),
            eps=float(opt.eps),
            weight_decay=float(opt.weight_decay),
            mask=decay_mask_fn if opt.decay_mask else None,
        ),
    )


def make_train_step(model, loss_fn_of_logits, tx):
    """Jitted (params, opt_state, x, y, step_key) -> (params, opt_state, metrics). `loss_fn_of_logits`
    maps (logits, targets) -> (loss, metrics): make_loss_and_metrics(config) (train.py) or make_train_loss."""
    def train_step(params, opt_state, x, y, step_key):
        def loss_fn(params):
            logits = model.apply({"params": params}, x, deterministic=False, rngs={"dropout": step_key})
            return loss_fn_of_logits(logits, y)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, metrics

    return jax.jit(train_step)


def next_dropout_key(dropout_key, legacy=False):
    """Return (carried key, key for this step). Legacy mode reuses one key on every step,
    which is what the original notebook effectively did (identical dropout masks every step)."""
    if legacy:
        return dropout_key, dropout_key
    carried, step_key = jax.random.split(dropout_key)
    return carried, step_key


def make_apply_fn(model):
    return jax.jit(lambda params, x: model.apply({"params": params}, x, deterministic=True))


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.ndarray, jax.Array)):
        return to_jsonable(np.asarray(x).tolist()) if np.ndim(x) else float(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def peak_memory_gib():
    try:
        return round(jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, 2)
    except Exception:
        return None


def run_paths(config, run_name):
    ckpt_dir = (REPO_ROOT / config.output.checkpoint_dir / run_name).resolve()
    run_dir = (REPO_ROOT / config.output.results_dir / run_name).resolve()
    return run_dir, ckpt_dir / "stage", ckpt_dir / "best"


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def train(config, model, data, seed, legacy_dropout_rng, run_dir, stage_dir, best_dir):
    train_int, val_int, _ = data
    B = config.training.batch_size
    T = config.training.sequence_length
    epochs = config.training.epochs
    val_interval = config.training.validation_interval

    init_key, dropout_key = jax.random.split(jax.random.PRNGKey(seed))
    np_rng = np.random.default_rng(seed)

    params = init_params(model, config, init_key)
    print(f"Model created: {config.model_name} ({config.model.type}), parameters: {count_params(params):_}")

    loss_and_metrics = make_loss_and_metrics(config)  # training step and validation: full metric set
    tx = make_optimizer(config)
    opt_state = tx.init(params)
    train_step = make_train_step(model, loss_and_metrics, tx)
    apply_fn = make_apply_fn(model)

    val_x, val_y = make_fixed_val_set(val_int, T)
    print(f"Fixed validation set: {val_x.shape[0]} windows x {val_x.shape[1]} chars (seed {VAL_SET_SEED})")
    print(f"Dropout RNG: {'LEGACY (same key every step)' if legacy_dropout_rng else 'split every step'}; "
          f"decay mask: {config.optimizer.decay_mask}")

    metrics_logger = logger.Logger(epochs)
    checkpoints_to_save = {"stage": [], "best_loss_all": None, "best_acc": None}
    step_times = np.full(epochs, np.nan)

    time_start = time.time()
    for it in range(epochs):
        t0 = time.perf_counter()
        dropout_key, step_key = next_dropout_key(dropout_key, legacy_dropout_rng)
        x, y = get_batch(train_int, B, T, np_rng)
        params, opt_state, metrics = train_step(params, opt_state, x, y, step_key)
        jax.block_until_ready((params, opt_state, metrics))
        step_times[it] = time.perf_counter() - t0

        train_metrics = jax.device_get({k: metrics[k] for k in ("loss_train", "loss_all", "loss_last")})
        metrics_logger.log_train({k: float(v) for k, v in train_metrics.items()})
        if not np.isfinite(train_metrics["loss_train"]):
            raise FloatingPointError(f"non-finite training loss {train_metrics['loss_train']} at iteration {it}")

        if it % val_interval == 0 or it == epochs - 1:
            time_since_start = time.time() - time_start
            val_logits = apply_fn(params, val_x)
            _, val_metrics = loss_and_metrics(val_logits, val_y)
            val_metrics = jax.device_get(val_metrics)

            metrics_logger.log_eval(it, val_metrics)
            metrics_logger.print_metrics(time_since_start)
            sys.stdout.flush()

            is_best_loss_all, is_best_acc, is_best_acc_last = metrics_logger.isBest(val_metrics)
            checkpoint_state = {
                "params": params,
                "opt_state": opt_state,
                "rng": dropout_key,
                "step": it,
                "val_loss": float(val_metrics["loss_all"]),
                "val_bpc": float(val_metrics["bpc"]),
                "val_perplexity": float(val_metrics["perplexity"]),
                "val_acc": float(val_metrics["acc"]),
                "val_acc_last": float(val_metrics["acc_last"]),
                "ECE": float(val_metrics["ece"][0]),
            }
            # (unchanged from the notebook: this condition is only checked on validation iterations)
            if (it + 1) % config.training.stage_checkpoint_interval == 0 or (it + 1) == epochs:
                checkpoints_to_save["stage"].append((it, checkpoint_state.copy()))
                print(f"\t \t Saved stage checkpoint @ {it:,}")
            if is_best_loss_all:
                checkpoints_to_save["best_loss_all"] = (it, checkpoint_state.copy())
                print(f"\t \t Saving best loss checkpoint at iteration {it}...")
            if is_best_acc:
                checkpoints_to_save["best_acc"] = (it, checkpoint_state.copy())
                print(f"\t \t Saving best accuracy checkpoint at iteration {it}...")
            if is_best_acc_last:
                checkpoints_to_save["best_acc_last"] = (it, checkpoint_state.copy())
                print(f"\t \t Saving best last-character accuracy checkpoint at iteration {it}...")

    total_time = time.time() - time_start

    save_collected_checkpoints(checkpoints_to_save, stage_dir, best_dir)
    analyze_training_performance(
        metrics_logger=metrics_logger,
        total_time=total_time,
        niter=epochs,
        n_final=10,
        save_results=True,
        results_path=str(run_dir / "training_results.json"),
    )
    plot_training_curves(metrics_logger=metrics_logger, save_path=str(run_dir / "training_curves.pdf"))
    with open(run_dir / "metrics_history.json", "w") as f:
        json.dump(to_jsonable(metrics_logger.get_metrics_history()), f)

    # Throughput: median wall time of (batch sampling + step + block_until_ready) over
    # iterations [1000, 2000), excluding the iterations that also ran validation.
    lo, hi = TIMING_WINDOW
    window_its = [i for i in range(lo, min(hi, epochs)) if i % val_interval != 0 and i != epochs - 1]
    if window_its:
        timing_its, timing_note = window_its, f"iterations [{lo}, {hi}) excluding validation iterations"
    else:  # smoke tests shorter than the window: use everything after the first (compiling) step
        timing_its = [i for i in range(1, epochs) if i % val_interval != 0 and i != epochs - 1]
        timing_note = "SHORT RUN: all iterations after the first, excluding validation iterations"
    s_per_step = float(np.median(step_times[timing_its])) if timing_its else float("nan")
    timing = {
        "s_per_step_median": s_per_step,
        "tokens_per_s": float(B * T / s_per_step),
        "timing_iterations": timing_note,
        "n_timed_steps": len(timing_its),
        "s_per_step_mean_all_iterations_incl_validation": float(total_time / epochs),
        "total_train_time_s": float(total_time),
        "batch_size": B,
        "sequence_length": T,
        "device": jax.devices()[0].device_kind,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_device_memory_gib_training": peak_memory_gib(),
    }
    with open(run_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    np.save(run_dir / "step_times.npy", step_times)
    print(f"Throughput: {s_per_step:.4f} s/step (median, {timing_note}), {timing['tokens_per_s']:.3e} tokens/s")
    return timing


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
LEGACY_KEYS = ("perplexity", "bpc", "acc", "acc_last", "acc_rare", "acc_common", "distinct", "coherence")


def run_test_checkpoint(model, params, test_int, sequence_length):
    t0 = time.time()
    out = test_checkpoint(model, params, test_int, INT_TO_CHAR, sequence_length=sequence_length)
    res = {k: float(v) for k, v in zip(LEGACY_KEYS, out)}
    res.update({
        "sequence_length": sequence_length,
        "rare_common_definition": "per-batch target frequency (utils.eval.token_frequency_analysis)",
        "seconds": time.time() - t0,
    })
    return res


def generate_sample(model, config, params):
    rng = jax.random.PRNGKey(config.evaluation.seed)
    # (generate_tokens keeps the last max_len tokens of a longer prompt)
    prompt_int = jnp.array(
        [[CHAR_TO_INT.get(c, len(CHAR_SET)) for c in config.evaluation.prompt.lower()]],
        dtype=jnp.int32,
    )
    out_ids = generation.generate_tokens(
        model, params, rng, prompt_int, config.evaluation.gen_len,
        block_size=config.model.max_len,
        temperature=config.evaluation.temperature,
        sample=config.evaluation.sample,
        ngram_size=None,
    )
    return config.evaluation.prompt + "".join(INT_TO_CHAR.get(int(x), "?") for x in list(out_ids[0]))


def write_text_report(path, name, checkpoint, entry, config, full_text):
    lg = entry["legacy32"]
    with open(path, "w") as f:
        f.write(f"{name}\n\n")
        f.write(f"Step: {checkpoint['step']}\n")
        f.write(f"Validation Loss: {checkpoint['val_loss']:.4f}\n")
        f.write(f"Validation BPC: {checkpoint['val_bpc']:.4f}\n")
        f.write(f"Validation Perplexity: {checkpoint['val_perplexity']:.4f}\n")
        f.write(f"Validation Acc (Last): {100 * checkpoint['val_acc_last']:.2f}%\n")
        f.write(f"Validation Acc: {100 * checkpoint['val_acc']:.2f}%\n")
        f.write(f"Validation ECE: {checkpoint['ECE']:.4f}\n\n")
        f.write("[legacy32: non-overlapping 32-char blocks]\n")
        f.write(f"Test Perplexity: {lg['perplexity']:.4f}\n")
        f.write(f"Test BPC: {lg['bpc']:.4f}\n")
        f.write(f"Test Acc: {100 * lg['acc']:.2f}%\n")
        f.write(f"Test Acc (Last): {100 * lg['acc_last']:.2f}%\n")
        f.write(f"Test Avg Rare Acc: {100 * lg['acc_rare']:.2f}%\n")
        f.write(f"Test Avg Common Acc: {100 * lg['acc_common']:.2f}%\n")
        f.write(f"Test Avg Distinct: {lg['distinct']:.2f}\n")
        f.write(f"Test Coherence: {lg['coherence']:.2f}\n")
        c128 = entry["chunked128"]
        f.write(f"\n[chunked{c128['sequence_length']}] BPC {c128['bpc']:.4f}  Acc {100 * c128['acc']:.2f}%  "
                f"Acc (Last) {100 * c128['acc_last']:.2f}%\n")
        if "full_context" in entry:
            fc = entry["full_context"]
            f.write(f"[full_context W={fc['window']} S={fc['stride']}] BPC {fc['bpc']:.4f}  PPL {fc['perplexity']:.4f}  "
                    f"Acc {100 * fc['acc']:.2f}%  ECE {fc['ece']:.4f}  Rare {100 * fc['acc_rare']:.2f}%  "
                    f"Common {100 * fc['acc_common']:.2f}%\n")
        if "self_bleu_4" in entry:
            f.write(f"Self-BLEU-{config.evaluation.self_belu_n_grams}: {entry['self_bleu_4']:.4f}\n")
        if "self_bleu_4_multi" in entry:
            sbm = entry["self_bleu_4_multi"]
            f.write(f"Self-BLEU-{config.evaluation.self_belu_n_grams} (multi-prompt, {sbm['n_prompts']} validation prompts x "
                    f"{sbm['n_samples']} samples): {sbm['mean']:.4f} +- {sbm['std']:.4f}\n")
        f.write("\n")
        f.write(full_text)


def multi_prompt_self_bleu(model, config, params, val_int):
    ev = config.evaluation
    offsets, prompts = make_self_bleu_prompts(val_int, ev.self_bleu_multi_n_prompts, ev.self_bleu_multi_prompt_len)
    out = eval_utils.self_bleu_multi_prompt(
        model, params, prompts, INT_TO_CHAR,
        block_size=config.model.max_len,
        gen_len=ev.gen_len,
        temperature=ev.temperature,
        sample=ev.sample,
        seed=ev.seed,
        n_grams=ev.self_belu_n_grams,
        n_samples=ev.self_bleu_multi_n_samples,
        ngram_size=None,
    )
    out.update({
        "prompt_source": "validation split",
        "prompt_len": int(ev.self_bleu_multi_prompt_len),
        "prompt_offsets": [int(o) for o in offsets],
        "prompt_offset_seed": SELF_BLEU_PROMPT_SEED,
        "std_ddof": 1,
    })
    return out


def evaluate(config, model, data, run_dir, stage_dir, best_dir, meta):
    train_int, val_int, test_int = data
    groups = frequency_groups(train_int, config.model.vocab_size, INT_TO_CHAR)
    print("Rare characters (training-corpus frequency):", [g["char"] for g in groups["rare"]])
    print("Common characters (training-corpus frequency):", [g["char"] for g in groups["common"]])

    results = dict(meta)
    results["frequency_groups"] = groups
    timing_path = run_dir / "timing.json"
    results["throughput"] = json.load(open(timing_path)) if timing_path.exists() else None
    results["checkpoints"] = {}

    for prefix, name, full in CHECKPOINT_TYPES:
        print(f"\n{name} ({prefix})...")
        checkpoint = load_checkpoint(prefix, name, best_dir, stage_dir)
        if checkpoint is None:
            print(f"\t no checkpoint found for {prefix}")
            continue
        params = checkpoint["params"]
        entry = {k: to_jsonable(checkpoint[k]) for k in
                 ("step", "val_loss", "val_bpc", "val_perplexity", "val_acc", "val_acc_last", "ECE")}
        entry["legacy32"] = run_test_checkpoint(model, params, test_int, 32)
        entry["chunked128"] = run_test_checkpoint(model, params, test_int, config.model.max_len)
        if full:
            fc = full_context_eval(
                model, params, test_int,
                window=config.model.max_len,
                stride=config.evaluation.full_context_stride,
                batch_size=config.evaluation.full_context_batch_size,
                vocab_size=config.model.vocab_size,
                rare_ids=groups["rare_ids"],
                common_ids=groups["common_ids"],
            )
            assert fc["n_scored"] == len(test_int) - 1, (fc["n_scored"], len(test_int) - 1)
            entry["full_context"] = fc
            print(f"\t full_context (W={fc['window']}, S={fc['stride']}, {fc['seconds']:.0f}s): BPC {fc['bpc']:.4f}, "
                  f"acc {100 * fc['acc']:.2f}%, ECE {fc['ece']:.4f}, rare {100 * fc['acc_rare']:.2f}%, "
                  f"common {100 * fc['acc_common']:.2f}%")
            entry["self_bleu_4"] = float(eval_utils.self_bleu(
                model, config, INT_TO_CHAR, CHAR_TO_INT, CHAR_SET, params,
                prompt=config.evaluation.prompt,
                gen_len=config.evaluation.gen_len,
                temperature=config.evaluation.temperature,
                sample=config.evaluation.sample,
                seed=config.evaluation.seed,
                n_grams=config.evaluation.self_belu_n_grams,
                n_samples=config.evaluation.self_belu_n_samples,
                ngram_size=None,
            ))
            entry["self_bleu_4_multi"] = multi_prompt_self_bleu(model, config, params, val_int)
            full_text = generate_sample(model, config, params)
            sample_path = run_dir / (prefix.replace("_", "") + ".txt")
            write_text_report(sample_path, name, checkpoint, entry, config, full_text)
            entry["sample_file"] = sample_path.name
            print(f"\tSaved to {sample_path}\n\tPreview: {full_text[:100]}...")
        results["checkpoints"][prefix] = entry
        results["peak_device_memory_gib"] = peak_memory_gib()
        with open(run_dir / "eval.json", "w") as f:  # write after every checkpoint
            json.dump(to_jsonable(results), f, indent=2)
    return results


def evaluate_tuning(config, model, data, run_dir, stage_dir, best_dir, meta):
    """--tuning: score the best-loss checkpoint with full-context evaluation on the validation split.
    Never touches the test split (data[2] is None) and writes no samples or Self-BLEU."""
    train_int, val_int, test_int = data
    assert test_int is None, "tuning mode must not load the test split"
    groups = frequency_groups(train_int, config.model.vocab_size, INT_TO_CHAR)
    results = dict(meta)
    results["split"] = "validation"
    results["frequency_groups"] = groups
    timing_path = run_dir / "timing.json"
    results["throughput"] = json.load(open(timing_path)) if timing_path.exists() else None

    prefix, name, _ = CHECKPOINT_TYPES[0]
    checkpoint = load_checkpoint(prefix, name, best_dir, stage_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"no {prefix} checkpoint in {best_dir}")
    entry = {k: to_jsonable(checkpoint[k]) for k in
             ("step", "val_loss", "val_bpc", "val_perplexity", "val_acc", "val_acc_last", "ECE")}
    fc = full_context_eval(
        model, checkpoint["params"], val_int,
        window=config.model.max_len,
        stride=config.evaluation.tuning_val_stride,
        batch_size=config.evaluation.full_context_batch_size,
        vocab_size=config.model.vocab_size,
        rare_ids=groups["rare_ids"],
        common_ids=groups["common_ids"],
    )
    assert fc["n_scored"] == len(val_int) - 1, (fc["n_scored"], len(val_int) - 1)
    entry["full_context"] = fc
    print(f"\t validation full_context (W={fc['window']}, S={fc['stride']}, {fc['seconds']:.0f}s): BPC {fc['bpc']:.4f}, "
          f"acc {100 * fc['acc']:.2f}%, ECE {fc['ece']:.4f}")
    results["checkpoints"] = {prefix: entry}
    results["peak_device_memory_gib"] = peak_memory_gib()
    with open(run_dir / "val_eval.json", "w") as f:
        json.dump(to_jsonable(results), f, indent=2)
    return results


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-name", default=None, help="default: <model_name>_s<seed>")
    p.add_argument("--legacy-dropout-rng", action="store_true",
                   help="reuse one dropout key on every step (reproduces the original bug; ablation only)")
    p.add_argument("--epochs", type=int, default=None, help="override training.epochs (smoke tests only)")
    p.add_argument("--val-interval", type=int, default=None, help="override training.validation_interval (smoke tests only)")
    p.add_argument("--fc-stride", type=int, default=None, help="override evaluation.full_context_stride (smoke tests only)")
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate the run's saved checkpoints")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a dotted config key, e.g. training.learning_rate=6e-4 (repeatable)")
    p.add_argument("--tuning", action="store_true",
                   help="hyperparameter-search mode: never open the test file; write val_eval.json instead of eval.json")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"), help="directory with text8_train.txt / text8_test.txt")
    return p.parse_args(argv)


def build_run_config(args):
    """Resolved config dict for a new run: YAML + defaults, then --seed, --set, then the smoke-test flags."""
    d = resolve_config(load_config(args.config)).to_dict()
    if any(item.partition("=")[0].strip() == "training.seed" for item in args.overrides):
        raise KeyError("--set training.seed: use --seed")
    d["training"]["seed"] = args.seed
    overrides = apply_overrides(d, args.overrides)
    if args.epochs is not None:
        d["training"]["epochs"] = overrides["training.epochs"] = args.epochs
    if args.val_interval is not None:
        d["training"]["validation_interval"] = overrides["training.validation_interval"] = args.val_interval
    if args.fc_stride is not None:
        d["evaluation"]["full_context_stride"] = overrides["evaluation.full_context_stride"] = args.fc_stride
    if d["training"]["epochs"] <= d["scheduler"]["warmup_iters"]:
        # The warmup-cosine schedule needs epochs > warmup; shrink warmup for smoke tests only.
        d["scheduler"]["warmup_iters"] = overrides["scheduler.warmup_iters"] = max(1, d["training"]["epochs"] // 10)
    return d, overrides


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    d, overrides = build_run_config(args)
    run_name = args.run_name or f"{d['model_name']}_s{args.seed}"
    d["run"] = {
        "run_name": run_name,
        "seed": args.seed,
        "legacy_dropout_rng": bool(args.legacy_dropout_rng),
        "tuning": bool(args.tuning),
        "config_file": str(args.config),
        "overrides": overrides,
        "git_commit": git_commit(),
    }
    config = Config(d)
    run_dir, stage_dir, best_dir = run_paths(config, run_name)

    if args.eval_only:
        # Evaluate with the run's own resolved config (only the full-context stride may be overridden).
        d = load_config(run_dir / "config.yaml").to_dict()
        if args.fc_stride is not None:
            d["evaluation"]["full_context_stride"] = args.fc_stride
            d["run"]["overrides"]["evaluation.full_context_stride"] = args.fc_stride
        d["run"]["eval_git_commit"] = git_commit()
        d["run"]["tuning"] = bool(args.tuning)
        config = resolve_config(Config(d))

    for p in (run_dir, stage_dir, best_dir):
        p.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / ("eval.log" if args.eval_only else "train.log"), "a" if args.eval_only else "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        run(args, config, run_name, run_dir, stage_dir, best_dir)
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_file.close()


def run(args, config, run_name, run_dir, stage_dir, best_dir):
    tuning = bool(args.tuning)
    print(f"Run {run_name}: config {args.config}, seed {config.run.seed}, devices {jax.devices()}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, eval_only={args.eval_only}, tuning={tuning}")
    if not args.eval_only:
        save_config(config, run_dir / "config.yaml")
    print(json.dumps(config.to_dict(), indent=2))

    data = load_data(args.data_dir, include_test=not tuning)
    print(f"train {len(data[0]):_} / val {len(data[1]):_} / test "
          f"{'NOT LOADED (tuning mode)' if tuning else f'{len(data[2]):_}'} characters")
    model = build_model(config)

    if not args.eval_only:
        train(config, model, data, config.run.seed, config.run.legacy_dropout_rng, run_dir, stage_dir, best_dir)

    num_params = count_params(init_params(model, config, jax.random.PRNGKey(0)))
    meta = {
        "run_name": run_name,
        "model_name": config.model_name,
        "model_type": config.model.type,
        "pos_encoding_type": getattr(config.model, "pos_encoding_type", None),
        "seed": config.run.seed,
        "legacy_dropout_rng": bool(config.run.legacy_dropout_rng),
        "decay_mask": bool(config.optimizer.decay_mask),
        "num_params": int(num_params),
        "config_file": config.run.config_file,
        "overrides": config.run.overrides.to_dict() if isinstance(config.run.overrides, Config) else config.run.overrides,
        "git_commit": config.run.git_commit,
        "eval_git_commit": getattr(config.run, "eval_git_commit", config.run.git_commit),
        "device": jax.devices()[0].device_kind,
    }
    if tuning:
        evaluate_tuning(config, model, data, run_dir, stage_dir, best_dir, meta)
        print(f"\nDone: {run_dir / 'val_eval.json'}")
    else:
        evaluate(config, model, data, run_dir, stage_dir, best_dir, meta)
        print(f"\nDone: {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
