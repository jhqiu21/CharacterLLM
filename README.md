# CharacterLLM


A character-level language modeling project implementing Transformer and LSTM baselines for next-character prediction on the text8 dataset. This project explores various architectural choices, positional encodings, and training strategies to achieve publication-quality results.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Project Layout](#project-layout)
- [Dataset](#dataset)
- [Key Training Parameters](#key-training-parameters)
- [Configuration System](#configuration-system)
- [Tests](#tests)
- [Code Style (Check-Style)](#code-style-check-style)
- [Branches](#branches)

## Overview

This project implements and trains small language models for character-level text generation using the text8 dataset (100M characters from Wikipedia).

**Key Features:**
- Decoder-only Transformer and multi-layer LSTM in one code base, selected by `model.type`
- Multiple positional encoding strategies (Learned, Sinusoidal, RoPE, ALiBi, Hybrid, Relative), all sharing one attention implementation
- Comprehensive evaluation metrics (BPC, perplexity, accuracy, ECE, rare/common-character accuracy, Self-BLEU)
- Sliding-window ("full context") evaluation that scores every target character exactly once
- Advanced loss weighting schemes (linear, sqrt, uniform), label smoothing, entropy bonus
- N-gram repetition prevention during generation
- A tuning mode that never touches the test set
- JAX/Flax implementation for efficient training

## Setup

### Environment

This project uses **Conda** for environment management.

Create the environment using the provided [`environment.yml`](environment.yml) file:

```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate charllm-env
```

Deactivate when done:
```bash
conda deactivate
```

**Note**: The default environment uses CPU. For GPU training:
- Replace `cpuonly` with `pytorch-cuda=11.8` in `environment.yml`
- Change `jax[cpu]` to `jax[cuda12]` or `jax[cuda11]` depending on your CUDA version

## Usage

### Training and evaluation: `train.py`

`train.py` is the canonical entry point. It trains one model from a YAML config, keeps the best checkpoints
(selected on one fixed validation set of 1024 × `sequence_length` windows), and then evaluates them on the test set.

```bash
python train.py --config configs/TR2o.yaml --seed 4212
python train.py --config configs/LR-2.yaml --seed 4212 --run-name LR-2_s4212
```

GPU selection comes from the environment (`CUDA_VISIBLE_DEVICES=0 python train.py ...`).
Outputs:
- `runs/<run_name>/`: `config.yaml` (the resolved config, including overrides), `train.log`, `metrics_history.json`,
  `training_results.json`, `training_curves.pdf`, `timing.json`, `eval.json`, and a text report with a generated sample;
- `checkpoints/<run_name>/{best,stage}/`.

The default run name is `<model_name>_s<seed>`. `eval.json` holds test metrics in three modes (`legacy32`,
`chunked128`, `full_context`), single-prompt Self-BLEU (`self_bleu_4`), and multi-prompt Self-BLEU
(`self_bleu_4_multi`: Self-BLEU within the samples of each of 10 validation-split prompts, mean ± std over prompts).

### Overriding config values: `--set`

`--set` overrides one dotted config key and can be repeated. Values are parsed as int, float, bool (`true`/`false`) or string.
An unknown key is an error. Overrides are recorded under `run.overrides` in the run's `config.yaml`.

```bash
python train.py --config configs/TR2o.yaml --seed 4212 \
    --set training.learning_rate=6e-4 --set model.attn_dropout=0.1 --set loss.ls_eps=0.0
```

### Hyperparameter search: `--tuning`

`--tuning` never opens the test file. Checkpoint selection works as in a normal run. At the end, the best-loss checkpoint
is scored with full-context evaluation on the **validation** split (stride `evaluation.tuning_val_stride`, default 8),
and the results go to `runs/<run_name>/val_eval.json`. Tuning mode writes no test metrics, no samples and no Self-BLEU.

```bash
python train.py --config configs/TR2o.yaml --seed 4212 --tuning --set training.learning_rate=6e-4 --run-name TR2o_lr6e-4
```

### Other flags

- `--legacy-dropout-rng`: reuse one dropout key on every step (reproduces an old bug; ablation only).
- `--eval-only`: re-run the final evaluation from a finished run's checkpoints.
- `--data-dir DIR`: directory with `text8_train.txt` / `text8_test.txt` (default `data/`).
- `--epochs N`, `--val-interval N`, `--fc-stride S`: for smoke tests only, e.g. `--epochs 300 --val-interval 100 --fc-stride 64`.

`scripts/launch_all.py` queues a batch of runs, one per free GPU (`--smoke` for 300-iteration checks, `--dry-run` to print the queue).

### Notebook

`transformer.ipynb` is an interactive version of `train.py`. It imports its building blocks (data loading, model, loss,
optimizer, training step, evaluation) from `train.py`, so both train the same way.

## Project Layout

```
.
├── train.py                    # Training + evaluation entry point (--set, --tuning, --eval-only)
├── transformer.ipynb           # Interactive notebook built on train.py
│
├── configs/                    # YAML configuration files
│   ├── baseline.yaml          # Default transformer configuration
│   ├── TR2o.yaml, TR8o.yaml   # Transformer runs (RoPE / relative position bias)
│   ├── LR-1.yaml, LR-2.yaml   # LSTM runs (hidden 256 / 314)
│   └── TR2o-nomask.yaml       # TR2o with weight decay on every parameter (ablation)
│
├── models/                     # Core model implementations
│   ├── models.py              # Decoder-only Transformer (MultiHeadAttentionWithPE for every PE type)
│   ├── LSTM.py                # Character-level LSTM (CharLSTM)
│   └── positional_encodings.py # Positional encoding methods
│
├── utils/                      # Training and evaluation utilities
│   ├── config.py              # Configuration loader
│   ├── logger.py              # Metrics logging
│   ├── analysis.py            # Performance analysis
│   ├── plot.py                # Training curve visualization
│   ├── eval.py                # Evaluation metrics (BPC, perplexity, ECE, Self-BLEU, ...)
│   ├── full_context_eval.py   # Sliding-window evaluation (every target scored once)
│   ├── generation.py          # Text generation (exact prompt conditioning, n-gram blocking)
│   ├── test.py                # Block-wise test set evaluation (legacy32 / chunked128)
│   └── checkpoint_saver.py    # Checkpoint management
│
├── tests/                      # pytest suite (runs on CPU)
├── scripts/                    # launch_all.py (run queue), gpu_sampler.py (GPU usage log)
├── results/                    # Committed result tables and reports
│
├── data/
│   ├── text8_train.txt        # Training data (train/validation split 95/5 in train.py)
│   └── text8_test.txt         # Test data
│
├── checkpoints/                # Model checkpoints (auto-created)
├── runs/                       # Experiment outputs (auto-created)
│
├── .github/workflows/          # CI/CD pipelines
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── pyproject.toml              # Ruff and pytest configuration
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Dataset

The **text8** dataset contains the first 100 million characters from a cleaned English Wikipedia dump. It includes only lowercase letters (a-z) and spaces, making it compact and ideal for benchmarking character-level models.

- **Vocabulary size**: 27 tokens (26 letters + space)
- **Training set**: ~85.5M characters (first 95% of `text8_train.txt`)
- **Validation set**: ~4.5M characters (last 5% of `text8_train.txt`)
- **Test set**: 5M characters (`text8_test.txt`)

## Key Training Parameters

**Transformer (`model.type: transformer`):**

| Parameter | Description | Default (`baseline.yaml`) |
|-----------|-------------|---------|
| `max_len` | Sequence length (context window) | 128 |
| `d_model` | Hidden dimension | 256 |
| `n_layers` | Number of transformer layers | 4 |
| `n_heads` | Number of attention heads | 4 |
| `pos_encoding_type` | `learned`, `sinusoidal`, `hybrid`, `rope`, `alibi` or `relative` | `learned` |
| `sinusoidal_scale` | `unit` (sinusoidal rows have norm 1) or `raw` (unscaled, norm √(d_model/2)) | `unit` |
| `emb/attn/mlp/resid_dropout` | Dropout rates | 0.1 |

**LSTM (`model.type: lstm`):**

| Parameter | Description | Default (`LR-1.yaml`) |
|-----------|-------------|---------|
| `hidden_size` | LSTM hidden dimension | 256 |
| `n_layers` | Number of stacked LSTM layers | 4 |
| `dropout` | Variational dropout rate | 0.1 |
| `forget_bias` | Initial forget-gate bias (0.0 = all-zero bias) | 1.0 |
| `max_len` | Sequence length | 128 |

**Shared:**

| Parameter | Description | Default (`baseline.yaml`) |
|-----------|-------------|---------|
| `training.learning_rate` | Peak learning rate | 0.0003 |
| `training.batch_size` | Training batch size | 128 |
| `training.epochs` | Total training **iterations** (random batches) | 5000 |
| `scheduler.warmup_iters` | Linear warmup, then cosine decay to `end_value` at the last iteration | 2000 |
| `loss.ls_eps` | Label smoothing (0 = off) | 0.05 |

## Configuration System

The project uses YAML configuration files for reproducible experiments. All hyperparameters are centralized in config files,
with these sections: `model`, `loss`, `scheduler`, `optimizer` (AdamW settings and the weight-decay mask), `training`,
`evaluation` (sampling, Self-BLEU, full-context and tuning settings) and `output`. Keys missing from older configs get
defaults from `train.resolve_config`. Any key can be overridden from the command line with `--set`.

## Tests

The test suite runs on CPU (`tests/conftest.py` sets `JAX_PLATFORMS=cpu`) and takes a few minutes:

```bash
pytest            # whole suite
pytest tests/test_generation.py -q
```

It covers causality, dropout RNG, the weight-decay mask, parameter counts, full-context evaluation, the learning-rate
schedule, the training loss, generation, Self-BLEU prompts, and the `--set` / `--tuning` command-line options.

## Code Style (Check-Style)

We enforce consistent Python/Notebook style with **Ruff + nbQA + pre-commit**.

- 📄 Full guide: See **[docs/check-style.md](docs/check-style.md)** for details
- 🔁 CI runs style checks automatically on every **Push/Pull Request**
- ✅ **Strongly recommended**: Run local checks before pushing

```bash
ruff check .
nbqa ruff .
pre-commit run --all-files
```

If violations are found in CI, **the PR cannot be merged** until all issues are fixed.

## Branches

This repository contains three main branches for different experimental approaches:

### [`main`](https://github.com/jhqiu21/CharacterLLM/tree/main) - Transformer Baseline
The main branch implements a decoder-only transformer with:
- Multiple positional encoding options (RoPE, ALiBi, Learned, etc.)
- Character-level tokenization (27 tokens)
- Comprehensive evaluation metrics

The LSTM model (`models/LSTM.py`) and `train.py` live alongside the Transformer on the development branches built on `main`.

### [`LSTM-baseline`](https://github.com/jhqiu21/CharacterLLM/tree/LSTM-baseline) - LSTM Baseline
Alternative architecture using LSTM for comparison:
- Character-level LSTM model
- Useful for comparing transformer vs recurrent approaches
- May achieve competitive results with simpler architecture

### [`subword-tokenization-baseline`](https://github.com/jhqiu21/CharacterLLM/tree/LSTM-baseline) - Subword Tokenization
Transformer with subword tokenization:
- BPE/WordPiece tokenization instead of character-level
- Larger vocabulary, shorter sequences
- Different trade-offs in modeling granularity


## Acknowledgments

- Text8 dataset from Matt Mahoney's compression benchmark
- The naive model baseline from alexxthiery's [char_transformer repository](https://github.com/alexxthiery/char_transformer)

---

**Questions or Issues?** Open an issue on GitHub or check the documentation in `docs/`.
