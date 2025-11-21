# CharacterLLM

A character-level language modeling project implementing Transformer and LSTM baselines for next-character prediction on the text8 dataset. This project explores various architectural choices, positional encodings, and training strategies to achieve publication-quality results.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Project Layout](#project-layout)
- [Data](#data)
- [Run & Develop](#run--develop)
- [Configuration System](#configuration-system)
- [Code Style (Check-Style)](#code-style-check-style)
- [Branches](#branches)

## Overview

This project implements and trains small language models for character-level text generation using the text8 dataset (100M characters from Wikipedia). 

**Key Features:**
- Multiple positional encoding strategies (Learned, Sinusoidal, RoPE, ALiBi, Hybrid, Relative)
- Comprehensive evaluation metrics (BPC, perplexity, accuracy, ECE, Self-BLEU)
- Advanced loss weighting schemes (linear, sqrt, uniform)
- N-gram repetition prevention during generation
- Systematic checkpointing and experiment tracking
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

## Project Layout

```
.
├── configs/                    # YAML configuration files
│   └── baseline.yaml          # Default transformer configuration
│
├── models/                     # Core model implementations
│   ├── models.py              # Transformer architecture
│   └── positional_encodings.py # Various positional encoding methods
│
├── utils/                      # Training and evaluation utilities
│   ├── config.py              # Configuration loader
│   ├── logger.py              # Metrics logging
│   ├── analysis.py            # Performance analysis
│   ├── plot.py                # Training curve visualization
│   ├── eval.py                # Evaluation metrics (BPC, perplexity, etc.)
│   ├── generation.py          # Text generation with n-gram blocking
│   ├── test.py                # Test set evaluation
│   └── checkpoint_saver.py    # Checkpoint management
│
├── transformer.ipynb           # Main training notebook
│
├── data/                       # Data directory (create this)
│   ├── text8_train.txt        # Training data (download required)
│   └── text8_test.txt         # Test data (download required)
│
├── checkpoints/                # Model checkpoints (auto-created)
├── runs/                       # Experiment results (auto-created)
│
├── .github/workflows/          # CI/CD pipelines
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── pyproject.toml              # Ruff linter configuration
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Data

### Dataset Information

The **text8** dataset contains the first 100 million characters from a cleaned English Wikipedia dump. It includes only lowercase letters (a-z) and spaces, making it compact and ideal for benchmarking character-level models.

- **Vocabulary size**: 27 tokens (26 letters + space)
- **Training set**: ~90M characters
- **Validation set**: ~5M characters  
- **Test set**: ~5M characters

## Run & Develop

### Quick Start

1. **Open the training notebook**:
   ```bash
   jupyter lab transformer.ipynb
   ```

1. **Configure your experiment**:
   - Edit `configs/baseline.yaml` or create a new config file
   - Key parameters to adjust:
     - `model.max_len`: Sequence length (L)
     - `model.pos_encoding_type`: Positional encoding method
     - `training.epochs`: Number of training iterations
     - `loss.tail_scheme`: Position weighting scheme

1. **Run training**:
   - Execute cells sequentially in the notebook
   - Training progress will be logged every `validation_interval` iterations
   - Checkpoints are automatically saved to `checkpoints/`
   - Results and plots are saved to `runs/`




### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_len` | Sequence length (context window) | 128 |
| `d_model` | Hidden dimension | 256 |
| `n_layers` | Number of transformer layers | 4 |
| `n_heads` | Number of attention heads | 4 |
| `pos_encoding_type` | Positional encoding method | 'learned' |
| `learning_rate` | Peak learning rate | 0.0003 |
| `batch_size` | Training batch size | 128 |
| `epochs` | Total training iterations | 5000 |

## Configuration System

The project uses YAML configuration files for reproducible experiments. All hyperparameters are centralized in config files.

## Code Style (Check-Style)

We enforce consistent Python/Notebook style with **Ruff + nbQA + pre-commit**.

- 📄 Full guide: See **[docs/check-style.md](docs/check-style.md)** for details
- 🔁 CI runs style checks automatically on every **Push/Pull Request**
- ✅ **Strongly recommended**: Run local checks before pushing

If violations are found in CI, **the PR cannot be merged** until all issues are fixed.

## Branches

This repository contains three main branches for different experimental approaches:

### [`main`](https://github.com/jhqiu21/CharacterLLM/tree/main) - Transformer Baseline
The current branch implements a decoder-only transformer with:
- Multiple positional encoding options (RoPE, ALiBi, Learned, etc.)
- Character-level tokenization (27 tokens)
- Comprehensive evaluation metrics

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