# CharacterLLM


## Table of Contents

- [Setup](#setup)
- [Project layout (high-level)](#project-layout-high-level)
- [Data](#data)
- [Run & Develop](#run--develop)
- [Code Style (Check-Style)](#code-style-check-style)



## Setup

### Environment

This project uses **Conda** for environment management.

You can create the environment using the provided [`environment.yml`](environment.yml) file

```bash
conda env create -f environment.yml
```

Once created, activate the environment
```bash
conda activate charllm-env
```

To deactivate the environment
```bash
conda deactivate
```

## Project layout (high-level)
[TODO]
```bash
.
├─ src/                  # Core code (model, training utilities, tokenization, etc.)
├─ notebooks/            # Experiments and EDA in Jupyter
├─ data/                 # Data directory (raw / processed)
├─ tests/                # Unit/integration tests (if applicable)
├─ .github/workflows/    # CI (CheckStyle pipeline)
├─ .pre-commit-config.yaml
├─ pyproject.toml
└─ README.md

```

## Data
[TODO]

## Run & Develop
[TODO]

## Code Style (Check-Style)
We enforce a consistent Python/Notebook style with **Ruff + nbQA + pre-commit**.

- 📄 Full guide: You may find more details **[here](docs/check-style.md/#check-style-guide)**.
- 🔁 CI runs style checks automatically on every **Push/Pull Request**.
  If violations are found, **the PR cannot be merged** until all issues are fixed.
- ✅ Strongly recommended before pushing: run local checks and fix issues.
