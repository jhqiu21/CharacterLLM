# Check-Style Guide

## Tools & repo files

- **Ruff**: fast linter/formatter. We enable **preview** + **E2** so whitespace errors are caught.
- **nbQA**: runs Ruff on Jupyter notebooks.
- **pre-commit**: unified entry for local & CI.
- **GitHub Actions**: runs on PR/Push.

Key files in the repo:
- `pyproject.toml`: Ruff config file
- `.pre-commit-config.yaml`: pre-commit config file
- `.github/workflows/lint.yml`: CI workflow

## CI behavior (PR/Push)

Our GitHub Action runs automatically whenever you **push** or open/update a **pull request**.
If the job finds style violations, **the PR cannot be merged** until all issues are fixed and the check passes.

**Strongly recommended:** before committing/pushing, run the local style checks and fix all reported issues (see the [**Quick start**](#quick-start-local) section for environment setup and commands).
For additional safety, enable branch protection to require the **CheckStyle** job to pass before merge.


## Quick start (local)
1. Set up your working environment. You may refer [here](../README.md/#environment)

1. activate the environment:
    ```bash
    conda activate charllm-env
    ```

1. Install the Git hook (we default to manual stage; this is not required)
    ```bash
    pre-commit install -f --hook-type pre-commit
    ```

1. Check code style locally

    - Run on the entire working tree
        ```bash
        ruff check .
        ```
    - Run on specific files
        ```bash
        ruff check filename.py
        ```
1. (Optional) Fix style bug automatically
    ```bash
    ruff check . --fix
    ```
