"""
Configuration loader for model training.
Supports both JSON and YAML configuration files.
"""

import json
import yaml
from pathlib import Path


class Config:
    """Configuration container with attribute access."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert configuration back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        """String representation of configuration."""
        return f"Config({self.to_dict()})"

    def __str__(self):
        """Pretty print configuration."""
        return json.dumps(self.to_dict(), indent=2)


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine file type and load
    suffix = config_path.suffix.lower()

    if suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .json, .yaml, or .yml")

    return Config(config_dict)


def save_config(config, save_path):
    """
    Save configuration to file.
    Args:
        config: Config object or dictionary
        save_path: Path to save configuration file
    """
    save_path = Path(save_path)

    # Convert Config to dict if necessary
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    # Save based on file extension
    suffix = save_path.suffix.lower()

    if suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif suffix in ['.yaml', '.yml']:
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"Unsupported save format: {suffix}. Use .json, .yaml, or .yml")

    print(f"✓ Configuration saved to '{save_path}'")


def print_config(config):
    """
    Pretty print configuration.
    Args:
        config: Config object or dictionary
    """
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)

    def print_nested(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_nested(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_nested(config_dict)
    print("=" * 60 + "\n")


# Default configuration template
DEFAULT_CONFIG = {
    "model_name": "transformer_baseline",
    "model": {
        "vocab_size": 27,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 2,
        "max_len": 128,
        "mlp_ratio": 4,
        "emb_dropout": 0.2,
        "mlp_dropout": 0.2,
        "attn_dropout": 0.1,
        "resid_dropout": 0.2
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0003,
        "max_iters": 50000,
        "eval_interval": 500,
        "eval_iters": 100,
        "warmup_iters": 2000,
        "checkpoint_interval": 5000,
        "log_interval": 100,
        "seed": 0
    },
    "data": {
        "train_path": "./data/text8_train.txt",
        "test_path": "./data/text8_test.txt"
    },
    "output": {
        "checkpoint_dir": "./checkpoints",
        "results_dir": "./results"
    }
}


def create_default_config(save_path):
    """
    Create and save a default configuration file.
    Args:
        save_path: Path where to save the default config
    """
    save_config(DEFAULT_CONFIG, save_path)
    print(f"✓ Default configuration created at '{save_path}'")
