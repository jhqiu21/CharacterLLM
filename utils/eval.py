import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import optax


def perplexity(logits, targets):
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    
    # Calculate log probabilities
    log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
    
    # Get log probability of correct tokens
    correct_log_probs = jnp.take_along_axis(
        log_probs, 
        flat_targets[:, None], 
        axis=-1
    ).squeeze(-1)
    
    # Average negative log likelihood (cross-entropy)
    avg_loss = -jnp.mean(correct_log_probs)
    
    # Perplexity = exp(loss)
    perplexity = jnp.exp(avg_loss)
    
    return perplexity, avg_loss


def bits_per_character(loss):
    return loss / jnp.log(2)


def accuracy(logits, targets):
    preds = jnp.argmax(logits, axis=-1)
    is_match = (preds == targets)
    acc_all = is_match.astype(jnp.float32).mean()
    acc_last = is_match.astype(jnp.float32)[:, -1].mean()
    return acc_all, acc_last


def cross_entropy_last_token_only(logits, targets):
    last_logits = logits[:, -1, :]
    last_targets = targets[:, -1]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        last_logits, last_targets
    )
    return jnp.mean(loss)


def self_BELU():
    pass


def ECE():
    pass


def distinctN():
    pass


def coherence():
    pass