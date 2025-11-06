import jax
import jax.numpy as jnp

import optax

from typing import List, Sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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


def expected_calibration_error(logits, targets, n_bins=10):
    """
    Expected Calibration Error (ECE) measures how well-calibrated the model's confidence is.
    Lower is better (0 = perfectly calibrated).
    n_bins: Number of bins for calibration
    Returns:
        ece: Expected Calibration Error (scalar)
        calibration_data: Dict with detailed calibration info
    """
    probs = jax.nn.softmax(logits, axis=-1)
    max_probs = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)

    # Flatten
    confidences = max_probs.reshape(-1)
    predictions_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    correct = (predictions_flat == targets_flat).astype(jnp.float32)

    # Bin edges
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)

    # Calculate ECE
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        # Find samples in this bin
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == n_bins - 1:  # Last bin includes right boundary
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        bin_size = jnp.sum(in_bin)

        # Skip empty bins
        if bin_size > 0:
            # Average confidence in bin
            bin_confidence = jnp.sum(confidences * in_bin) / bin_size

            # Average accuracy in bin
            bin_accuracy = jnp.sum(correct * in_bin) / bin_size

            # Weighted contribution to ECE
            ece += (bin_size / len(confidences)) * jnp.abs(bin_confidence - bin_accuracy)

            bin_data.append({
                'confidence': float(bin_confidence),
                'accuracy': float(bin_accuracy),
                'count': int(bin_size),
                'gap': float(jnp.abs(bin_confidence - bin_accuracy))
            })

    calibration_data = {
        'ece': float(ece),
        'n_bins': n_bins,
        'bins': bin_data,
        'mean_confidence': float(jnp.mean(confidences)),
        'mean_accuracy': float(jnp.mean(correct)),
    }

    return ece, calibration_data


def brier_score(logits, targets):
    # Measures the MSE between predicted probabilities and actual outcomes.
    probs = jax.nn.softmax(logits, axis=-1)
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)

    squared_diff = (probs - targets_one_hot) ** 2
    brier = jnp.mean(squared_diff)

    return brier


def prediction_entropy(logits):
    # Entropy of predictions (uncertainty measure).
    # High: model is uncertain
    # Low: model is confident

    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)

    avg_entropy = jnp.mean(entropy)
    entropy_by_position = jnp.mean(entropy, axis=0)

    return avg_entropy, entropy_by_position


def confidence_distribution(logits, targets):
    # Analyze confidence distribution.
    probs = jax.nn.softmax(logits, axis=-1)

    # Top-1 confidence
    top1_conf = jnp.max(probs, axis=-1)

    # Top-5 cumulative confidence
    top5_probs = jnp.sort(probs, axis=-1)[:, :, -5:]
    top5_conf = jnp.sum(top5_probs, axis=-1)

    # Correct predictions
    predictions = jnp.argmax(probs, axis=-1)
    correct = (predictions == targets).astype(jnp.float32)

    # Confidence of correct predictions
    correct_mask = correct[..., None]
    correct_conf = jnp.sum(top1_conf * correct) / jnp.sum(correct)

    # Confidence of incorrect predictions
    incorrect_mask = (1 - correct)
    incorrect_conf = jnp.sum(top1_conf * incorrect_mask) / jnp.sum(incorrect_mask)

    return {
        'mean_confidence': float(jnp.mean(top1_conf)),
        'median_confidence': float(jnp.median(top1_conf)),
        'mean_top5_confidence': float(jnp.mean(top5_conf)),
        'correct_confidence': float(correct_conf),
        'incorrect_confidence': float(incorrect_conf),
        'confidence_gap': float(correct_conf - incorrect_conf),
    }


def token_frequency_analysis(logits, targets, token_counts):
    """
    Analyze accuracy for common vs rare tokens.
    Args:
        logits: (B, T, V) array
        targets: (B, T) array
        token_counts: Array of shape (V,) with token frequencies in training data
    Returns:
        freq_stats: Dict with frequency-based statistics
    """
    pass


def self_bleu(
    texts: Sequence[Sequence[str]],
    n_grams: int = 4) -> float:
    """
    texts: list of tokenized sentences, e.g., [['this','is','a','test'], ...]
    weights: BLEU n-gram weights (default BLEU-4)
    smoothing: apply NLTK smoothing (method3 is common)

    ref: list(list(str))
    cand: list(str)
    """

    weights = [(1.0 / n_grams) for _ in range(n_grams)]

    n = len(texts)
    if n < 2:
        return 0.0
    
    scores = []
    for i in range(n):
        cand = list(texts[i])
        refs = [list(texts[j]) for j in range(n) if j != i]
        if not refs:
            continue
        smoothie = SmoothingFunction().method3
        score = sentence_bleu(refs, cand, weights=weights, smoothing_function=smoothie)
        scores.append(score)

    return sum(scores) / len(scores)


def distinct_n(tokens, n=2):
    """
    Calculate Distinct-N metric for generated text.
    Measures vocabulary diversity. Higher is better.
    Args:
        tokens: List or array of token IDs
        n: N-gram size (1, 2, or 4 typically)
    Returns:
        distinct_score: Ratio of unique n-grams to total n-grams
    """
    pass


def coherence_score(tokens, int_to_char):
    """
    Simple coherence metric based on valid English-like patterns.
    This is a placeholder - you can implement more sophisticated coherence measures.
    Args:
        tokens: List of token IDs
        int_to_char: Mapping from token ID to character
    Returns:
        coherence: Simple coherence score (0-1)
    """
    pass
