import jax
import jax.numpy as jnp

import optax

def loss_all(logits, targets):
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    per_pos = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets)
    loss = per_pos.mean()
    return loss


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


def expected_calibration_error(logits, targets, n_bins=20):
    probs = jax.nn.softmax(logits, axis=-1)
    max_probs = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)

    # Flatten all arrays
    confidences = max_probs.reshape(-1)
    predictions_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    correct = (predictions_flat == targets_flat).astype(jnp.float32)

    # Create bin edges
    bin_boundaries = jnp.linspace(0, 1, n_bins + 1)

    # Assign each confidence to a bin
    # digitize returns bin indices in range [0, n_bins]
    # We use bins[1:-1] to get interior boundaries
    bin_indices = jnp.digitize(confidences, bin_boundaries[1:-1])

    # Vectorized computation of bin statistics
    def compute_bin_stats(bin_idx):
        """Compute statistics for a single bin using pure JAX operations."""
        # Create mask for samples in this bin
        in_bin = (bin_indices == bin_idx).astype(jnp.float32)
        bin_size = jnp.sum(in_bin)

        # Prevent division by zero: if bin is empty, use 1.0 as denominator
        # The contribution will be masked out anyway
        safe_bin_size = jnp.where(bin_size > 0, bin_size, 1.0)

        # Average confidence and accuracy in this bin
        bin_confidence = jnp.sum(confidences * in_bin) / safe_bin_size
        bin_accuracy = jnp.sum(correct * in_bin) / safe_bin_size

        # Contribution to ECE (weighted by proportion of samples in bin)
        weight = bin_size / confidences.shape[0]
        gap = jnp.abs(bin_confidence - bin_accuracy)
        contribution = weight * gap

        # If bin is empty, set contribution to 0
        contribution = jnp.where(bin_size > 0, contribution, 0.0)

        return contribution, bin_confidence, bin_accuracy, bin_size, gap

    # Apply function to all bins using vmap
    bin_results = jax.vmap(compute_bin_stats)(jnp.arange(n_bins))
    contributions, bin_confidences, bin_accuracies, bin_sizes, gaps = bin_results

    # Sum all contributions to get final ECE
    ece = jnp.sum(contributions)

    # Prepare detailed calibration data
    # Note: This dict contains JAX arrays and is suitable for returning
    calibration_data = {
        'ece': ece,  # Keep as JAX array for now
        'n_bins': n_bins,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_sizes': bin_sizes,
        'bin_gaps': gaps,
        'bin_edges': bin_boundaries,
        'mean_confidence': jnp.mean(confidences),
        'mean_accuracy': jnp.mean(correct),
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
        'mean_confidence': jnp.mean(top1_conf),
        'median_confidence': jnp.median(top1_conf),
        'mean_top5_confidence': jnp.mean(top5_conf),
        'correct_confidence': correct_conf,
        'incorrect_confidence': incorrect_conf,
        'confidence_gap': correct_conf - incorrect_conf,
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


def self_BLEU(generated_samples, n_gram=4):
    """
    Calculate Self-BLEU score to measure diversity between samples.
    Lower is better (more diverse samples).
    Args:
        generated_samples: List of token sequences
        n_gram: Maximum n-gram size for BLEU
    Returns:
        self_bleu: Average BLEU score between each sample and others
    """
    pass


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
