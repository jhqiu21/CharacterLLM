import jax
import jax.numpy as jnp
import numpy as np

import optax
import itertools

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import generation


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


def token_frequency_analysis(logits, targets, token_counts, top_percent=0.2):
    """
    Analyze accuracy for common vs rare tokens.
    Args:
        logits: (B, T, V) array
        targets: (B, T) array
        token_counts: Array of shape (V,) with token frequencies in training data
        top_percent: fraction of tokens considered 'common' or 'rare'
    Returns:
        freq_stats: Dict with frequency-based statistics (overall, common, and rare token accuracy)
    """
    # Step 1: Predictions and correctness
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets).astype(jnp.float32)

    # Step 2: Flatten batch and sequence dimensions
    targets_flat = targets.flatten()
    correct_flat = correct.flatten()

    # Step 3: Determine thresholds for common and rare tokens
    V = token_counts.shape[0]
    sorted_indices = jnp.argsort(token_counts)  # ascending: rare → common
    n_top = max(1, int(V * top_percent))

    rare_tokens = sorted_indices[:n_top]
    common_tokens = sorted_indices[-n_top:]

    # Step 4: Accuracy for rare tokens
    rare_mask = jnp.isin(targets_flat, rare_tokens)
    rare_acc = correct_flat[rare_mask].mean() if rare_mask.sum() > 0 else jnp.nan

    # Step 5: Accuracy for common tokens
    common_mask = jnp.isin(targets_flat, common_tokens)
    common_acc = correct_flat[common_mask].mean() if common_mask.sum() > 0 else jnp.nan

    # Step 6: Overall accuracy
    overall_acc = correct_flat.mean()

    freq_stats = {
        "overall_accuracy": float(overall_acc),
        "common_accuracy": float(common_acc),
        "rare_accuracy": float(rare_acc),
    }

    return freq_stats


def self_bleu(
    model,
    config,
    int_to_char: dict,
    char_to_int: dict,
    char_set,
    params,
    prompt: str,
    gen_len: int,
    temperature: float,
    sample: bool,
    seed: int,
    n_grams: int = 4,
    n_samples: int = 20) -> float:
    """
    model, config, int_to_char, char_to_int, char_set: defined else where in the cells before
    params: defined in the loop
    prompt, gen_len, temperature, sample, seed: defined at the start of the evaluation cell
    n_grams: number of grams for self-BLEU (default: 4)
    n_samples: number of samples to generate for self-BLEU (default: 20)
    """
    # Generate continuations
    rng_base = jax.random.PRNGKey(seed)
    prompt_int = jnp.array(
        [[char_to_int.get(c, len(char_set)) for c in prompt.lower()[:config.model.max_len]]],
        dtype=jnp.int32
    )
    keys = jax.random.split(rng_base, n_samples)

    continuations = []
    for k in keys:
        out_ids_i = generation.generate_tokens(
            model, params, k, prompt_int, gen_len,
            block_size=config.model.max_len,
            temperature=temperature,
            sample=sample
        )
        cont_i = ''.join(int_to_char.get(int(x), '?') for x in list(out_ids_i[0])).strip()
        continuations.append(cont_i)
    texts = [c.split() for c in continuations]

    # Compute self-BLEU score
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

    self_sb = sum(scores) / len(scores)
    print(f"\t \tSelf-BLEU-{n_grams}: {self_sb:.4f}")
    return self_sb


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

    # flatten 2D list if needed
    if isinstance(tokens, np.ndarray) and tokens.ndim == 2:
        tokens = tokens.flatten()
    elif isinstance(tokens[0], list) or isinstance(tokens[0], np.ndarray):
        tokens = np.array(tokens).flatten()

    # extract n-grams
    n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if len(n_grams) == 0:
        return 0.0  # avoid division by zero

    unique_n_grams = set(n_grams)
    distinct_score = len(unique_n_grams) / len(n_grams)
    return distinct_score


def coherence_score(tokens, int_to_char):
    """
    Simple coherence metric based on valid English-like patterns.
    Args:
        tokens: List of token IDs
        int_to_char: Mapping from token ID to character
    Returns:
        coherence: Simple coherence score (0-1)
    """
    IDEAL_WORD_LENGTH = 5.0
    WORD_LENGTH_RANGE = 10.0
    REPEAT_PENALTY_SCALE = 10.0

    text = ''.join(int_to_char.get(int(t), '?') for t in tokens)
    max_repeat = max((len(list(g)) for k, g in itertools.groupby(text)), default=0)
    repeat_penalty = 1.0 / (1.0 + max_repeat / REPEAT_PENALTY_SCALE)
    words = text.split(' ')
    avg_word_len = np.mean([len(w) for w in words if w]) if any(w for w in words) else 0
    word_len_score = 1.0 - abs(avg_word_len - IDEAL_WORD_LENGTH) / WORD_LENGTH_RANGE

    word_len_score = np.clip(word_len_score, 0, 1)

    coherence = (repeat_penalty + word_len_score) / 2.0

    return float(coherence)
