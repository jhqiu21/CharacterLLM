# JITted token generator using jax.lax.scan - adds autoregressive sampling for the trained model
import jax
import jax.numpy as jnp
from collections import defaultdict


def sample_categorical(rng, logits, temperature=1.0):
    """Sample from categorical distribution given by logits with temperature."""
    # logits: (B, V)
    if temperature != 1.0:
        logits = logits / temperature
    return jax.random.categorical(rng, logits, axis=-1)


def initial_window(context, block_size, fill_token=None):
    """Place the prompt at positions 0..S-1 of a fixed (B, block_size) window.

    Prompts longer than the window keep their last `block_size` tokens. The positions after the
    prompt only hold a filler (by default the last prompt token repeated; `fill_token` overrides it).
    The models are causal, so the filler never affects the logits at positions < S that are read.
    Returns (window, number of real tokens S).
    """
    context = jnp.asarray(context, dtype=jnp.int32)
    assert context.ndim == 2 and context.shape[1] >= 1, "context must have shape (B, S) with S >= 1"
    context = context[:, -block_size:]
    S = context.shape[1]
    if S == block_size:
        return context, S
    if fill_token is None:
        window = jnp.pad(context, ((0, 0), (0, block_size - S)), mode="edge")
    else:
        window = jnp.pad(context, ((0, 0), (0, block_size - S)), constant_values=fill_token)
    return window, S


def next_token_logits(model, params, window, cur_len):
    """Logits (B, V) for the token after the cur_len real tokens at the start of `window`."""
    logits = model.apply({"params": params}, window, deterministic=True)  # (B, block_size, V)
    return jax.lax.dynamic_index_in_dim(logits, cur_len - 1, axis=1, keepdims=False)


def _advance(window, cur_len, next_token, block_size):
    """Write `next_token` (B,) at position cur_len, or slide the window left by one once it is full."""
    written = jax.lax.dynamic_update_slice_in_dim(window, next_token[:, None], jnp.minimum(cur_len, block_size - 1), axis=1)
    slid = jnp.concatenate([window[:, 1:], next_token[:, None]], axis=1)
    return jnp.where(cur_len < block_size, written, slid), jnp.minimum(cur_len + 1, block_size)


def generate_tokens(model, params, rng, context, length, block_size=64, temperature=1.0, sample=True, ngram_size=None,
                    fill_token=None):
    """Generate `length` new tokens autoregressively starting from `context`.

    The prompt occupies positions 0..S-1 of a fixed-size window (no padding before it); the logits are
    read at the last real position cur_len-1 and the sampled token is written at cur_len. Once the window
    is full it slides by one token per step. Because the models are causal, this conditions exactly on the
    real tokens, while every forward pass keeps the shape (B, block_size).

    Args:
      model: Flax/linen model object with apply signature model.apply({'params': params}, tokens, deterministic=True)
      params: model parameters pytree.
      rng: jax PRNGKey (e.g., jax.random.PRNGKey(0)).
      context: int32 array shape (B, S), S >= 1. If S > block_size only the last block_size tokens are used.
      length: number of tokens to generate (int).
      block_size: model context window (static).
      temperature: sampling temperature (static).
      sample: if True use sampling; if False use argmax (greedy).
      ngram_size: if set, block repeated n-grams (slow Python loop, batch size 1 only).
      fill_token: token placed after the prompt (never read; default: the last prompt token).

    Returns:
      generated: int32 array shape (B, length) of generated token ids.
    """
    context = jnp.asarray(context, dtype=jnp.int32)
    B = context.shape[0]
    window, S = initial_window(context, block_size, fill_token)
    temp = temperature if temperature > 0 else 1.0

    if ngram_size is None:
        def _step(carry, _):
            rng, ctx, cur_len = carry  # ctx: (B, block_size); positions < cur_len hold real tokens
            last_logits = next_token_logits(model, params, ctx, cur_len)  # (B, V)
            rng, subkey = jax.random.split(rng)
            if sample:
                next_token = jax.random.categorical(subkey, last_logits / temp, axis=-1)
            else:
                next_token = jnp.argmax(last_logits, axis=-1)
            next_token = next_token.astype(jnp.int32)
            ctx, cur_len = _advance(ctx, cur_len, next_token, block_size)
            return (rng, ctx, cur_len), next_token

        _, tokens = jax.lax.scan(_step, (rng, window, jnp.int32(S)), None, length=length)
        # tokens: shape (length, B) -> (B, length)
        return tokens.transpose(1, 0)

    assert B == 1, "ngram_size is currently implemented only for batch size 1"

    # history = prompt tokens only (no filler)
    history = list(map(int, context[0].tolist()))
    ctx, cur_len = window, S
    generated_tokens = []

    for _ in range(length):
        # forward pass: logits at the last real position
        last_logits = next_token_logits(model, params, ctx, cur_len)[0]  # (V,)

        # apply n-gram blocking
        if ngram_size is not None and ngram_size > 1:
            last_logits = block_repeated_ngrams(last_logits, history, ngram_size)

        # sampling / greedy (same calls and shapes as the scan path, so both paths agree without blocking)
        rng, subkey = jax.random.split(rng)
        if sample:
            next_token = jax.random.categorical(subkey, last_logits[None] / temp, axis=-1)[0]
        else:
            next_token = jnp.argmax(last_logits, axis=-1)

        next_token = int(next_token)
        generated_tokens.append(next_token)
        history.append(next_token)

        # write at cur_len, or slide once the window is full
        if cur_len < block_size:
            ctx = ctx.at[0, cur_len].set(next_token)
            cur_len += 1
        else:
            ctx = jnp.concatenate([ctx[:, 1:], jnp.array([[next_token]], dtype=jnp.int32)], axis=1)

    # return shape (1, length) to match original interface
    return jnp.array([generated_tokens], dtype=jnp.int32)
# example usage:
# generated = generate_tokens(model, params, rng, context, length=100, block_size=64, temperature=1.0, sample=True)
# print(generated)  # (B, 100) array of generated token ids


def block_repeated_ngrams(logits, history, n):
    """
    logits: 1D jnp.array, shape (V,)  — logits for next token
    history: list[int]               — prompt + all generated token ids so far
    n: int                           — n-gram size to block (e.g. 3 or 4)

    Returns:
        blocked_logits: 1D jnp.array (V,) with some positions set to -1e9
    """
    seq_len = len(history)
    if seq_len < n - 1:
        # not enough tokens to form an n-gram yet
        return logits

    # Build: prefix (tuple of n-1 tokens) -> set of forbidden next tokens
    forbidden = defaultdict(set)
    for i in range(seq_len - n + 1):
        prefix = tuple(history[i : i + n - 1])
        next_tok = history[i + n - 1]
        forbidden[prefix].add(next_tok)

    # Current prefix = last n-1 tokens
    current_prefix = tuple(history[-(n - 1):])
    bad_tokens = forbidden.get(current_prefix, set())
    if not bad_tokens:
        return logits

    blocked_logits = logits.copy()
    neg_inf = -1e9
    for t in bad_tokens:
        if 0 <= t < blocked_logits.shape[0]:
            blocked_logits = blocked_logits.at[t].set(neg_inf)

    return blocked_logits
