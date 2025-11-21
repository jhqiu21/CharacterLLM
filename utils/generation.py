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


# @functools.partial(jax.jit, static_argnames=("model", "block_size", "temperature", "sample"))
def generate_tokens(model, params, rng, context, length, block_size=64, temperature=1.0, sample=True, ngram_size=None):
    """Generate `length` new tokens autoregressively starting from `context`.

    Args:
      model: Flax/linen model object with apply signature model.apply({'params': params}, tokens)
      params: model parameters pytree.
      rng: jax PRNGKey (e.g., jax.random.PRNGKey(0)).
      context: int32 array shape (B, S) where S <= block_size. If S < block_size, left-pad with zeros or use full context.
      length: number of tokens to generate (int).
      block_size: model context window (static).
      temperature: sampling temperature (static).
      sample: if True use sampling; if False use argmax (greedy).

    Returns:
      generated: int32 array shape (B, length) of generated token ids.
    """
    B = context.shape[0]
    S = context.shape[1]
    assert S <= block_size, "context length must be <= block_size"

    # initialize running context: if S < block_size, left-pad with zeros (or use a preferred pad token)
    if ngram_size is None:
        if S < block_size:
            pad_len = block_size - S
            context = jnp.concatenate([jnp.zeros((B, pad_len), dtype=jnp.int32), context], axis=1)

        def _step(carry, _):
            rng, ctx = carry  # rng: PRNGKey, ctx: (B, block_size)
            # forward pass: get logits for all positions, take last position
            logits = model.apply({"params": params}, ctx, deterministic=True)  # (B, block_size, V)
            last_logits = logits[:, -1, :]  # (B, V)
            rng, subkey = jax.random.split(rng)
            if sample:
                next_token = jax.random.categorical(subkey, last_logits / (temperature if temperature > 0 else 1.0), axis=-1)
            else:
                next_token = jnp.argmax(last_logits, axis=-1)
            next_token = next_token.astype(jnp.int32)
            # append to context: drop first token, append new token at end
            next_token_col = next_token.reshape(B, 1)
            new_ctx = jnp.concatenate([ctx[:, 1:], next_token_col], axis=1)
            return (rng, new_ctx), next_token_col

        # run scan for `length` steps
        (rng_final, ctx_final), tokens = jax.lax.scan(_step, (rng, context), None, length=length)
        # tokens: shape (length, B, 1) -> reshape to (B, length)
        tokens = tokens.squeeze(-1).transpose(1, 0)
        return tokens

    assert B == 1, "ngram_size is currently implemented only for batch size 1"

    # history = prompt tokens only (no padding)
    history = list(map(int, context[0].tolist()))

    # prepare padded context for the model
    if S < block_size:
        pad_len = block_size - S
        ctx = jnp.concatenate(
            [jnp.zeros((B, pad_len), dtype=jnp.int32), context],
            axis=1
        )
    else:
        ctx = context

    generated_tokens = []

    for _ in range(length):
        # forward pass
        logits = model.apply({"params": params}, ctx, deterministic=True)  # (1, block_size, V)
        last_logits = logits[:, -1, :][0]  # (V,)

        # apply n-gram blocking
        if ngram_size is not None and ngram_size > 1:
            last_logits = block_repeated_ngrams(last_logits, history, ngram_size)

        # sampling / greedy
        if sample:
            rng, subkey = jax.random.split(rng)
            next_token = jax.random.categorical(
                subkey,
                last_logits / (temperature if temperature > 0 else 1.0),
                axis=-1,
            )
        else:
            next_token = jnp.argmax(last_logits, axis=-1)

        next_token = int(next_token)
        generated_tokens.append(next_token)
        history.append(next_token)

        # update ctx: drop first token, append new one
        next_token_arr = jnp.array([[next_token]], dtype=jnp.int32)
        ctx = jnp.concatenate([ctx[:, 1:], next_token_arr], axis=1)

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
