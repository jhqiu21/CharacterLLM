"""
Different Positional Encoding Method implementations for Transformer models.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import math


"""Sinusoidal positional encodings"""


def get_sinusoidal_embeddings(max_len: int, d_model: int) -> jnp.ndarray:
    position = jnp.arange(max_len)[:, None]  # (max_len, 1)
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))

    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    return pe


class SinusoidalPositionalEncoding(nn.Module):

    d_model: int
    max_len: int

    def setup(self):
        # Pre-compute sinusoidal encodings (not learnable)
        self.pe = get_sinusoidal_embeddings(self.max_len, self.d_model)

    def __call__(self, x):
        # x: Input embeddings of shape (B, T, D)
        T = x.shape[1]
        return x + self.pe[:T]


"""RoPE Positional Encodings"""


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(max_len)
    freqs = jnp.outer(t, freqs)  # (max_len, dim//2)
    # Return as complex numbers for rotation
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    # Reshape x to treat pairs of dimensions as complex numbers
    x_complex = jax.lax.complex(x[..., ::2], x[..., 1::2])  # (B, T, n_heads, head_dim//2)

    # Apply rotation
    freqs_cis = freqs_cis[:x.shape[1]]  # Truncate to sequence length
    freqs_cis = freqs_cis[None, :, None, :]  # (1, T, 1, head_dim//2)

    x_rotated = x_complex * freqs_cis

    # Convert back to real representation
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1)
    x_out = x_out.reshape(x.shape)

    return x_out


class RotaryPositionalEmbedding(nn.Module):
    # Applied within the attention mechanism, not to embeddings directly.
    head_dim: int
    max_len: int
    theta: float = 10000.0

    def setup(self):
        self.freqs_cis = precompute_freqs_cis(self.head_dim, self.max_len, self.theta)

    def __call__(self, q, k):
        """Apply RoPE to queries and keys."""
        q_rotated = apply_rotary_emb(q, self.freqs_cis)
        k_rotated = apply_rotary_emb(k, self.freqs_cis)
        return q_rotated, k_rotated


"""ALiBi Positional Encodings"""


def get_alibi_slopes(n_heads: int):
    # Get ALiBi slopes for each attention head.

    def get_slopes_power_of_2(n):
        # n is a Python int here, safe to use
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return jnp.array([start * (ratio ** i) for i in range(n)])

    # Check if power of 2 using Python math (not JAX)
    if math.log2(n_heads) % 1 == 0:
        return get_slopes_power_of_2(n_heads)
    else:
        # Closest power of 2 - use Python math throughout
        closest_power_of_2 = int(2 ** math.floor(math.log2(n_heads)))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
        extra_slopes = extra_slopes[0::2][:n_heads - closest_power_of_2]
        slopes = jnp.concatenate([slopes, extra_slopes])
        return slopes


def get_alibi_bias(n_heads: int, seq_len: int):
    """Generate ALiBi bias matrix for attention scores."""
    # Distance matrix: distance[i, j] = i - j (how far back in sequence)
    distances = jnp.arange(seq_len)[None, :] - jnp.arange(seq_len)[:, None]

    # Get slopes for each head
    slopes = get_alibi_slopes(n_heads)  # (n_heads,)

    # Compute bias: slope * (-|distance|)
    # Shape: (n_heads, seq_len, seq_len)
    bias = slopes[:, None, None] * (-jnp.abs(distances)[None, :, :])

    return bias


class ALiBiPositionalBias(nn.Module):
    """ALiBi positional bias for attention."""

    n_heads: int
    max_len: int

    def setup(self):
        # Precompute slopes (these are fixed, not learned)
        self.slopes = get_alibi_slopes(self.n_heads)

    def __call__(self, seq_len: int):
        """Generate bias for given sequence length."""
        return get_alibi_bias(self.n_heads, seq_len)


"""Combine learned and sinusoidal encodings."""


class HybridPositionalEncoding(nn.Module):
    d_model: int
    max_len: int
    learned_weight: float = 0.5  # Weight for learned component

    def setup(self):
        # Learned component
        self.learned_embed = self.param(
            "learned_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model)
        )

        # Sinusoidal component (fixed)
        self.sinusoidal_embed = get_sinusoidal_embeddings(self.max_len, self.d_model)

    def __call__(self, x):
        T = x.shape[1]
        learned = self.learned_embed[:T]
        sinusoidal = self.sinusoidal_embed[:T]

        combined = (self.learned_weight * learned +
                   (1 - self.learned_weight) * sinusoidal)

        return x + combined


def get_positional_encoding(pe_type, d_model, max_len, n_heads=None):
    if pe_type == 'learned':
        # default positional embedding in baseline model
        return None
    elif pe_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
    elif pe_type == 'rope':
        head_dim = d_model // n_heads
        return RotaryPositionalEmbedding(head_dim=head_dim, max_len=max_len)
    elif pe_type == 'alibi':
        return ALiBiPositionalBias(n_heads=n_heads, max_len=max_len)
    elif pe_type == 'hybrid':
        return HybridPositionalEncoding(d_model=d_model, max_len=max_len)
    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")
