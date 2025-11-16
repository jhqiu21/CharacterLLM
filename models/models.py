"""
Decoder-only Transformer blocks in Flax/JAX, commented for learning.

Tensor shape conventions used below:
- B: batch size
- T: sequence length (time/positions)
- D: hidden size / embedding dimension (d_model)
- V: vocabulary size
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import attention as attn
from .positional_encodings import (
    SinusoidalPositionalEncoding,
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    HybridPositionalEncoding
)


class MLP(nn.Module):
        """Transformer feed-forward network (a.k.a. MLP block).

        Structure: Dense(D -> 4D), GELU, Dense(4D -> D) by default.
        The expansion factor can be adjusted with `mlp_ratio`.

        Args:
            d_model: Hidden size D.
            mlp_ratio: Expansion factor for the intermediate hidden size.

        Input shape:  (B, T, D)
        Output shape: (B, T, D)
        """

        d_model: int
        mlp_dropout: float
        mlp_ratio: int

        @nn.compact
        def __call__(self, x, *, deterministic: bool):
                # Expand channel dimension (D -> hidden), apply non-linearity, project back to D.
                hidden = int(self.d_model * self.mlp_ratio)
                x = nn.Dense(hidden)(x)
                x = nn.gelu(x)
                x = nn.Dense(self.d_model)(x)
                # Post-MLP dropout (before residual add)
                x = nn.Dropout(rate=self.mlp_dropout)(x, deterministic=deterministic)
                return x


class MultiHeadAttentionWithPE(nn.Module):
    """Multi-head attention with support for various positional encodings.

    This is a custom implementation that allows:
    - RoPE to be applied to Q and K
    - ALiBi/Relative biases to be added to attention scores
    """

    d_model: int
    n_heads: int
    attn_dropout: float
    pos_encoding_type: str
    max_len: int

    def setup(self):
        self.head_dim = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        # Q, K, V projections
        self.q_proj = nn.Dense(self.d_model, use_bias=False)
        self.k_proj = nn.Dense(self.d_model, use_bias=False)
        self.v_proj = nn.Dense(self.d_model, use_bias=False)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

        # Position encoding specific modules
        if self.pos_encoding_type == 'rope':
            self.rope = RotaryPositionalEmbedding(
                head_dim=self.head_dim,
                max_len=self.max_len
            )
        elif self.pos_encoding_type == 'alibi':
            self.alibi = ALiBiPositionalBias(
                n_heads=self.n_heads,
                max_len=self.max_len
            )

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool):
        """
        Args:
            x: Input of shape (B, T, D)
            mask: Attention mask (B, 1, T, T) or (1, 1, T, T)
            deterministic: Whether to apply dropout

        Returns:
            Output of shape (B, T, D)
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)

        # Reshape to multi-head format: (B, T, H, d)
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.reshape(B, T, self.n_heads, self.head_dim)

        # Apply RoPE if specified
        if self.pos_encoding_type == 'rope':
            q, k = self.rope(q, k)

        # Transpose to (B, H, T, d) for attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores: (B, H, T, T)
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        # Add positional biases if using ALiBi
        if self.pos_encoding_type == 'alibi':
            alibi_bias = self.alibi(T)  # (H, T, T)
            attn_scores = attn_scores + alibi_bias[None, :, :, :]  # (B, H, T, T)

        # Apply causal mask
        if mask is not None:
            if mask.shape[1] == 1:
                mask = jnp.broadcast_to(mask, (B, self.n_heads, T, T))
            attn_scores = jnp.where(mask, attn_scores, -1e10)

        # Softmax
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Dropout on attention weights
        attn_weights = nn.Dropout(rate=self.attn_dropout)(
            attn_weights, deterministic=deterministic
        )

        # Apply attention to values: (B, H, T, d)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape back: (B, T, D)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = self.out_proj(attn_out)

        return out


class DecoderBlock(nn.Module):
    """A single decoder block.

    Pre-LayerNorm improves training stability. Residual connections are used after
    attention and MLP sublayers. The attention is causal when a causal mask is passed
    (so each position can only attend to previous or current positions).

    Args:
      d_model: Hidden size D.
      n_heads: Number of attention heads.

    Input/Output shape: (B, T, D)
    """

    d_model: int
    n_heads: int
    mlp_dropout: float
    attn_dropout: float
    resid_dropout: float
    mlp_ratio: int
    pos_encoding_type: str
    max_len: int

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool):
        h = nn.LayerNorm()(x)

        if self.pos_encoding_type in ['rope', 'alibi']:
            h = MultiHeadAttentionWithPE(
                d_model=self.d_model,
                n_heads=self.n_heads,
                attn_dropout=self.attn_dropout,
                pos_encoding_type=self.pos_encoding_type,
                max_len=self.max_len
            )(h, mask=mask, deterministic=deterministic)
        else:
            # default SelfAttention for sinusoidal, hybrid
            h = nn.SelfAttention(
                num_heads=self.n_heads,
                use_bias=False,
                dropout_rate=self.attn_dropout,
            )(h, mask=mask, deterministic=deterministic)

        h = nn.Dropout(rate=self.resid_dropout)(h, deterministic=deterministic)
        x = x + h  # residual connection

        # MLP sublayer: Pre-LayerNorm -> MLP -> Residual add
        h = nn.LayerNorm()(x)
        h = MLP(self.d_model,
                mlp_ratio=self.mlp_ratio,
                mlp_dropout=self.mlp_dropout,
        )(h, deterministic=deterministic)
        x = x + h  # residual connection

        return x


class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.

    Components:
      - Token embeddings: maps token ids to D-dim vectors
      - Learned positional embeddings: adds position information (0..T-1)
      - N stacked decoder blocks with causal self-attention
      - Final LayerNorm
      - Output projection:
          * If tie_weights=True (default), reuse token embedding matrix E to
            compute logits via x @ E^T (implemented via einsum).
          * Else, use a separate linear head to project to V logits.

    Args:
      vocab_size: Vocabulary size V.
      d_model: Hidden size D.
      n_layers: Number of decoder blocks.
      n_heads: Attention heads per block.
      max_len: Maximum supported sequence length for positional embeddings.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int

    emb_dropout: float        # embedding dropout
    mlp_dropout: float        # MLP dropout
    attn_dropout: float       # attention weights dropout
    resid_dropout: float      # post-attention/MLP dropout
    pos_encoding_type: str    # positional encoding type: 'learned', 'sinusoidal', 'hybrid', 'rope', 'alibi'

    def setup(self):
        # Token embedding table E with shape (V, D)
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)

        self.pos_encoding = None

        if self.pos_encoding_type == 'learned':
            # Learned positional embeddings (baseline)
            self.positional_embed = self.param(
                "positional_embed",
                nn.initializers.normal(stddev=0.02),
                (self.max_len, self.d_model)
            )
        elif self.pos_encoding_type == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=self.d_model,
                max_len=self.max_len
            )
        elif self.pos_encoding_type == 'hybrid':
            self.pos_encoding = HybridPositionalEncoding(
                d_model=self.d_model,
                max_len=self.max_len
            )
        elif self.pos_encoding_type in ['rope', 'alibi']:
            # These are handled within attention mechanism
            self.pos_encoding = None
        else:
            raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}")

        # Stack of decoder blocks
        self.blocks = [
            DecoderBlock(
                  d_model=self.d_model,
                  n_heads=self.n_heads,
                  mlp_ratio=self.mlp_ratio,
                  mlp_dropout=self.mlp_dropout,
                  attn_dropout=self.attn_dropout,
                  resid_dropout=self.resid_dropout,
                  pos_encoding_type=self.pos_encoding_type,
                  max_len=self.max_len
            ) for _ in range(self.n_layers)
        ]

        # Final LayerNorm before projecting to logits
        self.layerNorm_final = nn.LayerNorm()

        # Optional separate output head if not weight-tying
        self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    @nn.compact
    def __call__(self, idx, *, deterministic: bool):
        """Forward pass (causal-only).

        Args:
          idx: Token ids of shape (B, T), dtype int32/int64.

        Returns:
          logits: (B, T, V) unnormalized vocabulary scores for next-token prediction.
        """
        B, T = idx.shape

        # Token embeddings
        x = self.tok_embed(idx)
        if self.pos_encoding_type == 'learned':
            x = x + self.positional_embed[:T]
        elif self.pos_encoding_type in ['sinusoidal', 'hybrid']:
            x = self.pos_encoding(x)

        # Embedding dropout
        x = nn.Dropout(rate=self.emb_dropout)(x, deterministic=deterministic)

        # Build attention mask: strictly causal (lower-triangular), no padding mask.
        causal = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        mask = causal

        # Run the stack of decoder blocks
        for blk in self.blocks:
            x = blk(x, mask=mask, deterministic=deterministic)

        # Final LayerNorm before output projection
        x = self.layerNorm_final(x)

        # Output projection to logits over V tokens.
        logits = self.project_to_vocab(x)

        return logits
