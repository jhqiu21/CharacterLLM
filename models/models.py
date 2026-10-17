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
    SELF_EXTEND_MODES,
    SinusoidalPositionalEncoding,
    RotaryPositionalEmbedding,
    ALiBiPositionalBias,
    HybridPositionalEncoding,
    RelativePositionalEncoding,
    self_extend_positions,
)

POS_SCALE_TYPES = ("rope", "sinusoidal")  # encodings that support position interpolation (pos_scale)


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

    Used by every pos_encoding_type, so all encodings share one attention implementation
    (including independent attention-dropout masks per batch element and head). It allows:
    - RoPE to be applied to Q and K
    - ALiBi/Relative biases to be added to attention scores
    For 'learned', 'sinusoidal' and 'hybrid' (applied to the embeddings) and 'none' (NoPE: no positional
    signal at all, only the causal mask) it is plain causal scaled dot-product attention.

    Evaluation-only options (the defaults leave training, outputs and parameters unchanged):
    - capture_attn: sow the post-softmax attention weights (B, H, T, T) into the 'intermediates'
      collection as 'attn_weights' (call apply with mutable=['intermediates']).
    - rel_table_len: length the 'relative' table was trained for (its rel_embed has 2*rel_table_len-1
      columns); 0 means max_len. Lets a model rebuilt with a larger max_len load the trained table.
    - rel_clamp: clamp relative offsets to +-(rel_table_len-1) (T5-style), so 'relative' runs on sequences
      longer than rel_table_len. Offsets inside the table are untouched, so for T <= rel_table_len the
      output is identical with or without the clamp.
    - pos_scale: 'rope' only here; rotation angles use position p * pos_scale (position interpolation).
    - self_extend_window / self_extend_group / self_extend_mode: 'rope' only; pairs with distance
      i - j < self_extend_window keep their exact positions, farther pairs use the compressed positions of
      positional_encodings.self_extend_positions (SelfExtend 'floor' or continuous 'linear'). 0 = off.
    """

    d_model: int
    n_heads: int
    attn_dropout: float
    pos_encoding_type: str
    max_len: int
    capture_attn: bool = False
    rel_table_len: int = 0
    rel_clamp: bool = False
    pos_scale: float = 1.0
    self_extend_window: int = 0
    self_extend_group: float = 1.0
    self_extend_mode: str = "floor"

    def setup(self):
        self.head_dim = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        # Q, K, V projections
        self.q_proj = nn.Dense(self.d_model, use_bias=False)
        self.k_proj = nn.Dense(self.d_model, use_bias=False)
        self.v_proj = nn.Dense(self.d_model, use_bias=False)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)

        # Position encoding specific modules
        if self.pos_encoding_type == "relative":
            self.rel_bias = RelativePositionalEncoding(
                n_heads=self.n_heads,
                max_len=self.rel_table_len or self.max_len
            )
        elif self.pos_encoding_type == 'rope':
            self.rope = RotaryPositionalEmbedding(
                head_dim=self.head_dim,
                max_len=self.max_len,
                pos_scale=self.pos_scale,
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

        # Apply RoPE if specified (SelfExtend: also the far-band rotation of the unrotated q, k)
        if self.pos_encoding_type == 'rope':
            if self.self_extend_window:
                q_pos, k_pos = self_extend_positions(T, self.self_extend_window, self.self_extend_group, self.self_extend_mode)
                q_far, k_far = self.rope.rotate_at(q, k, q_pos, k_pos)
            q, k = self.rope(q, k)

        # Transpose to (B, H, T, d) for attention computation
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention scores: (B, H, T, T)
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if self.self_extend_window:
            far_scores = jnp.einsum('bqhd,bkhd->bhqk', q_far, k_far) * scale
            pos = jnp.arange(T)
            attn_scores = jnp.where((pos[:, None] - pos[None, :]) < self.self_extend_window, attn_scores, far_scores)

        # Add positional biases if using ALiBi or Relative
        if self.pos_encoding_type == 'alibi':
            alibi_bias = self.alibi(T)  # (H, T, T)
            attn_scores = attn_scores + alibi_bias[None, :, :, :]  # (B, H, T, T)
        elif self.pos_encoding_type == 'relative':
            table_len = self.rel_table_len or self.max_len
            if T > table_len and not self.rel_clamp:
                raise ValueError(f"'relative' was trained for offsets up to {table_len - 1}; sequence length {T} "
                                 f"needs rel_clamp=True")
            pos = jnp.arange(T)
            offset = pos[:, None] - pos[None, :]  # (T, T), i - j
            if self.rel_clamp:
                offset = jnp.clip(offset, -(table_len - 1), table_len - 1)
            rel_idx = offset + (table_len - 1)
            bias_matrix = self.rel_bias()  # call the module to get (H, 2*table_len-1)
            bias = bias_matrix[:, rel_idx]  # (H, T, T)
            attn_scores = attn_scores + bias[None, :, :, :]  # (B, H, T, T)

        # Apply causal mask
        if mask is not None:
            if mask.shape[1] == 1:
                mask = jnp.broadcast_to(mask, (B, self.n_heads, T, T))
            attn_scores = jnp.where(mask, attn_scores, -1e10)

        # Softmax
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        if self.capture_attn:
            self.sow('intermediates', 'attn_weights', attn_weights)

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
      mlp_dropout, attn_dropout, resid_dropout: dropout rates of the MLP output, the attention weights and the
        attention output.
      mlp_ratio: MLP expansion factor.
      pos_encoding_type, max_len: passed to MultiHeadAttentionWithPE.
      capture_attn, rel_table_len, rel_clamp, pos_scale, self_extend_*: options of MultiHeadAttentionWithPE.

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
    capture_attn: bool = False
    rel_table_len: int = 0
    rel_clamp: bool = False
    pos_scale: float = 1.0
    self_extend_window: int = 0
    self_extend_group: float = 1.0
    self_extend_mode: str = "floor"

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool):
        h = nn.LayerNorm()(x)

        # One attention implementation for every pos_encoding_type (Flax nn.SelfAttention is not used:
        # its default broadcast_dropout=True shares one attention-dropout mask across batch and heads).
        h = MultiHeadAttentionWithPE(
            d_model=self.d_model,
            n_heads=self.n_heads,
            attn_dropout=self.attn_dropout,
            pos_encoding_type=self.pos_encoding_type,
            max_len=self.max_len,
            capture_attn=self.capture_attn,
            rel_table_len=self.rel_table_len,
            rel_clamp=self.rel_clamp,
            pos_scale=self.pos_scale,
            self_extend_window=self.self_extend_window,
            self_extend_group=self.self_extend_group,
            self_extend_mode=self.self_extend_mode,
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
      - Position information, depending on pos_encoding_type: added to the embeddings
        ('learned', 'sinusoidal', 'hybrid'), applied inside attention ('rope', 'alibi', 'relative'),
        or none at all ('none', NoPE: the causal mask is the only source of order information)
      - N stacked decoder blocks with causal self-attention
      - Final LayerNorm
      - Output projection: a separate (untied) linear head `project_to_vocab` to V logits;
        the token embedding matrix is not reused.

    Args:
      vocab_size: Vocabulary size V.
      d_model: Hidden size D.
      n_layers: Number of decoder blocks.
      n_heads: Attention heads per block.
      max_len: Maximum supported sequence length for positional embeddings.
      sinusoidal_scale: 'unit' (rows of the sinusoidal table have norm 1) or 'raw' (norm sqrt(D/2));
        used by 'sinusoidal' and the sinusoidal half of 'hybrid'.
      pos_scale: position interpolation for 'rope' and 'sinusoidal': position p is replaced by p * pos_scale
        (e.g. 128 / W maps a window of W onto the trained range). 1.0 = unchanged. Set at evaluation time by the
        extrapolation tools, or in training through model.pos_scale (PI fine-tuning, train.py --init-from).
      attn_window: evaluation-only local attention; query i sees only keys j with i - j < attn_window (0 = full
        causal attention). Applies to every encoding.
      self_extend_window, self_extend_group, self_extend_mode: evaluation-only SelfExtend for 'rope' (see
        MultiHeadAttentionWithPE); 0 = off.
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
    pos_encoding_type: str    # 'learned', 'sinusoidal', 'hybrid', 'rope', 'alibi', 'relative', 'none'
    sinusoidal_scale: str = "unit"  # 'unit' or 'raw' (see positional_encodings.sinusoidal_scale_factor)
    # Evaluation-only options, see MultiHeadAttentionWithPE (defaults leave training unchanged).
    capture_attn: bool = False
    rel_table_len: int = 0
    rel_clamp: bool = False
    pos_scale: float = 1.0
    attn_window: int = 0
    self_extend_window: int = 0
    self_extend_group: float = 1.0
    self_extend_mode: str = "floor"

    def setup(self):
        if self.pos_scale != 1.0 and self.pos_encoding_type not in POS_SCALE_TYPES:
            raise ValueError(f"pos_scale (position interpolation) applies to {POS_SCALE_TYPES} only, "
                             f"not to pos_encoding_type {self.pos_encoding_type!r}")
        if self.self_extend_window:
            if self.pos_encoding_type != "rope" or self.pos_scale != 1.0:
                raise ValueError("self_extend applies to 'rope' with pos_scale 1.0 only, not to "
                                 f"{self.pos_encoding_type!r} with pos_scale {self.pos_scale}")
            if self.self_extend_mode not in SELF_EXTEND_MODES:
                raise ValueError(f"self_extend_mode must be one of {SELF_EXTEND_MODES}, got {self.self_extend_mode!r}")
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
                max_len=self.max_len,
                scale=self.sinusoidal_scale,
                pos_scale=self.pos_scale,
            )
        elif self.pos_encoding_type == 'hybrid':
            self.pos_encoding = HybridPositionalEncoding(
                d_model=self.d_model,
                max_len=self.max_len,
                scale=self.sinusoidal_scale,
            )
        elif self.pos_encoding_type in ['rope', 'alibi', 'relative', 'none']:
            # Handled within the attention mechanism ('none': no positional signal, plain causal attention)
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
                  max_len=self.max_len,
                  capture_attn=self.capture_attn,
                  rel_table_len=self.rel_table_len,
                  rel_clamp=self.rel_clamp,
                  pos_scale=self.pos_scale,
                  self_extend_window=self.self_extend_window,
                  self_extend_group=self.self_extend_group,
                  self_extend_mode=self.self_extend_mode,
            ) for _ in range(self.n_layers)
        ]

        # Final LayerNorm before projecting to logits
        self.layerNorm_final = nn.LayerNorm()

        # Separate (untied) output head
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
        if T > self.max_len and self.pos_encoding_type in ('learned', 'hybrid'):
            raise ValueError(f"pos_encoding_type {self.pos_encoding_type!r} has a learned table of {self.max_len} positions; "
                             f"it cannot be evaluated on sequences of length {T}")

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
        if self.attn_window:  # local attention: only the last attn_window positions (the query included)
            pos = jnp.arange(T)
            mask = jnp.logical_and(mask, ((pos[:, None] - pos[None, :]) < self.attn_window)[None, None])

        # Run the stack of decoder blocks
        for blk in self.blocks:
            x = blk(x, mask=mask, deterministic=deterministic)

        # Final LayerNorm before output projection
        x = self.layerNorm_final(x)

        # Output projection to logits over V tokens.
        logits = self.project_to_vocab(x)

        return logits
