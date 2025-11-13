# lstm_baseline.py
"""
Character-level LSTM baseline (robust, extensible):
  - Input:  (B, T) int32 token ids
  - Output: (B, T, V) logits
Design notes:
  * No Modules are called inside lax.scan's step to avoid tracer leaks.
  * All parameters are created once via `self.param` inside `@nn.compact`.
  * Variational dropout: one mask per layer per batch (shared across time steps).
  * Optional weight tying with the input embedding.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class CharLSTM(nn.Module):
    vocab_size: int
    hidden_size: int
    n_layers: int
    dropout: float
    max_len: int
    tie_weights: bool = False  # if True: use E^T for projection

    @nn.compact
    def __call__(self, idx: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
          idx: (B, T) int32 token ids
          train: enable dropout behavior when True
        Returns:
          logits: (B, T, V)
        """
        # 1) Embedding parameters and lookup
        embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size, name="embed")
        x = embed(idx)  # (B, T, H)
        B, T, H = x.shape

        # 2) LSTM stacked parameters (create once as raw tensors)
        #    For each layer i, we use: gates = x @ W_x[i] + h @ W_h[i] + b[i]
        #    where gates split into (i, f, g, o)
        Wx = []  # input-to-gates weights: (H, 4H)
        Wh = []  # hidden-to-gates weights: (H, 4H)
        b = []   # gates bias: (4H,)
        for i in range(self.n_layers):
            Wx.append(self.param(f"Wx_{i}", nn.initializers.lecun_normal(), (H, 4 * H)))
            Wh.append(self.param(f"Wh_{i}", nn.initializers.orthogonal(), (H, 4 * H)))
            b.append(self.param(f"b_{i}", nn.initializers.zeros, (4 * H,)))

        # 3) Variational dropout masks (one per layer, shared across time)
        #    This avoids calling nn.Dropout inside scan.
        if train and self.dropout > 0.0:
            rng = self.make_rng("dropout")
            rngs = jax.random.split(rng, max(self.n_layers - 1, 1))  # no mask for last layer by default
            keep = 1.0 - self.dropout
            masks = []
            for i in range(self.n_layers):
                if i < self.n_layers - 1:
                    m = jax.random.bernoulli(rngs[min(i, len(rngs)-1)], p=keep, shape=(B, H))
                    masks.append(m.astype(x.dtype) / keep)  # rescale to keep expectation
                else:
                    masks.append(jnp.ones((B, H), dtype=x.dtype))
        else:
            masks = [jnp.ones((B, H), dtype=x.dtype) for _ in range(self.n_layers)]

        # 4) Initial (c, h) states for each layer
        init_states = tuple(
            (jnp.zeros((B, H), dtype=x.dtype), jnp.zeros((B, H), dtype=x.dtype))
            for _ in range(self.n_layers)
        )

        # 5) One time step: pure JAX math (no Module calls inside)
        def step(carry, x_t):
            """
            carry: tuple of (c_i, h_i) for i in [0..n_layers-1]
            x_t:   (B, H)
            returns: (new_carry, top_h)
            """
            new_states = []
            h_in = x_t
            for i in range(self.n_layers):
                c_prev, h_prev = carry[i]
                gates = h_in @ Wx[i] + h_prev @ Wh[i] + b[i]  # (B, 4H)
                i_gate, f_gate, g_gate, o_gate = jnp.split(gates, 4, axis=-1)
                i_gate = jax.nn.sigmoid(i_gate)
                f_gate = jax.nn.sigmoid(f_gate)
                g_gate = jnp.tanh(g_gate)
                o_gate = jax.nn.sigmoid(o_gate)

                c_new = f_gate * c_prev + i_gate * g_gate
                h_new = o_gate * jnp.tanh(c_new)

                # apply variational dropout mask between layers (skip on last layer)
                h_out = h_new * masks[i]
                new_states.append((c_new, h_new))
                h_in = h_out  # feed to next layer

            top_h = h_in  # (B, H)
            return tuple(new_states), top_h

        # 6) Scan over time: (B,T,H) -> (T,B,H) -> scan -> (B,T,H)
        x_tm = jnp.swapaxes(x, 0, 1)      # (T, B, H)
        _, hs_tm = jax.lax.scan(step, init_states, x_tm)
        hs = jnp.swapaxes(hs_tm, 0, 1)    # (B, T, H)

        # 7) Output projection (weight tying optional)
        if self.tie_weights:
            # embed.embedding: (V, H) -> use E^T: (H, V)
            logits = hs @ embed.embedding.T  # (B, T, V)
        else:
            W_out = self.param("W_out", nn.initializers.lecun_normal(), (H, self.vocab_size))
            logits = hs @ W_out  # (B, T, V)

        return logits
