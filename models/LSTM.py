# lstm_baseline.py
"""
Character-level LSTM baseline:
    - Input:  (B, T) token ids
    - Output: (B, T, V) logits (predicting next token for each position)
    - Training/validation can share the same loss_fn / train_step as Transformer
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class CharLSTM(nn.Module):
    vocab_size: int
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    max_len: int = 128        # not used but for compatibility
    # if you want to share weights with output layer, can be extended later
    tie_weights: bool = False
    
    def setup(self):
        # Map token ids to continuous vector space with dimension hidden_size for LSTM input
        self.embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)

        # Stack multiple LSTMCells; we will use lax.scan to iterate these cells over time
        self.cells = [nn.LSTMCell() for _ in range(self.num_layers)]

        # output projection layer: map final hidden state h at each time step to vocab size V to get logits
        self.proj = nn.Dense(self.vocab_size, use_bias=False)

        # FIXME: If you want to implement tie weights, you should use a shared parameter here.
        # But for simplicity, we skip it now.

    def __call__(self, idx: jnp.ndarray, *, train: bool = True):
        """
        前向计算：
        idx: (B, T) int32 的 token ids
        return: logits (B, T, V)
        """
        # Embed the integer token ids into vectors of shape (B, T, H)
        x = self.embed(idx)
        B, T, H = x.shape

        # initialize the (c, h) states for each LSTM layer
        init_states = tuple(
            (jnp.zeros((B, self.hidden_size), dtype=x.dtype),
             jnp.zeros((B, self.hidden_size), dtype=x.dtype))
            for _ in range(self.num_layers)
        )

        # set dropout rate based on train/eval mode
        dropout_rate = self.dropout if train else 0.0

        def step(carry, x_t):
            """
            Single-step recurrence function: inputs the states of all layers at the previous time step
            + the embedding at the current time step, outputs the new states + the top output 

            carry: tuple[(c_i, h_i)] * num_layers, each of shape (B, H)
            x_t:   (B, H) input vector at current time step t
            return:
                new_carry: updated (c_i, h_i)
                top_h:     hidden state h_t of the top LSTM (as the representation at this time step)
            """
            new_states = []
            h = x_t
            for i, cell in enumerate(self.cells):
                c_prev, h_prev = carry[i]
                (c_new, h_new), y = cell((c_prev, h_prev), h)  # y == h_new
                h = y
                if dropout_rate > 0.0:
                    # Apply dropout to the outputs of each layer except the last
                    h = nn.Dropout(rate=dropout_rate)(h, deterministic=not train)
                new_states.append((c_new, h_new))
            return tuple(new_states), h

        # Use jax.lax.scan to unroll the LSTM over time steps
        x_time_major = jnp.swapaxes(x, 0, 1)                    # (T, B, H)
        states_final, hs_time_major = jax.lax.scan(step, init_states, x_time_major)
        # hs_time_major: (T, B, H) -- hidden states from top LSTM layer at all time steps

        # Convert back to (B, T, H)
        hs = jnp.swapaxes(hs_time_major, 0, 1)                  # (B, T, H)

        # Project the hidden states to vocabulary logits
        logits = self.proj(hs)                                  # (B, T, V)
        return logits
