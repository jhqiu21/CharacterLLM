import sys
from pathlib import Path

import jax
import numpy as np
import pytest

import train
from helpers import make_config, model_and_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_attention import SPACE, SPAN_MASS, make_stats_fn, summarize  # noqa: E402

W, B = 16, 3


@pytest.mark.parametrize("pe", ["rope", "alibi"])
def test_attention_stats_match_brute_force(pe):
    config = make_config(pos_encoding_type=pe, dropout=0.0, max_len=W)
    _, params = model_and_params(config)
    model = train.build_model(config, capture_attn=True)
    x = np.array(jax.random.randint(jax.random.PRNGKey(5), (B, W), 0, 27), dtype=np.int32)
    x[:, [3, 9]] = SPACE
    acc = {k: np.asarray(v, np.float64) for k, v in jax.device_get(make_stats_fn(model, W)(params, x)).items()}
    _, inter = model.apply({"params": params}, x, deterministic=True, mutable=["intermediates"])
    w = np.stack([np.asarray(inter["intermediates"][f"blocks_{k}"]["MultiHeadAttentionWithPE_0"]["attn_weights"][0])
                  for k in range(2)]).astype(np.float64)  # (Lyr,B,H,W,W)

    n_l, _, n_h = w.shape[:3]
    prof = np.zeros((n_l, n_h, W))
    span, ent, l8, sp, sp_n = (np.zeros((n_l, n_h, B, W)) for _ in range(5))
    for li in range(n_l):
        for b in range(B):
            for h in range(n_h):
                for i in range(W):
                    row = w[li, b, h, i, : i + 1][::-1]  # by distance 0..i
                    prof[li, h, : i + 1] += row
                    span[li, h, b, i] = np.argmax(np.cumsum(row) >= SPAN_MASS - 1e-6)
                    r = row[row > 0]
                    ent[li, h, b, i] = -(r * np.log(r)).sum()
                    l8[li, h, b, i] = row[:8].sum()
                    prev = [j for j in range(i) if x[b, j] == SPACE]
                    if prev:
                        sp[li, h, b, i] = w[li, b, h, i, prev[-1]]
                        sp_n[li, h, b, i] = 1
    np.testing.assert_allclose(acc["profile_sum"], prof, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(acc["span_q"], span.sum(2))
    np.testing.assert_allclose(acc["ent_q"], ent.sum(2), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(acc["local8_q"], l8.sum(2), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(acc["space_q"], sp.sum(2), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(acc["space_n_q"], sp_n[0, 0].sum(0))

    s = summarize(acc, B, W, 0)
    np.testing.assert_allclose(s["span_mean"], span.mean((2, 3)), rtol=1e-6)
    np.testing.assert_array_equal(s["span_median"], np.quantile(span.reshape(n_l, n_h, -1), 0.5, axis=-1, method="inverted_cdf"))
    np.testing.assert_allclose(s["space_mass"], sp.sum((2, 3)) / sp_n[0, 0].sum(), rtol=1e-5)
    past = summarize(acc, B, W, W // 2)
    np.testing.assert_allclose(past["local8"], l8[..., W // 2:].mean((2, 3)), rtol=1e-5)
