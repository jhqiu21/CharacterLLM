import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import PE_TYPES, make_config, model_and_params
from models.positional_encodings import HybridPositionalEncoding, SinusoidalPositionalEncoding

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_pe_tables import pe_tables  # noqa: E402

L = 16


@pytest.mark.parametrize("pe", PE_TYPES)
def test_pe_tables(pe):
    config = make_config(pos_encoding_type=pe, max_len=L)
    _, params = model_and_params(config)
    t = pe_tables(config, params)
    zeros = np.zeros((1, L, 32), np.float32)
    if pe == "learned":
        np.testing.assert_array_equal(t["table"], params["positional_embed"])
    elif pe == "hybrid":
        mod = HybridPositionalEncoding(d_model=32, max_len=L)
        added = mod.apply({"params": params["pos_encoding"]}, zeros)[0]
        np.testing.assert_allclose(t["table"], added, rtol=1e-6, atol=1e-7)
    elif pe == "sinusoidal":
        added = SinusoidalPositionalEncoding(d_model=32, max_len=L).apply({}, zeros)[0]
        np.testing.assert_allclose(t["table"], added, rtol=1e-6, atol=1e-7)
    elif pe == "relative":
        rel = np.asarray(params["blocks_1"]["MultiHeadAttentionWithPE_0"]["rel_bias"]["rel_embed"])
        assert np.asarray(t["bias_by_distance"]).shape == (2, 2, L)
        np.testing.assert_array_equal(np.asarray(t["bias_by_distance"])[1, :, 3], rel[:, L - 1 + 3])
    elif pe == "alibi":
        b = np.asarray(t["bias_by_distance"])
        assert b.shape == (2, L) and np.all(b[:, 0] == 0) and np.all(np.diff(b, axis=1) < 0)
    else:
        assert len(t["frequencies"]) == 8
