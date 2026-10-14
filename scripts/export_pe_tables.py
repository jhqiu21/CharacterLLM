"""
Export a Transformer run's positional-encoding tables to runs/<run>/pe_tables.json (reads the best-loss
checkpoint; a few seconds). results/v3/aggregate_v3.py draws the PE figures from these files. Needs the GPU
backend: orbax cannot restore the GPU-saved checkpoints on the CPU backend ("sharding ... Got None").

    CUDA_VISIBLE_DEVICES=3 python scripts/export_pe_tables.py --run TR-final-learned_s42

- learned:    the trained (max_len, d_model) table positional_embed;
- hybrid:     learned_embed, the fixed sinusoidal component and the table actually added to the embeddings;
- sinusoidal: the fixed table (with the run's sinusoidal_scale);
- relative:   rel_embed of every layer, (n_layers, H, 2L-1), and its causal part as bias[layer][head][d] for d = i - j;
- alibi:      the fixed slopes, and bias[head][d] = -slope * d;
- rope:       the rotation frequencies (no trained table).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from models.positional_encodings import get_alibi_slopes, get_sinusoidal_embeddings, sinusoidal_scale_factor  # noqa: E402


def pe_tables(config, params):
    m = config.model
    pe, L, D = m.pos_encoding_type, int(m.max_len), int(m.d_model)
    out = {"pos_encoding_type": pe, "max_len": L, "d_model": D, "n_heads": int(m.n_heads), "n_layers": int(m.n_layers)}
    sin = np.asarray(get_sinusoidal_embeddings(L, D)) * sinusoidal_scale_factor(getattr(m, "sinusoidal_scale", "unit"), D)
    if pe == "learned":
        out["table"] = np.asarray(params["positional_embed"]).tolist()
    elif pe == "hybrid":
        learned = np.asarray(params["pos_encoding"]["learned_embed"])
        out["learned"] = learned.tolist()
        out["sinusoidal"] = sin.tolist()
        out["table"] = (0.5 * learned + 0.5 * sin).tolist()  # HybridPositionalEncoding.learned_weight = 0.5
    elif pe == "sinusoidal":
        out["table"] = sin.tolist()
    elif pe == "relative":
        rel = np.stack([np.asarray(params[f"blocks_{k}"]["MultiHeadAttentionWithPE_0"]["rel_bias"]["rel_embed"])
                        for k in range(m.n_layers)])  # (Lyr, H, 2L-1)
        out["rel_embed"] = rel.tolist()
        out["bias_by_distance"] = rel[:, :, L - 1:].tolist()  # index L-1+d, d = i - j = 0..L-1
    elif pe == "alibi":
        slopes = np.asarray(get_alibi_slopes(int(m.n_heads)))
        out["slopes"] = slopes.tolist()
        out["bias_by_distance"] = (-slopes[:, None] * np.arange(L)[None]).tolist()
    elif pe == "rope":
        hd = D // int(m.n_heads)
        out["frequencies"] = (1.0 / 10000.0 ** (np.arange(0, hd, 2) / hd)).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    from eval_extrapolation import load_run

    run_dir, config, ckpt = load_run(args.run)
    out = pe_tables(config, ckpt["params"])
    out.update(run=args.run, step=int(ckpt["step"]))
    json.dump(out, open(run_dir / "pe_tables.json", "w"))
    print(f"{args.run}: {out['pos_encoding_type']} -> {run_dir / 'pe_tables.json'}")


if __name__ == "__main__":
    main()
