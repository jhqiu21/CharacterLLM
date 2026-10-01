import jax
import numpy as np

import train
import utils.eval as eval_utils
from helpers import make_config, model_and_params
from utils.generation import generate_tokens

TR2O = str(train.REPO_ROOT / "configs" / "TR2o.yaml")


def _val(n=5000):
    return np.random.default_rng(123).integers(0, 27, size=n).astype(np.uint8)


def test_prompts_come_from_validation_split_at_deterministic_offsets():
    val = _val()
    offsets, prompts = train.make_self_bleu_prompts(val, 10, 128)
    offsets2, prompts2 = train.make_self_bleu_prompts(val, 10, 128)
    np.testing.assert_array_equal(offsets, offsets2)
    np.testing.assert_array_equal(prompts, prompts2)
    assert prompts.shape == (10, 128) and len(set(offsets.tolist())) == 10
    assert np.all(np.diff(offsets) > 0) and offsets.min() >= 0 and offsets.max() + 128 <= len(val)
    for o, p in zip(offsets, prompts):
        np.testing.assert_array_equal(p, val[o:o + 128])
    np.testing.assert_array_equal(offsets, np.sort(np.random.default_rng(0).choice(len(val) - 127, 10, replace=False)))


def test_prompts_do_not_depend_on_the_training_seed():
    val = _val()
    results = []
    for seed in (4212, 4213):
        d, _ = train.build_run_config(train.parse_args(["--config", TR2O, "--seed", str(seed)]))
        ev = d["evaluation"]
        assert (ev["self_bleu_multi_n_prompts"], ev["self_bleu_multi_n_samples"], ev["self_bleu_multi_prompt_len"]) == (10, 20, 128)
        results.append(train.make_self_bleu_prompts(val, ev["self_bleu_multi_n_prompts"], ev["self_bleu_multi_prompt_len"]))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_multi_prompt_self_bleu_scores_each_prompt_batch():
    config = make_config("transformer", dropout=0.0, max_len=16)
    model, params = model_and_params(config)
    _, prompts = train.make_self_bleu_prompts(_val(), 3, 20)
    kw = dict(block_size=16, gen_len=30, temperature=1.0, sample=True, seed=5, n_grams=4, n_samples=4)
    out = eval_utils.self_bleu_multi_prompt(model, params, prompts, train.INT_TO_CHAR, **kw)
    assert out["n_prompts"] == 3 and out["n_samples"] == 4 and len(out["per_prompt"]) == 3
    assert np.isclose(out["mean"], np.mean(out["per_prompt"])) and np.isclose(out["std"], np.std(out["per_prompt"], ddof=1))
    # prompt 1 by hand: one batch of n_samples copies of the prompt, key fold_in(PRNGKey(seed), 1)
    ctx = np.repeat(prompts[1:2], 4, axis=0)
    gen = np.asarray(generate_tokens(model, params, jax.random.fold_in(jax.random.PRNGKey(5), 1), ctx, 30, block_size=16, temperature=1.0))
    texts = ["".join(train.INT_TO_CHAR[int(t)] for t in row).strip() for row in gen]
    assert np.isclose(out["per_prompt"][1], eval_utils.self_bleu_from_texts(texts, 4))
