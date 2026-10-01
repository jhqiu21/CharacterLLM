import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import make_config, model_and_params
from utils.generation import generate_tokens, initial_window, next_token_logits

W = 16
PROMPT_LENGTHS = [1, 12, W, W + 5]
GEN = 24  # > W, so generation also exercises the sliding window
MODELS = [("transformer", "learned"), ("transformer", "rope"), ("lstm", None)]


@pytest.fixture(scope="module", params=MODELS, ids=lambda m: m[1] or m[0])
def model_setup(request):
    model_type, pe = request.param
    config = make_config(model_type, pos_encoding_type=pe or "rope", dropout=0.0, max_len=W)
    model, params = model_and_params(config, seed=1)
    return model, params


def _prompt(S, seed=0, B=1):
    # tokens 1..26, so a stray token-0 filler would be visible
    return np.asarray(jax.random.randint(jax.random.PRNGKey(100 + seed), (B, S), 1, 27), dtype=np.int32)


def _reference(model, params, prompt, length, rng, sample, temperature=0.7):
    """Brute force: at every step run the model on exactly the real tokens (last W of them), unpadded."""
    seq = list(prompt[0])
    out = []
    for _ in range(length):
        ctx = jnp.asarray([seq[-W:]], dtype=jnp.int32)
        logits = model.apply({"params": params}, ctx, deterministic=True)[:, -1, :]
        rng, sub = jax.random.split(rng)
        tok = int(jax.random.categorical(sub, logits / temperature, axis=-1)[0]) if sample else int(jnp.argmax(logits[0]))
        seq.append(tok)
        out.append(tok)
    return np.array(out)


@pytest.mark.parametrize("S", PROMPT_LENGTHS)
def test_first_step_logits_equal_unpadded_forward_pass(model_setup, S):
    model, params = model_setup
    prompt = _prompt(S)
    window, cur_len = initial_window(prompt, W)
    assert window.shape == (1, W) and cur_len == min(S, W)
    np.testing.assert_array_equal(np.asarray(window[:, :cur_len]), prompt[:, -W:])  # prompt at positions 0..S-1
    got = next_token_logits(model, params, window, cur_len)
    ref = model.apply({"params": params}, jnp.asarray(prompt[:, -W:]), deterministic=True)[:, -1, :]
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("S", PROMPT_LENGTHS)
@pytest.mark.parametrize("sample", [False, True])
def test_scan_path_matches_unpadded_brute_force(model_setup, S, sample):
    model, params = model_setup
    prompt = _prompt(S)
    rng = jax.random.PRNGKey(7)
    got = np.asarray(generate_tokens(model, params, rng, prompt, GEN, block_size=W, temperature=0.7, sample=sample))
    assert got.shape == (1, GEN)
    np.testing.assert_array_equal(got[0], _reference(model, params, prompt, GEN, rng, sample))


@pytest.mark.parametrize("S", PROMPT_LENGTHS)
def test_no_padding_token_influences_generation(model_setup, S):
    """The filler after the prompt is never read: token 0, another token and the default all give the same output."""
    model, params = model_setup
    prompt = _prompt(S)
    window, cur_len = initial_window(prompt, W)
    if cur_len < W:
        assert not np.any(np.asarray(window) == 0)  # the default filler is not token 0 ('a')
    rng = jax.random.PRNGKey(3)
    outs = [np.asarray(generate_tokens(model, params, rng, prompt, GEN, block_size=W, temperature=0.7, fill_token=f))
            for f in (None, 0, 13)]
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])
    outs_ng = [np.asarray(generate_tokens(model, params, rng, prompt, 6, block_size=W, temperature=0.7, ngram_size=1, fill_token=f))
               for f in (None, 0)]
    np.testing.assert_array_equal(outs_ng[0], outs_ng[1])


@pytest.mark.parametrize("sample", [False, True])
def test_batch_greater_than_one_in_scan_path(model_setup, sample):
    model, params = model_setup
    prompts = _prompt(12, B=3)
    rng = jax.random.PRNGKey(5)
    out = np.asarray(generate_tokens(model, params, rng, prompts, GEN, block_size=W, temperature=0.7, sample=sample))
    assert out.shape == (3, GEN)
    if not sample:  # greedy rows are independent of the batch
        for b in range(3):
            single = np.asarray(generate_tokens(model, params, rng, prompts[b:b + 1], GEN, block_size=W, sample=False))
            np.testing.assert_array_equal(out[b], single[0])
    else:  # identical prompts still give different samples per row
        same = np.repeat(prompts[:1], 3, axis=0)
        out_same = np.asarray(generate_tokens(model, params, rng, same, GEN, block_size=W, temperature=0.7))
        assert len({tuple(r) for r in out_same}) > 1


@pytest.mark.parametrize("S", PROMPT_LENGTHS)
@pytest.mark.parametrize("sample", [False, True])
def test_ngram_path_matches_scan_path_conditioning(model_setup, S, sample):
    """ngram_size=1 takes the Python (n-gram) path without blocking: same tokens as the scan path."""
    model, params = model_setup
    prompt = _prompt(S)
    rng = jax.random.PRNGKey(11)
    scan = np.asarray(generate_tokens(model, params, rng, prompt, GEN, block_size=W, temperature=0.7, sample=sample))
    loop = np.asarray(generate_tokens(model, params, rng, prompt, GEN, block_size=W, temperature=0.7, sample=sample, ngram_size=1))
    np.testing.assert_array_equal(loop, scan)


def test_ngram_blocking_still_blocks(model_setup):
    model, params = model_setup
    prompt = _prompt(12)
    n = 3
    out = np.asarray(generate_tokens(model, params, jax.random.PRNGKey(0), prompt, GEN, block_size=W, sample=False, ngram_size=n))[0]
    seq = list(prompt[0]) + list(out)
    seen = set()
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i:i + n])
        if i + n > len(prompt[0]):  # n-grams ending in a generated token must be new
            assert gram not in seen, gram
        seen.add(gram)
