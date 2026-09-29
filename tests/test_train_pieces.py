import numpy as np

import train


def test_fixed_validation_set_is_deterministic_and_seed_independent():
    val = np.random.default_rng(123).integers(0, 27, size=10_000).astype(np.uint8)
    a = train.make_fixed_val_set(val, 128)
    b = train.make_fixed_val_set(val, 128)
    assert a[0].shape == (1024, 128)
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_array_equal(np.asarray(a[0])[:, 1:], np.asarray(a[1])[:, :-1])  # targets shifted by one


def test_get_batch_is_reproducible_from_the_seed():
    text = np.random.default_rng(0).integers(0, 27, size=5_000).astype(np.uint8)
    x1, _ = train.get_batch(text, 8, 16, np.random.default_rng(4212))
    x2, _ = train.get_batch(text, 8, 16, np.random.default_rng(4212))
    x3, _ = train.get_batch(text, 8, 16, np.random.default_rng(4213))
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
    assert not np.array_equal(np.asarray(x1), np.asarray(x3))


def test_encode_matches_notebook_encoding():
    s = "hello world abcxyz"
    ref = np.array([train.CHAR_TO_INT[c] for c in s], dtype=np.uint8)
    np.testing.assert_array_equal(train.encode(s), ref)
