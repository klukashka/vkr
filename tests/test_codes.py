"""Tests for coding-theory utilities and the channel."""

import numpy as np
import pytest

from self_organized_codes.channel import transmit
from self_organized_codes.codes import hamming74, hamming_bound, min_distance, singleton_bound


def test_hamming74_is_perfect_code():
    """Check that the Hamming [7,4] code has 16 distinct words, d = 3, and meets the Hamming bound."""
    words = hamming74()
    assert words.shape == (16, 7)
    assert len({w.tobytes() for w in words}) == 16
    assert min_distance(words) == 3
    assert hamming_bound(7, 3) == 16  # perfect: meets the sphere-packing bound


def test_bounds():
    """Check the Hamming and Singleton bounds against hand-computed values."""
    assert hamming_bound(12, 1) == 2**12
    assert hamming_bound(12, 5) == 2**12 // (1 + 12 + 66)
    assert singleton_bound(12, 5) == 2**8


def test_min_distance_needs_two_words():
    """Check that the minimum distance of a single word is undefined."""
    assert min_distance(np.zeros((1, 5), np.uint8)) is None


def test_channel_noiseless_and_rate():
    """Check that the channel copies vectors at zero noise and flips bits at the requested rate otherwise."""
    rng = np.random.default_rng(0)
    v = rng.integers(0, 2, 10_000, dtype=np.uint8)
    assert np.array_equal(transmit(v, 0.0, rng), v)
    flipped = np.count_nonzero(transmit(v, 0.1, rng) != v) / v.size
    assert flipped == pytest.approx(0.1, abs=0.01)
