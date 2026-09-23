"""Coding-theory utilities: distances, bounds, reference codes."""

import itertools
from math import comb

import numpy as np


def pairwise_distances(words: np.ndarray) -> np.ndarray:
    """Hamming distance matrix of a ``(K, n)`` binary array."""
    return np.count_nonzero(words[:, None, :] != words[None, :, :], axis=2)


def min_distance(words: np.ndarray) -> int | None:
    """Minimum Hamming distance between distinct rows, or ``None`` if there are fewer than two words."""
    if len(words) < 2:
        return None
    dist = pairwise_distances(words)
    return int(dist[np.triu_indices(len(words), k=1)].min())


def hamming_bound(n: int, d: int) -> int:
    """Sphere-packing upper bound on the size of a binary code with length ``n`` and minimum distance ``d``."""
    t = (d - 1) // 2
    return 2**n // sum(comb(n, i) for i in range(t + 1))


def singleton_bound(n: int, d: int) -> int:
    """Singleton upper bound on the size of a binary code with length ``n`` and minimum distance ``d``."""
    return 2 ** (n - d + 1)


def hamming74() -> np.ndarray:
    """All 16 codewords of the Hamming [7,4] code (d = 3), as a ``(16, 7)`` ``uint8`` array."""
    generator = np.array(
        [
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    messages = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    return (messages @ generator % 2).astype(np.uint8)
