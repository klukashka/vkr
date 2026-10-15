"""Coding-theory utilities: distances, bounds, reference codes."""

import itertools
from math import comb

import numpy as np


def pairwise_distances(words: np.ndarray) -> np.ndarray:
    """Compute the Hamming distance matrix of a set of binary words.

    :param words: ``(K, n)`` binary array, one word per row.
    :return: ``(K, K)`` integer array of pairwise Hamming distances.
    """
    return np.count_nonzero(words[:, None, :] != words[None, :, :], axis=2)


def min_distance(words: np.ndarray) -> int | None:
    """Compute the minimum Hamming distance between distinct rows.

    :param words: ``(K, n)`` binary array, one word per row.
    :return: the minimum distance, or ``None`` if there are fewer than two words.
    """
    if len(words) < 2:
        return None
    dist = pairwise_distances(words)
    return int(dist[np.triu_indices(len(words), k=1)].min())


def hamming_bound(n: int, d: int) -> int:
    """Compute the sphere-packing (Hamming) upper bound on the size of a binary code.

    :param n: code length.
    :param d: minimum distance.
    :return: the largest number of words a binary code with these parameters can have by this bound.
    """
    t = (d - 1) // 2
    return 2**n // sum(comb(n, i) for i in range(t + 1))


def singleton_bound(n: int, d: int) -> int:
    """Compute the Singleton upper bound on the size of a binary code.

    :param n: code length.
    :param d: minimum distance.
    :return: ``2 ** (n - d + 1)``.
    """
    return 2 ** (n - d + 1)


def hamming74() -> np.ndarray:
    """Build all codewords of the Hamming [7,4] code (minimum distance 3).

    :return: ``(16, 7)`` ``uint8`` array, one codeword per row.
    """
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
