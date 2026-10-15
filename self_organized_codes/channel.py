"""Binary symmetric channel."""

import numpy as np


def transmit(vector: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Send a binary vector through a binary symmetric channel.

    Each bit flips independently with probability ``noise``.

    :param vector: 1-D ``uint8`` array of zeros and ones; not modified.
    :param noise: per-bit flip probability.
    :param rng: random generator.
    :return: a new ``uint8`` array.
    """
    if noise == 0.0:
        return vector.copy()
    flips = rng.random(vector.shape[0]) < noise
    return vector ^ flips.astype(np.uint8)


def random_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly random binary vector.

    :param n: vector length.
    :param rng: random generator.
    :return: a new ``uint8`` array of length ``n``.
    """
    return rng.integers(0, 2, size=n, dtype=np.uint8)
