"""Extract the emergent code from a population and measure it."""

from dataclasses import dataclass
from math import log2
from typing import Any, Sequence

import numpy as np

from emergent_codes.channel import transmit
from emergent_codes.codes import hamming_bound, min_distance, pairwise_distances, singleton_bound
from emergent_codes.node import Node


@dataclass
class Codebook:
    """The candidate code ``C``: cluster centers of all pooled records.

    ``roles[i]`` is the weight-dominant role of cluster ``i`` and ``purity[i]`` the share of its weight carried by
    that role.
    """

    words: np.ndarray  # (K, n) uint8
    weights: np.ndarray  # (K,) total record weight per cluster
    roles: np.ndarray  # (K,) dominant role
    purity: np.ndarray  # (K,) in (0, 1]

    @property
    def size(self) -> int:
        return len(self.words)


def extract_code(nodes: Sequence[Node], n_roles: int, radius: int, min_share: float) -> Codebook:
    """Pool the records of all nodes and cluster them greedily by Hamming distance.

    Records are visited in decreasing weight order; each joins the nearest existing cluster within ``radius`` or
    starts a new one. A cluster's center is the weighted bitwise majority of its members. Clusters carrying less
    than ``min_share`` of the total weight are treated as transient noise and dropped.

    :param nodes: population to analyse.
    :param n_roles: number of role labels ``R``.
    :param radius: clustering radius (normally the merge radius ``theta``).
    :param min_share: minimum fraction of total pooled weight for a cluster to count as a codeword.
    :return: the codebook, possibly empty.
    """
    n = nodes[0].n
    vectors = np.concatenate([node.vectors for node in nodes]) if nodes else np.zeros((0, n), np.uint8)
    weights = np.concatenate([node.weights for node in nodes]) if nodes else np.zeros(0)
    roles = np.concatenate([node.roles for node in nodes]) if nodes else np.zeros(0, np.int64)
    empty = Codebook(np.zeros((0, n), np.uint8), np.zeros(0), np.zeros(0, np.int64), np.zeros(0))
    if len(weights) == 0:
        return empty

    centers: list[np.ndarray] = []
    bit_sums: list[np.ndarray] = []
    totals: list[float] = []
    role_weights: list[np.ndarray] = []
    for i in np.argsort(-weights, kind="stable"):
        v, w, r = vectors[i], weights[i], roles[i]
        if centers:
            dist = np.count_nonzero(np.asarray(centers) != v, axis=1)
            j = int(dist.argmin())
            if dist[j] <= radius:
                bit_sums[j] += w * v
                totals[j] += w
                role_weights[j][r] += w
                centers[j] = (bit_sums[j] / totals[j] >= 0.5).astype(np.uint8)
                continue
        centers.append(v.copy())
        bit_sums.append(w * v.astype(np.float64))
        totals.append(float(w))
        rw = np.zeros(n_roles)
        rw[r] = w
        role_weights.append(rw)

    total = np.asarray(totals)
    keep = np.flatnonzero(total >= min_share * total.sum())
    if len(keep) == 0:
        return empty
    rw = np.asarray(role_weights)[keep]
    return Codebook(
        words=np.asarray(centers, dtype=np.uint8)[keep],
        weights=total[keep],
        roles=rw.argmax(axis=1),
        purity=rw.max(axis=1) / total[keep],
    )


def decoding_error(
    code: Codebook, noise: float, trials: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """Push random codewords of ``C`` through the channel and decode by minimum distance.

    :return: ``(word_error_rate, role_error_rate)``; ``(None, None)`` for an empty code. A word error is decoding to
        a different codeword; a role error is decoding to a codeword whose dominant role differs.
    """
    if code.size == 0:
        return None, None
    word_errors = role_errors = 0
    for _ in range(trials):
        sent = int(rng.integers(code.size))
        received = transmit(code.words[sent], noise, rng)
        dist = np.count_nonzero(code.words != received, axis=1)
        candidates = np.flatnonzero(dist == dist.min())
        decoded = int(rng.choice(candidates))
        word_errors += decoded != sent
        role_errors += code.roles[decoded] != code.roles[sent]
    return word_errors / trials, role_errors / trials


def cross_role_distance(code: Codebook) -> int | None:
    """Minimum distance between codewords of different dominant roles (synonym pairs are ignored)."""
    if len(set(code.roles.tolist())) < 2:
        return None
    dist = pairwise_distances(code.words)
    return int(dist[code.roles[:, None] != code.roles[None, :]].min())


def top_word_indices(code: Codebook) -> list[int]:
    """Index of the heaviest codeword of each covered role, in increasing role order."""
    return [
        int(np.flatnonzero(code.roles == r)[np.argmax(code.weights[code.roles == r])]) for r in np.unique(code.roles)
    ]


def summarize(
    nodes: Sequence[Node],
    n_roles: int,
    noise: float,
    radius: int,
    rng: np.random.Generator,
    min_share: float = 0.02,
    trials: int = 2000,
) -> dict[str, Any]:
    """All end-of-run metrics as a flat dict (suitable for a CSV row)."""
    code = extract_code(nodes, n_roles, radius, min_share)
    n = nodes[0].n
    d = min_distance(code.words)
    word_err, role_err = decoding_error(code, noise, trials, rng)
    sizes = [node.size for node in nodes]
    top = top_word_indices(code)
    return {
        "records_mean": float(np.mean(sizes)),
        "code_size": code.size,
        "k": log2(code.size) if code.size else None,
        "d": d,
        "d_roles": cross_role_distance(code),
        "d_top": min_distance(code.words[top]),
        "roles_covered": len(top),
        "purity_mean": float(np.average(code.purity, weights=code.weights)) if code.size else None,
        "word_error": word_err,
        "role_error": role_err,
        "hamming_bound": hamming_bound(n, d) if d else None,
        "singleton_bound": singleton_bound(n, d) if d else None,
    }
