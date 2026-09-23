"""A single node: a store of weighted, role-labelled binary records."""

from enum import Enum

import numpy as np

from emergent_codes.config import SimConfig


class Outcome(Enum):
    """What happened to an incoming message at the receiver."""

    REINFORCED = "reinforced"  # matched a record of the same role
    CONFUSED = "confused"  # matched a record of a different role
    CREATED = "created"  # nothing close enough, a new record was stored


class Node:
    """Holds records ``(bit-mass, weight, role)``.

    Each record keeps a per-bit *mass* in ``[0, 1]`` — the weighted fraction of merged inputs that had a one in that
    position. The record's vector is the bitwise majority ``mass >= 0.5``. Storage is a set of preallocated arrays
    that grow on demand; only the first ``size`` rows are live.
    """

    def __init__(self, n: int, capacity: int = 16) -> None:
        self.n = n
        self.size = 0
        self._mass = np.zeros((capacity, n), dtype=np.float64)
        self._bits = np.zeros((capacity, n), dtype=np.uint8)
        self._weight = np.zeros(capacity, dtype=np.float64)
        self._role = np.zeros(capacity, dtype=np.int64)

    # ------------------------------------------------------------------ views

    @property
    def vectors(self) -> np.ndarray:
        return self._bits[: self.size]

    @property
    def weights(self) -> np.ndarray:
        return self._weight[: self.size]

    @property
    def roles(self) -> np.ndarray:
        return self._role[: self.size]

    # ---------------------------------------------------------------- storage

    def add(self, vector: np.ndarray, role: int, weight: float) -> int:
        """Append a record and return its index."""
        if self.size == len(self._weight):
            self._grow()
        i = self.size
        self._mass[i] = vector
        self._bits[i] = vector
        self._weight[i] = weight
        self._role[i] = role
        self.size += 1
        return i

    def _grow(self) -> None:
        capacity = 2 * len(self._weight)
        self._mass = np.resize(self._mass, (capacity, self.n))
        self._bits = np.resize(self._bits, (capacity, self.n))
        self._weight = np.resize(self._weight, capacity)
        self._role = np.resize(self._role, capacity)

    # --------------------------------------------------------------- decoding

    def nearest(self, vector: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        """Index of the closest record by Hamming distance (ties broken at random) and that distance.

        :return: ``(-1, n + 1)`` if the store is empty.
        """
        if self.size == 0:
            return -1, self.n + 1
        dist = np.count_nonzero(self.vectors != vector, axis=1)
        best = dist.min()
        candidates = np.flatnonzero(dist == best)
        idx = int(candidates[0]) if len(candidates) == 1 else int(rng.choice(candidates))
        return idx, int(best)

    # -------------------------------------------------------------- behaviour

    def receive(self, vector: np.ndarray, role: int, cfg: SimConfig, rng: np.random.Generator) -> Outcome:
        """Process an incoming message: minimum-distance decoding against the node's own records.

        * Within ``merge_radius`` of a same-role record — the record is reinforced and pulled toward the input.
        * Within ``merge_radius`` of a different-role record — that record is penalized and the input is stored
          as a new record of the correct role (so the two start competing).
        * Otherwise — the input is stored as a new record.
        """
        idx, dist = self.nearest(vector, rng)
        if idx >= 0 and dist <= cfg.merge_radius:
            if self._role[idx] == role:
                self._reinforce(idx, vector, cfg.reinforce)
                return Outcome.REINFORCED
            self._weight[idx] *= cfg.role_penalty
            self.add(vector, role, cfg.new_weight)
            return Outcome.CONFUSED
        self.add(vector, role, cfg.new_weight)
        return Outcome.CREATED

    def _reinforce(self, idx: int, vector: np.ndarray, amount: float) -> None:
        w = self._weight[idx]
        self._mass[idx] = (w * self._mass[idx] + amount * vector) / (w + amount)
        self._bits[idx] = self._mass[idx] >= 0.5
        self._weight[idx] = w + amount

    def choose(self, role: int, rng: np.random.Generator) -> np.ndarray | None:
        """Draw a record of the given role with probability proportional to weight.

        :return: a copy of the record's vector, or ``None`` if the node has no record for this role.
        """
        candidates = np.flatnonzero(self.roles == role)
        if len(candidates) == 0:
            return None
        w = self._weight[candidates]
        idx = candidates[rng.choice(len(candidates), p=w / w.sum())]
        return self._bits[idx].copy()

    def forget(self, decay: float, min_weight: float) -> None:
        """Decay all weights and drop records that fell below ``min_weight``."""
        if self.size == 0:
            return
        self._weight[: self.size] *= 1.0 - decay
        keep = np.flatnonzero(self._weight[: self.size] >= min_weight)
        if len(keep) == self.size:
            return
        k = len(keep)
        self._mass[:k] = self._mass[keep]
        self._bits[:k] = self._bits[keep]
        self._weight[:k] = self._weight[keep]
        self._role[:k] = self._role[keep]
        self.size = k
