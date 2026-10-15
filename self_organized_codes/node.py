"""A single node: a store of weighted, role-labelled binary records."""

from enum import Enum

import numpy as np

from self_organized_codes.config import SimConfig


class Outcome(Enum):
    """Describe what happened to an incoming message at the receiver.

    :cvar REINFORCED: the nearest record within the merge radius has the same role and was reinforced.
    :cvar CONFUSED: the nearest record within the merge radius has a different role and was penalized.
    :cvar CREATED: nothing was close enough, so the input was stored as a new record.
    """

    REINFORCED = "reinforced"
    CONFUSED = "confused"
    CREATED = "created"


class Node:
    """Store a node's records: binary vector, per-bit mass, weight and role.

    Each record keeps a per-bit *mass* in ``[0, 1]`` — the weighted fraction of merged inputs that had a one in that
    position. The record's vector is the bitwise majority ``mass >= 0.5``. Storage is a set of preallocated arrays
    that grow on demand; only the first ``size`` rows are live.

    :ivar n: vector length.
    :ivar size: number of live records.
    """

    def __init__(self, n: int, capacity: int = 16) -> None:
        """Create an empty store.

        :param n: vector length.
        :param capacity: initial number of preallocated rows; the store grows beyond it on demand.
        """
        self.n = n
        self.size = 0
        self._mass = np.zeros((capacity, n), dtype=np.float64)
        self._bits = np.zeros((capacity, n), dtype=np.uint8)
        self._weight = np.zeros(capacity, dtype=np.float64)
        self._role = np.zeros(capacity, dtype=np.int64)

    # ------------------------------------------------------------------ views

    @property
    def vectors(self) -> np.ndarray:
        """Vectors of the live records.

        :return: ``(size, n)`` ``uint8`` view; not a copy.
        """
        return self._bits[: self.size]

    @property
    def weights(self) -> np.ndarray:
        """Weights of the live records.

        :return: ``(size,)`` float view; not a copy.
        """
        return self._weight[: self.size]

    @property
    def roles(self) -> np.ndarray:
        """Roles of the live records.

        :return: ``(size,)`` integer view; not a copy.
        """
        return self._role[: self.size]

    # ---------------------------------------------------------------- storage

    def add(self, vector: np.ndarray, role: int, weight: float) -> int:
        """Append a record whose mass equals its vector.

        :param vector: binary vector of the record.
        :param role: role of the record.
        :param weight: initial weight.
        :return: index of the new record.
        """
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
        """Double the capacity of the storage arrays, keeping the live records."""
        capacity = 2 * len(self._weight)
        self._mass = np.resize(self._mass, (capacity, self.n))
        self._bits = np.resize(self._bits, (capacity, self.n))
        self._weight = np.resize(self._weight, capacity)
        self._role = np.resize(self._role, capacity)

    # --------------------------------------------------------------- decoding

    def nearest(self, vector: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        """Find the record closest to a vector by Hamming distance.

        Ties are broken uniformly at random, regardless of weight and role.

        :param vector: binary vector to decode.
        :param rng: random generator used to break ties.
        :return: ``(index, distance)`` of the closest record, or ``(-1, n + 1)`` if the store is empty.
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
        """Process an incoming message by minimum-distance decoding against the node's own records.

        * Within ``merge_radius`` of a same-role record — the record is reinforced and pulled toward the input.
        * Within ``merge_radius`` of a different-role record — that record is penalized and the input is stored
          as a new record of the correct role, so the two start competing.
        * Otherwise — the input is stored as a new record.

        :param vector: received (possibly corrupted) binary vector.
        :param role: role label of the message; delivered without errors.
        :param cfg: simulation parameters (merge radius, reinforcement, penalty, new-record weight).
        :param rng: random generator used to break ties in :meth:`nearest`.
        :return: what happened to the message.
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
        """Merge a vector into a record and recompute its bits by weighted bitwise majority.

        The mass is updated as the weighted mean ``(w * mass + amount * vector) / (w + amount)``, a bit is one when
        its mass is at least ``0.5``, and the weight grows by ``amount``.

        :param idx: index of the record.
        :param vector: binary vector being merged.
        :param amount: weight given to the new vector and added to the record's weight.
        """
        w = self._weight[idx]
        self._mass[idx] = (w * self._mass[idx] + amount * vector) / (w + amount)
        self._bits[idx] = self._mass[idx] >= 0.5
        self._weight[idx] = w + amount

    def choose(self, role: int, rng: np.random.Generator) -> np.ndarray | None:
        """Draw a record of the given role with probability proportional to its weight.

        :param role: role to speak about.
        :param rng: random generator.
        :return: a copy of the chosen record's vector, or ``None`` if the node has no record for this role.
        """
        candidates = np.flatnonzero(self.roles == role)
        if len(candidates) == 0:
            return None
        w = self._weight[candidates]
        idx = candidates[rng.choice(len(candidates), p=w / w.sum())]
        return self._bits[idx].copy()

    def forget(self, decay: float, min_weight: float) -> None:
        """Decay all weights and drop the records that fell below a threshold.

        :param decay: fraction of weight lost by every record.
        :param min_weight: records with a smaller weight after the decay are removed.
        """
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
