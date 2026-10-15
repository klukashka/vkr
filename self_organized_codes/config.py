"""Simulation parameters.

Symbols in the field descriptions refer to the notation used in the README and the v0.1 report.
"""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

INIT_MODES = ("random", "empty", "hamming74")


@dataclass(frozen=True)
class SimConfig:
    """Hold all parameters of a single simulation run.

    The instance is immutable; use :meth:`with_` to derive a modified copy.

    :ivar n: binary vector length, ``n``.
    :ivar n_roles: number of roles (messages), ``R``.
    :ivar noise: per-bit flip probability of the channel, ``p``.
    :ivar n_nodes: number of nodes in the population, ``N``.
    :ivar turnover: per-tick probability that a node is replaced by an empty newcomer, ``L``.
    :ivar interactions_per_tick: transmissions per tick; ``0`` means one per node, i.e. ``n_nodes``.
    :ivar merge_radius: largest Hamming distance at which an input is merged into a record, ``θ``.
    :ivar decay: fraction of every record's weight lost per tick, ``λ``.
    :ivar min_weight: records whose weight falls below this value are forgotten, ``w_min``.
    :ivar new_weight: weight of a freshly created record, ``w_0``.
    :ivar reinforce: weight added to a record on a same-role match, ``a``.
    :ivar role_penalty: weight multiplier applied to a record matched under a different role, ``κ``.
    :ivar init: initial state of the nodes' memory, one of :data:`INIT_MODES`.
    :ivar init_records_per_role: random records per role per node when ``init == "random"``, ``m``.
    :ivar ticks: run length in ticks, ``T``.
    :ivar seed: seed of the random generator.
    :raises ValueError: if a parameter is out of its admissible range.
    """

    # --- channel and messages ---
    n: int = 12
    n_roles: int = 4
    noise: float = 0.05

    # --- population ---
    n_nodes: int = 20
    turnover: float = 0.0
    interactions_per_tick: int = 0

    # --- memory dynamics ---
    merge_radius: int = 2
    decay: float = 0.03
    min_weight: float = 0.1
    new_weight: float = 1.0
    reinforce: float = 1.0
    role_penalty: float = 0.1

    # --- initialization ---
    init: str = "random"
    init_records_per_role: int = 3

    # --- run ---
    ticks: int = 3000
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the parameters.

        :raises ValueError: if a parameter is out of its admissible range.
        """
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.n_roles < 1:
            raise ValueError("n_roles must be positive")
        if not 0.0 <= self.noise <= 0.5:
            raise ValueError("noise must be in [0, 0.5]")
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if not 0.0 <= self.turnover <= 1.0:
            raise ValueError("turnover must be in [0, 1]")
        if self.interactions_per_tick < 0:
            raise ValueError("interactions_per_tick must be non-negative")
        if not 0 <= self.merge_radius <= self.n:
            raise ValueError("merge_radius must be in [0, n]")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if not 0.0 < self.role_penalty <= 1.0:
            raise ValueError("role_penalty must be in (0, 1]")
        if self.init not in INIT_MODES:
            raise ValueError(f"init must be one of {INIT_MODES}")
        if self.init == "hamming74" and self.n != 7:
            raise ValueError("init='hamming74' requires n=7")
        if self.ticks < 0:
            raise ValueError("ticks must be non-negative")

    @property
    def effective_interactions(self) -> int:
        """Number of speaker-to-listener transmissions per tick.

        :return: ``interactions_per_tick``, or ``n_nodes`` when it is ``0``.
        """
        return self.interactions_per_tick or self.n_nodes

    def with_(self, **changes: Any) -> "SimConfig":
        """Derive a copy with some fields replaced.

        :param changes: field names and their new values.
        :return: a new, re-validated configuration.
        :raises ValueError: if a new value is out of its admissible range.
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a plain dictionary.

        :return: field names mapped to their values.
        """
        return asdict(self)

    @classmethod
    def field_types(cls) -> dict[str, type]:
        """Map every field to its Python type; used by the CLI to parse overrides.

        :return: field names mapped to their types.
        """
        return {f.name: type(getattr(cls(), f.name)) for f in fields(cls)}
