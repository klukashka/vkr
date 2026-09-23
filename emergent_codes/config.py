"""Simulation parameters.

Symbols in comments refer to the notation used in the README.
"""

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

INIT_MODES = ("random", "empty", "hamming74")


@dataclass(frozen=True)
class SimConfig:
    """All parameters of a single simulation run.

    :raises ValueError: if a parameter is out of its admissible range.
    """

    # --- channel and messages ---
    n: int = 12  # vector length n
    n_roles: int = 4  # R, number of role labels
    noise: float = 0.05  # p, per-bit flip probability

    # --- population ---
    n_nodes: int = 20  # N
    turnover: float = 0.0  # L, per-tick probability that a node is replaced by an empty newcomer
    interactions_per_tick: int = 0  # 0 means "one per node", i.e. n_nodes

    # --- memory dynamics ---
    merge_radius: int = 2  # theta, max Hamming distance at which an input is merged into a record
    decay: float = 0.03  # lambda, per-tick multiplicative weight decay
    min_weight: float = 0.1  # records below this weight are forgotten
    new_weight: float = 1.0  # weight of a freshly created record
    reinforce: float = 1.0  # weight added to a record on a same-role match
    role_penalty: float = 0.1  # weight multiplier applied to a record matched under a different role

    # --- initialization ---
    init: str = "random"  # one of INIT_MODES
    init_records_per_role: int = 3  # random records per role per node (init="random")

    # --- run ---
    ticks: int = 3000
    seed: int = 0

    def __post_init__(self) -> None:
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
        """Number of speaker -> listener transmissions per tick."""
        return self.interactions_per_tick or self.n_nodes

    def with_(self, **changes: Any) -> "SimConfig":
        """Return a copy with some fields replaced (and re-validated)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_types(cls) -> dict[str, type]:
        """Map of field name to its Python type, used by the CLI to parse overrides."""
        return {f.name: type(getattr(cls(), f.name)) for f in fields(cls)}
