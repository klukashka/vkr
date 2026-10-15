"""Population dynamics: nodes exchanging role-labelled binary vectors over a noisy channel."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from self_organized_codes.analysis import summarize
from self_organized_codes.channel import random_vector, transmit
from self_organized_codes.codes import hamming74
from self_organized_codes.config import SimConfig
from self_organized_codes.node import Node, Outcome

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Hold the outcome of one run.

    :ivar config: parameters of the run.
    :ivar metrics: end-of-run metrics, as returned by :func:`~self_organized_codes.analysis.summarize`.
    :ivar history: periodic snapshots of the metrics, each with a ``"tick"`` key; empty if snapshots were disabled.
    :ivar outcomes: number of reinforcements, confusions and created records over the whole run.
    """

    config: SimConfig
    metrics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)


class Simulation:
    """Simulate a fully mixed population of ``N`` nodes.

    One tick is ``interactions_per_tick`` transmissions, then forgetting, then turnover. In a transmission a random
    speaker picks a role uniformly, draws one of its records of that role by weight and sends it through the channel
    to a random listener. A speaker with no record for the role invents a random vector and stores it as its own
    record for that role — this is the only source of novelty besides channel noise.

    :ivar cfg: simulation parameters.
    :ivar rng: random generator seeded with ``cfg.seed``.
    :ivar tick: number of completed ticks.
    :ivar outcomes: running count of reception outcomes, keyed by :class:`~self_organized_codes.node.Outcome` value.
    :ivar nodes: the population.
    """

    def __init__(self, cfg: SimConfig) -> None:
        """Create the population in its initial state.

        :param cfg: simulation parameters.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.tick = 0
        self.outcomes: Counter[str] = Counter()
        self.nodes = [self._initial_node() for _ in range(cfg.n_nodes)]

    # ---------------------------------------------------------- construction

    def _initial_node(self) -> Node:
        """Create one node according to ``cfg.init``.

        ``"random"`` gives ``init_records_per_role`` random records per role, ``"empty"`` gives none, and
        ``"hamming74"`` gives all Hamming [7,4] codewords distributed over the roles in turn.

        :return: the new node.
        """
        cfg = self.cfg
        node = Node(cfg.n)
        if cfg.init == "random":
            for role in range(cfg.n_roles):
                for _ in range(cfg.init_records_per_role):
                    node.add(random_vector(cfg.n, self.rng), role, cfg.new_weight)
        elif cfg.init == "hamming74":
            for i, word in enumerate(hamming74()):
                node.add(word, i % cfg.n_roles, cfg.new_weight)
        return node

    # -------------------------------------------------------------- dynamics

    def transmit_once(self) -> Outcome:
        """Perform one speaker-to-listener transmission.

        :return: what happened to the message at the listener.
        """
        cfg, rng = self.cfg, self.rng
        s, t = rng.choice(cfg.n_nodes, size=2, replace=False)
        speaker, listener = self.nodes[s], self.nodes[t]
        role = int(rng.integers(cfg.n_roles))
        vector = speaker.choose(role, rng)
        if vector is None:
            vector = random_vector(cfg.n, rng)
            speaker.add(vector, role, cfg.new_weight)
        outcome = listener.receive(transmit(vector, cfg.noise, rng), role, cfg, rng)
        self.outcomes[outcome.value] += 1
        return outcome

    def step(self) -> None:
        """Advance the simulation by one tick: transmissions, forgetting, then turnover."""
        cfg = self.cfg
        for _ in range(cfg.effective_interactions):
            self.transmit_once()
        for node in self.nodes:
            node.forget(cfg.decay, cfg.min_weight)
        if cfg.turnover > 0:
            for i in np.flatnonzero(self.rng.random(cfg.n_nodes) < cfg.turnover):
                self.nodes[i] = Node(cfg.n)
        self.tick += 1

    def metrics(self, trials: int = 2000) -> dict[str, Any]:
        """Measure the code currently held by the population.

        :param trials: number of codewords transmitted in the decoding test.
        :return: flat dictionary of metrics, see :func:`~self_organized_codes.analysis.summarize`.
        """
        return summarize(self.nodes, self.cfg.n_roles, self.cfg.noise, self.cfg.merge_radius, self.rng, trials=trials)

    def run(self, snapshot_every: int = 0) -> RunResult:
        """Run ``cfg.ticks`` ticks and measure the result.

        :param snapshot_every: record intermediate metrics every this many ticks (0 disables snapshots).
        :return: final metrics, snapshot history and outcome counts.
        """
        history: list[dict[str, Any]] = []
        for _ in range(self.cfg.ticks):
            self.step()
            if snapshot_every and self.tick % snapshot_every == 0:
                snap = {"tick": self.tick, **self.metrics(trials=200)}
                history.append(snap)
                logger.debug(
                    "tick %d: |C|=%s d=%s records=%.1f", self.tick, snap["code_size"], snap["d"], snap["records_mean"]
                )
        return RunResult(self.cfg, self.metrics(), history, dict(self.outcomes))
