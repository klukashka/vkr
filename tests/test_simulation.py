"""Tests for the population dynamics and code extraction."""

import numpy as np

from self_organized_codes.analysis import extract_code
from self_organized_codes.codes import hamming74
from self_organized_codes.config import SimConfig
from self_organized_codes.node import Node
from self_organized_codes.simulation import Simulation

SMALL = SimConfig(n=10, n_roles=3, n_nodes=8, ticks=300, seed=1)


def test_same_seed_is_reproducible():
    """Check that two runs with the same seed give identical metrics."""
    a = Simulation(SMALL).run().metrics
    b = Simulation(SMALL).run().metrics
    assert a == b


def test_empty_init_bootstraps_a_repertoire():
    """Check that a population starting with empty memory still builds a code."""
    result = Simulation(SMALL.with_(init="empty")).run()
    assert result.metrics["code_size"] >= 1
    assert result.metrics["records_mean"] > 0


def test_turnover_replaces_nodes():
    """Check that full turnover replaces every node with an empty one."""
    sim = Simulation(SMALL.with_(turnover=1.0, ticks=1))
    sim.run()
    assert all(node.size == 0 for node in sim.nodes)


def test_extract_code_recovers_seeded_codebook():
    """Check that code extraction recovers a known codebook exactly, with full purity."""
    words = hamming74()
    node = Node(7)
    for i, w in enumerate(words):
        node.add(w, i % 4, weight=1.0)
    code = extract_code([node, node], n_roles=4, radius=1, min_share=0.01)
    assert code.size == 16
    assert {w.tobytes() for w in code.words} == {w.tobytes() for w in words}
    assert np.all(code.purity == 1.0)
