import numpy as np

from emergent_codes.analysis import extract_code
from emergent_codes.codes import hamming74
from emergent_codes.config import SimConfig
from emergent_codes.node import Node
from emergent_codes.simulation import Simulation

SMALL = SimConfig(n=10, n_roles=3, n_nodes=8, ticks=300, seed=1)


def test_same_seed_is_reproducible():
    a = Simulation(SMALL).run().metrics
    b = Simulation(SMALL).run().metrics
    assert a == b


def test_empty_init_bootstraps_a_repertoire():
    result = Simulation(SMALL.with_(init="empty")).run()
    assert result.metrics["code_size"] >= 1
    assert result.metrics["records_mean"] > 0


def test_turnover_replaces_nodes():
    sim = Simulation(SMALL.with_(turnover=1.0, ticks=1))
    sim.run()
    assert all(node.size == 0 for node in sim.nodes)


def test_extract_code_recovers_seeded_codebook():
    words = hamming74()
    node = Node(7)
    for i, w in enumerate(words):
        node.add(w, i % 4, weight=1.0)
    code = extract_code([node, node], n_roles=4, radius=1, min_share=0.01)
    assert code.size == 16
    assert {w.tobytes() for w in code.words} == {w.tobytes() for w in words}
    assert np.all(code.purity == 1.0)
