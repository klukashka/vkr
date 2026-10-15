"""Tests for a node's reception, choice and forgetting rules."""

import numpy as np

from self_organized_codes.config import SimConfig
from self_organized_codes.node import Node, Outcome

CFG = SimConfig(n=6, merge_radius=1, role_penalty=0.5, reinforce=1.0, new_weight=1.0)
RNG = np.random.default_rng(0)


def vec(s: str) -> np.ndarray:
    """Build a binary vector from a string of zeros and ones.

    :param s: bits, e.g. ``"110100"``.
    :return: ``uint8`` array of the same length.
    """
    return np.array([int(c) for c in s], dtype=np.uint8)


def test_far_input_creates_record():
    """Check that an input farther than the merge radius is stored as a new record."""
    node = Node(6)
    assert node.receive(vec("000000"), 0, CFG, RNG) is Outcome.CREATED
    assert node.receive(vec("111111"), 0, CFG, RNG) is Outcome.CREATED
    assert node.size == 2


def test_close_same_role_reinforces_and_moves_by_majority():
    """Check that a close same-role input adds weight and moves the record by bitwise majority."""
    node = Node(6)
    node.add(vec("000000"), 0, weight=1.0)
    assert node.receive(vec("100000"), 0, CFG, RNG) is Outcome.REINFORCED
    assert node.size == 1
    assert node.weights[0] == 2.0
    # tie (1 of 2 inputs has the bit) resolves to 1; a further zero pulls it back
    assert node.vectors[0].tolist() == [1, 0, 0, 0, 0, 0]
    node.receive(vec("000000"), 0, CFG, RNG)
    assert node.vectors[0].tolist() == [0, 0, 0, 0, 0, 0]


def test_close_other_role_penalizes_and_stores():
    """Check that a close input of another role penalizes the record and is stored under its own role."""
    node = Node(6)
    node.add(vec("000000"), 0, weight=1.0)
    assert node.receive(vec("000001"), 1, CFG, RNG) is Outcome.CONFUSED
    assert node.weights.tolist() == [0.5, 1.0]
    assert node.roles.tolist() == [0, 1]


def test_forget_drops_light_records_and_keeps_order():
    """Check that forgetting removes light records, keeps the order of the rest, and survives storage growth."""
    node = Node(6, capacity=1)  # also exercises growth
    for i, w in enumerate([1.0, 0.05, 2.0]):
        node.add(vec("000000"), i, weight=w)
    node.forget(decay=0.5, min_weight=0.1)
    assert node.roles.tolist() == [0, 2]
    assert node.weights.tolist() == [0.5, 1.0]


def test_choose_respects_role():
    """Check that a node speaks only with records of the requested role and returns ``None`` for unknown roles."""
    node = Node(6)
    node.add(vec("111000"), 0, weight=1.0)
    node.add(vec("000111"), 1, weight=1.0)
    assert node.choose(1, RNG).tolist() == [0, 0, 0, 1, 1, 1]
    assert node.choose(2, RNG) is None
