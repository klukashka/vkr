"""v0.2 follow-ups to the v0.1 exploratory series.

Parts (each can be run alone, results go to results/v02/<part>.json):

* ``general``  — does d_top grow with noise for other n and R?
* ``single``   — R=1 over long runs: why no collapse to one vector?
* ``hamming``  — control: n=7, R=16, theta=1, seeded with Hamming [7,4] vs random
* ``ceiling``  — why does d_top stall around 5.5 at n=12, R=4?

Usage::

    .venv/bin/python experiments/v02.py general single hamming ceiling
"""

import argparse
import itertools
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from emergent_codes.analysis import extract_code, top_word_indices
from emergent_codes.codes import hamming74, min_distance, pairwise_distances
from emergent_codes.config import SimConfig
from emergent_codes.simulation import Simulation

logger = logging.getLogger("v02")
OUT = Path(__file__).resolve().parent.parent / "results" / "v02"
MIN_SHARE = 0.02


# --------------------------------------------------------------------- helpers


def end_state(cfg: SimConfig) -> dict[str, Any]:
    """Run to the end and return the metrics plus the top word of each role (for inspection)."""
    sim = Simulation(cfg)
    result = sim.run()
    code = extract_code(sim.nodes, cfg.n_roles, cfg.merge_radius, MIN_SHARE)
    top = top_word_indices(code)
    return {
        **cfg.to_dict(),
        **result.metrics,
        "top_words": code.words[top].tolist(),
        "top_roles": code.roles[top].tolist(),
    }


def trajectory(cfg: SimConfig, every: int) -> dict[str, Any]:
    """Run and snapshot structure and event rates every ``every`` ticks."""
    sim = Simulation(cfg)
    prev: Counter[str] = Counter()
    rows = []
    for _ in range(cfg.ticks):
        sim.step()
        if sim.tick % every:
            continue
        code = extract_code(sim.nodes, cfg.n_roles, cfg.merge_radius, MIN_SHARE)
        top = top_word_indices(code)
        events = sim.outcomes - prev
        prev = Counter(sim.outcomes)
        total = sum(events.values())
        shares = code.weights / code.weights.sum() if code.size else np.zeros(0)
        tw = code.words[top]
        pair = pairwise_distances(tw)[np.triu_indices(len(tw), 1)] if len(tw) > 1 else np.zeros(0)
        rows.append(
            {
                "tick": sim.tick,
                "code_size": code.size,
                "d_top": min_distance(tw),
                "pair_mean": float(pair.mean()) if len(pair) else None,
                "top_share": float(shares.max()) if code.size else None,
                "entropy_bits": float(-(shares * np.log2(shares)).sum()) if code.size else None,
                "confused_rate": events["confused"] / total,
                "created_rate": events["created"] / total,
            }
        )
    return {**cfg.to_dict(), "trajectory": rows}


def random_code_min_distance(n: int, size: int, samples: int, rng: np.random.Generator, exclude: int = 0) -> float:
    """Mean min distance of ``size`` uniform random words, rejecting codes with min distance below ``exclude``."""
    values = []
    while len(values) < samples:
        words = rng.integers(0, 2, size=(size, n), dtype=np.uint8)
        d = min_distance(words)
        if d is not None and d >= exclude:
            values.append(d)
    return float(np.mean(values))


def confusion_probability(n: int, d: int, p: float, theta: int) -> float:
    """P(a BSC(p) copy of word A lands within ``theta`` of word B, where dist(A, B) = d).

    Flips inside the d differing positions move toward B, flips in the n-d agreeing ones move away.
    """
    total = 0.0
    for i in range(d + 1):  # flips toward B
        for j in range(n - d + 1):  # flips away
            if d - i + j <= theta:
                total += comb(d, i) * p**i * (1 - p) ** (d - i) * comb(n - d, j) * p**j * (1 - p) ** (n - d - j)
    return total


def pool_map(fn, items) -> list[Any]:
    with ProcessPoolExecutor() as pool:
        return list(pool.map(fn, items))


def save(name: str, data: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1))
    logger.info("saved %s", OUT / f"{name}.json")


def mean_sd(xs: list[float | None]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"{np.mean(xs):.2f}±{np.std(xs, ddof=1) if len(xs) > 1 else 0:.2f}"


# ----------------------------------------------------------------------- parts


def part_general(seeds: int, ticks: int) -> None:
    ns, rs, ps = (8, 12, 16, 20), (2, 4, 8), (0.0, 0.01, 0.02, 0.03, 0.04)
    configs = [
        SimConfig(n=n, n_roles=r, noise=p, ticks=ticks, seed=s)
        for n, r, p, s in itertools.product(ns, rs, ps, range(seeds))
    ]
    rows = pool_map(end_state, configs)
    save("general", rows)
    rng = np.random.default_rng(0)
    print(
        f"{'n':>3} {'R':>2} {'random':>6} " + " ".join(f"{'p=' + str(p):>10}" for p in ps) + "   slope±se   purity@.04"
    )
    for n, r in itertools.product(ns, rs):
        group = [x for x in rows if x["n"] == n and x["n_roles"] == r]
        base = random_code_min_distance(n, r, 2000, rng)
        cells = [mean_sd([x["d_top"] for x in group if x["noise"] == p]) for p in ps]
        pts = [(x["noise"], x["d_top"]) for x in group if x["d_top"] is not None]
        x, y = np.array(pts).T
        slope, _ = np.polyfit(x, y, 1)
        resid = y - np.polyval(np.polyfit(x, y, 1), x)
        se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
        purity = mean_sd([g["purity_mean"] for g in group if g["noise"] == ps[-1]])
        print(
            f"{n:>3} {r:>2} {base:>6.2f} " + " ".join(f"{c:>10}" for c in cells) + f"  {slope:6.1f}±{se:<5.1f} {purity}"
        )


def part_single(seeds: int, ticks: int) -> None:
    configs = [
        SimConfig(n_roles=1, noise=p, ticks=ticks, seed=s) for p, s in itertools.product((0.0, 0.05), range(seeds))
    ]
    every = ticks // 12
    runs = pool_map(partial(trajectory, every=every), configs)
    save("single", runs)
    for p in (0.0, 0.05):
        print(f"# noise={p}: code_size / top_share / created_rate over time (mean over {seeds} seeds)")
        group = [r["trajectory"] for r in runs if r["noise"] == p]
        for i, snap in enumerate(group[0]):
            print(
                f"  tick {snap['tick']:>6}: |C|={mean_sd([g[i]['code_size'] for g in group]):>10}"
                f" top_share={mean_sd([g[i]['top_share'] for g in group]):>10}"
                f" created={np.mean([g[i]['created_rate'] for g in group]):.4f}"
            )


def part_hamming(seeds: int, ticks: int) -> None:
    configs = [
        SimConfig(n=7, n_roles=16, merge_radius=1, init=init, noise=p, ticks=ticks, seed=s)
        for init, p, s in itertools.product(("hamming74", "random"), (0.0, 0.02, 0.05, 0.1), range(seeds))
    ]
    rows = pool_map(end_state, configs)
    reference = {w.tobytes(): i for i, w in enumerate(hamming74())}
    for row in rows:
        words = [np.array(w, dtype=np.uint8).tobytes() for w in row["top_words"]]
        row["kept_in_place"] = sum(reference.get(w) == r for w, r in zip(words, row["top_roles"]))
        row["hamming_words"] = sum(w in reference for w in words)
    save("hamming", rows)
    cols = ("code_size", "roles_covered", "d", "d_top", "purity_mean", "role_error", "hamming_words", "kept_in_place")
    print(f"{'init':>10} {'p':>5} " + " ".join(f"{c:>14}" for c in cols))
    for init, p in itertools.product(("hamming74", "random"), (0.0, 0.02, 0.05, 0.1)):
        group = [x for x in rows if x["init"] == init and x["noise"] == p]
        print(f"{init:>10} {p:>5} " + " ".join(f"{mean_sd([g[c] for g in group]):>14}" for c in cols))


def part_ceiling(seeds: int, ticks: int) -> None:
    rng = np.random.default_rng(0)
    print("# random 4-word codes, n=12: mean min distance, unconditioned and with min >= k")
    for k in (0, 3, 4, 5, 6):
        print(f"  min>={k}: {random_code_min_distance(12, 4, 2000, rng, exclude=k):.2f}")
    print("  best possible (Plotkin, A(12,8)=4): 8")

    print("# P(copy of A lands within theta of B | dist(A,B)=d), n=12, p=0.04")
    print("  " + "theta \\ d".rjust(9) + "".join(f"{d:>10}" for d in range(3, 9)))
    for theta in (2, 3, 4):
        print(f"  {theta:>9}" + "".join(f"{confusion_probability(12, d, 0.04, theta):>10.1e}" for d in range(3, 9)))

    configs = [
        SimConfig(noise=0.04, merge_radius=t, ticks=ticks, seed=s)
        for t, s in itertools.product((2, 3, 4), range(seeds))
    ]
    runs = pool_map(partial(trajectory, every=ticks // 10), configs)
    save("ceiling", runs)
    for t in (2, 3, 4):
        print(f"# p=0.04 theta={t}: d_top, mean pairwise distance of top words, confusion rate (mean over {seeds})")
        group = [r["trajectory"] for r in runs if r["merge_radius"] == t]
        for i, snap in enumerate(group[0]):
            print(
                f"  tick {snap['tick']:>6}: d_top={mean_sd([g[i]['d_top'] for g in group]):>10}"
                f" pair_mean={mean_sd([g[i]['pair_mean'] for g in group]):>10}"
                f" confused={np.mean([g[i]['confused_rate'] for g in group]):.4f}"
            )


PARTS = {
    "general": (part_general, 10, 10_000),
    "single": (part_single, 5, 60_000),
    "hamming": (part_hamming, 10, 10_000),
    "ceiling": (part_ceiling, 5, 50_000),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parts", nargs="+", choices=PARTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in args.parts:
        fn, seeds, ticks = PARTS[name]
        print(f"### {name} (seeds={seeds}, ticks={ticks})", flush=True)
        fn(seeds, ticks)


if __name__ == "__main__":
    main()
