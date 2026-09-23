"""v0.3: re-run the roles series with forgetting scaled to R, and the large-n series with 30 seeds.

Parts (results go to results/v03/<part>.json):

* ``roles``   — n=12, theta=2, decay = 0.12 / R (so R=4 keeps the old 0.03), R in 2..32, p in {0.02, 0.05}
* ``large_n`` — n in {16, 20, 24}, theta = round(n / 6), decay = 0.12 / R, R in {2, 4, 8}, p in 0..0.04, 30 seeds

Usage::

    .venv/bin/python experiments/v03.py roles large_n
"""

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v02  # noqa: E402

from emergent_codes.codes import hamming_bound  # noqa: E402
from emergent_codes.config import SimConfig  # noqa: E402

v02.OUT = Path(__file__).resolve().parent.parent / "results" / "v03"
DECAY_PER_ROLE = 0.12  # decay * R; R=4 gives the v0.1 default 0.03


def slope(rows: list[dict]) -> tuple[float, float]:
    """Least-squares slope of d_top on noise and its standard error."""
    x = np.array([r["noise"] for r in rows])
    y = np.array([r["d_top"] for r in rows], dtype=float)
    fit = np.polyfit(x, y, 1)
    resid = y - np.polyval(fit, x)
    return float(fit[0]), float(np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum()))


def part_roles(seeds: int, ticks: int) -> None:
    rs, ps = (2, 4, 8, 16, 32), (0.02, 0.05)
    configs = [
        SimConfig(n_roles=r, decay=DECAY_PER_ROLE / r, noise=p, ticks=ticks, seed=s)
        for r, p, s in itertools.product(rs, ps, range(seeds))
    ]
    rows = v02.pool_map(v02.end_state, configs)
    v02.save("roles", rows)
    rng = np.random.default_rng(0)
    cols = ("roles_covered", "code_size", "d_top", "purity_mean", "role_error")
    print(f"{'p':>5} {'R':>3} {'decay':>8} " + " ".join(f"{c:>14}" for c in cols) + f" {'random d':>9}")
    for p, r in itertools.product(ps, rs):
        group = [x for x in rows if x["noise"] == p and x["n_roles"] == r]
        base = v02.random_code_min_distance(12, r, 1000, rng)
        cells = " ".join(f"{v02.mean_sd([g[c] for g in group]):>14}" for c in cols)
        print(f"{p:>5} {r:>3} {DECAY_PER_ROLE / r:>8.5f} {cells} {base:>9.2f}")
    print("Hamming bound on |C| at n=12: " + ", ".join(f"d={d}: {hamming_bound(12, d)}" for d in (3, 4, 5, 6)))


def part_large_n(seeds: int, ticks: int) -> None:
    ns, rs, ps = (16, 20, 24), (2, 4, 8), (0.0, 0.01, 0.02, 0.03, 0.04)
    configs = [
        SimConfig(n=n, n_roles=r, merge_radius=round(n / 6), decay=DECAY_PER_ROLE / r, noise=p, ticks=ticks, seed=s)
        for n, r, p, s in itertools.product(ns, rs, ps, range(seeds))
    ]
    rows = v02.pool_map(v02.end_state, configs)
    v02.save("large_n", rows)
    rng = np.random.default_rng(0)
    print("d_top mean±sd (runs with all R roles covered / total); slope over full-coverage runs only")
    print(
        f"{'n':>3} {'th':>2} {'R':>2} {'random':>6} "
        + " ".join(f"{'p=' + str(p):>15}" for p in ps)
        + "  slope±se     t"
    )
    for n, r in itertools.product(ns, rs):
        group = [x for x in rows if x["n"] == n and x["n_roles"] == r]
        full = [x for x in group if x["roles_covered"] == r]
        base = v02.random_code_min_distance(n, r, 2000, rng)
        cells = []
        for p in ps:
            at_p = [x for x in full if x["noise"] == p]
            cells.append(f"{v02.mean_sd([x['d_top'] for x in at_p])}({len(at_p)}/{seeds})")
        b, se = slope(full) if len({x["noise"] for x in full}) > 1 else (float("nan"), float("nan"))
        print(
            f"{n:>3} {round(n / 6):>2} {r:>2} {base:>6.2f} "
            + " ".join(f"{c:>15}" for c in cells)
            + f"  {b:6.1f}±{se:<5.1f} {b / se:5.1f}"
        )


PARTS = {"roles": (part_roles, 10, 10_000), "large_n": (part_large_n, 30, 10_000)}


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
