"""v0.2 checks: [7,4] control with decay scaled by R; generality with theta scaled to n."""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v02 import end_state, mean_sd, pool_map

from emergent_codes import SimConfig
from emergent_codes.codes import hamming74


def main():
    ref = {w.tobytes(): i for i, w in enumerate(hamming74())}
    print("# [7,4] control, R=16, theta=1: decay scaled by R/4")
    cfgs = [
        SimConfig(n=7, n_roles=16, merge_radius=1, init=i, noise=p, decay=dec, ticks=10000, seed=s)
        for dec, i, p, s in itertools.product((0.0075, 0.003), ("hamming74", "random"), (0.0, 0.02, 0.05), range(10))
    ]
    rows = pool_map(end_state, cfgs)
    for r in rows:
        ws = [np.array(w, np.uint8).tobytes() for w in r["top_words"]]
        r["kept"] = sum(ref.get(w) == ro for w, ro in zip(ws, r["top_roles"]))
    print(
        f"{'decay':>6} {'init':>10} {'p':>5} {'|C|':>11} {'roles':>11} {'d_top':>11} {'purity':>11} {'role_err':>11} {'kept':>11}"
    )
    for dec, i, p in itertools.product((0.0075, 0.003), ("hamming74", "random"), (0.0, 0.02, 0.05)):
        g = [r for r in rows if r["decay"] == dec and r["init"] == i and r["noise"] == p]
        print(
            f"{dec:>6} {i:>10} {p:>5} "
            + " ".join(
                f"{mean_sd([x[c] for x in g]):>11}"
                for c in ("code_size", "roles_covered", "d_top", "purity_mean", "role_error", "kept")
            )
        )

    print("\n# generality with theta scaled to n (theta = n/6), R=4; (k) = runs with all 4 roles")
    cfgs = [
        SimConfig(n=n, n_roles=4, merge_radius=n // 6, noise=p, ticks=10000, seed=s)
        for n, p, s in itertools.product((12, 18, 24), (0.0, 0.01, 0.02, 0.03, 0.04), range(10))
    ]
    rows = pool_map(end_state, cfgs)
    for n in (12, 18, 24):
        g = [r for r in rows if r["n"] == n and r["roles_covered"] == 4]
        x = np.array([r["noise"] for r in g])
        y = np.array([r["d_top"] for r in g], float)
        b = np.polyfit(x, y, 1)
        se = np.sqrt((y - np.polyval(b, x)).var(ddof=2) / ((x - x.mean()) ** 2).sum())
        cells = [
            f"{mean_sd([r['d_top'] for r in g if r['noise']==p])}({sum(r['noise']==p for r in g)})"
            for p in (0.0, 0.01, 0.02, 0.03, 0.04)
        ]
        print(f"n={n} theta={n//6}: " + "  ".join(cells) + f"   slope {b[0]:.1f}±{se:.1f} t={b[0]/se:.1f}")


if __name__ == "__main__":
    main()
