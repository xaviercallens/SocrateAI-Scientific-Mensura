"""Skeptic follow-up: how seed-dependent are the three H3 known-answer verdicts?

The shipped tests fix one seed each. This measures the verdict distribution over
20 independent redraws of each of the three clouds, so the ledger can record the
verdicts as "N of 20 draws" rather than as properties of the cloud family.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from socrates.hypergraph.comparison import compare_accuracy_at_max_n  # noqa: E402


def brownian(n: int, seed: int) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    inc = rng.normal(0.0, 1.0, size=(n * 4, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(inc, axis=0)])
    return [tuple(p) for p in path[::4][:n]]


def uniform(n: int, dim: int, seed: int) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(0.0, 1.0, size=(n, dim))]


def main() -> int:
    seeds = list(range(20))

    print("WIN case -- planar Brownian n=1600 k=10 tol=0.15 (true dim 2)")
    wins = 0
    for s in seeds:
        r = compare_accuracy_at_max_n(brownian(1600, s), 2.0, 1600, k=10, tolerance=0.15)
        wins += r.poly_algebraic_accuracy_win
        print(
            f"  seed {s:2d} poly={r.poly_algebraic_estimate:.4f} "
            f"trad={r.traditional_estimate:.4f} win={r.poly_algebraic_accuracy_win}"
        )
    print(f"  -> poly criterion-(c) wins on {wins}/20 draws\n")

    print("LOSS case -- uniform 3-cube n=400 k=12 tol=0.15 (true dim 3)")
    poly_win = trad_win = 0
    for s in seeds:
        r = compare_accuracy_at_max_n(uniform(400, 3, s), 3.0, 400, k=12, tolerance=0.15)
        poly_win += r.poly_algebraic_accuracy_win
        trad_win += r.traditional_accuracy_win
        print(
            f"  seed {s:2d} poly={r.poly_algebraic_estimate:.4f} "
            f"trad={r.traditional_estimate:.4f} poly_win={r.poly_algebraic_accuracy_win} "
            f"trad_win={r.traditional_accuracy_win}"
        )
    print(f"  -> poly wins {poly_win}/20, traditional wins {trad_win}/20\n")

    print("TIE case -- uniform square n=800 k=10 tol=0.15 margin=0.2 (true dim 2)")
    ties = 0
    for s in seeds:
        r = compare_accuracy_at_max_n(
            uniform(800, 2, s), 2.0, 800, k=10, tolerance=0.15, margin=0.2
        )
        is_tie = (
            r.more_accurate == "tie"
            and not r.poly_algebraic_accuracy_win
            and not r.traditional_accuracy_win
        )
        ties += is_tie
        print(
            f"  seed {s:2d} poly={r.poly_algebraic_estimate:.4f} "
            f"trad={r.traditional_estimate:.4f} verdict={r.more_accurate} tie={is_tie}"
        )
    print(f"  -> tie on {ties}/20 draws\n")

    print("DEGENERACY case -- circle, n in {200,400,800}, k in {4,6,8} (true dim 1)")
    all_refused = True
    for n in (200, 400, 800):
        for k in (4, 6, 8):
            pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
            r = compare_accuracy_at_max_n(pts, 1.0, n, k=k, tolerance=0.05, margin=0.05)
            refused = not r.poly_algebraic_accuracy_win
            all_refused &= refused
            print(
                f"  n={n} k={k} poly={r.poly_algebraic_estimate:.4f} "
                f"trad={r.traditional_estimate:.4f} "
                f"degen={r.poly_algebraic_degenerate_fraction:.3f} "
                f"refused={refused}"
            )
    print(f"  -> degeneracy guard refused every circle configuration: {all_refused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
