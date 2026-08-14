"""E4: is the shell estimator's +0.02..+0.17 error on a uniform square a
BOUNDARY effect, a FINITE-SIZE effect, or k-dependent variance?

This decides whether B3 (boundary correction) fixes it, and therefore whether
W_family caps should be frozen before or after that fix. The three hypotheses
make different, separable predictions:

  BOUNDARY   -- a node whose r-ball reaches the edge of the square sees a
                truncated ball, so its volume grows slower than r^2 and the
                fitted slope is biased. PREDICTION: restricting to interior
                nodes (whose whole ball fits inside) removes most of the bias,
                and the bias does NOT vanish as n grows, because at fixed
                max_radius the affected FRACTION of nodes is roughly constant.
  FINITE-SIZE -- the graph is too small for the shells to reach the asymptotic
                regime. PREDICTION: bias shrinks monotonically with n for
                every k, and interior restriction does not help much.
  VARIANCE   -- no systematic bias; the spread across k is sampling noise.
                PREDICTION: the sign of the error is not stable across seeds.

Truth is exactly 2.0 by construction (i.i.d. uniform on the unit square).
Run: python3 e4_uniform_square_bias.py
"""

from __future__ import annotations

import math
import statistics as st

import numpy as np

from socrates.hypergraph.dimension import _sampled_nodes, local_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

TRUTH = 2.0
SAMPLES = 80


def square(n: int, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.random((n, 2))]


def _interior_margin(points, hg, max_radius: int) -> float:
    """A ball of `max_radius` hops spans roughly max_radius * (median edge
    length). A node further than that from every wall has an untruncated ball."""
    pts = np.asarray(points)
    lengths = [
        float(np.linalg.norm(pts[i] - pts[j]))
        for i, j in list(hg.edges)[: min(4000, len(hg.edges))]
    ]
    return max_radius * st.median(lengths)


def measure(points, k: int, max_radius: int) -> tuple[float, float, int, int]:
    """(all-node mean, interior-only mean, n_all, n_interior)."""
    hg = knn_hypergraph(points, k=k)
    margin = _interior_margin(points, hg, max_radius)
    pts = np.asarray(points)
    every, inner = [], []
    for node in _sampled_nodes(hg, SAMPLES):
        est = local_dimension(hg, node, max_radius=max_radius)
        if not est.is_well_fit(0.9):
            continue
        every.append(est.dimension)
        x, y = pts[node]
        if min(x, y, 1.0 - x, 1.0 - y) > margin:
            inner.append(est.dimension)
    return (
        st.mean(every) if every else float("nan"),
        st.mean(inner) if inner else float("nan"),
        len(every),
        len(inner),
    )


def main() -> None:
    print("=" * 78)
    print("E4  uniform square, truth = 2.0 exactly.  err = estimate - 2.0")
    print("=" * 78)

    print("\n--- 1. FINITE-SIZE: does the bias shrink with n?  (k=10, mr=6) ---")
    print(f"{'n':>7} {'all':>9} {'err':>8} {'interior':>10} {'err':>8} {'n_all':>6} {'n_int':>6}")
    for n in (800, 1600, 3200, 6400, 12800):
        a, i, na, ni = measure(square(n, 11), k=10, max_radius=6)
        print(f"{n:>7} {a:>9.4f} {a - TRUTH:>+8.4f} {i:>10.4f} {i - TRUTH:>+8.4f} {na:>6} {ni:>6}")

    print("\n--- 2. BOUNDARY: all-node vs interior-only across (k, max_radius) ---")
    print(f"{'k':>3} {'mr':>3} {'all':>9} {'err':>8} {'interior':>10} {'err':>8} {'removed':>8}")
    pts = square(6400, 11)
    deltas = []
    for k in (6, 8, 10, 15):
        for mr in (4, 6):
            a, i, na, ni = measure(pts, k=k, max_radius=mr)
            if math.isfinite(a) and math.isfinite(i):
                deltas.append(abs(a - TRUTH) - abs(i - TRUTH))
            print(
                f"{k:>3} {mr:>3} {a:>9.4f} {a - TRUTH:>+8.4f} "
                f"{i:>10.4f} {i - TRUTH:>+8.4f} {na - ni:>8}"
            )
    if deltas:
        print(f"\n  median reduction in |err| from interior restriction: {st.median(deltas):+.4f}")

    print("\n--- 3. VARIANCE: is the sign of the error stable across seeds? ---")
    print(f"{'seed':>5} {'all':>9} {'err':>8}")
    errs = []
    for seed in (1, 2, 3, 4, 5, 6):
        a, _, _, _ = measure(square(6400, seed), k=10, max_radius=6)
        errs.append(a - TRUTH)
        print(f"{seed:>5} {a:>9.4f} {a - TRUTH:>+8.4f}")
    same_sign = all(e > 0 for e in errs) or all(e < 0 for e in errs)
    print(f"\n  all six seeds share the sign of the error: {same_sign}")
    print(f"  mean err {st.mean(errs):+.4f}   sd {st.stdev(errs):.4f}")
    print("\n  -> a stable sign with sd much smaller than |mean| means BIAS, not variance.")


if __name__ == "__main__":
    main()
