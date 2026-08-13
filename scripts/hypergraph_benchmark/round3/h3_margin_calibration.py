"""H3 / criterion (c) calibration: DEFAULT_ACCURACY_MARGIN and
DEFAULT_MAX_SENTINEL_FRACTION.

This script is the measurement behind the two module constants in
`socrates.hypergraph.comparison`. It is cited from their calibration comments;
if either constant is changed, re-run this and update the comment with the new
table rather than editing the number alone.

Two questions, two tables.

TABLE 1 -- how noisy is the accuracy gap?
    The margin gates `traditional_abs_error - poly_algebraic_abs_error`. If the
    margin sits below that quantity's own run-to-run spread, criterion (c)
    scores resampling luck. We redraw each point cloud `N_REPEATS` times at
    fixed n / k / max_radius (reseeding the stochastic clouds, re-phasing the
    deterministic one -- a deterministic cloud has no seed, so the honest
    analogue of a redraw is a different window of the same trajectory) and
    report each method's standard deviation and the root-sum-square gap noise.

TABLE 2 -- where does the sentinel fraction actually sit?
    The F1 guard must accept every non-degenerate cloud and reject the ring
    lattice. This measures the must-accept and must-reject sides directly, so
    the threshold is bracketed by data rather than asserted.

Standing rule: this is a calibration script, not a result. Nothing here is a
benchmark win; it only fixes two thresholds used by one.
"""

from __future__ import annotations

import math
import statistics
import sys
from collections.abc import Callable

import numpy as np

from socrates.hypergraph.baseline import correlation_dimension
from socrates.hypergraph.comparison import (
    DEFAULT_ACCURACY_MARGIN,
    DEFAULT_MAX_SENTINEL_FRACTION,
)
from socrates.hypergraph.dimension import (
    degenerate_fraction,
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph

N_REPEATS = 10
SAMPLES = 40
MAX_RADIUS = 6  # comparison's own default

OMEGA2 = math.sqrt(2.0)
TORUS_DENSITY = 500 / 1800.0  # round-2 problem 05's own time-density

Cloud = Callable[[int, int], list[tuple[float, ...]]]


def brownian(n: int, seed: int) -> list[tuple[float, ...]]:
    """Planar Brownian motion, subsampled by stride 4 (round-2 problem 06's shape)."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 1.0, size=(n * 4, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)])
    return [tuple(p) for p in path[::4][:n]]


def uniform_square(n: int, seed: int) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(0.0, 1.0, size=(n, 2))]


def uniform_cube(n: int, seed: int) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(0.0, 1.0, size=(n, 3))]


def quasiperiodic_torus(n: int, seed: int) -> list[tuple[float, ...]]:
    """(cos t, cos(sqrt(2) t)) at constant time-density.

    Deterministic, so `seed` shifts the start time instead of reseeding: a
    different window of the same non-repeating trajectory is the honest
    analogue of a redraw here.
    """
    t_max = n / TORUS_DENSITY
    t0 = 13.7 * seed
    return [
        (math.cos(t0 + t_max * i / n), math.cos(OMEGA2 * (t0 + t_max * i / n))) for i in range(n)
    ]


def circle(n: int, seed: int) -> list[tuple[float, ...]]:
    """Exact circulant ring lattice under k-NN -- the F1 degeneracy, must reject."""
    phase = 0.017 * seed
    return [
        (math.cos(phase + 2 * math.pi * i / n), math.sin(phase + 2 * math.pi * i / n))
        for i in range(n)
    ]


def straight_line(n: int, seed: int) -> list[tuple[float, ...]]:
    return [(i / n, 0.0) for i in range(n)]


# (name, generator, n, k, true dimension, sentinel side)
CONFIGS: list[tuple[str, Cloud, int, int, float, str]] = [
    ("Brownian 2D", brownian, 1600, 10, 2.0, "must accept"),
    ("uniform square 2D", uniform_square, 800, 10, 2.0, "must accept"),
    ("quasiperiodic torus", quasiperiodic_torus, 800, 10, 2.0, "must accept"),
    ("uniform cube 3D", uniform_cube, 400, 12, 3.0, "must accept"),
    ("circle (ring lattice)", circle, 400, 6, 1.0, "MUST REJECT"),
    ("straight line", straight_line, 400, 6, 1.0, "MUST REJECT"),
]


def main() -> int:
    rows = []
    for name, gen, n, k, true_dim, side in CONFIGS:
        polys, trads, sentinels = [], [], []
        for seed in range(N_REPEATS):
            points = gen(n, seed)
            hg = knn_hypergraph(points, k=k, dedupe=True)
            polys.append(mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS))
            trads.append(correlation_dimension(points).dimension)
            sentinels.append(
                degenerate_fraction(hg, samples=SAMPLES, max_radius=MAX_RADIUS)
                + near_degenerate_fraction(hg, samples=SAMPLES, max_radius=MAX_RADIUS)
            )
        rows.append((name, n, k, true_dim, side, polys, trads, sentinels))

    print(f"TABLE 1 -- redraw noise at fixed n (N_REPEATS = {N_REPEATS})")
    print(f"{'cloud':24s} {'n':>5s} {'k':>3s}  {'poly sd':>8s} {'trad sd':>8s} {'gap sd':>8s}")
    worst_gap_sd = 0.0
    for name, n, k, _true, _side, polys, trads, _sent in rows:
        poly_sd = statistics.pstdev(polys)
        trad_sd = statistics.pstdev(trads)
        gap_sd = math.hypot(poly_sd, trad_sd)
        worst_gap_sd = max(worst_gap_sd, gap_sd)
        print(f"{name:24s} {n:5d} {k:3d}  {poly_sd:8.4f} {trad_sd:8.4f} {gap_sd:8.4f}")
    print(
        f"\nworst gap sd = {worst_gap_sd:.4f}; DEFAULT_ACCURACY_MARGIN = "
        f"{DEFAULT_ACCURACY_MARGIN} = {DEFAULT_ACCURACY_MARGIN / worst_gap_sd:.2f} sd"
    )

    print("\nTABLE 2 -- sentinel fraction (degenerate + near-degenerate)")
    print(f"{'cloud':24s} {'side':>12s}  {'min':>7s} {'mean':>7s} {'max':>7s}")
    accept_max, reject_min = 0.0, 1.0
    for name, _n, _k, _true, side, _polys, _trads, sent in rows:
        print(
            f"{name:24s} {side:>12s}  {min(sent):7.3f} "
            f"{statistics.mean(sent):7.3f} {max(sent):7.3f}"
        )
        if side == "must accept":
            accept_max = max(accept_max, max(sent))
        else:
            reject_min = min(reject_min, min(sent))
    print(
        f"\nmust-accept max = {accept_max:.3f}; must-reject min = {reject_min:.3f}; "
        f"DEFAULT_MAX_SENTINEL_FRACTION = {DEFAULT_MAX_SENTINEL_FRACTION}"
    )

    ok = accept_max <= DEFAULT_MAX_SENTINEL_FRACTION < reject_min
    print(f"\nthreshold brackets both sides: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
