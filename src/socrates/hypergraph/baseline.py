"""The classical (traditional) correlation-dimension estimator, for comparison only.

This module is NOT part of the "Poly-Algebraic Calculus" toolkit -- it is
the honest baseline everything in `dimension.py`/`pointcloud.py` gets
compared against. Without a real traditional-method implementation,
"better than traditional" has no concrete referent; this module exists so
that claim can be checked rather than asserted.

The method is the original Grassberger & Procaccia (1983) correlation
integral:

    C(r) = (2 / (N(N-1))) * #{(i,j) : i<j, |x_i - x_j| < r}
    D2 = d log C(r) / d log r   (slope in the scaling region)

Built on the same `scipy.spatial.cKDTree` primitive as `pointcloud.py`
(`cKDTree.count_neighbors`, which is a well-known O(n log n)-ish pair-count,
not the naive O(n^2) most textbook descriptions imply) so that any
compute-cost comparison against the shell-growth estimator isolates the
*algorithm* difference (global pairwise scaling vs local graph growth), not
an implementation-quality difference between a hand-rolled O(n^2) loop and
an optimized library call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CorrelationDimensionEstimate:
    """Result of a classical Grassberger-Procaccia correlation-dimension fit."""

    radii: tuple[float, ...]
    correlation_sums: tuple[float, ...]
    dimension: float
    r_squared: float
    n_points: int
    n_pair_counts: int  # number of cKDTree.count_neighbors calls -- the cost unit

    def is_well_fit(self, threshold: float = 0.9) -> bool:
        return self.r_squared >= threshold


def correlation_dimension(
    points: list[tuple[float, ...]],
    *,
    n_radii: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.2,
) -> CorrelationDimensionEstimate:
    """Classical correlation dimension via the Grassberger-Procaccia correlation integral.

    Radii are log-spaced between `r_min_frac` and `r_max_frac` of the point
    cloud's bounding-box diagonal (a standard choice: too small and C(r) is
    dominated by discreteness noise; too large and it saturates toward 1).
    The fit uses only radii where 0 < C(r) < 1 (points at either boundary
    contribute no information to a log-log slope and are excluded, not
    zero-padded).
    """
    arr = np.asarray(points, dtype=float)
    n = len(arr)
    if n < 3:
        raise ValueError("need at least 3 points for a correlation-sum estimate")

    tree = cKDTree(arr)
    diag = float(np.sqrt(np.sum((arr.max(axis=0) - arr.min(axis=0)) ** 2)))
    if diag == 0:
        raise ValueError("all points are identical; correlation dimension is undefined")

    radii = np.logspace(math.log10(r_min_frac * diag), math.log10(r_max_frac * diag), n_radii)

    total_pairs = n * (n - 1) / 2
    correlation_sums = []
    for r in radii:
        # count_neighbors(self, r) counts ordered pairs (i, j) with i != j at
        # distance < r, PLUS each point counted against itself at distance 0;
        # subtract n self-pairs and halve to match the standard i<j convention.
        count = tree.count_neighbors(tree, r) - n
        correlation_sums.append(count / 2 / total_pairs)

    log_r, log_c = [], []
    for r, c in zip(radii, correlation_sums, strict=True):
        if 0 < c < 1:
            log_r.append(math.log(r))
            log_c.append(math.log(c))

    if len(log_r) < 2:
        return CorrelationDimensionEstimate(
            tuple(radii),
            tuple(correlation_sums),
            dimension=float("nan"),
            r_squared=0.0,
            n_points=n,
            n_pair_counts=n_radii,
        )

    slope, r_squared = _log_log_fit(log_r, log_c)
    return CorrelationDimensionEstimate(
        tuple(radii),
        tuple(correlation_sums),
        dimension=slope,
        r_squared=r_squared,
        n_points=n,
        n_pair_counts=n_radii,
    )


def _log_log_fit(log_x: list[float], log_y: list[float]) -> tuple[float, float]:
    """Least-squares slope and R^2, given logs already taken (inputs may be negative)."""
    n = len(log_x)
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in log_x)
    if var_x == 0:
        return 0.0, 0.0
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in log_y)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_x, log_y, strict=True))
    scale = max(1.0, sum(y * y for y in log_y))
    r_squared = 1.0 if ss_tot <= 1e-24 * scale else 1.0 - ss_res / ss_tot
    return slope, r_squared


def minimum_points_for_target_accuracy(
    points: list[tuple[float, ...]],
    true_dimension: float,
    tolerance: float,
    *,
    n_grid: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200),
    n_radii: int = 20,
) -> int | None:
    """Smallest n in `n_grid` (a prefix of `points`) whose correlation-dimension
    estimate is within `tolerance` of `true_dimension` AND stays within tolerance
    at every larger n tested (guards against a lucky single crossing, the exact
    failure mode docs/POLY_ALGEBRAIC_BENCHMARK.md findings F4/F4b documented for
    the shell-growth estimator -- the traditional method needs the same
    convergence discipline applied to it, not a pass).

    Returns None if no n in the grid achieves stable convergence.
    """
    results = []
    for n in n_grid:
        if n > len(points):
            break
        est = correlation_dimension(points[:n], n_radii=n_radii)
        results.append((n, est.dimension))

    for i, (n, dim) in enumerate(results):
        if not math.isfinite(dim):
            continue
        if abs(dim - true_dimension) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dimension) <= tolerance for _, d in results[i:]):
            return n
    return None
