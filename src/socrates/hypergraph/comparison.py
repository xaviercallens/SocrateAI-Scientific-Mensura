"""Apples-to-apples comparison: Poly-Algebraic Calculus vs the traditional baseline.

Tier B. Answers one specific, checkable question per problem: at how many
points does each method's dimension estimate first become, and then STAY,
within `tolerance` of the known/traditional dimension? Fewer points needed
is a genuine compute-cost claim (both methods build a k-NN or pair-count
structure in O(n log n) via the same `scipy.spatial.cKDTree`, so this
isolates sample efficiency, not implementation quality). Point-count parity
is enforced: both methods are evaluated on identical prefixes of the same
point cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .baseline import correlation_dimension, minimum_points_for_target_accuracy
from .dimension import mean_dimension
from .pointcloud import DuplicatePointsError, knn_hypergraph


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing the two methods on one point cloud."""

    true_dimension: float
    tolerance: float
    poly_algebraic_min_n: int | None
    traditional_min_n: int | None
    poly_algebraic_estimate_at_max_n: float
    traditional_estimate_at_max_n: float
    max_n_tested: int

    @property
    def poly_algebraic_wins(self) -> bool:
        """Strictly fewer points needed, with BOTH methods actually converging.

        A method that never stably converges in the tested grid cannot be
        said to "win" on efficiency -- that would reward failure to
        converge as if it were failure to need many points. See
        `compute_savings_fraction` for the case both converge.
        """
        if self.poly_algebraic_min_n is None or self.traditional_min_n is None:
            return False
        return self.poly_algebraic_min_n < self.traditional_min_n

    @property
    def compute_savings_fraction(self) -> float | None:
        """Fraction of points saved by the poly-algebraic method, if both converged.

        None if either method never converged in the tested grid (a savings
        percentage is not meaningful when the denominator is undefined).
        """
        if self.poly_algebraic_min_n is None or self.traditional_min_n is None:
            return None
        return 1.0 - (self.poly_algebraic_min_n / self.traditional_min_n)


def poly_algebraic_minimum_points(
    points: list[tuple[float, ...]],
    true_dimension: float,
    tolerance: float,
    *,
    k: int,
    n_grid: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200),
    max_radius: int = 6,
    samples: int = 40,
) -> int | None:
    """Analogous to `baseline.minimum_points_for_target_accuracy`, for the shell-growth method.

    Requires stable convergence (within tolerance at every larger n tested
    in the grid, not just a single lucky crossing), applying the same
    discipline docs/POLY_ALGEBRAIC_BENCHMARK.md findings F4/F4b demanded.
    Duplicate point clusters are auto-deduplicated (`dedupe=True`) rather
    than raising, since this function is meant to run unattended over many
    (n, k) combinations.
    """
    results = []
    for n in n_grid:
        if n > len(points) or k >= n:
            break
        subset = points[:n]
        try:
            hg = knn_hypergraph(subset, k=k, dedupe=True)
        except DuplicatePointsError:
            continue
        dim = mean_dimension(hg, samples=min(n, samples), max_radius=max_radius)
        results.append((n, dim))

    for i, (_, dim) in enumerate(results):
        if not math.isfinite(dim):
            continue
        if abs(dim - true_dimension) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dimension) <= tolerance for _, d in results[i:]):
            return results[i][0]
    return None


def compare(
    points: list[tuple[float, ...]],
    true_dimension: float,
    tolerance: float,
    *,
    k: int,
    n_grid: tuple[int, ...] = (100, 200, 400, 800, 1600, 3200),
    max_radius: int = 6,
) -> ComparisonResult:
    """Run both methods on the same point cloud and report the comparison."""
    max_n = max(n for n in n_grid if n <= len(points))

    poly_n = poly_algebraic_minimum_points(
        points, true_dimension, tolerance, k=k, n_grid=n_grid, max_radius=max_radius
    )
    trad_n = minimum_points_for_target_accuracy(points, true_dimension, tolerance, n_grid=n_grid)

    try:
        hg_final = knn_hypergraph(points[:max_n], k=k, dedupe=True)
        poly_final = mean_dimension(hg_final, samples=40, max_radius=max_radius)
    except (DuplicatePointsError, ValueError):
        poly_final = float("nan")

    trad_final = correlation_dimension(points[:max_n]).dimension

    return ComparisonResult(
        true_dimension=true_dimension,
        tolerance=tolerance,
        poly_algebraic_min_n=poly_n,
        traditional_min_n=trad_n,
        poly_algebraic_estimate_at_max_n=poly_final,
        traditional_estimate_at_max_n=trad_final,
        max_n_tested=max_n,
    )
