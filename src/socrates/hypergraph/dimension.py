"""Emergent dimension estimation via ball-volume growth (Pre-Geometric Dimension Variable).

Tier B: this is a well-defined statistic of a graph, not a new physical
claim. If a hypergraph's adjacency graph behaves like a d-dimensional
lattice at some scale, the number of nodes within radius r of a source grows
as |ball(r)| ~ r^d, so d is recoverable from the log-log slope. This is the
standard volume-growth dimension used in causal set theory and in Wolfram's
own work; nothing here is novel except the packaging.

The fit is done on *shell sizes* (dV(r) = |ball(r)| - |ball(r-1)|), not on
cumulative volume directly. This matters: cumulative volume on a lattice is
affine rather than homogeneous (a 1D path has |ball(r)| = 2r+1 from an
interior point, not r^1 -- the "+1" is a real offset, not noise), which
biases a naive log-log fit on volume itself well below the true dimension
at any radius small enough to be computationally tractable. The shell size
of a d-dimensional lattice scales as r^(d-1) with *no* additive offset (a
path's shell is the exact constant 2; a 2D grid's shell is exactly 4r), so
`dimension = shell_slope + 1` converges immediately rather than
asymptotically. Verified against both cases directly, not assumed.

The "dynamic"/"fluid" dimension idea (concept #4 in the brief) is realized
directly: `local_dimension` computes this per-source, so a hypergraph can
have different estimated dimension at different regions, and
`dimension_profile` tracks it over an evolution history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core import Hypergraph, Node


@dataclass(frozen=True)
class DimensionEstimate:
    """Result of a log-log fit of ball volume vs radius.

    `degenerate=True` means the shell sequence being fit was constant (to
    within floating-point noise) -- e.g. a k-NN graph on a smooth closed
    curve, which is an exact circulant ring lattice. In that regime
    `r_squared=1.0` is a *sentinel* meaning "the input had no variation to
    fit," not a measurement of fit quality on a diverse shell sequence. This
    distinction exists because a 10-problem physics benchmark
    (docs/POLY_ALGEBRAIC_BENCHMARK.md, finding F1/N2) found that 6 of 10
    "passes" were exactly this case, misread by the estimator's own API as
    a perfect fit -- `is_well_fit()` alone could not distinguish them.
    """

    source: Node
    radii: tuple[int, ...]
    volumes: tuple[int, ...]
    dimension: float
    r_squared: float
    degenerate: bool = False

    def is_well_fit(self, threshold: float = 0.95) -> bool:
        return self.r_squared >= threshold

    def is_genuinely_well_fit(self, threshold: float = 0.95) -> bool:
        """Well-fit AND not a degenerate constant-shell sentinel."""
        return self.is_well_fit(threshold) and not self.degenerate


def _log_log_fit(xs: list[float], ys: list[float]) -> tuple[float, float, bool]:
    """Least-squares slope, R^2, and a degeneracy flag for log(ys) vs log(xs)."""
    n = len(xs)
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
    mean_x = sum(log_x) / n
    mean_y = sum(log_y) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in log_x)
    if var_x == 0:
        return 0.0, 0.0, True
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in log_y)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_x, log_y, strict=True))
    # ss_tot is exactly 0 for a perfectly constant y-sequence in exact arithmetic,
    # but floating-point rounding in mean_y (sum(log_y) / n) can leave a tiny
    # nonzero residual -- observed as small as ~1e-31 for some fit lengths/values
    # -- that makes a literal `ss_tot > 0` check take the general branch and
    # divide by that residual, collapsing r_squared to ~0 by pure numerical noise
    # even though the underlying fit is exact. A relative-scale tolerance avoids
    # treating that noise as a real residual.
    scale = max(1.0, sum(y * y for y in log_y))
    degenerate = ss_tot <= 1e-24 * scale
    r_squared = 1.0 if degenerate else 1.0 - ss_res / ss_tot
    return slope, r_squared, degenerate


def local_dimension(
    hg: Hypergraph,
    source: Node,
    *,
    max_radius: int = 6,
    min_radius: int = 1,
) -> DimensionEstimate:
    """Estimate the volume-growth dimension around `source`.

    Fits log(shell size) vs log(radius) and reports `slope + 1` as the
    dimension (a d-dimensional lattice has shell size ~ r^(d-1)). Radii past
    saturation (the ball has stopped growing, e.g. the whole connected
    component has been reached, so the shell drops to zero) are excluded --
    a saturated shell has no well-defined log and would otherwise crash or
    silently bias the fit.

    Computes the whole radii=0..max_radius volume sequence in a single
    incremental BFS pass (each radius extends the previous frontier rather
    than recomputing `ball()` from the source each time) -- calling `ball()`
    once per radius, as an earlier version of this function did, redid all
    of radius r-1's work at every step, which compounded with the
    O(hypergraph) cost of an uncached `adjacency()` call into the dominant
    cost of a dimension-vs-n convergence sweep (see `core.py`'s
    `_adjacency_cached`, fixed alongside this).
    """
    if source not in hg.nodes:
        raise ValueError(f"node {source} is not present in this hypergraph")
    adj = hg.adjacency()
    volumes = [1]
    visited = {source}
    frontier = {source}
    for _ in range(max_radius):
        next_frontier: set[Node] = set()
        for u in frontier:
            next_frontier |= adj.get(u, set()) - visited
        visited |= next_frontier
        volumes.append(len(visited))
        if not next_frontier:
            volumes.extend([len(visited)] * (max_radius - len(volumes) + 1))
            break
        frontier = next_frontier

    radii = list(range(0, max_radius + 1))
    # shells[i - 1] is the shell size (new nodes) at radius i.
    shells = [volumes[i] - volumes[i - 1] for i in range(1, len(volumes))]

    fit_radii, fit_shells = [], []
    for r, shell in zip(radii[1:], shells, strict=True):
        if r < min_radius:
            continue
        if shell <= 0:
            break  # saturated; stop here and beyond
        fit_radii.append(r)
        fit_shells.append(shell)

    if len(fit_radii) < 2:
        return DimensionEstimate(
            source,
            tuple(radii[1:]),
            tuple(volumes[1:]),
            dimension=0.0,
            r_squared=0.0,
            degenerate=False,
        )

    slope, r_squared, degenerate = _log_log_fit(
        [float(r) for r in fit_radii], [float(s) for s in fit_shells]
    )
    return DimensionEstimate(
        source,
        tuple(radii[1:]),
        tuple(volumes[1:]),
        dimension=slope + 1.0,
        r_squared=r_squared,
        degenerate=degenerate,
    )


def mean_dimension(hg: Hypergraph, *, samples: int | None = None, max_radius: int = 6) -> float:
    """Average local dimension over (a sample of) nodes, ignoring poor fits.

    Includes degenerate (constant-shell) fits in the average -- their
    dimension value is still correct, only their r_squared is a sentinel
    (see `DimensionEstimate.degenerate`). Callers who need to distinguish a
    genuine measurement from a degenerate one (e.g. before citing accuracy,
    per docs/POLY_ALGEBRAIC_BENCHMARK.md finding F1) should call
    `local_dimension` directly and check `.degenerate` themselves; this
    convenience wrapper answers "what does the estimator say", not "is that
    answer informative."
    """
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    estimates = [local_dimension(hg, n, max_radius=max_radius) for n in nodes]
    well_fit = [e.dimension for e in estimates if e.is_well_fit(threshold=0.9)]
    if not well_fit:
        return float("nan")
    return sum(well_fit) / len(well_fit)


def degenerate_fraction(
    hg: Hypergraph, *, samples: int | None = None, max_radius: int = 6
) -> float:
    """Fraction of sampled nodes whose fit was degenerate (constant shell sequence).

    A high value (e.g. all of `exact_periodic` in the physics benchmark) is
    itself informative: it means the r_squared column for this hypergraph is
    largely sentinel-valued, and `mean_dimension`'s apparent "perfect fit"
    should not be cited as evidence of estimator accuracy (finding F1).
    """
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    if not nodes:
        return float("nan")
    estimates = [local_dimension(hg, n, max_radius=max_radius) for n in nodes]
    return sum(1 for e in estimates if e.degenerate) / len(estimates)


def dimension_profile(
    history: list[Hypergraph], *, samples: int = 8, max_radius: int = 5
) -> list[float]:
    """Mean estimated dimension at each step of an evolution history."""
    return [mean_dimension(hg, samples=samples, max_radius=max_radius) for hg in history]
