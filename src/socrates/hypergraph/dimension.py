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

from .core import Hypergraph, Node, ball


@dataclass(frozen=True)
class DimensionEstimate:
    """Result of a log-log fit of ball volume vs radius."""

    source: Node
    radii: tuple[int, ...]
    volumes: tuple[int, ...]
    dimension: float
    r_squared: float

    def is_well_fit(self, threshold: float = 0.95) -> bool:
        return self.r_squared >= threshold


def _log_log_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and R^2 of log(ys) vs log(xs)."""
    n = len(xs)
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
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
    # ss_tot is exactly 0 for a perfectly constant y-sequence in exact arithmetic,
    # but floating-point rounding in mean_y (sum(log_y) / n) can leave a tiny
    # nonzero residual -- observed as small as ~1e-31 for some fit lengths/values
    # -- that makes a literal `ss_tot > 0` check take the general branch and
    # divide by that residual, collapsing r_squared to ~0 by pure numerical noise
    # even though the underlying fit is exact. A relative-scale tolerance avoids
    # treating that noise as a real residual.
    scale = max(1.0, sum(y * y for y in log_y))
    r_squared = 1.0 if ss_tot <= 1e-24 * scale else 1.0 - ss_res / ss_tot
    return slope, r_squared


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
    """
    radii = list(range(0, max_radius + 1))
    volumes = [len(ball(hg, source, r)) for r in radii]
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
            source, tuple(radii[1:]), tuple(volumes[1:]), dimension=0.0, r_squared=0.0
        )

    slope, r_squared = _log_log_fit([float(r) for r in fit_radii], [float(s) for s in fit_shells])
    return DimensionEstimate(
        source, tuple(radii[1:]), tuple(volumes[1:]), dimension=slope + 1.0, r_squared=r_squared
    )


def mean_dimension(hg: Hypergraph, *, samples: int | None = None, max_radius: int = 6) -> float:
    """Average local dimension over (a sample of) nodes, ignoring poor fits."""
    nodes = sorted(hg.nodes)
    if samples is not None and samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    estimates = [local_dimension(hg, n, max_radius=max_radius) for n in nodes]
    well_fit = [e.dimension for e in estimates if e.is_well_fit(threshold=0.9)]
    if not well_fit:
        return float("nan")
    return sum(well_fit) / len(well_fit)


def dimension_profile(
    history: list[Hypergraph], *, samples: int = 8, max_radius: int = 5
) -> list[float]:
    """Mean estimated dimension at each step of an evolution history."""
    return [mean_dimension(hg, samples=samples, max_radius=max_radius) for hg in history]
