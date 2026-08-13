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

Theiler window (H1, round 3)
----------------------------
Theiler, J. (1986), "Spurious dimension from correlation algorithms applied
to limited time-series data", Physical Review A 34(3), 2427-2432, showed
that when the points are samples of a *trajectory*, pairs that are close in
TIME are close in space for a trivial reason -- the trajectory has not had
time to go anywhere -- and counting them inflates C(r) at small r. The
inflation is a roughly constant additive offset in C(r), which flattens the
log-log curve at the small-r end and therefore biases D2 DOWNWARD. The
standard fix is a Theiler window W: drop every pair whose original
trajectory time-index difference is < W from both the numerator and the
denominator of the correlation sum.

This is a FAIRNESS CORRECTION TO THE BASELINE, not a trick to help the
shell-growth estimator. Applying it generally makes the traditional method
*more* accurate on trajectory data, i.e. a HARDER baseline to beat. Under
this repository's standing rule -- a superiority claim is only testable
against a real baseline, never a strawman -- that is the expected and
desired outcome, not a regression.

It is strictly opt-in. `theiler_window=0` (the default) takes the original
code path verbatim, so every previously recorded number reproduces exactly.
Pass an explicit integer, or `theiler_window="auto"` for the
autocorrelation-based heuristic in `theiler_window_from_autocorrelation`.

MEASURED CAVEAT, small n. The window is an asymptotic accuracy correction; it
is not a small-sample fix and can make small-n estimates markedly NOISIER.
Measured on a Lorenz trajectory (dt=0.005) by
scripts/hypergraph_benchmark/round3/n8f_theiler_window_h1.py: at n=12800 the
window moves the estimate from 1.943 to 2.010 against a literature D2 of 2.05,
but at n=200-400 it moves it from ~1.72-1.81 to ~2.8-3.1. The reason is not
the window mis-firing: a few-hundred-point prefix of a finely sampled flow is
a short ARC, so once the temporally-local pairs are gone there is very little
genuine structure left at the small-r end. The uncorrected ~1.75 there was not
a better estimate, it was the autocorrelation artifact pinning the slope. Both
are unreliable at that n; only the corrected one is honest about it. Anything
scoring "points needed to converge" must therefore re-derive that number under
the corrected baseline rather than reuse the uncorrected one.

Time indices
------------
"Close in time" means close in ORIGINAL TRAJECTORY TIME INDEX, which is not
the same as close in array position once a caller has strided, resampled,
sorted, or otherwise reordered the cloud. Two ways to say which is which:

* Pass `time_indices` explicitly -- one integer per point, its index in the
  original trajectory. This survives any reordering the caller applies and
  is the correct choice after resampling/reordering.
* Omit it, in which case the DOCUMENTED ASSUMPTION is that `points` is in
  original trajectory order with uniform sampling, so point `i` has time
  index `i`. Any reordering must then be applied by the caller *after* this
  function runs.

The implementation sorts by time index internally, so `time_indices` need
not be sorted and the result is invariant to how the caller permuted the
array (verified by test).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# Largest number of (i, j) pair distances materialised at once when counting
# time-local pairs. Bounds peak memory independently of n and W.
_PAIR_CHUNK = 1_000_000


@dataclass(frozen=True)
class CorrelationDimensionEstimate:
    """Result of a classical Grassberger-Procaccia correlation-dimension fit."""

    radii: tuple[float, ...]
    correlation_sums: tuple[float, ...]
    dimension: float
    r_squared: float
    n_points: int
    n_pair_counts: int  # number of cKDTree.count_neighbors calls -- the cost unit
    # Theiler-window provenance. Defaults reproduce the pre-H1 record exactly:
    # window 0 means the correction was not applied and no pairs were dropped.
    theiler_window: int = 0
    n_excluded_pairs: int = 0

    def is_well_fit(self, threshold: float = 0.9) -> bool:
        return self.r_squared >= threshold


def correlation_dimension(
    points: list[tuple[float, ...]],
    *,
    n_radii: int = 20,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.2,
    theiler_window: int | str = 0,
    time_indices: Sequence[int] | None = None,
) -> CorrelationDimensionEstimate:
    """Classical correlation dimension via the Grassberger-Procaccia correlation integral.

    Radii are log-spaced between `r_min_frac` and `r_max_frac` of the point
    cloud's bounding-box diagonal (a standard choice: too small and C(r) is
    dominated by discreteness noise; too large and it saturates toward 1).
    The fit uses only radii where 0 < C(r) < 1 (points at either boundary
    contribute no information to a log-log slope and are excluded, not
    zero-padded).

    Parameters
    ----------
    theiler_window:
        Theiler (1986) exclusion window, in units of ORIGINAL TRAJECTORY TIME
        INDEX difference. Every pair with ``|t_i - t_j| < W`` is removed from
        both the numerator and the denominator of C(r).

        * ``0`` (default) -- correction OFF. Takes the original code path
          verbatim; results are bit-for-bit identical to the pre-H1 estimator.
          This is deliberate: the window is a new capability, never a silent
          behaviour change to numbers already on the record.
        * ``1`` -- excludes only ``|t_i - t_j| < 1``, i.e. nothing beyond the
          self-pairs already excluded. Numerically identical to ``0``, and is
          the conventional "no window" value in the literature.
        * any ``W >= 2`` -- the correction, with that window.
        * ``"auto"`` -- use `theiler_window_from_autocorrelation` on this
          cloud. Opt-in only; never reached unless the caller asks for it.
    time_indices:
        One integer per point: its index in the ORIGINAL trajectory. Required
        for correctness whenever `points` has been reordered/resampled, since
        array position is then no longer time. If omitted, point `i` is
        assumed to have time index `i` -- i.e. `points` is in original
        trajectory order and any reordering is the caller's to apply *after*
        this call. Unused when the window is off, but still validated there:
        a malformed `time_indices` is a caller bug worth reporting whether or
        not this particular call happens to consume it.
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

    window = _resolve_theiler_window(theiler_window, arr, time_indices)
    times = _resolve_time_indices(time_indices, n)

    total_pairs = n * (n - 1) / 2
    correlation_sums: list[float] = []

    if window <= 1:
        # Pre-H1 path, preserved verbatim so recorded numbers reproduce exactly.
        for r in radii:
            # count_neighbors(self, r) counts ordered pairs (i, j) with i != j at
            # distance < r, PLUS each point counted against itself at distance 0;
            # subtract n self-pairs and halve to match the standard i<j convention.
            count = tree.count_neighbors(tree, r) - n
            correlation_sums.append(count / 2 / total_pairs)
        excluded_pairs = 0
    else:
        excluded_pairs, excluded_per_radius = _time_local_pair_counts(arr, times, window, radii)
        kept_pairs = total_pairs - excluded_pairs
        if kept_pairs <= 0:
            raise ValueError(
                f"theiler_window={window} excludes all {int(total_pairs)} pairs of the "
                f"{n}-point cloud; no correlation sum is left to fit"
            )
        for r, drop in zip(radii, excluded_per_radius, strict=True):
            count = tree.count_neighbors(tree, r) - n
            # `drop` is recomputed with numpy's Euclidean norm rather than read
            # back out of the tree, so a last-ULP disagreement on a pair sitting
            # exactly at r could in principle overshoot. Clamp at 0 rather than
            # emit a negative count.
            kept = max(count / 2 - float(drop), 0.0)
            correlation_sums.append(kept / kept_pairs)

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
            theiler_window=window,
            n_excluded_pairs=excluded_pairs,
        )

    slope, r_squared = _log_log_fit(log_r, log_c)
    return CorrelationDimensionEstimate(
        tuple(radii),
        tuple(correlation_sums),
        dimension=slope,
        r_squared=r_squared,
        n_points=n,
        n_pair_counts=n_radii,
        theiler_window=window,
        n_excluded_pairs=excluded_pairs,
    )


def _resolve_time_indices(time_indices: Sequence[int] | None, n: int) -> np.ndarray:
    """Validate `time_indices`, or fall back to the documented `t_i = i` assumption."""
    if time_indices is None:
        return np.arange(n, dtype=np.int64)
    times = np.asarray(time_indices)
    if times.ndim != 1 or len(times) != n:
        raise ValueError(
            f"time_indices must be one integer per point: got {times.shape} for {n} points"
        )
    if not np.issubdtype(times.dtype, np.integer):
        rounded = np.rint(times)
        if not np.allclose(times, rounded):
            raise ValueError(
                "time_indices must be integer trajectory indices, not fractional times"
            )
        times = rounded
    return times.astype(np.int64, copy=False)


def _resolve_theiler_window(
    theiler_window: int | str,
    arr: np.ndarray,
    time_indices: Sequence[int] | None,
) -> int:
    """Turn the `theiler_window` argument into a concrete non-negative window."""
    if isinstance(theiler_window, str):
        if theiler_window != "auto":
            raise ValueError(
                f"theiler_window must be a non-negative int or 'auto', got {theiler_window!r}"
            )
        return theiler_window_from_autocorrelation(arr, time_indices=time_indices)
    if isinstance(theiler_window, bool) or not isinstance(theiler_window, int):
        raise ValueError(
            f"theiler_window must be a non-negative int or 'auto', got {theiler_window!r}"
        )
    if theiler_window < 0:
        raise ValueError(f"theiler_window must be non-negative, got {theiler_window}")
    return theiler_window


def theiler_window_from_autocorrelation(
    points: Sequence[tuple[float, ...]] | np.ndarray,
    *,
    time_indices: Sequence[int] | None = None,
    max_window: int | None = None,
) -> int:
    """Default Theiler-window heuristic: the first lag whose autocorrelation is <= 1/e.

    CHOICE AND REASONING. Theiler's own prescription is "longer than the
    correlation time of the trajectory", which is a property of the data, not
    a constant -- so a fixed literal default (W=10, W=50, ...) would be
    arbitrary and would silently be wrong whenever the sampling rate changed.
    This heuristic instead reads the correlation time off the trajectory:

      1. Center each coordinate and form the variance-pooled autocorrelation
         rho(L) = sum_dims cov_dim(L) / sum_dims var_dim, the standard
         multivariate generalisation. Pooling by variance (rather than
         averaging per-coordinate correlations) keeps a nearly-constant
         coordinate from dominating a well-mixing one.
      2. Take W = the first lag L >= 1 with rho(L) <= 1/e. The 1/e crossing is
         the conventional decorrelation time, the same statistic used to pick
         an embedding delay, and it is exact for an AR(1)/Ornstein-Uhlenbeck
         process, which is the local model Theiler's argument assumes.
      3. Clamp to [1, max_window], default max_window = min(n // 10, 250).
         The n // 10 arm keeps the window from eating a large fraction of the
         pairs (at W = n/10 roughly 10% of pairs are already discarded, and
         the variance of the surviving correlation sum starts to matter more
         than the bias being removed); the 250 arm bounds the O(n * W) cost.

    Returns max_window if rho never crosses 1/e within that range -- e.g. a
    strictly periodic orbit sampled at high rate, where no finite lag is truly
    decorrelated and the clamp is the honest answer.

    Sampling assumption: lag is measured in positions of the time-sorted
    sequence. For a uniformly sampled trajectory (`time_indices` omitted, or a
    contiguous index range) position-lag and time-index difference coincide.
    With gaps, this returns a window in units of "samples", which is then
    applied as a time-index difference -- so pass an explicit integer instead
    if the trajectory is unevenly sampled and the distinction matters.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        arr = arr.reshape(len(arr), -1)
    n = len(arr)
    if n < 3:
        raise ValueError("need at least 3 points to estimate a decorrelation time")

    if time_indices is not None:
        order = np.argsort(_resolve_time_indices(time_indices, n), kind="stable")
        arr = arr[order]

    if max_window is None:
        max_window = min(n // 10, 250)
    max_window = max(1, int(max_window))

    centered = arr - arr.mean(axis=0)
    var = float(np.sum(centered * centered))
    if var <= 0:
        return 1  # constant trajectory: nothing to decorrelate

    threshold = math.exp(-1.0)
    for lag in range(1, max_window + 1):
        cov = float(np.sum(centered[:-lag] * centered[lag:]))
        if cov / var <= threshold:
            return lag
    return max_window


def _time_local_pair_counts(
    arr: np.ndarray,
    times: np.ndarray,
    window: int,
    radii: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Count pairs with |t_i - t_j| < window: total, and how many fall within each radius.

    Works in the time-sorted view, where "within `window` in time" is a
    contiguous forward run, so the pairs to drop can be enumerated with a
    sliding window instead of an O(n^2) scan. Cost is O(n log n + n * window),
    and peak memory is bounded by `_PAIR_CHUNK` pair distances regardless of n
    or window.

    `radii` must be ascending. Counting uses `d <= r`, matching
    `cKDTree.count_neighbors`, so the two counts are subtractable.
    """
    order = np.argsort(times, kind="stable")
    t_sorted = times[order]
    p_sorted = arr[order]
    n = len(t_sorted)

    # For each i, the first position whose time is >= t_i + window.
    hi = np.searchsorted(t_sorted, t_sorted + window, side="left")
    counts = np.maximum(hi - np.arange(n) - 1, 0).astype(np.int64)
    total_excluded = int(counts.sum())

    per_radius = np.zeros(len(radii), dtype=np.int64)
    if total_excluded == 0:
        return 0, per_radius

    cumulative = np.cumsum(counts)
    start = 0
    while start < n:
        consumed = int(cumulative[start - 1]) if start > 0 else 0
        stop = int(np.searchsorted(cumulative, consumed + _PAIR_CHUNK, side="left")) + 1
        stop = min(max(stop, start + 1), n)

        block = counts[start:stop]
        block_total = int(block.sum())
        if block_total:
            rows = np.repeat(np.arange(start, stop, dtype=np.int64), block)
            # Ragged forward offsets 1..block[i] for each row i.
            offsets = np.arange(block_total, dtype=np.int64) - np.repeat(
                np.cumsum(block) - block, block
            )
            cols = rows + 1 + offsets
            dist = np.linalg.norm(p_sorted[rows] - p_sorted[cols], axis=1)
            per_radius += np.searchsorted(np.sort(dist), radii, side="right")
        start = stop

    return total_excluded, per_radius


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
    theiler_window: int | str = 0,
    time_indices: Sequence[int] | None = None,
) -> int | None:
    """Smallest n in `n_grid` (a prefix of `points`) whose correlation-dimension
    estimate is within `tolerance` of `true_dimension` AND stays within tolerance
    at every larger n tested (guards against a lucky single crossing, the exact
    failure mode docs/POLY_ALGEBRAIC_BENCHMARK.md findings F4/F4b documented for
    the shell-growth estimator -- the traditional method needs the same
    convergence discipline applied to it, not a pass).

    Returns None if no n in the grid achieves stable convergence.

    `theiler_window` / `time_indices` are forwarded to `correlation_dimension`
    unchanged; `theiler_window=0` (the default) reproduces the pre-H1 result
    exactly. Note that each grid point uses the PREFIX `points[:n]`, so with
    the default `t_i = i` assumption the prefix's time indices are just
    `0..n-1` -- consistent at every n. An explicit `time_indices` is sliced to
    the same prefix. `"auto"` is re-evaluated per prefix, which is intended:
    the decorrelation time is a property of the sample actually being fitted.
    """
    results = []
    for n in n_grid:
        if n > len(points):
            break
        est = correlation_dimension(
            points[:n],
            n_radii=n_radii,
            theiler_window=theiler_window,
            time_indices=None if time_indices is None else time_indices[:n],
        )
        results.append((n, est.dimension))

    for i, (n, dim) in enumerate(results):
        if not math.isfinite(dim):
            continue
        if abs(dim - true_dimension) > tolerance:
            continue
        if all(math.isfinite(d) and abs(d - true_dimension) <= tolerance for _, d in results[i:]):
            return n
    return None
