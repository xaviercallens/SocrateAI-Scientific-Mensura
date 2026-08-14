"""Bridge from a continuous point cloud (e.g. a physical trajectory) to a Hypergraph.

Tier B. This is the standard k-nearest-neighbour proximity-graph
construction used throughout manifold learning and correlation-dimension
estimation (Grassberger & Procaccia 1983; the same idea underlies
`socrates.tda`'s Vietoris-Rips complexes) -- nothing novel about the
construction itself. What is new here is only that it feeds into this
module's `local_dimension`, letting the *same* shell-growth estimator
validated on discrete lattices (`dimension.py`) be applied to continuous
trajectories, so a physics benchmark can compare it against literature
values for correlation dimension.

Graph distance on a sufficiently dense k-NN graph approximates geodesic
distance on the underlying manifold (this is the standard justification for
Isomap-style dimension/embedding estimates); it is *not* exact, and is
sensitive to k and to sample density -- see the known-answer tests in
`tests/test_hypergraph.py` for the regime where it actually works.

Built on `scipy.spatial.cKDTree` (O(n log n) construction, O(log n) query)
rather than the brute-force O(n^2) pairwise search this module started with
-- a 10-problem physics benchmark (docs/MENSURA_BENCHMARK.md, finding
N5) found the O(n^2) cost was the binding constraint on point count for the
two fractal (chaotic-attractor) cases, whose dimension estimates were still
visibly unconverged at the largest n the O(n^2) cost made affordable.

Theiler window on the GRAPH side (AR1, round 3)
-----------------------------------------------
Theiler, J. (1986), Physical Review A 34(3), 2427-2432, made his argument
about the correlation sum: on trajectory data, pairs that are close in TIME
are close in space for a trivial reason (the trajectory has not had time to
go anywhere), so counting them measures the sampling rate rather than the
geometry. `baseline.correlation_dimension` already carries the fix for the
correlation sum (see its `theiler_window`).

The SAME defect afflicts a k-nearest-neighbour graph built on a trajectory,
and more severely, because k-NN is a *ranking* rather than a count: when the
along-trajectory spacing is much smaller than the transverse spacing between
successive passes of the attractor, a point's k nearest neighbours are all
its own temporal neighbours, and the "proximity graph" is a 1-dimensional
chain no matter what the underlying set is. Measured directly
(scripts/hypergraph_benchmark/round3/ar1_theiler_knn.py): on the round-2
Lorenz cloud at n=100 **100%** of k-NN edges join array-index-adjacent
points, still 57% at n=400; on Rossler 99% at n=100 and 83% at n=400. Over
exactly that range the shell-growth estimator reports 1.01-1.17 on a
2.05-dimensional attractor -- it is measuring the trajectory, not the
attractor.

`theiler_window` restricts each point's candidate neighbours to those with
``|t_i - t_j| >= W``, so the k retained neighbours are the k nearest points
the trajectory reached at a genuinely different time.

STRICTLY OPT-IN, AND THAT IS A MEASURED DECISION, NOT CAUTION. The default
``theiler_window=0`` takes the original code path verbatim, so every recorded
number reproduces bit-for-bit. Turning it on unconditionally is NOT safe:
where the temporal neighbours are the *only* genuine spatial neighbours -- a
1-D closed orbit sampled once, which is problems 03/04/07 of the benchmark
and two of its four verified wins -- the exclusion replaces local geometry
with long-range chords and destroys the estimate outright (measured: those
three problems go from `poly_algebraic_min_n` 100 to never converging, with
the median k-th-neighbour distance inflated 4x at n=100 rising to 84x at
n=6400). The discriminating statistic is exactly that inflation factor, and
on the ten benchmark clouds the must-fire and must-not-fire sides are only
1.55x and 2.39x -- too narrow a bracket to promote into an automatic rule on
this evidence, so the caller decides and the numbers stay reproducible. See
`ar1_theiler_knn.py` for the full table.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .baseline import theiler_window_from_autocorrelation
from .core import Hypergraph


class DuplicatePointsError(ValueError):
    """Raised when `knn_hypergraph` finds near-duplicate points in the input.

    Fed by finding F3 of docs/MENSURA_BENCHMARK.md: sampling multiple
    periods of a closed orbit (or any repeated traversal of the same curve)
    produces near-exact duplicate points at ~1e-9 separation -- typically in
    clusters of size equal to the number of periods sampled, not just pairs.
    A k-NN graph then wires each point to its own duplicate cluster instead
    of its along-curve neighbours, corrupting the shell-growth structure the
    dimension estimator depends on -- silently, in three of the ten
    benchmark problems, with severity ranging from a degraded well-fit rate
    to a complete `nan`. This is deliberately a hard error, not a warning:
    the benchmark showed the corruption is easy to miss by inspection.
    """


def knn_hypergraph(
    points: list[tuple[float, ...]],
    k: int,
    *,
    dedupe: bool = False,
    duplicate_tolerance: float = 1e-6,
    theiler_window: int | str = 0,
    time_indices: Sequence[int] | None = None,
) -> Hypergraph:
    """Build an undirected k-nearest-neighbour proximity graph over `points`.

    Each point becomes a node (indexed by its position in `points`, or in
    the deduplicated list if `dedupe=True` and duplicates were found); an
    edge (i, j) is added whenever j is among i's k nearest neighbours OR i
    is among j's (the standard symmetrization of a directed k-NN graph,
    needed because "nearest neighbour of" is not a symmetric relation).

    `duplicate_tolerance` is a *relative* threshold against the point
    cloud's own bounding-box diagonal (not the local nearest-neighbour
    spacing -- that scale is itself corrupted by the duplicates being
    detected, since a duplicated point's "nearest neighbour" becomes its
    own duplicate; verified directly: for a synthetic circle with 3 exact
    period-copies offset by ~1e-9, the naive nearest-neighbour-based scale
    collapses to ~1e-9 itself, silently disabling detection). Any group of
    points mutually within `duplicate_tolerance * bbox_diagonal` of each
    other is treated as one duplicate cluster (found via connected
    components on the within-threshold pair graph, so clusters of 3+
    near-identical points -- e.g. from sampling 3 periods of a closed orbit
    -- are caught as a single cluster, not missed by only checking each
    point's single nearest neighbour). The default (1e-6) catches the
    ~1e-9-separation duplicates the benchmark actually produced (against a
    bounding box of order 1-10 for these trajectories) while leaving
    genuinely close (but distinct) samples of a densely-sampled manifold
    untouched.

    - `dedupe=False` (default): raise `DuplicatePointsError` naming the
      duplicate clusters, forcing the caller to notice and decide -- per the
      benchmark's finding that this corruption was easy to miss silently.
    - `dedupe=True`: keep one representative point per duplicate cluster and
      drop the rest before building the graph (a warning-free convenience
      for callers who have already decided duplicates are expected and
      should just be removed, e.g. an automated sweep over many point
      clouds).

    Parameters
    ----------
    theiler_window:
        Theiler (1986) exclusion window, in units of ORIGINAL TRAJECTORY TIME
        INDEX difference: point `i` may only take as a neighbour a point `j`
        with ``|t_i - t_j| >= W``. Same meaning, same units and the same
        ``"auto"`` heuristic as `baseline.correlation_dimension`'s argument of
        the same name -- deliberately, so the two estimators can be given the
        identical correction and compared fairly.

        * ``0`` (default) -- OFF. Takes the original code path verbatim; the
          resulting graph is identical edge-for-edge to the pre-AR1 one.
        * ``1`` -- excludes only ``|t_i - t_j| < 1``, i.e. nothing beyond the
          self-match already excluded. Identical to ``0``, and is the
          conventional "no window" value in the literature.
        * any ``W >= 2`` -- the correction, with that window.
        * ``"auto"`` -- `baseline.theiler_window_from_autocorrelation` on this
          cloud (the first lag whose autocorrelation falls to 1/e). Opt-in
          only; never reached unless the caller asks for it.

        Read the module docstring before turning this on: it is the right
        correction for a trajectory whose along-path sampling is finer than
        its transverse structure, and the wrong one for a curve whose temporal
        neighbours ARE its spatial neighbours.
    time_indices:
        One integer per point: its index in the ORIGINAL trajectory. Required
        for correctness whenever `points` has been reordered or resampled,
        since array position is then no longer time. If omitted, point `i` is
        assumed to have time index `i`. Validated even when the window is off,
        because a malformed `time_indices` is a caller bug worth reporting
        whether or not this particular call consumes it. Entries need not be
        sorted or contiguous; only differences are used.

    Raises
    ------
    ValueError
        If the window is so wide that some point has fewer than `k` eligible
        neighbours left. That is reported rather than silently accepted: a
        point starved of candidates would otherwise be wired to whatever
        distant points remain, which is the exact failure the module docstring
        describes, and silently produces a plausible-looking graph.
    """
    n = len(points)
    if k < 1:
        raise ValueError("k must be at least 1")

    times = _resolve_time_indices(time_indices, n)

    arr = np.asarray(points, dtype=float)
    tree = cKDTree(arr)

    bbox_diagonal = _bbox_diagonal(arr)
    threshold = duplicate_tolerance * bbox_diagonal if bbox_diagonal > 0 else duplicate_tolerance

    clusters = _duplicate_clusters(tree, n, threshold)
    if clusters:
        if not dedupe:
            sizes = sorted((len(c) for c in clusters), reverse=True)
            raise DuplicatePointsError(
                f"{len(clusters)} near-duplicate point cluster(s) found "
                f"(mutual distance < {threshold:.3e}, {duplicate_tolerance:.0e} of "
                f"the bounding-box diagonal {bbox_diagonal:.3e}); "
                f"cluster sizes: {sizes[:10]}{'...' if len(sizes) > 10 else ''}. "
                f"This is the exact failure mode docs/MENSURA_BENCHMARK.md "
                f"finding F3 documents (e.g. sampling multiple periods of a closed "
                f"orbit) -- pass dedupe=True to drop duplicates automatically, or "
                f"resample your point cloud without repeated traversals."
            )
        arr, points, times = _drop_duplicate_clusters(arr, clusters, n, times)
        n = len(points)
        tree = cKDTree(arr)

    if k >= n:
        raise ValueError(f"k ({k}) must be less than the number of points ({n})")

    window = _resolve_theiler_window(theiler_window, arr, times)

    if window <= 1:
        # Pre-AR1 path, preserved verbatim so recorded graphs reproduce exactly.
        # query k+1 to include each point itself, then drop the self-match.
        _, neighbour_idx = tree.query(arr, k=k + 1)
        neighbours: list[list[int]] = [
            [int(j) for j in row if int(j) != i][:k] for i, row in enumerate(neighbour_idx)
        ]
    else:
        neighbours = _windowed_neighbours(tree, arr, times, k, window)

    edges: set[tuple[int, int]] = set()
    for i, row in enumerate(neighbours):
        for j in row:
            edges.add((i, j) if i < j else (j, i))

    return Hypergraph(tuple(sorted(edges)))


def _windowed_neighbours(
    tree: cKDTree, arr: np.ndarray, times: np.ndarray, k: int, window: int
) -> list[list[int]]:
    """Each point's k nearest neighbours among those at time distance >= `window`.

    Correctness of the query size, which is the only subtle part: at most
    ``2 * window - 1`` points (the time-local run around `i`, inclusive of `i`)
    can be excluded, so the k nearest ELIGIBLE neighbours are guaranteed to lie
    within the ``k + 2 * window - 1`` nearest points overall. Querying exactly
    that many is therefore exact, not a heuristic truncation.
    """
    n = len(arr)
    kq = min(n, k + 2 * window - 1)
    _, neighbour_idx = tree.query(arr, k=kq)
    if neighbour_idx.ndim == 1:  # scipy returns 1-D when kq == 1
        neighbour_idx = neighbour_idx[:, None]

    out: list[list[int]] = []
    starved: list[int] = []
    for i, row in enumerate(neighbour_idx):
        t_i = times[i]
        keep = [int(j) for j in row if abs(times[int(j)] - t_i) >= window][:k]
        if len(keep) < k:
            starved.append(i)
        out.append(keep)

    if starved:
        raise ValueError(
            f"theiler_window={window} leaves {len(starved)} of {n} points with fewer "
            f"than k={k} eligible neighbours (first offender: index {starved[0]}). The "
            f"window is excluding a large fraction of the cloud, so the surviving "
            f"neighbours would be arbitrary distant points rather than a proximity "
            f"structure. Use a smaller window, more points, or a smaller k."
        )
    return out


def _resolve_time_indices(time_indices: Sequence[int] | None, n: int) -> np.ndarray:
    """Validate `time_indices`, or fall back to the documented `t_i = i` assumption.

    Mirrors `baseline._resolve_time_indices`; kept as its own function rather
    than imported so the two estimators' argument validation cannot drift apart
    silently if either module's contract changes.
    """
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
    theiler_window: int | str, arr: np.ndarray, times: np.ndarray
) -> int:
    """Turn the `theiler_window` argument into a concrete non-negative window."""
    if isinstance(theiler_window, str):
        if theiler_window != "auto":
            raise ValueError(
                f"theiler_window must be a non-negative int or 'auto', got {theiler_window!r}"
            )
        return theiler_window_from_autocorrelation(arr, time_indices=times)
    if isinstance(theiler_window, bool) or not isinstance(theiler_window, int):
        raise ValueError(
            f"theiler_window must be a non-negative int or 'auto', got {theiler_window!r}"
        )
    if theiler_window < 0:
        raise ValueError(f"theiler_window must be non-negative, got {theiler_window}")
    return theiler_window


def _bbox_diagonal(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    span = arr.max(axis=0) - arr.min(axis=0)
    return float(np.sqrt(np.sum(span**2)))


def _duplicate_clusters(tree: cKDTree, n: int, threshold: float) -> list[list[int]]:
    """Connected components of the graph "distance < threshold", excluding singletons."""
    if threshold <= 0:
        return []
    pairs = tree.query_pairs(r=threshold, output_type="ndarray")
    if pairs.size == 0:
        return []
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=bool)
    adjacency = coo_matrix((data, (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adjacency, directed=False)
    clusters: dict[int, list[int]] = {}
    touched = set(rows.tolist()) | set(cols.tolist())
    for idx in touched:
        clusters.setdefault(int(labels[idx]), []).append(int(idx))
    return [sorted(members) for members in clusters.values() if len(members) > 1]


def _drop_duplicate_clusters(
    arr: np.ndarray, clusters: list[list[int]], n: int, times: np.ndarray
) -> tuple[np.ndarray, list[tuple[float, ...]], np.ndarray]:
    """Keep the lowest-indexed point of each duplicate cluster; drop the rest.

    `times` is carried through the same selection, so a surviving point keeps
    its ORIGINAL trajectory time index rather than being renumbered by its new
    array position -- which is the whole point of tracking time separately.
    """
    to_drop: set[int] = set()
    for cluster in clusters:
        to_drop.update(cluster[1:])  # keep cluster[0] (lowest index), drop the rest
    keep = [i for i in range(n) if i not in to_drop]
    kept_arr = arr[keep]
    kept_points = [tuple(row) for row in kept_arr]
    return kept_arr, kept_points, times[keep]
