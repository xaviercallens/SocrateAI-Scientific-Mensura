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
from dataclasses import dataclass

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


# ==========================================================================
# SCALE-AWARE (LOCAL-MAHALANOBIS) NEIGHBOUR SELECTION
#
# WHY THIS EXISTS. Dimension is a bi-Lipschitz invariant: D_0, D_1 and D_2
# are all unchanged by an invertible linear map. A plain Euclidean k-NN graph
# is not, and docs/MENSURA_BENCH_V2.md section 7.1 records the refutation --
# rescaling one coordinate of a uniform square by 0.01 (metres -> kilometres)
# flipped `cic.certify` from MEASURED [1.59, 2.28] to MEASURED [0.64, 1.36],
# two DISJOINT measured intervals for the same set with no abstention signal
# raised. The mechanism on this side is the k-NN graph: under strong
# anisotropy every point's nearest neighbours lie along the long axis, so the
# proximity graph degenerates into a chain and the shell-growth estimator
# reads ~1 no matter what the set is. A textbook Takens delay embedding at
# lag 1 reaches that regime, so it is a defect reachable by ordinary use.
#
# THE FIX, chosen by the owner over "normalise in the adapter" and
# "detect anisotropy and abstain" (v2 section 7.2, option 3): make the
# neighbour selection itself metric-adaptive. For each point estimate a local
# covariance from its current neighbourhood, use its inverse as a per-point
# Mahalanobis metric, re-select neighbours under that metric, and iterate to
# a fixed point. Only the EDGE SET changes; `dimension.py`'s shell-growth
# estimator reads it unchanged.
#
# LITERATURE. The local-covariance-from-k-nearest-neighbours step is LLE's
# local-linear-patch construction verbatim (Roweis & Saul 2000, "Nonlinear
# Dimensionality Reduction by Locally Linear Embedding", Science 290(5500)),
# including its ridge regularisation of a rank-deficient local covariance.
# The adapt-the-neighbourhood-to-the-sample principle is Farahmand,
# Szepesvari & Audibert (2007), "Manifold-adaptive dimension estimation",
# already cited in `dimension.py`; this applies it to the local SHAPE rather
# than only the local SIZE. Theiler (1986) is this codebase's own precedent
# for "a hard-coded scale computed from raw coordinates silently encodes an
# assumption" -- the identical defect one abstraction level over, in time
# rather than space.
#
# MEASURED REACH, and its ceiling, stated up front (v2 sections 8.1-8.2,
# 10.4). This construction extends the MEASURED region from isotropic-only
# to ~100:1 anisotropy at n=1600 (~200:1 at n=3200), with or without an
# arbitrary rotation, and is exactly rotation-equivariant: transforms with
# the same condition number produce byte-identical output. It does NOT
# reach further. Beyond ~100:1 the shell arm collapses to the chain artifact
# again and NO ridge-floor value rescues it -- that was measured, not
# assumed, and is a structural limit of a locally-LINEAR metric. Past the
# ceiling the answer is `cic.py`'s abstention stack, which converts those
# rows into honest UNDECIDED verdicts rather than silent violations.
# ==========================================================================

# Row-chunk size for the batched pairwise Mahalanobis quadratic form. A pure
# memory/speed knob: it partitions a sum, so it cannot change a result.
_MAHALANOBIS_CHUNK = 256

# --- the three ridge floors, which are NOT one constant wearing three hats --
#
# Every one of them clips a covariance's eigenvalues to a fraction of that
# same covariance's own largest eigenvalue before it is inverted. They differ
# because they regularise three different estimates doing three different
# jobs, and collapsing them was measured to break the construction.
#
# GLOBAL -- regularises the WHOLE-CLOUD covariance, estimated from all n
# points, so a low-variance and reliable statistic. Its only job is to stop a
# literally singular direction (a constant coordinate) from producing an
# infinity. It must NOT cap how much genuine anisotropy is corrected, so it
# sits near machine-precision-relative. MEASURED: with one shared 0.05 floor
# the correctable ratio was capped at 20:1 EVERYWHERE, including on the
# global bootstrap -- and the refutation case's true ratio is 10000:1 (0.01
# squared), so the bootstrap was only ever correcting a 20:1 slice of it.
# Splitting this floor out moved the correlation-sum arm on the refutation
# case from 1.34 (barely off the unfixed 0.64-1.36 bracket) to ~1.9.
#
# TOPOLOGY -- regularises the per-point covariance that SELECTS EDGES. This
# one must stay conservative, and 0.05 is the value both known-answer
# batteries passed under. MEASURED (v2 section 8.2, and the round-2 floor
# sweep): loosening it below ~0.02 makes the shell arm WORSE, not better,
# collapsing back toward ~1.0 on a Lorenz cloud that is not 1-D. The
# mechanism was confirmed by inspecting the graphs: an aggressively
# up-weighted near-degenerate direction turns the k-NN search into a search
# for points merely CLOSE ALONG THE THIN AXIS however far away they are along
# the well-resolved ones, and on a curved support that relation is not
# transitive with "close on the manifold". The result is spurious long-range
# shortcut edges, which a BFS shell count is far more sensitive to than any
# distance-sum statistic.
#
# METRIC -- regularises the per-point covariance that supplies the
# CORRELATION-SUM arm's distances. No graph, no BFS, no shortcuts to create,
# so it tolerates much more aggressive correction: holding edge selection at
# 0.05 and tightening only this floor from 0.05 to 0.001 moved the arm
# monotonically and then plateaued. It is never used to select an edge.
#
# DO NOT MERGE THESE BACK INTO ONE CONSTANT. The split is load-bearing and
# each half is separately justified above.
GLOBAL_RIDGE_FLOOR_FRAC = 1e-9
TOPOLOGY_RIDGE_FLOOR_FRAC = 0.05
METRIC_RIDGE_FLOOR_FRAC = 0.001

# Iteration auto-stop. The refinement halts when the neighbour sets stop
# moving, not after a caller-chosen number of rounds: >= 97% neighbour-set
# overlap between consecutive rounds is the convergence criterion, and the
# prototype measured 1-2 rounds sufficient on ordinary targets, with the
# hardest case (Lorenz at lag 1) plateauing well inside 8.
METRIC_CONVERGENCE_OVERLAP = 0.97
MIN_METRIC_ITERATIONS = 1
MAX_METRIC_ITERATIONS = 8


def regularize_covariance(cov: np.ndarray, floor_frac: float) -> np.ndarray:
    """Clip `cov`'s eigenvalues to `floor_frac` of its own largest eigenvalue.

    Relative to the covariance's own scale, so the operation commutes with
    rescaling the cloud -- which is what makes the whole construction exactly
    scale-equivariant rather than approximately so.
    """
    evals, evecs = np.linalg.eigh(cov)
    floor = max(float(evals.max()), 1e-300) * floor_frac
    evals = np.maximum(evals, floor)
    return (evecs * evals) @ evecs.T


def regularize_covariance_batch(covs: np.ndarray, floor_frac: float) -> np.ndarray:
    """Batched `regularize_covariance` over an (n, d, d) stack.

    Bit-identical to looping the scalar form: batched `eigh` runs the same
    LAPACK routine per slice, `evals[..., -1]` is the largest eigenvalue
    because `eigh` returns them ascending, and the reconstruction is the same
    per-slice contraction. One gufunc call instead of n, which matters
    because at d=2-3 the per-call overhead dominated the arithmetic.
    """
    evals, evecs = np.linalg.eigh(covs)
    floor = np.maximum(evals[..., -1], 1e-300) * floor_frac
    evals = np.maximum(evals, floor[..., None])
    return np.matmul(evecs * evals[..., None, :], np.swapaxes(evecs, -1, -2))


def global_covariance(
    arr: np.ndarray, *, floor_frac: float = GLOBAL_RIDGE_FLOOR_FRAC
) -> np.ndarray:
    """The whole cloud's covariance, regularised at the global floor.

    THIS IS THE BOOTSTRAP, AND THE CHOICE IS MEASURED, NOT STYLISTIC.
    Bootstrapping the first local covariance from a plain EUCLIDEAN k-NN
    neighbourhood is circular under exactly the anisotropy this exists to
    correct: that neighbourhood already IS the chain artifact, so the
    first-round covariance comes out with its axes backwards and the fixed
    point it converges to is the wrong one. The global covariance needs no
    neighbour selection at all, so it cannot inherit the defect.

    That it is a single global matrix does not make this "normalise and
    declare": it is round zero of a construction that then keeps refining
    per point, and on curved supports it demonstrably keeps refining.
    """
    centered = arr - arr.mean(axis=0)
    cov = (centered.T @ centered) / len(arr)
    return regularize_covariance(cov, floor_frac)


def local_covariances(arr: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    """Raw (UNregularised) per-point covariance of each point's neighbour set.

    Roweis & Saul's local-linear-patch step. Returned unfloored because the
    caller needs the SAME raw estimate floored two different ways -- once for
    edge selection and once for the distance arm (see the floor commentary
    above); flooring here would force one floor on both.
    """
    idx = np.asarray(neighbor_idx)
    pts = arr[idx]  # (n, k, d)
    centered = pts - pts.mean(axis=1, keepdims=True)
    return np.matmul(centered.transpose(0, 2, 1), centered) / idx.shape[1]


def mahalanobis_knn_indices(arr: np.ndarray, sigma_inv: np.ndarray, k: int) -> np.ndarray:
    """Exact k nearest neighbours of every point under a Mahalanobis metric,
    searched over the FULL population. Returns an (n, k) index array, each row
    ascending by distance.

    THE FULL SEARCH IS DELIBERATE, NOT A MISSED OPTIMISATION. Pre-filtering
    candidates with a Euclidean k-NN shortlist and re-ranking them -- the
    standard trick, and the first thing tried -- is unsafe here: under a 100x
    compression a point's true isotropic neighbours are demonstrably NOT among
    its top few hundred Euclidean neighbours, so the shortlist would exclude
    precisely the points the Mahalanobis re-ranking exists to surface. The
    full search is the correctness fix, and it is the dominant cost of the
    construction (O(n^2 d^2) per round).

    `sigma_inv` is either one (d, d) matrix broadcast to every point (the
    global bootstrap) or one (n, d, d) matrix per point.
    """
    n = len(arr)
    if not 1 <= k < n:
        raise ValueError(f"k ({k}) must satisfy 1 <= k < n ({n})")
    per_point = sigma_inv.ndim == 3
    out = np.empty((n, k), dtype=np.intp)
    for start in range(0, n, _MAHALANOBIS_CHUNK):
        end = min(start + _MAHALANOBIS_CHUNK, n)
        diffs = arr[None, :, :] - arr[start:end, None, :]  # (c, n, d)
        if per_point:
            s = sigma_inv[start:end]
        else:
            s = np.broadcast_to(sigma_inv, (end - start, *sigma_inv.shape))
        d2c = np.einsum("cnd,cde,cne->cn", diffs, s, diffs)
        for ci in range(end - start):
            i = start + ci
            d2 = d2c[ci]
            d2[i] = np.inf
            idx = np.argpartition(d2, k)[:k]
            out[i] = idx[np.argsort(d2[idx])]
    return out


def hypergraph_from_neighbours(neighbor_idx: np.ndarray) -> Hypergraph:
    """Symmetrised proximity graph over an (n, k) neighbour-index array.

    Same edge convention as `knn_hypergraph`: an undirected edge (i, j)
    whenever j is among i's k neighbours or i is among j's. `np.unique` on
    (lo, hi) pairs returns lexicographically sorted unique rows, which is
    exactly `sorted(set(...))` over integer pairs.
    """
    rows = np.asarray(neighbor_idx, dtype=np.int64)
    n = len(rows)
    i_col = np.repeat(np.arange(n, dtype=np.int64), rows.shape[1])
    j_col = rows.reshape(-1)
    lo, hi = np.minimum(i_col, j_col), np.maximum(i_col, j_col)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return Hypergraph(tuple(map(tuple, pairs.tolist())))


def auto_covariance_window(n: int, d: int) -> int:
    """`k0`, the covariance-estimation window, chosen from the data's shape alone.

    ZERO KNOB: this is a function of (n, d) only, so no caller can move it.
    A d-dimensional covariance has d(d+1)/2 free parameters and the usual
    rule of thumb wants several times that many samples for a low-variance
    estimate, giving the base rule `5 * d * (d + 1)` -- 30 at d=2, 60 at d=3.
    Floored at 30 (the smallest window the prototype tested) and capped so it
    never asks for an unreasonable share of a small cloud.

    Kept separate from the graph degree `k` on purpose: reusing `k` for both
    would couple graph sparsity to covariance stability for no reason. NOT
    tuned beyond this -- k0 in 20-60 was measured indistinguishable on 2-D
    and 3-D targets, and the k0 sweep in v2 section 10.1 moved the
    correlation-sum readout by less than its own seed-to-seed spread across
    k0 = 15 ... 480.
    """
    base = 5 * d * (d + 1)
    k0 = int(min(max(30, base), max(60, n // 20)))
    return max(1, min(k0, n - 2))


@dataclass(frozen=True)
class LocalMetric:
    """The converged local-Mahalanobis metric, plus what it took to get there.

    Two inverse-covariance stacks, from the SAME raw local covariances floored
    two different ways: `sigma_inv_topology` is what selects edges,
    `sigma_inv_metric` is what the correlation-sum arm measures distances
    with. See the ridge-floor commentary above for why they differ.
    """

    k0: int
    iterations: int
    converged: bool
    final_overlap: float
    sigma_inv_topology: np.ndarray
    sigma_inv_metric: np.ndarray
    global_cov_cond: float
    median_topology_cond: float
    median_metric_cond: float
    max_metric_cond: float
    round_overlaps: tuple[float, ...]


def local_mahalanobis_metric(arr: np.ndarray, *, k0: int | None = None) -> LocalMetric:
    """The fixed-point refinement: bootstrap globally, then re-select and
    re-estimate per point until the neighbour sets stop moving.

    Each round estimates every point's covariance FROM the neighbours it just
    selected, then selects the next round's neighbours under the inverse of
    that covariance. The loop stops when consecutive rounds agree on >= 97% of
    every point's neighbour set (`METRIC_CONVERGENCE_OVERLAP`), and always
    runs at least one refinement round so the returned metric is never merely
    the global bootstrap.

    A CONSEQUENCE WORTH NAMING, because it was a real bug found while
    validating: the returned covariance is measured from the LAST selected
    neighbour set, so it is the metric that would select the NEXT round's
    neighbours, not the one that selected the last round's. At convergence
    the two nearly coincide -- but "nearly" is not "exactly", and the
    rung-admissibility test was measured to be sensitive to exactly that
    residual gap (on a Lorenz cloud, the stale sets found NO admissible rung
    anywhere on the ladder while a fresh query under the reported metric
    found one). Callers must therefore re-query neighbours under
    `sigma_inv_topology` rather than reuse any earlier round's sets. This
    function deliberately returns only the metric, so that mistake is not
    reachable through its API.
    """
    n = len(arr)
    d = arr.shape[1]
    if k0 is None:
        k0 = auto_covariance_window(n, d)

    g_cov = global_covariance(arr, floor_frac=GLOBAL_RIDGE_FLOOR_FRAC)
    sigma_inv = np.linalg.inv(g_cov)

    neighbor_idx = mahalanobis_knn_indices(arr, sigma_inv, k0)
    prev_sets = [set(row.tolist()) for row in neighbor_idx]
    raw = local_covariances(arr, neighbor_idx)
    reg_topo = regularize_covariance_batch(raw, TOPOLOGY_RIDGE_FLOOR_FRAC)

    it = 0
    overlap = 0.0
    overlaps: list[float] = []
    while it < MAX_METRIC_ITERATIONS:
        it += 1
        sigma_inv = np.linalg.inv(reg_topo)
        new_idx = mahalanobis_knn_indices(arr, sigma_inv, k0)
        new_sets = [set(row.tolist()) for row in new_idx]
        overlap = float(
            np.mean([len(a & b) / k0 for a, b in zip(prev_sets, new_sets, strict=True)])
        )
        overlaps.append(overlap)
        raw = local_covariances(arr, new_idx)
        reg_topo = regularize_covariance_batch(raw, TOPOLOGY_RIDGE_FLOOR_FRAC)
        prev_sets = new_sets
        if it >= MIN_METRIC_ITERATIONS and overlap >= METRIC_CONVERGENCE_OVERLAP:
            break

    reg_metric = regularize_covariance_batch(raw, METRIC_RIDGE_FLOOR_FRAC)
    metric_cond = np.linalg.cond(reg_metric)
    return LocalMetric(
        k0=k0,
        iterations=it,
        converged=bool(overlap >= METRIC_CONVERGENCE_OVERLAP),
        final_overlap=overlap,
        sigma_inv_topology=np.linalg.inv(reg_topo),
        sigma_inv_metric=np.linalg.inv(reg_metric),
        global_cov_cond=float(np.linalg.cond(g_cov)),
        median_topology_cond=float(np.median(np.linalg.cond(reg_topo))),
        median_metric_cond=float(np.median(metric_cond)),
        max_metric_cond=float(np.max(metric_cond)),
        round_overlaps=tuple(overlaps),
    )
