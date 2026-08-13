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
-- a 10-problem physics benchmark (docs/POLY_ALGEBRAIC_BENCHMARK.md, finding
N5) found the O(n^2) cost was the binding constraint on point count for the
two fractal (chaotic-attractor) cases, whose dimension estimates were still
visibly unconverged at the largest n the O(n^2) cost made affordable.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .core import Hypergraph


class DuplicatePointsError(ValueError):
    """Raised when `knn_hypergraph` finds near-duplicate points in the input.

    Fed by finding F3 of docs/POLY_ALGEBRAIC_BENCHMARK.md: sampling multiple
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
    """
    n = len(points)
    if k < 1:
        raise ValueError("k must be at least 1")

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
                f"This is the exact failure mode docs/POLY_ALGEBRAIC_BENCHMARK.md "
                f"finding F3 documents (e.g. sampling multiple periods of a closed "
                f"orbit) -- pass dedupe=True to drop duplicates automatically, or "
                f"resample your point cloud without repeated traversals."
            )
        arr, points = _drop_duplicate_clusters(arr, clusters, n)
        n = len(points)
        tree = cKDTree(arr)

    if k >= n:
        raise ValueError(f"k ({k}) must be less than the number of points ({n})")

    # query k+1 to include each point itself, then drop the self-match.
    _, neighbour_idx = tree.query(arr, k=k + 1)

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in neighbour_idx[i]:
            j = int(j)
            if j == i:
                continue
            edges.add((i, j) if i < j else (j, i))

    return Hypergraph(tuple(sorted(edges)))


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
    arr: np.ndarray, clusters: list[list[int]], n: int
) -> tuple[np.ndarray, list[tuple[float, ...]]]:
    """Keep the lowest-indexed point of each duplicate cluster; drop the rest."""
    to_drop: set[int] = set()
    for cluster in clusters:
        to_drop.update(cluster[1:])  # keep cluster[0] (lowest index), drop the rest
    keep = [i for i in range(n) if i not in to_drop]
    kept_arr = arr[keep]
    kept_points = [tuple(row) for row in kept_arr]
    return kept_arr, kept_points
