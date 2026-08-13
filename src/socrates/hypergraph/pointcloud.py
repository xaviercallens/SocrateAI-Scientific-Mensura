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
"""

from __future__ import annotations

from .core import Hypergraph


def knn_hypergraph(points: list[tuple[float, ...]], k: int) -> Hypergraph:
    """Build an undirected k-nearest-neighbour proximity graph over `points`.

    Each point becomes a node (indexed by its position in `points`); an edge
    (i, j) is added whenever j is among i's k nearest neighbours OR i is
    among j's (the standard symmetrization of a directed k-NN graph, needed
    because "nearest neighbour of" is not a symmetric relation).

    O(n^2) distance computation -- fine for the benchmark sizes this module
    targets (hundreds to low thousands of points); not intended for large-
    scale point clouds.
    """
    n = len(points)
    if k < 1:
        raise ValueError("k must be at least 1")
    if k >= n:
        raise ValueError(f"k ({k}) must be less than the number of points ({n})")

    def dist2(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        dists = sorted(range(n), key=lambda j: dist2(points[i], points[j]))
        neighbours = [j for j in dists if j != i][:k]
        for j in neighbours:
            edges.add((i, j) if i < j else (j, i))

    return Hypergraph(tuple(sorted(edges)))
