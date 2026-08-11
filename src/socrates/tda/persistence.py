"""Persistent homology of point clouds.

Thin, typed layer over `ripser` that gives every domain module the same
interface: point cloud in, `PersistenceDiagram` out. Keeping the wrapper
narrow is deliberate -- the scientific claims live in the domain modules, and
this file only has to be correct about bookkeeping (infinite bars, empty
inputs, dimension indexing).

References
----------
Edelsbrunner & Harer, *Computational Topology* (2010).
Otter et al., "A roadmap for the computation of persistent homology" (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Metric = Literal["euclidean", "precomputed"]


@dataclass(frozen=True)
class PersistenceDiagram:
    """Persistence pairs of a filtration, grouped by homological dimension.

    `diagrams[k]` is an (n_k, 2) array of (birth, death) pairs in dimension k.
    Essential classes carry `death = inf` and are retained, not silently
    dropped -- they are the features that persist to the end of the filtration.
    """

    diagrams: list[np.ndarray]
    max_dimension: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k, d in enumerate(self.diagrams):
            if d.ndim != 2 or (d.size and d.shape[1] != 2):
                raise ValueError(f"diagram {k} must have shape (n, 2), got {d.shape}")

    def in_dimension(self, k: int) -> np.ndarray:
        if not 0 <= k < len(self.diagrams):
            raise IndexError(f"dimension {k} outside 0..{len(self.diagrams) - 1}")
        return self.diagrams[k]

    def lifetimes(self, k: int, *, finite_only: bool = True) -> np.ndarray:
        """Death - birth for each class in dimension k."""
        d = self.in_dimension(k)
        if d.size == 0:
            return np.empty(0)
        spans = d[:, 1] - d[:, 0]
        return spans[np.isfinite(spans)] if finite_only else spans

    def betti_numbers(self, threshold: float) -> list[int]:
        """Betti numbers of the complex at filtration value `threshold`.

        A class contributes when born at or before `threshold` and not yet
        dead -- the standard half-open convention [birth, death).
        """
        counts = []
        for d in self.diagrams:
            if d.size == 0:
                counts.append(0)
                continue
            alive = (d[:, 0] <= threshold) & (d[:, 1] > threshold)
            counts.append(int(np.count_nonzero(alive)))
        return counts

    def most_persistent(self, k: int, n: int = 5) -> np.ndarray:
        """The n longest-lived classes in dimension k, longest first.

        Infinite bars sort first; they are the most persistent features there are.
        """
        d = self.in_dimension(k)
        if d.size == 0:
            return np.empty((0, 2))
        order = np.argsort(-(d[:, 1] - d[:, 0]))
        return d[order][:n]

    def total_persistence(self, k: int) -> float:
        """Sum of finite lifetimes in dimension k -- a scalar summary of structure."""
        return float(np.sum(self.lifetimes(k)))

    def summary(self) -> dict[str, object]:
        return {
            "max_dimension": self.max_dimension,
            "n_classes": [int(d.shape[0]) for d in self.diagrams],
            "total_persistence": [self.total_persistence(k) for k in range(len(self.diagrams))],
            **self.metadata,
        }


def vietoris_rips(
    points: np.ndarray,
    *,
    max_dimension: int = 2,
    max_edge_length: float | None = None,
    metric: Metric = "euclidean",
    n_perm: int | None = None,
) -> PersistenceDiagram:
    """Persistent homology of the Vietoris-Rips filtration on `points`.

    `n_perm` enables greedy-permutation subsampling, which bounds the cost on
    large clouds at the price of an approximation with a known bottleneck
    guarantee -- necessary above ~2000 points, where exact VR becomes intractable.
    """
    from ripser import ripser

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"expected a 2-D array of points, got shape {pts.shape}")
    if pts.shape[0] == 0:
        return PersistenceDiagram(
            [np.empty((0, 2)) for _ in range(max_dimension + 1)], max_dimension
        )

    kwargs: dict[str, object] = {
        "maxdim": max_dimension,
        "distance_matrix": metric == "precomputed",
    }
    if max_edge_length is not None:
        kwargs["thresh"] = max_edge_length
    if n_perm is not None:
        kwargs["n_perm"] = min(n_perm, pts.shape[0])

    result = ripser(pts, **kwargs)  # type: ignore[arg-type]

    return PersistenceDiagram(
        diagrams=[np.asarray(d, dtype=float).reshape(-1, 2) for d in result["dgms"]],
        max_dimension=max_dimension,
        metadata={
            "n_points": int(pts.shape[0]),
            "filtration": "vietoris-rips",
            "subsampled": n_perm is not None,
        },
    )


def alpha_complex(
    points: np.ndarray, *, max_dimension: int = 2
) -> PersistenceDiagram:
    """Persistent homology of the Alpha filtration (Delaunay-based).

    For low-dimensional Euclidean data the alpha complex is far smaller than
    Vietoris-Rips at equal fidelity, which is what makes ~10^5-point cosmic-web
    clouds tractable. Falls back to Vietoris-Rips when GUDHI is unavailable.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return PersistenceDiagram(
            [np.empty((0, 2)) for _ in range(max_dimension + 1)], max_dimension
        )

    try:
        import gudhi
    except ImportError:
        return vietoris_rips(pts, max_dimension=max_dimension)

    complex_ = gudhi.AlphaComplex(points=pts)
    tree = complex_.create_simplex_tree()
    tree.compute_persistence()

    diagrams = []
    for k in range(max_dimension + 1):
        intervals = tree.persistence_intervals_in_dimension(k)
        arr = np.asarray(intervals, dtype=float).reshape(-1, 2)
        # GUDHI's alpha filtration values are squared circumradii.
        diagrams.append(np.sqrt(np.abs(arr)) * np.sign(arr))

    return PersistenceDiagram(
        diagrams=diagrams,
        max_dimension=max_dimension,
        metadata={"n_points": int(pts.shape[0]), "filtration": "alpha"},
    )


def bottleneck_distance(a: PersistenceDiagram, b: PersistenceDiagram, dimension: int) -> float:
    """Bottleneck distance between two diagrams in a fixed dimension.

    This is the metric that makes cross-domain comparison meaningful: it is
    stable under perturbation of the underlying point cloud, so a small
    distance reflects genuinely similar topology rather than sampling noise.
    """
    from persim import bottleneck

    return float(bottleneck(a.in_dimension(dimension), b.in_dimension(dimension)))


def persistence_entropy(diagram: PersistenceDiagram, dimension: int) -> float:
    """Shannon entropy of the normalized finite lifetimes in `dimension`.

    Low entropy means a few features dominate (one clear structure); high
    entropy means persistence is spread across many comparable features.
    """
    spans = diagram.lifetimes(dimension)
    spans = spans[spans > 0]
    if spans.size == 0:
        return 0.0
    p = spans / spans.sum()
    return float(-np.sum(p * np.log(p)))
