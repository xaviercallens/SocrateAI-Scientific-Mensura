"""Known-answer tests for the point-cloud -> Hypergraph bridge.

Calibration verified empirically before being written into assertions (see
docs/HYPERGRAPH_NOTES.md and the session that added this file): a k-NN graph
built from a uniform sample of a filled cube underestimates dimension near
the boundary (corners and faces have fewer true neighbours inside the
sample than an interior point does), so tests that need higher-dimensional
accuracy sample only interior points, matching standard practice in
correlation-dimension estimation.
"""

from __future__ import annotations

import math
import random

import pytest

from socrates.hypergraph.dimension import local_dimension, mean_dimension
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph


def test_knn_hypergraph_rejects_k_too_large():
    with pytest.raises(ValueError, match="must be less than"):
        knn_hypergraph([(0.0,), (1.0,), (2.0,)], k=3)


def test_circle_point_cloud_has_dimension_one():
    # A circle is a 1D manifold regardless of the ambient (2D) embedding.
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 300 for i in range(300))]
    hg = knn_hypergraph(points, k=6)
    dim = mean_dimension(hg, samples=10, max_radius=5)
    assert dim == pytest.approx(1.0, abs=0.05)


def test_filled_square_point_cloud_has_dimension_two():
    rng = random.Random(0)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(500)]
    hg = knn_hypergraph(points, k=10)
    dim = mean_dimension(hg, samples=10, max_radius=5)
    assert dim == pytest.approx(2.0, abs=0.15)


def test_filled_cube_interior_points_have_dimension_three():
    # Boundary points of a finite cube sample have fewer true neighbours
    # than interior points do (a real, well-known effect in correlation-
    # dimension estimation, not an artifact of this implementation) --
    # restricting to interior points is standard practice, not cherry-
    # picking a convenient result.
    rng = random.Random(0)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(1500)]
    hg = knn_hypergraph(points, k=20)
    interior = [i for i, p in enumerate(points) if all(0.3 < c < 0.7 for c in p)]
    assert len(interior) >= 50, "sanity check: enough interior points for a stable mean"

    estimates = [local_dimension(hg, i, max_radius=3) for i in interior[:20]]
    well_fit = [e.dimension for e in estimates if e.is_well_fit(threshold=0.9)]
    assert len(well_fit) >= 10
    mean_dim = sum(well_fit) / len(well_fit)
    assert mean_dim == pytest.approx(3.0, abs=0.3)


def test_boundary_points_of_a_cube_underestimate_dimension():
    # Documents the real effect the previous test works around, rather than
    # letting it be an unexplained "why interior-only" comment: a corner of
    # the cube has a k-NN neighbourhood that is geometrically a fraction of
    # a full ball, so the shell-growth estimator reads a lower effective
    # dimension there than the true ambient dimension.
    rng = random.Random(1)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(1500)]
    hg = knn_hypergraph(points, k=20)
    corner_idx = min(range(len(points)), key=lambda i: sum(c**2 for c in points[i]))
    corner_est = local_dimension(hg, corner_idx, max_radius=3)
    if corner_est.is_well_fit(threshold=0.9):
        assert corner_est.dimension < 2.9  # measurably below the true dimension of 3


# ---------------------------------------------------------------- N4: duplicate detection


def test_knn_hypergraph_raises_on_multi_period_duplicate_clusters():
    # Regression test for docs/POLY_ALGEBRAIC_BENCHMARK.md finding F3: sampling
    # multiple periods of a closed orbit produces near-exact duplicate points
    # at ~1e-9 separation, which corrupted 3 of the 10 benchmark problems
    # silently. Reproduces the exact defect (3 period-copies of a circle,
    # offset by ~1e-9) and confirms it is now caught as an error naming the
    # cluster count and sizes, not silently corrupting the graph.
    circle = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 300 for i in range(300))]
    tripled = (
        circle
        + [(x + 1e-9, y + 1e-9) for x, y in circle]
        + [(x - 1e-9, y - 1e-9) for x, y in circle]
    )
    with pytest.raises(DuplicatePointsError, match="300 near-duplicate point cluster"):
        knn_hypergraph(tripled, k=6)


def test_knn_hypergraph_dedupe_recovers_the_clean_result():
    # dedupe=True on the same tripled circle must recover the same dimension
    # as the clean, non-duplicated circle -- not merely "not crash."
    circle = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 300 for i in range(300))]
    tripled = (
        circle
        + [(x + 1e-9, y + 1e-9) for x, y in circle]
        + [(x - 1e-9, y - 1e-9) for x, y in circle]
    )
    hg = knn_hypergraph(tripled, k=6, dedupe=True)
    assert hg.num_nodes == 300  # exactly recovers the original point count
    dim = mean_dimension(hg, samples=10, max_radius=5)
    assert dim == pytest.approx(1.0, abs=0.05)


def test_knn_hypergraph_does_not_false_positive_on_dense_genuine_sampling():
    # A densely-sampled (5000-point) circle has genuine nearest-neighbour
    # spacing (~2*pi/5000 ~ 1.3e-3) many orders of magnitude larger than the
    # duplicate threshold (~1e-6 of the ~2.8 bounding-box diagonal, i.e.
    # ~2.8e-6) -- dense-but-distinct sampling must not be flagged.
    dense = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 5000 for i in range(5000))]
    hg = knn_hypergraph(dense, k=6)  # must not raise
    assert hg.num_nodes == 5000


# ---------------------------------------------------------------- N5: KD-tree scaling


def test_knn_hypergraph_handles_large_point_clouds():
    # The O(n^2) brute-force construction this module started with made
    # n=15000 impractical (the binding constraint on the two chaotic-
    # attractor benchmark problems, docs/POLY_ALGEBRAIC_BENCHMARK.md finding
    # N5). The scipy.spatial.cKDTree-based construction must handle it
    # directly -- this test is a scale smoke test, not a timing assertion
    # (no wall-clock bound is asserted, since CI hardware varies), but it
    # would have been impractically slow to even attempt before N5.
    points = [
        (math.cos(t), math.sin(t), math.sin(2 * t))
        for t in (2 * math.pi * i / 15000 for i in range(15000))
    ]
    hg = knn_hypergraph(points, k=8)
    assert hg.num_nodes == 15000
    assert hg.num_edges > 0
