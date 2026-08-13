"""Known-answer tests for the Theiler window on the k-NN GRAPH side (AR1).

Theiler, J. (1986), Phys. Rev. A 34(3), 2427-2432 is usually cited against the
correlation sum, where time-local pairs add a spurious offset to C(r)
(tests/test_hypergraph_baseline_theiler.py pins that side down). The k-NN
graph has the same disease in a sharper form, because k-NN is a RANKING and
not a count: if the along-trajectory spacing is finer than the transverse
spacing between successive passes, a point's k nearest neighbours are *all*
its own temporal neighbours and the proximity graph degenerates into a
1-dimensional chain regardless of the set it was sampled from.

What these tests pin down:

1. OFF BY DEFAULT MEANS BIT-FOR-BIT. `theiler_window=0`, `theiler_window=1`
   and the default all produce the identical edge set, so every graph and
   every dimension already recorded in docs/POLY_ALGEBRAIC_BENCHMARK.md still
   reproduces exactly.
2. IT WORKS, ON A CASE WITH A KNOWN ANSWER AND A KNOWN CORRECT WINDOW. The
   `sticky_square` fixture is a genuinely 2-dimensional point SET visited by a
   scanner that lingers in blobs of exactly `STICKY_BLOB_SIZE` samples. So the
   right window is known in advance rather than read off the result, and the
   uncorrected k-NN graph's failure is measurable: 56% of its edges join two
   samples of the SAME anchor, and it puts the dimension of a flatly
   2-dimensional set at 0.94.
3. IT IS NOT SAFE TO TURN ON EVERYWHERE, and that fact is a test rather than a
   comment. On a 1-D closed orbit sampled once -- benchmark problems 03/04/07,
   two of which are verified wins -- the temporal neighbours ARE the genuine
   spatial neighbours, and excluding them replaces local geometry with
   long-range chords. `test_window_harms_a_singly_traversed_closed_orbit`
   asserts the damage, so that a future edit promoting the window to a default
   has to delete an explicit test saying it must not be.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from socrates.hypergraph.dimension import mean_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

STICKY_BLOB_SIZE = 8
STICKY_K = 10


def sticky_square(n_anchors: int = 400, blob: int = STICKY_BLOB_SIZE, eps: float = 5e-3, seed=7):
    """A 2-dimensional point SET visited by a deliberately "sticky" scanner.

    Identical construction to the fixture of the same name in
    tests/test_hypergraph_baseline_theiler.py, reproduced rather than imported
    so the two estimators' test files stay independent. `n_anchors` anchors are
    drawn i.i.d. uniform from the unit square, so the set is exactly
    2-dimensional; the SEQUENCE emits `blob` consecutive samples within ~eps of
    one anchor before jumping to the next. Every pair with |t_i - t_j| < blob is
    intra-blob (spurious); every pair with |t_i - t_j| >= blob is a genuine
    anchor-to-anchor pair. The correct window is therefore `blob`, known in
    advance.

    `blob` is deliberately SMALLER than `STICKY_K`. A blob wider than k would
    be a different and much less interesting failure: after the exclusion all k
    survivors would come from whichever single OTHER blob happens to be
    nearest, so the corrected graph would be an anchor-to-NEAREST-anchor graph
    rather than an anchor k-NN graph, and would not recover 2 either. With 7
    blob-mates competing for 10 slots the uncorrected graph is dominated by the
    sampler (56% of its edges are intra-blob, measured) while the corrected one
    still reaches ~10 distinct anchors.
    """
    rng = np.random.default_rng(seed)
    anchors = rng.random((n_anchors, 2))
    points = []
    for anchor in anchors:
        for _ in range(blob):
            points.append(tuple(anchor + eps * rng.standard_normal(2)))
    return points


def closed_orbit(n: int = 400):
    """One traversal of a circle: a 1-D curve whose temporal neighbours are
    exactly its spatial neighbours. The regime the window must NOT be used in."""
    return [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]


def line(n: int = 300):
    return [(float(i), 0.0) for i in range(n)]


# --------------------------------------------------------------------------
# 1. Off by default, and "off" is exact
# --------------------------------------------------------------------------


def test_default_and_window_zero_and_window_one_are_bit_for_bit_identical():
    points = sticky_square()
    default = knn_hypergraph(points, k=STICKY_K)
    zero = knn_hypergraph(points, k=STICKY_K, theiler_window=0)
    one = knn_hypergraph(points, k=STICKY_K, theiler_window=1)
    assert default.edges == zero.edges
    assert default.edges == one.edges


def test_time_indices_alone_change_nothing_when_the_window_is_off():
    points = sticky_square()
    plain = knn_hypergraph(points, k=STICKY_K)
    with_times = knn_hypergraph(points, k=STICKY_K, time_indices=list(range(len(points))))
    assert plain.edges == with_times.edges


def test_auto_window_is_opt_in_only():
    """A cloud with strong temporal structure still gets the uncorrected graph
    unless the caller asks; otherwise 'off by default' would not be off."""
    points = sticky_square()
    assert knn_hypergraph(points, k=STICKY_K).edges == knn_hypergraph(
        points, k=STICKY_K, theiler_window=0
    ).edges
    assert knn_hypergraph(points, k=STICKY_K, theiler_window="auto").edges != knn_hypergraph(
        points, k=STICKY_K
    ).edges


# --------------------------------------------------------------------------
# 2. Known answer: the window recovers the true dimension of the sticky sampler
# --------------------------------------------------------------------------


def test_uncorrected_knn_graph_sees_only_the_blobs():
    """The failure mode, stated as a measurement rather than asserted: with the
    window off, the majority of edges join two samples of the SAME anchor, i.e.
    they record the sampler's dwell time and carry no geometric information."""
    points = sticky_square()
    hg = knn_hypergraph(points, k=STICKY_K)
    same_blob = [i // STICKY_BLOB_SIZE == j // STICKY_BLOB_SIZE for i, j in hg.edges]
    # Every point's 7 blob-mates are nearer than anything else, so they take 7
    # of the 10 slots; the sampler, not the geometry, is writing the majority of
    # this graph.
    assert sum(same_blob) / len(same_blob) > 0.5


def test_window_at_the_known_blob_size_removes_every_intra_blob_edge():
    points = sticky_square()
    hg = knn_hypergraph(points, k=STICKY_K, theiler_window=STICKY_BLOB_SIZE)
    assert all(abs(i - j) >= STICKY_BLOB_SIZE for i, j in hg.edges)
    assert all(i // STICKY_BLOB_SIZE != j // STICKY_BLOB_SIZE for i, j in hg.edges)


def test_window_recovers_dimension_two_on_the_sticky_square():
    """The known answer. The SET is 2-dimensional by construction; the
    uncorrected graph cannot see that at all, the corrected one does."""
    points = sticky_square()
    naive = mean_dimension(knn_hypergraph(points, k=STICKY_K), samples=40, max_radius=6)
    corrected = mean_dimension(
        knn_hypergraph(points, k=STICKY_K, theiler_window=STICKY_BLOB_SIZE),
        samples=40,
        max_radius=6,
    )
    assert not math.isfinite(naive) or abs(naive - 2.0) > 0.5
    assert math.isfinite(corrected)
    assert abs(corrected - 2.0) <= 0.3


def test_auto_window_also_recovers_it_without_being_told_the_blob_size():
    points = sticky_square()
    corrected = mean_dimension(
        knn_hypergraph(points, k=STICKY_K, theiler_window="auto"), samples=40, max_radius=6
    )
    assert abs(corrected - 2.0) <= 0.3


# --------------------------------------------------------------------------
# 3. Time is time, not array position
# --------------------------------------------------------------------------


def test_graph_is_invariant_under_reordering_when_time_indices_are_supplied():
    points = sticky_square(n_anchors=40)
    n = len(points)
    perm = np.random.default_rng(0).permutation(n)
    shuffled = [points[i] for i in perm]

    in_order = knn_hypergraph(points, k=STICKY_K, theiler_window=17)
    reordered = knn_hypergraph(shuffled, k=STICKY_K, theiler_window=17, time_indices=perm.tolist())

    # Translate the reordered graph's edges back to original indices.
    translated = {
        tuple(sorted((int(perm[i]), int(perm[j])))) for i, j in reordered.edges
    }
    assert translated == set(in_order.edges)


def test_omitting_time_indices_on_a_reordered_cloud_excludes_the_wrong_edges():
    """The documented assumption has teeth: without `time_indices`, a shuffled
    cloud's window is applied to array position, which is not time."""
    points = sticky_square(n_anchors=40)
    n = len(points)
    perm = np.random.default_rng(1).permutation(n)
    shuffled = [points[i] for i in perm]

    right = knn_hypergraph(shuffled, k=STICKY_K, theiler_window=17, time_indices=perm.tolist())
    wrong = knn_hypergraph(shuffled, k=STICKY_K, theiler_window=17)
    assert right.edges != wrong.edges

    # The correct call excludes exactly the intra-blob pairs; the wrong one does not.
    assert all(abs(int(perm[i]) - int(perm[j])) >= 17 for i, j in right.edges)
    assert any(abs(int(perm[i]) - int(perm[j])) < 17 for i, j in wrong.edges)


def test_non_contiguous_time_indices_are_measured_in_time_not_position():
    """Sampling every third step: a window of 6 in TIME is a window of 2 in
    array position, and the two calls must agree."""
    points = sticky_square(n_anchors=30)
    n = len(points)
    strided = knn_hypergraph(
        points, k=STICKY_K, theiler_window=6, time_indices=[3 * i for i in range(n)]
    )
    equivalent = knn_hypergraph(points, k=STICKY_K, theiler_window=2)
    assert strided.edges == equivalent.edges


def test_time_indices_survive_deduplication():
    """A dropped duplicate must not renumber its survivors' time indices."""
    points = sticky_square(n_anchors=30)
    n = len(points)
    duplicated = points + [points[0], points[5]]
    times = list(range(n)) + [0, 5]
    hg = knn_hypergraph(duplicated, k=STICKY_K, dedupe=True, theiler_window=8, time_indices=times)
    plain = knn_hypergraph(points, k=STICKY_K, theiler_window=8)
    assert hg.edges == plain.edges


# --------------------------------------------------------------------------
# 4. The window is NOT safe to turn on everywhere -- asserted, not merely noted
# --------------------------------------------------------------------------


def test_window_harms_a_singly_traversed_closed_orbit():
    """On a 1-D curve traversed once, the temporal neighbours ARE the spatial
    neighbours, so the exclusion has nothing legitimate to find and wires the
    graph with long-range chords instead.

    This is the measured reason `theiler_window` defaults to 0 rather than to
    "auto", and the reason benchmark problems 03/04/07 must not be run with it.
    If a future change makes the window automatic, this test must fail first.
    """
    points = closed_orbit()
    naive = mean_dimension(knn_hypergraph(points, k=6), samples=40, max_radius=6)
    windowed = mean_dimension(
        knn_hypergraph(points, k=6, theiler_window=20), samples=40, max_radius=6
    )
    assert abs(naive - 1.0) <= 0.15  # the curve is 1-dimensional and the plain graph knows it
    assert not (math.isfinite(windowed) and abs(windowed - 1.0) <= 0.15)


# --------------------------------------------------------------------------
# 5. Argument validation and the starvation guard
# --------------------------------------------------------------------------


def test_a_window_that_starves_points_of_neighbours_raises():
    points = line(60)
    with pytest.raises(ValueError, match="eligible neighbours"):
        knn_hypergraph(points, k=6, theiler_window=40)


def test_rejects_negative_window():
    with pytest.raises(ValueError, match="non-negative"):
        knn_hypergraph(line(), k=6, theiler_window=-1)


def test_rejects_unknown_string_window():
    with pytest.raises(ValueError, match="'auto'"):
        knn_hypergraph(line(), k=6, theiler_window="theiler")


@pytest.mark.parametrize("bad", [2.5, True, None])
def test_rejects_non_integer_window(bad):
    with pytest.raises(ValueError, match="non-negative int or 'auto'"):
        knn_hypergraph(line(), k=6, theiler_window=bad)


def test_rejects_time_indices_of_the_wrong_length():
    with pytest.raises(ValueError, match="one integer per point"):
        knn_hypergraph(line(300), k=6, time_indices=list(range(10)))


def test_rejects_fractional_time_indices():
    with pytest.raises(ValueError, match="not fractional times"):
        knn_hypergraph(line(300), k=6, time_indices=[i + 0.5 for i in range(300)])


def test_time_indices_are_validated_even_when_the_window_is_off():
    """A malformed argument is a caller bug whether or not this call consumes it."""
    with pytest.raises(ValueError, match="one integer per point"):
        knn_hypergraph(line(300), k=6, theiler_window=0, time_indices=[1, 2, 3])
