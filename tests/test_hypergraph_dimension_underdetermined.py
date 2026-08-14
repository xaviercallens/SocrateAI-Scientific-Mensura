"""Finding N9: a 2-radius fit window scores r_squared = 1.0 by algebra.

A least-squares line through m points has m - 2 residual degrees of freedom,
so at m = 2 the residual is identically zero and `r_squared` is 1.0 for EVERY
input. The R^2 acceptance branch is therefore not weak at m = 2, it is inert:
it certifies whatever slope those two points happen to imply.

This was found by an independent skeptic verifying the F4 dimension-type
diagnostic (docs/MENSURA_BENCH_V2.md). It is the mechanism behind the
estimator returning *confident negative dimensions* on Cantor dust: that
graph fragments into hundreds of small components, most sampled nodes exhaust
their component within 2-3 hops, and every one of those windows was being
admitted by `is_well_fit`.

The defect is the same class as N8 (a fit-quality gate certifying an input it
cannot judge), and N8's own NEAR_CONSTANT_MIN_FIT_LENGTH floor guarded only
the near-constant branch -- a fix applied to one branch of a shared weakness
and not the other. See docs/LL.md lessons 8 and 9.
"""

from __future__ import annotations

import math

from socrates.hypergraph.core import Hypergraph
from socrates.hypergraph.dimension import (
    MIN_RSQUARED_FIT_LENGTH,
    DimensionEstimate,
    _log_log_fit,
    local_dimension,
    mean_dimension,
)


def _estimate(volumes: tuple[int, ...]) -> DimensionEstimate:
    """A DimensionEstimate built exactly as `local_dimension` builds one."""
    radii = tuple(range(1, len(volumes) + 1))
    fit = _log_log_fit([float(r) for r in radii], [float(v) for v in volumes])
    return DimensionEstimate(
        0,
        radii,
        volumes,
        dimension=fit.slope + 1.0,
        r_squared=fit.r_squared,
        degenerate=fit.degenerate,
        near_degenerate=fit.near_degenerate,
        shell_cv=fit.shell_cv,
        slope_bound=fit.slope_bound,
        underdetermined=fit.underdetermined,
    )


# The exact shell sequences the skeptic measured on a fragmented k-NN graph.
# Each reported r_squared = 1.000000 with degenerate and near_degenerate both
# False, and each was admitted by is_well_fit(0.9) before this fix.
N9_MEASURED_CASES = ((19, 27), (28, 31), (10, 50))


def test_two_point_fits_still_report_r_squared_one_by_algebra():
    """The defect itself, pinned so it stays visible: r_squared is NOT fixed,
    because it is arithmetically correct. Only its interpretation is."""
    for volumes in N9_MEASURED_CASES:
        fit = _log_log_fit([1.0, 2.0], [float(v) for v in volumes])
        assert fit.r_squared == 1.0
        assert not fit.degenerate
        assert not fit.near_degenerate


def test_two_point_fits_are_flagged_underdetermined_and_refused():
    """The fix. Wildly different slopes, all perfectly 'fit', none usable."""
    dimensions = []
    for volumes in N9_MEASURED_CASES:
        est = _estimate(volumes)
        assert est.underdetermined
        assert est.r_squared == 1.0
        assert not est.is_well_fit(0.9)
        assert not est.is_genuinely_well_fit(0.9)
        dimensions.append(est.dimension)
    # The slopes these certified windows implied span more than two whole
    # dimensions -- the point of the finding.
    assert max(dimensions) - min(dimensions) > 2.0


def test_a_two_point_window_can_certify_a_negative_dimension():
    """The Cantor-dust symptom, reduced to its cause. A shrinking shell
    sequence yields a negative dimension with a perfect r_squared."""
    est = _estimate((40, 8))
    assert est.dimension < 0.0
    assert est.r_squared == 1.0
    assert est.underdetermined
    assert not est.is_well_fit(0.9)


def test_three_point_windows_are_still_judged_on_their_merits():
    """The fix must not simply raise the floor on everything: at the minimum
    admissible length, a genuine power law is accepted and a bad fit rejected."""
    good = _estimate((4, 8, 12))  # shells 4, 8, 12 -> exactly dimension 2
    assert not good.underdetermined
    assert math.isclose(good.dimension, 2.0, abs_tol=1e-9)
    assert good.is_well_fit(0.9)

    bad = _estimate((10, 90, 12))  # no power law through these
    assert not bad.underdetermined
    assert bad.r_squared < 0.9
    assert not bad.is_well_fit(0.9)


def test_min_fit_length_is_the_smallest_with_a_nonzero_residual_dof():
    """3 is not arbitrary: it is the shortest window in which r_squared can
    fail at all."""
    assert MIN_RSQUARED_FIT_LENGTH == 3
    perfect_but_meaningless = _log_log_fit([1.0, 2.0], [3.0, 97.0])
    assert perfect_but_meaningless.r_squared == 1.0
    can_actually_fail = _log_log_fit([1.0, 2.0, 3.0], [10.0, 90.0, 12.0])
    assert can_actually_fail.r_squared < 0.9


def test_fragmented_graph_no_longer_pools_confident_nonsense():
    """End to end on the structure that produced the finding: a graph of many
    tiny disconnected components, where every node exhausts its component in
    two hops. Before the fix these nodes were admitted and averaged."""
    # 60 disjoint 3-node paths: every ball saturates at radius 2.
    edges = []
    for c in range(60):
        base = 3 * c
        edges.append((base, base + 1))
        edges.append((base + 1, base + 2))
    hg = Hypergraph(tuple(tuple(e) for e in edges))

    estimates = [local_dimension(hg, node, max_radius=6) for node in range(180)]
    # Every node's usable window is too short to judge...
    assert all(e.underdetermined for e in estimates)
    # ...and none of them is admitted, however perfect its r_squared looks.
    assert not any(e.is_well_fit(0.9) for e in estimates)
    # So the aggregate honestly refuses to answer rather than averaging noise.
    assert math.isnan(mean_dimension(hg, samples=60, max_radius=6))
