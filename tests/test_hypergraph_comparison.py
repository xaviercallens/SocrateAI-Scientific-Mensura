"""Tests for the apples-to-apples Poly-Algebraic vs traditional comparison."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from socrates.hypergraph.baseline import minimum_points_for_target_accuracy
from socrates.hypergraph.comparison import (
    DEFAULT_ACCURACY_MARGIN,
    DEFAULT_MAX_SENTINEL_FRACTION,
    AccuracyComparisonResult,
    ComparisonResult,
    compare,
    compare_accuracy_at_max_n,
    poly_algebraic_minimum_points,
)


def test_poly_algebraic_minimum_points_converges_on_a_circle():
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 1600 for i in range(1600))]
    n = poly_algebraic_minimum_points(
        points, true_dimension=1.0, tolerance=0.1, k=6, n_grid=(50, 100, 200, 400, 800, 1600)
    )
    assert n is not None
    assert n <= 400


def test_compare_reports_consistent_result_on_a_circle():
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 1600 for i in range(1600))]
    result = compare(points, true_dimension=1.0, tolerance=0.1, k=6, n_grid=(100, 200, 400, 800))
    assert isinstance(result, ComparisonResult)
    assert result.poly_algebraic_min_n is not None
    assert result.traditional_min_n is not None
    assert result.poly_algebraic_estimate_at_max_n == pytest.approx(1.0, abs=0.1)
    assert result.traditional_estimate_at_max_n == pytest.approx(1.0, abs=0.15)
    # compute_savings_fraction must be consistent with the two min_n values
    expected_savings = 1.0 - (result.poly_algebraic_min_n / result.traditional_min_n)
    assert result.compute_savings_fraction == pytest.approx(expected_savings)


def test_comparison_result_wins_and_savings_are_none_when_a_method_never_converges():
    # Construct a result object directly (not via compare()) to test the
    # None-propagation logic in isolation from any particular point cloud.
    result = ComparisonResult(
        true_dimension=1.0,
        tolerance=0.01,
        poly_algebraic_min_n=200,
        traditional_min_n=None,  # never converged
        poly_algebraic_estimate_at_max_n=1.0,
        traditional_estimate_at_max_n=float("nan"),
        max_n_tested=1600,
    )
    assert result.poly_algebraic_wins is False  # cannot "win" against a non-convergent baseline
    assert result.compute_savings_fraction is None


def test_comparison_result_savings_is_positive_when_poly_algebraic_needs_fewer_points():
    result = ComparisonResult(
        true_dimension=1.0,
        tolerance=0.1,
        poly_algebraic_min_n=100,
        traditional_min_n=400,
        poly_algebraic_estimate_at_max_n=1.0,
        traditional_estimate_at_max_n=1.02,
        max_n_tested=1600,
    )
    assert result.poly_algebraic_wins is True
    assert result.compute_savings_fraction == pytest.approx(0.75)


def test_poly_algebraic_minimum_points_handles_duplicate_rich_input_via_dedupe():
    # poly_algebraic_minimum_points must not crash on a point cloud containing
    # near-duplicate clusters (unlike knn_hypergraph's default, which raises) --
    # it auto-dedupes internally so it can run unattended over many configs.
    rng = random.Random(0)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(300)]
    points += [(x + 1e-9, y + 1e-9) for x, y in points[:50]]  # 50 near-duplicates
    n = poly_algebraic_minimum_points(
        points, true_dimension=2.0, tolerance=0.3, k=10, n_grid=(100, 200, 300)
    )
    # Should not raise; may or may not converge depending on data, but must
    # return either an int or None, never propagate DuplicatePointsError.
    assert n is None or isinstance(n, int)


# =============================================================================
# Criterion (c): asymptotic accuracy at fixed n (H3 / finding R2-F9)
# =============================================================================
#
# The four integration cases below are KNOWN-ANSWER tests in the same sense as
# the rest of this suite: each point cloud has a dimension that is known
# independently of either estimator, and the expected verdict follows from
# published/derivable facts about that cloud, not from whatever the code
# currently prints.
#
#   clear win   -- planar Brownian motion has Hausdorff dimension 2 (Taylor
#                  1953). The correlation sum badly underestimates it at
#                  attainable n; the shell-growth estimator does not.
#   clear loss  -- uniformly sampled points in a 3-cube have dimension 3. The
#                  correlation sum recovers it; the shell-growth estimator has
#                  a documented downward bias (finding F5) and does not.
#   tie         -- uniformly sampled points in a unit square have dimension 2
#                  and BOTH methods get it, so neither may claim a win.
#   degeneracy  -- a k-NN graph on a circle is an exact circulant ring lattice,
#                  so the shell-growth estimator returns exactly 1.0 by
#                  construction (finding F1). It is closer to the truth than
#                  the baseline and that must NOT score as a win.
#
# Point counts are the smallest at which each expected verdict holds with a
# real cushion (see scripts/hypergraph_benchmark/round3/h3_margin_calibration.py
# for the redraw spread), so the suite stays fast.


def _brownian_2d(n: int, seed: int = 42) -> list[tuple[float, ...]]:
    """Planar Brownian path subsampled by stride 4; true dimension 2."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 1.0, size=(n * 4, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)])
    return [tuple(p) for p in path[::4][:n]]


def _uniform(n: int, dim: int, seed: int) -> list[tuple[float, ...]]:
    """Uniform sample of the unit `dim`-cube; true dimension `dim`."""
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(0.0, 1.0, size=(n, dim))]


def _circle(n: int) -> list[tuple[float, ...]]:
    """Evenly spaced points on a circle; true dimension 1, and an exact ring
    lattice under k-NN (the finding-F1 degenerate regime)."""
    return [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]


def _accuracy_result(**overrides) -> AccuracyComparisonResult:
    """An AccuracyComparisonResult with plausible measurements, for testing the
    verdict logic in isolation from any point cloud."""
    fields = {
        "true_dimension": 2.0,
        "tolerance": 0.15,
        "margin": DEFAULT_ACCURACY_MARGIN,
        "max_sentinel_fraction": DEFAULT_MAX_SENTINEL_FRACTION,
        "n_requested": 1600,
        "n_evaluated": 1600,
        "poly_algebraic_n_nodes": 1600,
        "poly_algebraic_estimate": 1.96,
        "traditional_estimate": 1.59,
        "poly_algebraic_degenerate_fraction": 0.0,
        "poly_algebraic_near_degenerate_fraction": 0.0,
        "traditional_r_squared": 0.998,
    }
    fields.update(overrides)
    return AccuracyComparisonResult(**fields)


# --- known-answer case 1: poly should clearly WIN ----------------------------


def test_accuracy_criterion_scores_a_win_on_planar_brownian_motion():
    # True dimension 2. The shell-growth estimator lands within ~0.04; the
    # correlation sum is short by ~0.41 -- the finding R2-F9 situation, at a
    # point count small enough to keep the test fast.
    points = _brownian_2d(1600)
    result = compare_accuracy_at_max_n(points, 2.0, 1600, k=10, tolerance=0.15)

    assert result.n_evaluated == 1600
    assert result.poly_algebraic_abs_error < 0.15
    assert result.traditional_abs_error > 0.30
    # The win is on a non-degenerate fit, not on the F1 sentinel.
    assert result.poly_algebraic_sentinel_fraction <= DEFAULT_MAX_SENTINEL_FRACTION
    assert result.more_accurate == "poly_algebraic"
    assert result.poly_algebraic_accuracy_win is True
    assert result.traditional_accuracy_win is False
    assert result.verdict_reason.startswith("win:")
    # The baseline is confidently wrong, which is the point of the finding:
    # a high R^2 is not evidence that the traditional estimate is right.
    assert result.traditional_r_squared > 0.99


def test_accuracy_criterion_credits_a_case_the_convergence_criterion_cannot():
    # The specific structural gap H3 exists to close. On this cloud the
    # traditional method never stably converges, so criterion (a) has no
    # "fewer points" comparison available and correctly reports no win --
    # while criterion (c) has a perfectly well-defined answer.
    points = _brownian_2d(1600)
    n_grid = (100, 200, 400, 800, 1600)

    assert minimum_points_for_target_accuracy(points, 2.0, 0.15, n_grid=n_grid) is None
    convergence = compare(points, 2.0, 0.15, k=10, n_grid=n_grid)
    assert convergence.traditional_min_n is None
    assert convergence.poly_algebraic_min_n is not None  # poly DID converge
    assert convergence.poly_algebraic_wins is False
    assert convergence.compute_savings_fraction is None

    accuracy = compare_accuracy_at_max_n(points, 2.0, 1600, k=10, tolerance=0.15)
    assert accuracy.poly_algebraic_accuracy_win is True


# --- known-answer case 2: poly should clearly LOSE ---------------------------


def test_accuracy_criterion_scores_a_loss_on_a_uniform_3_cube():
    # True dimension 3. The correlation sum recovers ~2.95; the shell-growth
    # estimator returns ~2.13 (finding F5's downward bias, severe in 3D).
    points = _uniform(400, 3, seed=1)
    result = compare_accuracy_at_max_n(points, 3.0, 400, k=12, tolerance=0.15)

    assert result.traditional_abs_error < 0.15
    assert result.poly_algebraic_abs_error > 0.5
    assert result.more_accurate == "traditional"
    assert result.poly_algebraic_accuracy_win is False
    assert result.traditional_accuracy_win is True
    assert "poly's own error" in result.verdict_reason


# --- known-answer case 3: TIE ------------------------------------------------


def test_accuracy_criterion_scores_a_tie_when_both_methods_are_right():
    # True dimension 2, and both methods get it (poly ~1.92, traditional
    # ~1.96). Neither may claim a win off a 0.03 difference. The margin is
    # widened past the default here so the tie verdict does not ride on that
    # 0.03 gap staying exactly where it is under a reseed.
    points = _uniform(800, 2, seed=1)
    result = compare_accuracy_at_max_n(points, 2.0, 800, k=10, tolerance=0.15, margin=0.2)

    assert result.poly_algebraic_abs_error < 0.15
    assert result.traditional_abs_error < 0.15
    assert result.more_accurate == "tie"
    assert result.poly_algebraic_accuracy_win is False
    assert result.traditional_accuracy_win is False


# --- known-answer case 4: the F1 degeneracy must not be scored as a win ------


def test_ring_lattice_degeneracy_is_not_scored_as_an_accuracy_win():
    # A k-NN graph on a circle is an exact circulant ring lattice, so every
    # shell sequence is constant and the estimator returns exactly 1.0 whatever
    # the data says. Its error against the true dimension 1 is therefore 0.0000
    # and it beats the baseline by every numeric condition -- (c1), (c3), (c4)
    # and (c5) all pass. Only the degeneracy guard (c2) stops it. This is the
    # test that distinguishes criterion (c) from "poly's number was closer".
    points = _circle(400)
    result = compare_accuracy_at_max_n(points, 1.0, 400, k=6, tolerance=0.05, margin=0.05)

    assert result.poly_algebraic_estimate == pytest.approx(1.0, abs=1e-9)
    assert result.poly_algebraic_degenerate_fraction == pytest.approx(1.0)
    assert result.poly_algebraic_abs_error < result.traditional_abs_error
    assert result.more_accurate == "poly_algebraic"  # closer, and it means nothing
    assert result.poly_algebraic_accuracy_win is False
    assert "sentinel fraction" in result.verdict_reason
    assert "F1" in result.verdict_reason

    # Confirm every OTHER condition really did pass, so the test is pinning the
    # degeneracy guard specifically and would fail if that guard were removed.
    ungated = _accuracy_result(
        true_dimension=1.0,
        tolerance=0.05,
        margin=0.05,
        poly_algebraic_estimate=result.poly_algebraic_estimate,
        traditional_estimate=result.traditional_estimate,
        poly_algebraic_degenerate_fraction=0.0,
    )
    assert ungated.poly_algebraic_accuracy_win is True


def test_near_degenerate_nodes_also_count_toward_the_sentinel_guard():
    # `near_degenerate` pins the dimension near 1.0 for the same structural
    # reason `degenerate` pins it at exactly 1.0, so both must feed the guard.
    split = _accuracy_result(
        true_dimension=1.0,
        tolerance=0.05,
        margin=0.05,
        poly_algebraic_estimate=1.0,
        traditional_estimate=1.2,
        poly_algebraic_degenerate_fraction=0.03,
        poly_algebraic_near_degenerate_fraction=0.03,
    )
    assert split.poly_algebraic_sentinel_fraction == pytest.approx(0.06)
    assert split.poly_algebraic_accuracy_win is False  # 0.06 > 0.05, only when summed


def test_sentinel_guard_boundary_is_inclusive():
    at_threshold = _accuracy_result(poly_algebraic_degenerate_fraction=0.05)
    just_over = _accuracy_result(poly_algebraic_degenerate_fraction=0.05 + 1e-9)
    assert at_threshold.poly_algebraic_accuracy_win is True
    assert just_over.poly_algebraic_accuracy_win is False


# --- the verdict logic itself, independent of any point cloud ----------------


def test_accuracy_win_requires_both_estimates_to_be_finite():
    # A baseline that produces no number at all is a baseline failure. Scoring
    # it as a poly win is exactly what standing rule 2 forbids, and is the same
    # rule ComparisonResult.poly_algebraic_wins applies to a None min_n.
    no_traditional = _accuracy_result(traditional_estimate=float("nan"))
    assert no_traditional.more_accurate == "poly_algebraic"
    assert no_traditional.poly_algebraic_accuracy_win is False
    assert "no finite estimate" in no_traditional.verdict_reason

    no_poly = _accuracy_result(poly_algebraic_estimate=float("nan"))
    assert no_poly.more_accurate == "traditional"
    assert no_poly.poly_algebraic_accuracy_win is False
    assert no_poly.traditional_accuracy_win is False

    neither = _accuracy_result(
        poly_algebraic_estimate=float("nan"), traditional_estimate=float("nan")
    )
    assert neither.more_accurate == "undetermined"
    assert neither.poly_algebraic_accuracy_win is False


def test_accuracy_win_requires_poly_to_be_accurate_not_merely_less_wrong():
    # poly 0.40 off, traditional 0.90 off: poly is much closer, and still wrong.
    less_wrong = _accuracy_result(
        poly_algebraic_estimate=1.60, traditional_estimate=1.10, tolerance=0.15
    )
    assert less_wrong.more_accurate == "poly_algebraic"
    assert less_wrong.poly_algebraic_accuracy_win is False
    assert "being less wrong than the baseline is not accuracy" in less_wrong.verdict_reason


def test_accuracy_win_requires_the_gap_to_exceed_the_margin():
    # poly 0.02 off, traditional 0.20 off, tolerance 0.10: both tolerance
    # conditions pass, but a 0.18 gap is inside the measured redraw spread.
    narrow = _accuracy_result(
        poly_algebraic_estimate=1.98, traditional_estimate=1.80, tolerance=0.10, margin=0.25
    )
    assert narrow.poly_algebraic_accuracy_win is False
    assert "below the margin" in narrow.verdict_reason
    assert narrow.more_accurate == "tie"  # 0.18 < margin 0.25

    # Same measurements, margin lowered below the gap: now a win. The margin is
    # a stated parameter of the criterion, not a hidden one.
    wide = _accuracy_result(
        poly_algebraic_estimate=1.98, traditional_estimate=1.80, tolerance=0.10, margin=0.10
    )
    assert wide.poly_algebraic_accuracy_win is True


def test_accuracy_criterion_cannot_be_manufactured_by_moving_the_tolerance():
    # The anti-gaming property claimed in the docstring, checked rather than
    # asserted. Conditions (c3) and (c4) pull in opposite directions, so for
    # fixed measurements a win exists only for tolerances in
    # [poly_err, trad_err) -- and only if that interval is at least `margin`
    # wide. Nothing about moving the tolerance manufactures a win.
    # Errors of exactly 0.25 and 0.75, chosen so the interval endpoints are
    # exactly representable and the boundary assertions below mean what they say.
    poly_err, trad_err = 0.25, 0.75
    base = dict(
        true_dimension=2.0,
        poly_algebraic_estimate=2.0 - poly_err,
        traditional_estimate=2.0 - trad_err,
        margin=0.25,
    )
    winning = [
        t / 100
        for t in range(0, 101)
        if _accuracy_result(tolerance=t / 100, **base).poly_algebraic_accuracy_win
    ]
    assert winning == pytest.approx([t / 100 for t in range(25, 75)])
    # Half-open: closed at poly's error, open at the traditional method's.
    assert _accuracy_result(tolerance=poly_err, **base).poly_algebraic_accuracy_win is True
    assert _accuracy_result(tolerance=trad_err, **base).poly_algebraic_accuracy_win is False

    # Now shrink the gap below the margin. NO tolerance yields a win.
    narrow = dict(base, traditional_estimate=2.0 - 0.40)
    assert not any(
        _accuracy_result(tolerance=t / 100, **narrow).poly_algebraic_accuracy_win
        for t in range(0, 101)
    )


def test_more_accurate_never_contradicts_the_win_flag_at_the_margin_boundary():
    # A gap sitting exactly on the margin must not be a "tie" ranking and a
    # win verdict at the same time: (c5) is `>= margin`, so the ranking's
    # tie band has to be strictly `< margin`.
    on_boundary = _accuracy_result(
        true_dimension=2.0,
        poly_algebraic_estimate=2.0,  # error 0.00
        traditional_estimate=1.75,  # error 0.25 == margin
        tolerance=0.10,
        margin=0.25,
    )
    assert on_boundary.traditional_abs_error - on_boundary.poly_algebraic_abs_error == 0.25
    assert on_boundary.poly_algebraic_accuracy_win is True
    assert on_boundary.more_accurate == "poly_algebraic"

    # And with margin=0 two identical errors are a tie, not a ranking.
    identical = _accuracy_result(poly_algebraic_estimate=1.9, traditional_estimate=2.1, margin=0.0)
    assert identical.poly_algebraic_abs_error == identical.traditional_abs_error
    assert identical.more_accurate == "tie"
    assert identical.poly_algebraic_accuracy_win is False
    assert identical.traditional_accuracy_win is False


def test_traditional_accuracy_win_is_the_mirror_and_can_fire():
    trad_better = _accuracy_result(
        true_dimension=3.0,
        poly_algebraic_estimate=2.13,
        traditional_estimate=2.95,
        tolerance=0.15,
    )
    assert trad_better.traditional_accuracy_win is True
    assert trad_better.poly_algebraic_accuracy_win is False
    # Never both at once: the tolerance conditions are mutually exclusive.
    poly_better = _accuracy_result()
    assert poly_better.poly_algebraic_accuracy_win is True
    assert poly_better.traditional_accuracy_win is False


def test_accuracy_result_errors_are_derived_not_stored():
    result = _accuracy_result(
        true_dimension=2.0, poly_algebraic_estimate=2.05, traditional_estimate=1.6357
    )
    assert result.poly_algebraic_abs_error == pytest.approx(0.05)
    assert result.traditional_abs_error == pytest.approx(0.3643)


# --- input handling ----------------------------------------------------------


def test_compare_accuracy_at_max_n_clamps_max_n_to_the_available_points():
    points = _uniform(300, 2, seed=2)
    result = compare_accuracy_at_max_n(points, 2.0, 10_000, k=10, tolerance=0.3)
    assert result.n_requested == 10_000
    assert result.n_evaluated == 300
    assert result.poly_algebraic_n_nodes <= 300


def test_compare_accuracy_at_max_n_rejects_undefined_inputs():
    points = _uniform(100, 2, seed=3)
    with pytest.raises(ValueError):
        compare_accuracy_at_max_n(points, 2.0, 2, k=10, tolerance=0.3)
    with pytest.raises(ValueError):
        compare_accuracy_at_max_n(points, 2.0, 100, k=0, tolerance=0.3)
    with pytest.raises(ValueError):
        compare_accuracy_at_max_n(points, 2.0, 100, k=10, tolerance=-0.1)
    with pytest.raises(ValueError):
        compare_accuracy_at_max_n(points, 2.0, 100, k=10, tolerance=0.3, margin=-0.1)
    with pytest.raises(ValueError):
        compare_accuracy_at_max_n(points, 2.0, 100, k=10, tolerance=0.3, max_sentinel_fraction=1.5)
    with pytest.raises(ValueError):
        compare_accuracy_at_max_n([(0.0, 0.0), (1.0, 1.0)], 2.0, 100, k=1, tolerance=0.3)


def test_compare_accuracy_at_max_n_records_a_failed_poly_fit_as_nan_not_an_exception():
    # k >= n makes knn_hypergraph raise; the accuracy comparison must record
    # "no answer" rather than propagate, since it runs unattended over a suite.
    points = _uniform(50, 2, seed=4)
    result = compare_accuracy_at_max_n(points, 2.0, 50, k=60, tolerance=0.3)
    assert math.isnan(result.poly_algebraic_estimate)
    assert result.poly_algebraic_n_nodes == 0
    assert math.isfinite(result.traditional_estimate)  # baseline still answered
    assert result.more_accurate == "traditional"
    assert result.poly_algebraic_accuracy_win is False


def test_compare_accuracy_at_max_n_does_not_require_either_method_to_converge():
    # 300 points of Brownian motion: neither method is near converged in the
    # criterion (a) sense at this n. The accuracy comparison must still return
    # a fully-populated result rather than None-ing out the way min_n does.
    points = _brownian_2d(300)
    result = compare_accuracy_at_max_n(points, 2.0, 300, k=10, tolerance=0.3)
    assert math.isfinite(result.poly_algebraic_estimate)
    assert math.isfinite(result.traditional_estimate)
    assert result.more_accurate in {"poly_algebraic", "traditional", "tie"}
    assert isinstance(result.poly_algebraic_accuracy_win, bool)
