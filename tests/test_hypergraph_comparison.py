"""Tests for the apples-to-apples Poly-Algebraic vs traditional comparison."""

from __future__ import annotations

import math
import random

import pytest

from socrates.hypergraph.comparison import ComparisonResult, compare, poly_algebraic_minimum_points


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
