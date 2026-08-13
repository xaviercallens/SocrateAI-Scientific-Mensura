"""Known-answer tests for the classical (traditional) correlation-dimension baseline.

This is the reference implementation everything in dimension.py/pointcloud.py
is compared against -- it needs its own known-answer verification just like
the module it is a baseline for, not a free pass because it is "the
traditional method."
"""

from __future__ import annotations

import math
import random

import pytest

from socrates.hypergraph.baseline import correlation_dimension, minimum_points_for_target_accuracy


def test_correlation_dimension_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        correlation_dimension([(0.0, 0.0), (1.0, 1.0)])


def test_correlation_dimension_rejects_degenerate_identical_points():
    with pytest.raises(ValueError, match="identical"):
        correlation_dimension([(1.0, 1.0)] * 10)


def test_circle_correlation_dimension_is_close_to_one():
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 1000 for i in range(1000))]
    est = correlation_dimension(points)
    assert est.is_well_fit()
    assert est.dimension == pytest.approx(1.0, abs=0.1)


def test_filled_square_correlation_dimension_is_close_to_two():
    rng = random.Random(0)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(1000)]
    est = correlation_dimension(points)
    assert est.is_well_fit()
    assert est.dimension == pytest.approx(2.0, abs=0.15)


def test_filled_cube_correlation_dimension_is_close_to_three():
    rng = random.Random(0)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(2000)]
    est = correlation_dimension(points)
    assert est.is_well_fit()
    assert est.dimension == pytest.approx(3.0, abs=0.2)


def test_minimum_points_for_target_accuracy_finds_stable_convergence():
    # A clean circle should converge to within a loose tolerance at a small n
    # and STAY converged at every larger n in the grid (the stability check
    # this function performs, unlike a single lucky crossing).
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 2000 for i in range(2000))]
    n = minimum_points_for_target_accuracy(
        points, true_dimension=1.0, tolerance=0.15, n_grid=(50, 100, 200, 400, 800, 1600)
    )
    assert n is not None
    assert n <= 400  # a circle is an easy target; should converge early


def test_minimum_points_for_target_accuracy_returns_none_when_unreachable():
    rng = random.Random(0)
    points = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(500)]
    # An impossibly tight tolerance against the wrong target dimension must
    # not be satisfiable.
    n = minimum_points_for_target_accuracy(
        points, true_dimension=1.0, tolerance=1e-6, n_grid=(100, 200, 400)
    )
    assert n is None
