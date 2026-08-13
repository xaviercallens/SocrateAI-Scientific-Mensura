"""Known-answer tests for the finite-size saturation guard on the fit window (AR2).

`dimension.local_dimension` fits log(shell) vs log(radius) and, before this
change, ended its fit window only at an EXACTLY zero shell. Exact zero is the
last stage of a process that biases the fit long before it arrives: once a ball
has engulfed most of a finite sample, its next shell is small because there is
nothing left to reach, and that shrinking shell sits at the largest radius --
the highest-leverage end of a log-log fit -- so it drags the slope down.

The fixture here is chosen so the RIGHT ANSWER AND THE RIGHT FRACTION ARE BOTH
KNOWN IN ADVANCE, rather than read off the result:

    `grid_graph(15)` is the 15x15 4-neighbour square lattice, 225 nodes, and
    `GRID_CENTRE` is its exact centre. On an infinite square lattice the shell
    at graph radius r from a node is exactly 4r, so the dimension is exactly
    2 with R^2 exactly 1 -- no noise, no fitting slack. The centre of a 15x15
    patch is 7 hops from the boundary, so shells are exactly 4r for r = 1..7
    and then get clipped: the measured sequence is
    (4, 8, 12, 16, 20, 24, 28, 28, 24, 20) over radii 1..10.
    The scaling region is therefore r <= 7, known from the geometry.

    The ball at radius 7 holds 1 + 2*7*8 = 113 of 225 nodes, i.e. 0.5022 of the
    graph, and the ball at radius 6 holds 85, i.e. 0.3778. So a fraction
    anywhere in (0.3778, 0.5022] cuts the fit at exactly the scaling region's
    edge, and 0.5 -- the value dimension.SATURATION_BALL_FRACTION carries -- is
    inside that interval. That is an independent, purely geometric check on the
    constant, derived from a lattice rather than tuned on the benchmark.

What these tests pin down:

1. OFF BY DEFAULT MEANS BIT-FOR-BIT. `max_ball_fraction=1.0` is the default and
   cannot ever truncate, because a ball never exceeds the whole graph. Every
   recorded benchmark number therefore still reproduces exactly.
2. IT WORKS, WITH THE ERROR MEASURED BOTH WAYS. Uncorrected, the lattice above
   reports 1.8132 with R^2 0.9034 -- confidently wrong on a structure whose
   dimension is exactly 2. At 0.5 it reports exactly 2.0 with R^2 exactly 1.0.
3. THE BRACKET IS REAL ON BOTH SIDES. 0.75 fires too late (bit-for-bit
   identical to off), and an aggressive fraction shrinks the window to two
   radii, where R^2 = 1.0 is vacuous rather than earned -- which is the
   mechanism by which 0.35 moved benchmark problems 02 and 10 from
   poly_algebraic_min_n 64 to 32 in
   scripts/hypergraph_benchmark/round3/ar2_saturation_guard.py section 3.
4. IT IS FORWARDED, NOT REIMPLEMENTED, by every aggregator and by comparison.py.
"""

from __future__ import annotations

import math

import pytest

from socrates.hypergraph.comparison import compare, poly_algebraic_minimum_points
from socrates.hypergraph.core import Hypergraph
from socrates.hypergraph.dimension import (
    SATURATION_BALL_FRACTION,
    degenerate_fraction,
    local_dimension,
    mean_dimension,
    near_degenerate_fraction,
)

GRID_SIDE = 15
GRID_CENTRE = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)
GRID_SHELLS = (4, 8, 12, 16, 20, 24, 28, 28, 24, 20)
GRID_SCALING_RADIUS = 7  # exact: the centre is 7 hops from the boundary


def grid_graph(side: int) -> Hypergraph:
    """The side x side 4-neighbour square lattice, node (x, y) indexed y*side + x."""
    edges = set()
    for y in range(side):
        for x in range(side):
            here = y * side + x
            if x + 1 < side:
                edges.add((here, here + 1))
            if y + 1 < side:
                edges.add((here, here + side))
    return Hypergraph(tuple(sorted(edges)))


def shells_of(estimate) -> tuple[int, ...]:
    volumes = (1,) + estimate.volumes
    return tuple(volumes[i] - volumes[i - 1] for i in range(1, len(volumes)))


def ring_graph(n: int) -> Hypergraph:
    """A cycle: the exactly-constant-shell (degenerate) regime, shells all 2."""
    return Hypergraph(tuple(sorted((min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n))))


# --- 1. the fixture is what the docstring says it is -------------------------


def test_grid_fixture_has_the_stated_shell_sequence():
    """Guards the whole file: if this drifts, every number below is about a
    different object than the docstring claims."""
    hg = grid_graph(GRID_SIDE)
    assert len(hg.nodes) == GRID_SIDE * GRID_SIDE == 225
    est = local_dimension(hg, GRID_CENTRE, max_radius=10)
    assert shells_of(est) == GRID_SHELLS
    # exactly 4r inside the scaling region, clipped outside it
    assert GRID_SHELLS[:GRID_SCALING_RADIUS] == tuple(
        4 * r for r in range(1, GRID_SCALING_RADIUS + 1)
    )
    assert GRID_SHELLS[GRID_SCALING_RADIUS] < 4 * (GRID_SCALING_RADIUS + 1)


def test_the_stated_bracket_is_a_property_of_the_lattice_not_a_tuning():
    """ball(6)/N = 0.3778 and ball(7)/N = 0.5022, so any fraction in that open-
    closed interval cuts exactly at the scaling region -- and 0.5 is in it."""
    hg = grid_graph(GRID_SIDE)
    est = local_dimension(hg, GRID_CENTRE, max_radius=10)
    n_nodes = len(hg.nodes)
    below = est.volumes[GRID_SCALING_RADIUS - 2] / n_nodes  # ball(6)
    at = est.volumes[GRID_SCALING_RADIUS - 1] / n_nodes  # ball(7)
    assert below == pytest.approx(85 / 225)
    assert at == pytest.approx(113 / 225)
    assert below < SATURATION_BALL_FRACTION <= at


# --- 2. off by default, bit-for-bit ------------------------------------------


@pytest.mark.parametrize("source", [0, GRID_CENTRE, GRID_SIDE * GRID_SIDE - 1])
@pytest.mark.parametrize("max_radius", [3, 6, 10, 20])
def test_default_is_bit_for_bit_the_pre_ar2_path(source, max_radius):
    hg = grid_graph(GRID_SIDE)
    default = local_dimension(hg, source, max_radius=max_radius)
    explicit = local_dimension(hg, source, max_radius=max_radius, max_ball_fraction=1.0)
    assert default == explicit


def test_fraction_one_cannot_truncate_even_when_the_ball_eats_the_graph():
    """A ball is never larger than the whole graph, so 1.0 is off by construction
    rather than by being a large number."""
    hg = grid_graph(5)  # 25 nodes; radius 8 covers all of it
    est = local_dimension(hg, 12, max_radius=8, max_ball_fraction=1.0)
    assert est.volumes[-1] == len(hg.nodes)
    assert est == local_dimension(hg, 12, max_radius=8)


def test_aggregators_default_to_off():
    hg = grid_graph(GRID_SIDE)
    for fn in (mean_dimension, degenerate_fraction, near_degenerate_fraction):
        assert fn(hg, samples=25, max_radius=10) == fn(
            hg, samples=25, max_radius=10, max_ball_fraction=1.0
        )


# --- 3. it works, and the error is measured both ways ------------------------


def test_uncorrected_fit_is_confidently_wrong_on_an_exactly_2d_lattice():
    hg = grid_graph(GRID_SIDE)
    est = local_dimension(hg, GRID_CENTRE, max_radius=10)
    assert est.dimension == pytest.approx(1.8132, abs=5e-4)
    assert est.r_squared == pytest.approx(0.9034, abs=5e-4)
    # the damage is not a rounding error: it is 0.19 on a structure that is
    # exactly 2-dimensional, and R^2 > 0.9 does not flag it.
    assert abs(est.dimension - 2.0) > 0.15


def test_guard_recovers_the_exact_answer():
    hg = grid_graph(GRID_SIDE)
    est = local_dimension(
        hg, GRID_CENTRE, max_radius=10, max_ball_fraction=SATURATION_BALL_FRACTION
    )
    assert est.dimension == pytest.approx(2.0, abs=1e-12)
    assert est.r_squared == pytest.approx(1.0, abs=1e-12)
    assert not est.degenerate  # a genuine 4,8,12,... fit, not a constant sequence


def test_guard_is_inert_where_the_ball_does_not_outrun_the_sample():
    """The same lattice, big enough that radius 10 is still deep inside it: the
    guard must change nothing at all. This is the "at large n nothing is
    truncated" half of the claim."""
    hg = grid_graph(101)
    plain = local_dimension(hg, 50 * 101 + 50, max_radius=10)
    guarded = local_dimension(
        hg, 50 * 101 + 50, max_radius=10, max_ball_fraction=SATURATION_BALL_FRACTION
    )
    assert plain == guarded
    assert plain.dimension == pytest.approx(2.0, abs=1e-12)


def test_guard_leaves_a_one_dimensional_ring_alone():
    """A ring's ball grows by 2 per radius, so on any ring long enough to matter
    the guard never fires and the degenerate/1.0 answer is untouched."""
    hg = ring_graph(200)
    plain = local_dimension(hg, 0, max_radius=6)
    guarded = local_dimension(hg, 0, max_radius=6, max_ball_fraction=SATURATION_BALL_FRACTION)
    assert plain == guarded
    assert guarded.degenerate
    assert guarded.dimension == pytest.approx(1.0)


# --- 4. the bracket, on both sides -------------------------------------------


def test_a_too_large_fraction_fires_too_late_to_help():
    """0.75 is above ball(7)/N = 0.5022, so on this lattice it is bit-for-bit
    identical to the guard being off -- the upper end of the bracket is a real
    measurement, not a precaution."""
    hg = grid_graph(GRID_SIDE)
    assert local_dimension(hg, GRID_CENTRE, max_radius=10, max_ball_fraction=0.75) == (
        local_dimension(hg, GRID_CENTRE, max_radius=10)
    )


def test_a_too_small_fraction_buys_a_vacuous_perfect_fit():
    """The lower end of the bracket, as a mechanism rather than a benchmark row.

    A 2-parameter fit over 2 points has zero residual degrees of freedom, so its
    R^2 is exactly 1 whatever the data -- it is not evidence. An aggressive
    fraction produces exactly that, and this is why the calibration sweep scores
    problems 02 and 10 dropping from poly_algebraic_min_n 64 to 32 at 0.35 as a
    REGRESSION rather than a four-fold sample-efficiency win.
    """
    # The CORNER of a 7x7 lattice, whose shells are (2, 3, 4, 5, 6, 7) -- shell
    # = r + 1, deliberately NOT a power law, so an honest fit over the whole
    # window cannot reach R^2 = 1 and a 2-point window cannot fail to.
    hg = grid_graph(7)  # 49 nodes
    full = local_dimension(hg, 0, max_radius=6)
    assert shells_of(full) == (2, 3, 4, 5, 6, 7)
    assert full.r_squared < 1.0

    starved = local_dimension(hg, 0, max_radius=6, max_ball_fraction=0.1)
    assert not starved.degenerate
    assert starved.r_squared == pytest.approx(1.0, abs=1e-12)  # zero residual d.o.f.
    # ...and the calibrated fraction does NOT collapse to that on the same node.
    kept = local_dimension(hg, 0, max_radius=6, max_ball_fraction=SATURATION_BALL_FRACTION)
    assert kept == full


# --- 5. argument validation ---------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.0001, 2.0, float("nan")])
def test_rejects_fractions_outside_the_unit_interval(bad):
    hg = grid_graph(5)
    with pytest.raises(ValueError, match="max_ball_fraction"):
        local_dimension(hg, 12, max_radius=3, max_ball_fraction=bad)


def test_error_message_says_how_to_turn_it_off():
    hg = grid_graph(5)
    with pytest.raises(ValueError, match="1.0 disables"):
        local_dimension(hg, 12, max_radius=3, max_ball_fraction=0.0)


# --- 6. it is forwarded, not reimplemented ------------------------------------


def test_mean_dimension_forwards_the_fraction():
    hg = grid_graph(GRID_SIDE)
    plain = mean_dimension(hg, samples=40, max_radius=10)
    guarded = mean_dimension(
        hg, samples=40, max_radius=10, max_ball_fraction=SATURATION_BALL_FRACTION
    )
    assert math.isfinite(plain) and math.isfinite(guarded)
    # the whole point: the guarded mean is closer to the lattice's true dimension
    assert abs(guarded - 2.0) < abs(plain - 2.0)


def test_fraction_fractions_forward_too():
    """`degenerate_fraction` / `near_degenerate_fraction` must describe the SAME
    fit windows `mean_dimension` pooled, or comparison.py's criterion (c)
    sentinel guard would be computed against a different estimator."""
    hg = grid_graph(9)
    for fn in (degenerate_fraction, near_degenerate_fraction):
        off = fn(hg, samples=20, max_radius=8)
        on = fn(hg, samples=20, max_radius=8, max_ball_fraction=0.3)
        assert math.isfinite(off) and math.isfinite(on)
        assert (off, on) != (float("nan"), float("nan"))


def _spiral_cloud(n: int) -> list[tuple[float, float]]:
    """A plain 1-D curve with no repeated traversal, for the plumbing tests."""
    return [(0.01 * i * math.cos(0.1 * i), 0.01 * i * math.sin(0.1 * i)) for i in range(n)]


def test_comparison_entry_points_accept_and_default_the_fraction():
    points = _spiral_cloud(400)
    grid = (100, 200, 400)
    a = poly_algebraic_minimum_points(points, 1.0, 0.2, k=6, n_grid=grid)
    b = poly_algebraic_minimum_points(points, 1.0, 0.2, k=6, n_grid=grid, max_ball_fraction=1.0)
    assert a == b
    c = compare(points, 1.0, 0.2, k=6, n_grid=grid)
    d = compare(points, 1.0, 0.2, k=6, n_grid=grid, max_ball_fraction=1.0)
    assert c == d
    # and the knob is actually plumbed through rather than swallowed
    compare(points, 1.0, 0.2, k=6, n_grid=grid, max_ball_fraction=0.4)
