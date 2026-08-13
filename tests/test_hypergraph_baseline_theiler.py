"""Known-answer tests for the Theiler-window correction to the traditional baseline (H1).

Theiler, J. (1986), Phys. Rev. A 34(3), 2427-2432: on a *trajectory*, pairs
that are close in original time index are close in space for a trivial
reason, and counting them adds a roughly constant offset to C(r). That offset
flattens the log-log curve at the small-r end, so the naive Grassberger-
Procaccia slope is biased DOWNWARD. Excluding pairs with |t_i - t_j| < W
removes the offset and moves the estimate back UP toward the truth.

Two things these tests are here to pin down, because both are load-bearing
for the benchmark's honesty:

1. The correction is OFF by default, and "off" means bit-for-bit identical to
   the pre-H1 estimator -- not "close enough". Every number already recorded
   in docs/POLY_ALGEBRAIC_BENCHMARK.md was produced with the window off and
   must still reproduce exactly.
2. The correction makes the TRADITIONAL method better, i.e. a harder baseline
   for the shell-growth estimator to beat. `test_theiler_window_makes_the_
   baseline_harder_to_beat_on_lorenz` asserts that explicitly, so that a
   future edit that quietly weakened the baseline would fail a test rather
   than flatter the headline claim.

The tests live in their own file so the original 47-test hypergraph suite
(tests/test_hypergraph_baseline.py included) stays untouched.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from socrates.hypergraph.baseline import (
    correlation_dimension,
    minimum_points_for_target_accuracy,
    theiler_window_from_autocorrelation,
)

# --------------------------------------------------------------------------
# Synthetic fixtures with a KNOWN answer and DELIBERATE temporal autocorrelation
# --------------------------------------------------------------------------

STICKY_BLOB_SIZE = 16


def sticky_square(n_anchors: int = 200, blob: int = STICKY_BLOB_SIZE, eps: float = 1e-3, seed=7):
    """A point set whose true D2 is 2, sampled by a deliberately "sticky" scanner.

    `n_anchors` anchors are drawn i.i.d. uniform from the unit square -- so the
    SET is a 2-dimensional sample, exactly. But the SEQUENCE visits them in
    runs: `blob` consecutive samples sit within ~eps of the same anchor before
    the sequence jumps to the next anchor. That is temporal autocorrelation
    with a known, exact extent: every spurious pair has |t_i - t_j| < blob, and
    every pair with |t_i - t_j| >= blob is a genuine anchor-to-anchor pair.

    So the known answer is sharp in both directions: W < blob must leave some
    spurious pairs behind, and W = blob must remove all of them and recover
    D2 ~= 2. Nothing about the fix is fitted to the data.
    """
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(0, 1, size=(n_anchors, 2))
    return [tuple(a + rng.normal(0.0, eps, size=2)) for a in anchors for _ in range(blob)]


def lorenz_trajectory(n: int = 3000, dt: float = 0.005, transient: int = 5000):
    """Oversampled Lorenz attractor (RK4). Accepted correlation dimension ~2.05.

    dt=0.005 is fine enough that consecutive samples are strongly correlated
    (that is the point) while the run still covers the attractor.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def deriv(v):
        x, y, z = v
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def step(v):
        k1 = deriv(v)
        k2 = deriv(v + dt / 2 * k1)
        k3 = deriv(v + dt / 2 * k2)
        k4 = deriv(v + dt * k3)
        return v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    v = np.array([1.0, 1.0, 1.0])
    for _ in range(transient):
        v = step(v)
    out = []
    for _ in range(n):
        v = step(v)
        out.append(tuple(v))
    return out


def brute_force_correlation_sums(points, window, *, n_radii=20, r_min_frac=0.01, r_max_frac=0.2):
    """Definitional O(n^2) reference: the correlation sum over pairs with |i-j| >= W.

    Deliberately written the slow, obvious way, from the formula rather than
    from the optimized implementation, so agreement is evidence and not a
    tautology.
    """
    arr = np.asarray(points, dtype=float)
    n = len(arr)
    diag = float(np.sqrt(np.sum((arr.max(axis=0) - arr.min(axis=0)) ** 2)))
    radii = np.logspace(math.log10(r_min_frac * diag), math.log10(r_max_frac * diag), n_radii)
    kept = [
        float(np.linalg.norm(arr[i] - arr[j]))
        for i in range(n)
        for j in range(i + 1, n)
        if (j - i) >= window
    ]
    kept_sorted = np.sort(np.array(kept))
    return radii, np.searchsorted(kept_sorted, radii, side="right") / len(kept_sorted)


# --------------------------------------------------------------------------
# 1. The default must not move -- bit-for-bit, not approximately
# --------------------------------------------------------------------------


def test_default_and_window_zero_and_window_one_are_bit_for_bit_identical():
    points = sticky_square()
    default = correlation_dimension(points)
    off = correlation_dimension(points, theiler_window=0)
    conventional_off = correlation_dimension(points, theiler_window=1)

    # Exact equality, not pytest.approx: this is the reproducibility guarantee
    # for every number already recorded in the benchmark docs.
    assert off.correlation_sums == default.correlation_sums
    assert off.dimension == default.dimension
    assert off.r_squared == default.r_squared
    # W=1 excludes only |t_i - t_j| < 1, i.e. self-pairs, which were already
    # excluded -- so it must be the same numbers as W=0, by construction.
    assert conventional_off.correlation_sums == default.correlation_sums
    assert conventional_off.dimension == default.dimension


def test_window_off_reports_zero_provenance():
    est = correlation_dimension(sticky_square())
    assert est.theiler_window == 0
    assert est.n_excluded_pairs == 0


def test_minimum_points_default_is_unchanged_by_the_new_parameters():
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 2000 for i in range(2000))]
    grid = (50, 100, 200, 400, 800, 1600)
    assert minimum_points_for_target_accuracy(
        points, true_dimension=1.0, tolerance=0.15, n_grid=grid
    ) == minimum_points_for_target_accuracy(
        points, true_dimension=1.0, tolerance=0.15, n_grid=grid, theiler_window=0
    )


# --------------------------------------------------------------------------
# 2. The known-answer regression: the window moves the estimate the RIGHT way
# --------------------------------------------------------------------------


def test_theiler_window_recovers_the_true_dimension_on_a_sticky_sampler():
    """KNOWN ANSWER. True D2 = 2 (i.i.d. uniform anchors in the unit square).

    Blob size 16 means every spurious pair has |t_i - t_j| < 16. So:
      * W=0  -- all 200 * C(16,2) = 24000 within-blob pairs are counted at
        d ~ 1e-3, an offset that swamps the genuine C(r) at the small-r end
        and drags the slope well below 2;
      * W=16 -- exactly those pairs are gone and the slope returns to ~2.
    """
    points = sticky_square()
    naive = correlation_dimension(points, theiler_window=0)
    corrected = correlation_dimension(points, theiler_window=STICKY_BLOB_SIZE)

    # The bias is real and large, and it is DOWNWARD, as Theiler predicts.
    assert naive.dimension < 1.5
    # ... and the window removes it, recovering the known answer.
    assert corrected.dimension == pytest.approx(2.0, abs=0.15)
    assert corrected.dimension > naive.dimension
    # The naive fit looks respectable while being wrong by ~0.75 -- which is
    # precisely why this correction matters for a fair baseline.
    assert naive.is_well_fit()
    assert corrected.r_squared > naive.r_squared

    # Provenance is recorded, and the number of dropped pairs is the exact
    # combinatorial count of pairs within 16 steps in a 3200-point sequence.
    n = len(points)
    w = STICKY_BLOB_SIZE
    assert corrected.theiler_window == w
    assert corrected.n_excluded_pairs == (w - 1) * n - w * (w - 1) // 2


def test_dimension_increases_monotonically_as_the_window_closes_on_the_blob_size():
    """Partial windows must give partial corrections -- no all-or-nothing cliff."""
    points = sticky_square()
    dims = [correlation_dimension(points, theiler_window=w).dimension for w in (0, 4, 8, 16)]
    assert all(later > earlier for earlier, later in itertools.pairwise(dims))


def test_theiler_window_makes_the_baseline_harder_to_beat_on_lorenz():
    """The fairness point, asserted rather than asserted-in-prose.

    Lorenz D2 ~= 2.05. Oversampled at dt=0.005 the naive correlation sum reads
    well low; the window moves it substantially closer to the accepted value.
    A future change that made the baseline *worse* here would fail this test.
    """
    points = lorenz_trajectory()
    true_d2 = 2.05
    naive = correlation_dimension(points, theiler_window=0)
    corrected = correlation_dimension(points, theiler_window="auto")

    assert naive.dimension < true_d2 - 0.1  # biased low, as predicted
    assert abs(corrected.dimension - true_d2) < abs(naive.dimension - true_d2)
    assert corrected.dimension == pytest.approx(true_d2, abs=0.2)


# --------------------------------------------------------------------------
# 3. It computes what it says it computes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", [2, 5, 17, 50])
def test_correlation_sums_match_a_brute_force_pair_enumeration(window):
    rng = np.random.default_rng(3)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(400, 3))]
    est = correlation_dimension(points, theiler_window=window)
    _, expected = brute_force_correlation_sums(points, window)
    assert np.array_equal(np.array(est.correlation_sums), expected)


def test_excluded_pair_count_matches_the_closed_form():
    rng = np.random.default_rng(11)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(300, 2))]
    for w in (2, 7, 40):
        est = correlation_dimension(points, theiler_window=w)
        assert est.n_excluded_pairs == (w - 1) * 300 - w * (w - 1) // 2


# --------------------------------------------------------------------------
# 4. "Time" means ORIGINAL TRAJECTORY TIME, not array position
# --------------------------------------------------------------------------


def test_result_is_invariant_under_reordering_when_time_indices_are_supplied():
    rng = np.random.default_rng(5)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(400, 3))]
    perm = rng.permutation(400)
    shuffled = [points[i] for i in perm]

    in_order = correlation_dimension(points, theiler_window=17)
    reordered = correlation_dimension(shuffled, theiler_window=17, time_indices=perm.tolist())

    assert reordered.correlation_sums == in_order.correlation_sums
    assert reordered.dimension == in_order.dimension
    assert reordered.n_excluded_pairs == in_order.n_excluded_pairs


def test_omitting_time_indices_on_a_reordered_cloud_excludes_the_wrong_pairs():
    """Documents WHY `time_indices` exists, by showing the failure it prevents.

    With the array shuffled and no `time_indices`, the estimator falls back to
    the documented `t_i = i` assumption -- which is now false -- and drops
    pairs chosen by array position instead of by trajectory time. Same number
    of pairs dropped, different pairs, different answer.
    """
    rng = np.random.default_rng(5)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(400, 3))]
    perm = rng.permutation(400)
    shuffled = [points[i] for i in perm]

    right = correlation_dimension(shuffled, theiler_window=17, time_indices=perm.tolist())
    wrong = correlation_dimension(shuffled, theiler_window=17)

    assert wrong.n_excluded_pairs == right.n_excluded_pairs  # same count ...
    assert wrong.dimension != right.dimension  # ... different pairs


def test_non_contiguous_time_indices_are_measured_in_time_not_position():
    """A strided/decimated trajectory: keeping every 3rd sample triples every gap.

    With indices 0, 3, 6, ... a window of 6 spans only one step, so it must
    behave exactly like a window of 2 on the un-strided index.
    """
    rng = np.random.default_rng(9)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(300, 2))]
    strided = correlation_dimension(
        points, theiler_window=6, time_indices=[3 * i for i in range(300)]
    )
    equivalent = correlation_dimension(points, theiler_window=2)
    assert strided.correlation_sums == equivalent.correlation_sums
    assert strided.n_excluded_pairs == equivalent.n_excluded_pairs == 299


# --------------------------------------------------------------------------
# 5. The "auto" heuristic
# --------------------------------------------------------------------------


def test_auto_window_is_opt_in_only():
    """The 'auto' path must never be reached implicitly -- that is what keeps the default exact."""
    points = sticky_square()
    assert correlation_dimension(points).theiler_window == 0
    assert correlation_dimension(points, theiler_window="auto").theiler_window > 1


def test_auto_window_finds_the_decorrelation_scale_of_the_sticky_sampler():
    """Blob size 16 -> the 1/e crossing of a linearly decaying overlap sits near 10."""
    w = theiler_window_from_autocorrelation(sticky_square())
    assert 5 <= w <= STICKY_BLOB_SIZE
    corrected = correlation_dimension(sticky_square(), theiler_window="auto")
    naive = correlation_dimension(sticky_square(), theiler_window=0)
    assert corrected.dimension > naive.dimension  # right direction, partial correction


def test_auto_window_is_one_when_there_is_no_temporal_structure():
    """No autocorrelation to remove -> the heuristic must not invent a correction.

    An i.i.d. cloud has no trajectory structure, so the honest window is the
    smallest one, and the estimate must be left where it was.
    """
    rng = np.random.default_rng(2)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(1500, 3))]
    assert theiler_window_from_autocorrelation(points) == 1
    assert (
        correlation_dimension(points, theiler_window="auto").dimension
        == correlation_dimension(points).dimension
    )


def test_auto_window_respects_its_clamp_on_a_pure_periodic_orbit():
    """A finely sampled circle never truly decorrelates; the clamp is the honest answer."""
    n = 1000
    points = [(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / n for i in range(n))]
    w = theiler_window_from_autocorrelation(points)
    assert 1 <= w <= min(n // 10, 250)
    assert theiler_window_from_autocorrelation(points, max_window=5) == 5


def test_auto_window_is_invariant_to_reordering_with_time_indices():
    rng = np.random.default_rng(13)
    points = sticky_square(seed=13)
    perm = rng.permutation(len(points))
    shuffled = [points[i] for i in perm]
    assert theiler_window_from_autocorrelation(
        shuffled, time_indices=perm.tolist()
    ) == theiler_window_from_autocorrelation(points)


# --------------------------------------------------------------------------
# 6. Refusals -- bad input must raise, never silently do something plausible
# --------------------------------------------------------------------------


def test_rejects_negative_window():
    with pytest.raises(ValueError, match="non-negative"):
        correlation_dimension(sticky_square(n_anchors=5), theiler_window=-1)


def test_rejects_unknown_string_window():
    with pytest.raises(ValueError, match="'auto'"):
        correlation_dimension(sticky_square(n_anchors=5), theiler_window="theiler")


def test_rejects_non_integer_window():
    with pytest.raises(ValueError, match="'auto'"):
        correlation_dimension(sticky_square(n_anchors=5), theiler_window=3.5)


def test_rejects_time_indices_of_the_wrong_length():
    points = sticky_square(n_anchors=5)
    with pytest.raises(ValueError, match="one integer per point"):
        correlation_dimension(points, theiler_window=2, time_indices=list(range(len(points) - 1)))


def test_rejects_fractional_time_indices():
    points = sticky_square(n_anchors=5)
    with pytest.raises(ValueError, match="integer trajectory indices"):
        correlation_dimension(
            points, theiler_window=2, time_indices=[0.5 * i for i in range(len(points))]
        )


def test_rejects_a_window_that_would_exclude_every_pair():
    rng = np.random.default_rng(1)
    points = [tuple(p) for p in rng.uniform(0, 1, size=(50, 2))]
    with pytest.raises(ValueError, match="excludes all"):
        correlation_dimension(points, theiler_window=50)
