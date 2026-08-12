"""Tests for the symplectic solver and the smooth T-dual geometry."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from socrates.dualscale.geometry import (
    certify_smooth_geometry,
    check_no_go_forces_max_form,
    check_smooth_bound,
    check_smooth_tduality,
    effective_wavenumber_smooth,
)
from socrates.solvers import convergence_order, leapfrog, leapfrog_exact

positive_rationals = st.fractions(
    min_value=Fraction(1, 64), max_value=Fraction(1000), max_denominator=64
)


# -- Smooth T-dual geometry ---------------------------------------------------


@given(alpha=positive_rationals, radius=positive_rationals)
@settings(max_examples=300, deadline=None)
def test_smooth_metric_bound_and_duality(alpha: Fraction, radius: Fraction) -> None:
    """R + a/R >= 2 sqrt(a) and exact invariance under R <-> a/R."""
    assert check_smooth_bound(alpha, radius)
    assert check_smooth_tduality(alpha, radius)


@given(alpha=positive_rationals, radius=positive_rationals)
@settings(max_examples=300, deadline=None)
def test_no_go_lemma_forcing(alpha: Fraction, radius: Fraction) -> None:
    """Exact duality + exact invisibility force the max-form below the cutoff."""
    assert check_no_go_forces_max_form(alpha, radius)


def test_smooth_geometry_certificate() -> None:
    cert = certify_smooth_geometry(Fraction(1, 4))
    assert cert["smooth_bound_2sqrt"]
    assert cert["smooth_exact_tduality"]
    assert cert["no_go_forces_max_form"]


def test_smooth_wavenumber_peak_and_duality() -> None:
    """k/(1+a k^2) peaks at 1/(2 sqrt(a)) and is invariant under k <-> 1/(a k)."""
    alpha = 1e-4
    k = np.logspace(-2, 4, 200)
    smooth = effective_wavenumber_smooth(alpha, k)
    assert smooth.max() <= 1.0 / (2 * np.sqrt(alpha)) + 1e-12
    dual = effective_wavenumber_smooth(alpha, 1.0 / (alpha * k))
    np.testing.assert_allclose(dual, smooth, rtol=1e-12)


# -- Symplectic solver --------------------------------------------------------


def test_leapfrog_is_second_order_on_oscillator() -> None:
    order = convergence_order(
        lambda q: -q, [1.0], [0.0],
        reference=lambda t: np.array([np.cos(t)]), t_end=5.0,
    )
    assert 1.9 <= order <= 2.1


def test_leapfrog_energy_bounded_on_long_run() -> None:
    """Symplecticity: energy error oscillates instead of drifting secularly."""
    run = leapfrog(lambda q: -q, [1.0], [0.0], dt=0.01, n_steps=100_000)
    assert run.energy_drift(lambda q: 0.5 * float(q @ q)) < 1e-4


def test_leapfrog_conserves_angular_momentum_for_central_force() -> None:
    def gravity(q: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(q))
        return -q / r**3

    run = leapfrog(gravity, [0.4, 0.0], [0.0, 2.0], dt=1e-3, n_steps=10_000)
    ell = run.positions[:, 0] * run.velocities[:, 1] - run.positions[:, 1] * run.velocities[:, 0]
    assert np.max(np.abs(ell - ell[0])) < 1e-12


def test_exact_leapfrog_matches_float_leapfrog() -> None:
    """The rational map and the float map agree to float precision."""
    n = 50
    dt_exact = Fraction(1, 100)
    exact_pos, _ = leapfrog_exact(lambda q: [-x for x in q], [1], [0], dt=dt_exact, n_steps=n)
    float_run = leapfrog(lambda q: -q, [1.0], [0.0], dt=0.01, n_steps=n)
    assert abs(float(exact_pos[-1][0]) - float_run.positions[-1, 0]) < 1e-12


def test_exact_leapfrog_is_exactly_time_reversible() -> None:
    """Run forward then backward in exact arithmetic: recover the start EXACTLY.

    This is a property float arithmetic cannot deliver and the sharpest
    demonstration of what exact mode buys: equality of Fractions, not epsilon.
    """
    force = lambda q: [-x for x in q]  # noqa: E731
    dt = Fraction(1, 50)
    pos, vel = leapfrog_exact(force, [1], [0], dt=dt, n_steps=40)
    back_pos, back_vel = leapfrog_exact(force, pos[-1], [-v for v in vel[-1]], dt=dt, n_steps=40)
    assert back_pos[-1] == [Fraction(1)]
    assert [-v for v in back_vel[-1]] == [Fraction(0)]
