"""Tests for T-dual geometry, the Sym^2 lock, and regularized cascades."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from socrates.dualscale.geometry import (
    certify_geometry,
    check_bounce,
    check_inertial_invisibility,
    check_tdual_bound,
    check_tduality,
    effective_radius,
    effective_wavenumber,
    minimal_scale,
)
from socrates.dualscale.shell import (
    dyadic_wavenumbers,
    enstrophy_bound,
    simulate_shell_model,
)
from socrates.operators.recurrence import (
    RecurrenceOperator,
    apery_zeta2,
    apery_zeta3,
    closed_form_symmetric_square,
    symmetric_square,
    verify_symmetric_square,
)

positive_rationals = st.fractions(
    min_value=Fraction(1, 64), max_value=Fraction(1000), max_denominator=64
)


# -- Section 2: T-dual geometry ---------------------------------------------


@given(alpha=positive_rationals, radius=positive_rationals)
@settings(max_examples=300, deadline=None)
def test_tdual_bound_holds_universally(alpha: Fraction, radius: Fraction) -> None:
    """Theorem 2.2: R_eff >= sqrt(alpha') for every positive radius."""
    assert check_tdual_bound(alpha, radius)


@given(alpha=positive_rationals, radius=positive_rationals)
@settings(max_examples=300, deadline=None)
def test_bounce_and_inertial_branches(alpha: Fraction, radius: Fraction) -> None:
    """Theorems 2.3 and 2.4: the two branches behave as claimed."""
    assert check_bounce(alpha, radius)
    assert check_inertial_invisibility(alpha, radius)


@given(alpha=positive_rationals, radius=positive_rationals)
@settings(max_examples=300, deadline=None)
def test_tduality_is_invariance(alpha: Fraction, radius: Fraction) -> None:
    """Corrected Theorem 2.5: R_eff is invariant under R <-> alpha'/R."""
    assert check_tduality(alpha, radius)


def test_draft_theorem_2_5_as_written_is_false() -> None:
    """Regression guard: the draft's extra alpha'/(.) factor makes the claim false.

    Pinning the counterexample prevents the erroneous form from being
    reintroduced during a later edit of the paper or the Lean development.
    """
    alpha, radius = Fraction(1, 4), Fraction(1000)
    dual = max(alpha / radius, alpha / (alpha / radius))
    assert alpha / dual != max(radius, alpha / radius)


def test_minimal_scale_is_the_fixed_point() -> None:
    alpha = 1e-6
    assert effective_radius(alpha, minimal_scale(alpha)) == pytest.approx(minimal_scale(alpha))


def test_effective_wavenumber_is_capped() -> None:
    alpha = 1e-8
    k = 2.0 ** np.arange(40)
    assert np.all(effective_wavenumber(alpha, k) <= 1.0 / np.sqrt(alpha) + 1e-9)


def test_zero_radius_is_refused_not_conventioned() -> None:
    """The division-by-zero edge case must raise, never silently return a value."""
    with pytest.raises(ValueError):
        effective_radius(1.0, 0.0)


def test_geometry_certificate_exercises_both_branches() -> None:
    cert = certify_geometry(Fraction(1, 4), n_samples=200)
    assert cert["n_below_cutoff"] > 0 and cert["n_above_cutoff"] > 0
    assert all(v for k, v in cert.items() if k.startswith("thm_"))


# -- Section 3: the symmetric-square lock -----------------------------------


@given(
    a=st.fractions(min_value=-8, max_value=8, max_denominator=12),
    b=st.fractions(min_value=-8, max_value=8, max_denominator=12),
)
@settings(max_examples=200, deadline=None)
def test_symmetric_square_matches_paper_theorem_3_1(a: Fraction, b: Fraction) -> None:
    """Sym^2 built from symmetric functions equals the paper's closed form."""
    operator = RecurrenceOperator.of(b, a)
    assert tuple(symmetric_square(operator).coefficients) == closed_form_symmetric_square(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [(1, 1), (3, -2), (0, 1), (5, 0), (-2, -3), (Fraction(1, 2), Fraction(-7, 5))],
)
def test_symmetric_square_annihilates_products(a, b) -> None:
    """Exact certification on squares AND cross products of independent solutions."""
    certificate = verify_symmetric_square(RecurrenceOperator.of(b, a), n_terms=25)
    assert certificate["certified"]
    assert certificate["cross_products_annihilated"]


def test_symmetric_square_rejects_wrong_order() -> None:
    with pytest.raises(ValueError):
        symmetric_square(RecurrenceOperator.of(1, 2, 3))


# -- Apery / sporadic sequences ---------------------------------------------


def test_apery_zeta3_reproduces_known_integers() -> None:
    assert [int(x) for x in apery_zeta3().iterate([1, 5], 6)] == [
        1, 5, 73, 1445, 33001, 819005,
    ]


def test_apery_zeta2_reproduces_known_integers() -> None:
    assert [int(x) for x in apery_zeta2().iterate([1, 3], 6)] == [
        1, 3, 19, 147, 1251, 11253,
    ]


def test_apery_sequences_stay_integral() -> None:
    """Integrality is the nontrivial arithmetic property of Apery numbers."""
    assert all(x.denominator == 1 for x in apery_zeta3().iterate([1, 5], 15))


# -- Section 4: regularized cascades ----------------------------------------


def test_inviscid_shell_model_conserves_energy() -> None:
    """The integration oracle: nonlinear terms telescope, so dE/dt = 0."""
    result = simulate_shell_model(dyadic_wavenumbers(12), t_max=2.0, cfl=0.05)
    assert result.energy_drift < 1e-5


def test_regularized_cascade_respects_theorem_4_2_bound() -> None:
    alpha = 1e-6
    k = effective_wavenumber(alpha, dyadic_wavenumbers(24))
    result = simulate_shell_model(k, t_max=6.0, cfl=0.05)
    assert result.max_enstrophy <= enstrophy_bound(alpha, result.energy[0])


def test_regularized_cascade_completes_while_classical_does_not() -> None:
    """Same integrator and data; only the wavenumber ladder differs."""
    k_raw = dyadic_wavenumbers(22)
    classical = simulate_shell_model(k_raw, t_max=6.0, cfl=0.05, max_steps=60_000)
    regularized = simulate_shell_model(
        effective_wavenumber(1e-6, k_raw), t_max=6.0, cfl=0.05, max_steps=60_000
    )
    assert regularized.terminated == "t_max"
    assert classical.terminated != "t_max"
    assert regularized.max_enstrophy < classical.max_enstrophy


def test_timestep_refinement_improves_energy_conservation() -> None:
    """Guards against reporting an integrator artifact as physics."""
    coarse = simulate_shell_model(dyadic_wavenumbers(16), t_max=1.5, cfl=0.4)
    fine = simulate_shell_model(dyadic_wavenumbers(16), t_max=1.5, cfl=0.025)
    assert fine.energy_drift < coarse.energy_drift
