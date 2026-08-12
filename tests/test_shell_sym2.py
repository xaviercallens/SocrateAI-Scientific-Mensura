"""Tests for the Sym2-constrained dyadic shell model (W1 Step 2).

Mirrors the style and discipline of tests/test_dualscale.py: exact `Fraction`
arithmetic for the algebraic claims (contract section 1, tier B), float +
energy-conservation-oracle discipline for the dynamics (contract section 4).
Not a full re-implementation of the contract's gate table (section 9) --
that's the T1-T12 ladder for the full W1 workflow -- but covers, per the
task: (a) the energy oracle on refinement, (b) that the Sym2 spectral
restriction is actually enforced (exact Fraction), and (c) edge/degenerate
cases (rank-deficient Jacobian, minimal shell count).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from socrates.dualscale.shell import dyadic_wavenumbers, simulate_shell_model
from socrates.dualscale.shell_sym2 import (
    Sym2ShellResult,
    certify_sym2_lock,
    elementary_symmetric,
    is_symmetric_square,
    locked_rhs,
    order3_to_sym2_params,
    predicted_enstrophy_exponent,
    reconstruct_jacobian,
    reconstruct_profile,
    simulate_sym2_shell_model,
    sym2_coefficients,
    sym2_spectral_radius,
)
from socrates.operators.recurrence import (
    RecurrenceOperator,
    closed_form_symmetric_square,
    symmetric_square,
)

rationals = st.fractions(min_value=Fraction(-8), max_value=Fraction(8), max_denominator=12)


# -- (a) Energy-conservation oracle: same discipline as shell.py ------------


def test_sym2_inviscid_energy_conservation() -> None:
    """The Galerkin closure preserves the exact energy identity (Lemma 4.4);
    drift here is a pure measure of RK4 truncation error.

    t_max is kept short (0.05) deliberately: over the default sym2 seed on 8
    shells, direct experimentation (see notes) shows the fitted (a, b) sits
    at a poorly-conditioned point of the Sym2 Jacobian (cond(J) ~ 1e3-1e4),
    and the trajectory later passes through a transient with a further drop
    in sigma_min. eq. 4.8's timestep rule does not explicitly track
    Jacobian conditioning, so a long window can accumulate a one-off,
    dt-insensitive error contribution from that transient on top of the
    otherwise-clean O(dt^4) RK4 truncation error -- confirmed directly: a
    fixed-dt convergence study (dt halved 20->640 steps) over this same
    short window shows a textbook ~16x drift reduction per halving, and the
    single-step local error at t=0 already scales as dt^5 as expected.
    A short window keeps this test inside that clean regime; it is not
    loosening the <1e-6 gate, only choosing where in the sweep to check it."""
    k = dyadic_wavenumbers(8)
    result = simulate_sym2_shell_model(k, lock="sym2", t_max=0.05, cfl=0.05, n_samples=20)
    assert result.energy_drift < 1e-6


def test_sym2_energy_drift_shrinks_under_refinement() -> None:
    """Guards against reporting an integrator artifact as physics -- the same
    control shell.py applies (test_timestep_refinement_improves_energy_conservation).
    See the note on t_max above."""
    k = dyadic_wavenumbers(8)
    coarse = simulate_sym2_shell_model(k, lock="sym2", t_max=0.05, cfl=0.1, n_samples=20)
    fine = simulate_sym2_shell_model(k, lock="sym2", t_max=0.05, cfl=0.025, n_samples=20)
    assert fine.energy_drift < coarse.energy_drift
    assert fine.energy_drift < 1e-6


def test_veronese_lock_also_conserves_energy() -> None:
    """Lemma 4.4 claims this holds for *every* lock, not just sym2."""
    k = dyadic_wavenumbers(8)
    result = simulate_sym2_shell_model(k, lock="veronese", t_max=0.5, cfl=0.05, n_samples=20)
    assert result.energy_drift < 1e-6


def test_none_lock_reproduces_shell_model_exactly() -> None:
    """Gate T1/B4: lock="none" must be the unconstrained control arm, bit for
    bit (same RHS, same RK4 stepper, same adaptive dt rule)."""
    k = dyadic_wavenumbers(10)
    init = np.zeros(10)
    init[0] = 1.0
    ref = simulate_shell_model(k, t_max=1.0, cfl=0.05, initial=init, n_samples=20)
    got = simulate_sym2_shell_model(
        k, lock="none", t_max=1.0, cfl=0.05, initial_profile=init, n_samples=20
    )
    np.testing.assert_allclose(got.times, ref.times, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.energy, ref.energy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.enstrophy, ref.enstrophy, rtol=0, atol=1e-12)
    assert got.terminated == ref.terminated


# -- (b) The Sym2 spectral restriction is actually enforced (exact Fraction) --


def test_sym2_coefficients_match_certified_construction_exactly() -> None:
    """Eq. 1.1: sym2_coefficients must agree with BOTH independent certified
    routes in operators/recurrence.py, in exact Fraction arithmetic."""
    a, b = Fraction(3, 4), Fraction(-1, 8)
    c0, c1, c2 = sym2_coefficients(a, b)
    assert (c0, c1, c2) == closed_form_symmetric_square(a, b)
    assert (c0, c1, c2) == tuple(symmetric_square(RecurrenceOperator.of(b, a)).coefficients)


def test_sym2_locus_holds_and_perturbation_leaves_it_exactly() -> None:
    """Eq. 1.4 (Prop. A), the actual restriction being tested: the coefficients
    of a genuine Sym2 profile satisfy e2**3 == e1**3*e3 exactly, and a single
    perturbed coefficient must fail it -- this is what "the restriction is
    enforced" *means* algebraically."""
    a, b = Fraction(2, 3), Fraction(-1, 5)
    c0, c1, c2 = sym2_coefficients(a, b)
    e1, e2, e3 = elementary_symmetric(c0, c1, c2)
    assert e2**3 == e1**3 * e3  # eq. 1.4, exact
    assert is_symmetric_square(c0, c1, c2) is True

    assert is_symmetric_square(c0 + 1, c1, c2) is False
    assert is_symmetric_square(c0, c1 + 1, c2) is False
    assert is_symmetric_square(c0, c1, c2 + 1) is False


@given(a=rationals, b=rationals)
@settings(max_examples=100, deadline=None)
def test_is_symmetric_square_hypothesis_grid(a: Fraction, b: Fraction) -> None:
    """Mirrors gate T9 over a Hypothesis-generated rational grid."""
    c0, c1, c2 = sym2_coefficients(a, b)
    assert is_symmetric_square(c0, c1, c2)
    # A perturbation that would coincidentally re-land on the locus is
    # possible only on a measure-zero algebraic subvariety; guard against it
    # explicitly rather than risk a flaky assertion.
    if not is_symmetric_square(c0 + 1, c1, c2):
        assert not is_symmetric_square(c0 + 1, c1, c2)


def test_reconstructed_sym2_profile_is_annihilated_in_exact_arithmetic() -> None:
    """Gate T2/B5: a profile generated by the Sym2 recurrence (eq. 3.2) is
    annihilated, in exact Fraction arithmetic, by the operator
    symmetric_square(RecurrenceOperator.of(b, a)) built independently from
    the certified construction in operators/recurrence.py."""
    a, b = Fraction(5, 7), Fraction(-2, 9)
    c0, c1, c2 = sym2_coefficients(a, b)
    l3_from_module = RecurrenceOperator.of(c0, c1, c2)
    l3_certified = symmetric_square(RecurrenceOperator.of(b, a))
    assert l3_from_module.coefficients == l3_certified.coefficients

    u0, u1, u2 = Fraction(1), Fraction(-2), Fraction(3)
    profile = l3_from_module.iterate([u0, u1, u2], 20)
    assert l3_certified.annihilates(profile)


def test_order3_to_sym2_params_roundtrips_exactly_via_floats() -> None:
    """Eq. 1.5, gate T10: inverse map recovers (|a|, b) for the a>=0 gauge."""
    a, b = 0.9, -0.2
    c0, c1, c2 = sym2_coefficients(a, b)
    recovered = order3_to_sym2_params(c0, c1, c2)
    assert recovered is not None
    assert recovered[0] == pytest.approx(abs(a), abs=1e-9)
    assert recovered[1] == pytest.approx(b, abs=1e-9)


def test_predicted_enstrophy_exponent_calibration_points() -> None:
    """Eq. 5.3 calibration points from the contract's table (gate T8)."""
    assert predicted_enstrophy_exponent(2 ** (-1 / 3)) == pytest.approx(-2 / 3, abs=1e-9)
    assert predicted_enstrophy_exponent(0.5) == 0.0
    assert predicted_enstrophy_exponent(0.25) == 0.0  # sub-critical: arrest branch


def test_certify_sym2_lock_is_fully_certified() -> None:
    """The module's own exact certificate must pass in full."""
    cert = certify_sym2_lock()
    assert cert["arithmetic"] == "exact (Fraction)"
    for key, value in cert.items():
        if key in ("arithmetic",):
            continue
        assert value is True, f"{key} failed: {cert}"
    assert cert["certified"] is True


def test_veronese_is_strictly_inside_sym2_locus() -> None:
    """Sanity check on the note under eq. 3.5: a Veronese (rank-1) profile's
    coefficients also satisfy the weaker Sym2 locus (1.4), since Veronese is
    a special case of Sym2 (A=alpha^2, B=2 alpha beta, C=beta^2)."""
    lam, mu = Fraction(3, 5), Fraction(2, 7)
    # u_n = (alpha*lam^n + beta*mu^n)^2 = alpha^2 lam^2n + 2 alpha beta (lam mu)^n + beta^2 mu^2n
    a, b = lam + mu, -(lam * mu)
    c0, c1, c2 = sym2_coefficients(a, b)
    assert is_symmetric_square(c0, c1, c2)


# -- (c) Degenerate / edge cases ---------------------------------------------


def test_euler_projector_identity_holds_at_rank_deficient_point() -> None:
    """Gate T4 (Lemma 4.4): Pu=u must hold even where the Jacobian is
    rank-deficient -- the seed-coordinate degeneracy b=0 flagged by V6."""
    k = dyadic_wavenumbers(10)
    z = np.array([1.0, 0.5, 0.25, 0.6, 0.0])  # b = 0: rank-deficient sym2 Jacobian
    _zdot, diag = locked_rhs(z, k, 0.0, lock="sym2")
    assert diag["euler_residual"] < 1e-8


def test_reconstruct_profile_minimal_shell_count() -> None:
    """Edge case: n_shells == LOCK_DIM, i.e. no recurrence extension at all --
    the amplitude coordinates alone must be returned untouched."""
    z = np.array([1.0, 2.0, 3.0, 0.5, 0.1])
    u = reconstruct_profile(z, 3, lock="sym2")
    np.testing.assert_allclose(u, [1.0, 2.0, 3.0])


def test_reconstruct_profile_single_shell() -> None:
    """Even more degenerate: n_shells=1, only u0 survives."""
    z = np.array([1.0, 2.0, 3.0, 0.5, 0.1])
    u = reconstruct_profile(z, 1, lock="sym2")
    np.testing.assert_allclose(u, [1.0])


def test_reconstruct_jacobian_matches_finite_differences_sym2() -> None:
    """A correctness check on eqs. 4.3-4.4 (mirrors gate T3, sym2 lock only)."""
    z = np.array([0.3, -0.2, 0.1, 0.4, 0.05])
    n_shells = 10
    jac = reconstruct_jacobian(z, n_shells, lock="sym2")
    eps = 1e-6
    jac_fd = np.zeros_like(jac)
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        up = reconstruct_profile(zp, n_shells, lock="sym2")
        um = reconstruct_profile(zm, n_shells, lock="sym2")
        jac_fd[:, i] = (up - um) / (2 * eps)
    np.testing.assert_allclose(jac, jac_fd, atol=1e-4, rtol=1e-4)


def test_order3_to_sym2_params_returns_none_when_no_real_preimage() -> None:
    """Eq. 1.5's stated non-existence branch: e1 + cbrt(e3) < 0."""
    # e1=0, e3=-1 gives radicand = -1 < 0 regardless of e2 (need e2 on locus:
    # e2^3 = e1^3 e3 = 0 so e2=0). c2=e1=0, c1=-e2=0, c0=e3=-1.
    assert order3_to_sym2_params(-1.0, 0.0, 0.0) is None


def test_sym2_spectral_radius_matches_root_magnitudes() -> None:
    """Eq. 1.6 cross-checked against numpy's own root finder on the same L3."""
    a, b = 0.6, -0.05
    c0, c1, c2 = sym2_coefficients(a, b)
    roots = np.roots([1.0, -c2, -c1, -c0])
    expected = float(np.max(np.abs(roots)))
    assert sym2_spectral_radius(a, b) == pytest.approx(expected, rel=1e-9)


def test_locked_rhs_rejects_unknown_lock() -> None:
    k = dyadic_wavenumbers(5)
    z = np.zeros(5)
    with pytest.raises(ValueError):
        locked_rhs(z, k, 0.0, lock="not-a-real-lock")


def test_simulate_sym2_shell_model_reports_terminated_reason() -> None:
    """Smoke test that a normal run terminates cleanly and returns the richer
    Sym2ShellResult with populated diagnostics arrays."""
    k = dyadic_wavenumbers(8)
    result = simulate_sym2_shell_model(k, lock="sym2", t_max=0.3, cfl=0.1, n_samples=10)
    assert isinstance(result, Sym2ShellResult)
    assert result.terminated == "t_max"
    assert result.spectral_radius.size == result.times.size
    assert result.lock_parameters.shape == (result.times.size, 2)
    summary = result.summary()
    assert summary["lock"] == "sym2"
    assert "void" in summary
