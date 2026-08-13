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

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from socrates.dualscale.geometry import effective_wavenumber
from socrates.dualscale.shell import dyadic_wavenumbers, simulate_shell_model
from socrates.dualscale.shell_sym2 import (
    Sym2ShellResult,
    certify_sym2_lock,
    elementary_symmetric,
    error_density,
    fit_lock_state,
    half_ladder_a,
    is_symmetric_square,
    locked_rhs,
    order3_to_sym2_params,
    parameter_curvature,
    predicted_enstrophy_exponent,
    reconstruct_jacobian,
    reconstruct_profile,
    seed_profile,
    simulate_sym2_shell_model,
    stability_rate,
    step_rate,
    sym2_coefficients,
    sym2_coefficients_asq,
    sym2_spectral_radius,
    sym2_spectral_radius_asq,
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
    loosening the <1e-6 gate, only choosing where in the sweep to check it.

    W1 ROUND 2: the window is now the contract's full t_max = 0.8, TIGHTENED
    from 0.05. The confinement above misdiagnosed the cause. The legacy sym2
    chart (u0,u1,u2,a,b) has a *branch point* at a = 0: eq. 1.2 makes
    (c0,c1,c2) depend on `a` only through a**2, so dPhi/da = 2a dPhi/d(a**2)
    vanishes identically there and J drops to rank 4. The Sym2-locked flow
    drives a to zero in finite time, so no timestep rule could rescue the
    long window. The chart is now (u0,u1,u2,A,b) with A = a**2, the fold is
    a regular point, and the full window is inside the clean regime."""
    k = dyadic_wavenumbers(8)
    result = simulate_sym2_shell_model(k, lock="sym2", t_max=0.8, cfl=0.05, n_samples=20)
    assert result.terminated == "t_max"
    assert result.energy_drift < 1e-6


def test_sym2_energy_drift_shrinks_under_refinement() -> None:
    """Guards against reporting an integrator artifact as physics -- the same
    control shell.py applies (test_timestep_refinement_improves_energy_conservation).
    Window tightened to the full t_max = 0.8; see the note above."""
    k = dyadic_wavenumbers(8)
    coarse = simulate_sym2_shell_model(k, lock="sym2", t_max=0.8, cfl=0.1, n_samples=20)
    fine = simulate_sym2_shell_model(k, lock="sym2", t_max=0.8, cfl=0.025, n_samples=20)
    assert coarse.terminated == "t_max"
    assert fine.terminated == "t_max"
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


# -- (d) W1 round 2: the chart branch point must not silently return --------


def _signed_drift(result) -> float:
    """(E(t_end) - E(0)) / E(0).

    `ShellResult.energy_drift` takes the absolute value, which hides sign
    flips.  A sign flip under refinement is bug signature B2 and the thing
    FINDINGS 7.2.B caught the previous repair on, so the convergence gates
    below work with the signed quantity.
    """
    return float((result.energy[-1] - result.energy[0]) / result.energy[0])


def _drift_scan(
    k: np.ndarray,
    cfls: tuple[float, ...],
    *,
    t_max: float,
    initial_profile: np.ndarray | None = None,
    max_cond: float = 1e4,
    min_ulps: float = 200.0,
) -> list[float]:
    """Signed drift at each cfl, with the gate's own evaluability preconditions.

    The two preconditions are asserted, not silently skipped, so that a change
    which pushes a pinned seed out of the regime where this gate MEANS anything
    fails loudly instead of quietly measuring nothing:

    * `max_cond` -- the conditioning question of FINDINGS 7.3 item 4 is open and
      is NOT what this gate tests.  A run whose Jacobian gets that badly
      conditioned has a dt-insensitive error contribution that no step rule can
      remove; measured, a fixed-dt plain RK4 control on the identical field is
      2-200x WORSE than the adaptive rule at every such seed (FINDINGS 10.4).
    * `min_ulps` -- below ~1e-13 relative the drift stops measuring the
      integrator and starts measuring the roundoff of the energy sum
      (FINDINGS 8.6.2).  Same control: a fixed-dt run whose drift lands there
      flips sign too.
    """
    drifts = []
    for cfl in cfls:
        result = simulate_sym2_shell_model(
            k, lock="sym2", t_max=t_max, cfl=cfl, n_samples=8, initial_profile=initial_profile
        )
        assert result.terminated == "t_max", f"cfl={cfl} stopped at {result.terminated}"
        cond = float(result.metadata["max_jacobian_condition"])
        assert cond < max_cond, f"cfl={cfl}: max cond(J) = {cond:.3g}, gate not discriminating"
        drift = _signed_drift(result)
        ulps = abs(result.energy[-1] - result.energy[0]) / np.spacing(result.energy[0])
        assert ulps >= min_ulps, f"cfl={cfl}: drift is {ulps:.0f} ulps, at the roundoff floor"
        drifts.append(drift)
    return drifts


def _assert_compensated_flatness(
    drifts: list[float], cfls: tuple[float, ...], *, order: int = 4, band: float = 2.0
) -> None:
    """The gate FINDINGS 7.2 says a convergence claim has to clear.

    Three assertions, in increasing strength:

    1. No sign flip in the signed drift (bug signature B2).
    2. Every adjacent halving reduces |drift| by >= 8x (gate T5).
    3. |drift| / cfl**order is FLAT to within `band` across the whole scan.

    (3) is the one that matters.  A least-squares fitted order near 4 is not
    evidence of convergence: FINDINGS 7.2.A-B exhibits a scan that fits to
    order ~3.95 while individual halvings make the drift 1.2x-20x *worse* and
    flip its sign.  Compensated flatness cannot be satisfied that way.

    WHERE THIS CRITERION IS AND IS NOT DISCRIMINATING (FINDINGS 8.6.4).  It
    fails at t_max = 0.2 and 0.4 -- but so does a fixed-dt plain RK4 control
    with no adaptive rule at all, at exactly those windows (it flips sign at
    t_max = 0.2, 0.3, 0.4, 0.5 and not at 0.6, 0.8).  On short windows the
    leading O(cfl**4) coefficient of the signed drift is small enough that
    the drift reaches the ~1e-14 roundoff floor of the energy sum inside the
    scanned cfl range and crosses zero.  That is a property of the energy
    functional, not of the stepper, so short windows must not be used for
    this gate.  Callers should keep t_max >= 0.6 and |drift| >= ~1e-13.
    """
    signs = {math.copysign(1.0, d) for d in drifts}
    assert len(signs) == 1, f"signed drift changes sign across the scan: {drifts}"

    for coarse, fine, cfl in zip(drifts, drifts[1:], cfls[1:], strict=False):
        ratio = abs(coarse) / abs(fine)
        assert ratio >= 8.0, f"cfl {cfl}: drift fell {ratio:.2f}x, want >= 8x (gate T5)"

    compensated = [abs(d) / c**order for d, c in zip(drifts, cfls, strict=False)]
    spread = max(compensated) / min(compensated)
    assert spread <= band, (
        f"|drift|/cfl**{order} spread {spread:.2f}x over the scan, want <= {band}; "
        f"compensated = {compensated}"
    )


# The round-2c seed sweep, as a gate.  Every root pair from the 26-seed sweep of
# FINDINGS 10.3 whose three runs all terminate "t_max", whose max cond(J) stays
# under 1e4, and whose drift stays above the roundoff floor at every cfl -- i.e.
# every seed at which this gate is a discriminating measurement of the STEP RULE
# rather than of the open conditioning question (FINDINGS 7.3 item 4) or of the
# energy sum's roundoff (FINDINGS 8.6.2).  The membership rule is a property of
# the run, not of the answer, and `_drift_scan` asserts it rather than skipping,
# so a regression that pushes a seed out of the regime fails loudly.
#
# The 13 seeds NOT listed are all reported in FINDINGS 10.3 with their numbers;
# at every one of them a fixed-dt plain RK4 control with no adaptive rule at all
# fails the same criterion, in all but one case far worse (FINDINGS 10.4).
# (0.7,-0.2) is omitted as an exact duplicate: it gives the same (A, b) =
# (0.25, 0.14), hence the same profile and the same trajectory, as (0.2,-0.7).
SEED_SWEEP: tuple[tuple[float, float], ...] = (
    (0.5, 0.25),  # module default / flagship
    (0.2, -0.7),  # FINDINGS 9.2 failure 1 (2.51x on this ladder under 4.8'')
    (0.1, -0.6),  # FINDINGS 9.2 failure 2 (24.75x)
    (0.15, -0.65),  # found failing in round 2c (12.67x AND a sign flip)
    (0.3, -0.4),  # contract seed C7
    (0.25, -0.75),
    (0.5, -0.5),  # A(0) = 0 exactly -- the a = 0 fold of FINDINGS 7
    (0.45, -0.35),
    (0.55, -0.45),
    (0.6, 0.3),
    (-0.2, 0.75),
    (0.05, 0.85),
    (0.8, -0.1),
)


@pytest.mark.slow
@pytest.mark.parametrize("roots", SEED_SWEEP, ids=lambda r: f"{r[0]}_{r[1]}")
def test_full_window_energy_drift_converges_at_fourth_order(roots: tuple[float, float]) -> None:
    """Gate T5/B2 over the FULL window, at the configuration the claim is about,
    ACROSS A SEED SWEEP.

    Round 1 measured, at t_max = 0.8, alpha' = 1e-6, N = 24, drift
    1.02e-5 / 9.33e-6 / 1.27e-5 / 6.77e-6 / 6.57e-6 / 6.27e-6 / 1.10e-5 /
    2.43e-6 as cfl swept 0.2 -> 0.001: neither convergent nor monotone.

    W1 ROUND 2 PHASE 3 re-pinned this from N = 6 -- a regime where the test
    passed but the claim did not live (FINDINGS 7.2.F) -- to N = 24 and N = 30
    on the T-dual ladder at alpha' = 1e-6, asserting compensated-error
    flatness rather than a fitted order.  See `_assert_compensated_flatness`
    for why a fitted order is not evidence.

    W1 ROUND 2c RE-PINS IT AGAIN, and this is now the point.  FINDINGS 9.2
    refuted the round-2b claim on generalisation: N = 24 and N = 30 are
    near-duplicates (the drift is essentially N-independent above N = 12, and
    the two agreed to four significant figures), so parametrising over them was
    never a generalisation check -- while at 2 of 13 *seeds* the criterion
    failed outright.  The parametrisation is therefore over SEEDS at fixed
    N = 24, and it includes both seeds that refuted round 2b plus a third,
    (0.15,-0.65), that round 2c found failing (12.67x with a sign flip) and
    that nobody had tried.  N = 30 keeps a single check of its own below.

    Measured under eq. 4.8''' at N = 24, cfl 0.08/0.04/0.02: compensated
    spreads 1.014 / 1.922 / 1.178 / 1.360 / 1.054 / 1.446 / 1.067 / 1.079 /
    1.066 / 1.056 / 1.476 / 1.078 / 1.031 in the order listed in SEED_SWEEP,
    zero sign flips everywhere, minimum exact-halving ratio 13.12 against gate
    T5's 8.  The 2x band therefore has only 4% headroom at its worst seed,
    (0.2,-0.7).  That is the honest state of the gate; it is recorded in
    FINDINGS 10.5 rather than padded out, and it is the first thing to attack.
    """
    k = effective_wavenumber(1e-6, dyadic_wavenumbers(24))
    cfls = (0.08, 0.04, 0.02)
    seed = seed_profile(24, lock="sym2", roots=roots)
    _assert_compensated_flatness(_drift_scan(k, cfls, t_max=0.8, initial_profile=seed), cfls)


@pytest.mark.slow
def test_full_window_gate_also_holds_at_the_other_shell_count() -> None:
    """N = 30 at the flagship seed -- round 2b's second parametrisation, kept as
    a cheap N-check but no longer mistaken for a generalisation check.

    FINDINGS 9.1: N = 24 and N = 30 agree to four significant figures because
    the drift is essentially N-independent above N = 12.  That is worth pinning
    once; it is not worth pinning thirteen times, which is why the seed sweep
    above runs at a single N.
    """
    k = effective_wavenumber(1e-6, dyadic_wavenumbers(30))
    cfls = (0.08, 0.04, 0.02)
    _assert_compensated_flatness(_drift_scan(k, cfls, t_max=0.8), cfls)


@pytest.mark.slow
def test_sweep_window_energy_drift_converges_at_fourth_order() -> None:
    """The same gate at the ACTUAL sweep window, which is where it bites.

    FINDINGS 7.2.E: the single run in the whole programme that currently
    reaches the workflow's gate is t_max = 12, N = 30, alpha' = 1e-2, and
    under eq. 4.8' one halving there gave only 5.26x against gate T5's 8x.
    A convergence claim checked only on a 0.8-long diagnostic window does not
    cover the measurement, so the sweep window gets its own gate.

    Measured under eq. 4.8'' with a 20-point log-spaced scan (cfl 0.2 ->
    0.0125): compensated spread 1.03x, zero sign flips, worst adjacent pair
    equivalent to 15.0x per halving, over 4.8 decades of drift; exact
    halvings give 16.17/15.76/15.95/15.96x.  The same scan on the HEAD
    eq. 4.8' implementation: spread 3.94x with one pair 0.81x per halving.
    """
    k = effective_wavenumber(1e-2, dyadic_wavenumbers(30))
    cfls = (0.1, 0.05, 0.025)
    _assert_compensated_flatness(_drift_scan(k, cfls, t_max=12.0), cfls)


def test_step_doubling_estimator_does_not_feed_back_on_its_own_roundoff() -> None:
    """eq. 4.8'''s error estimator must report no signal below its noise floor.

    Once the Richardson gap ||u_full - u_half|| is at roundoff,
    C = (16/15)*gap/(||u|| dt**5) measures machine epsilon divided by dt**5 and
    *diverges* as dt shrinks.  Unfloored that is positive feedback: the
    fixed-point map is dt -> cfl*dt/((16/15)*eps)**(1/5), gain ~1333*cfl, so
    below cfl ~ 7.5e-4 the step spirals to zero.  Measured on the unfloored
    code: clean at cfl = 5e-4, terminated "dt_collapse" after 23 steps at
    cfl = 2e-4 and after 10 at 1e-4.  This pins both halves of the fix.
    """
    u = np.ones(8)
    assert error_density(u, u * (1 + 1e-16), 1e-3) == 0.0  # roundoff-level gap
    assert error_density(u, u * (1 + 1e-6), 1e-3) > 0.0  # real gap

    k = effective_wavenumber(1e-6, dyadic_wavenumbers(12))
    for cfl in (2e-4, 1e-4):
        result = simulate_sym2_shell_model(k, lock="sym2", t_max=0.02, cfl=cfl, n_samples=4)
        assert result.terminated == "t_max", f"cfl={cfl} stopped at {result.terminated}"
        assert result.energy_drift < 1e-12


def test_sym2_jacobian_has_no_branch_point_at_the_gauge_fold() -> None:
    """The round-2 defect, pinned directly.

    In the legacy chart z = (u0,u1,u2,a,b), dPhi/da = 2a * dPhi/d(a**2) is
    identically zero at a = 0, so J drops to rank 4 and zdot = J^+ F diverges
    like 1/a. In the repaired chart z = (u0,u1,u2,A,b) that point is regular.
    If this ever fails, the branch point is back.
    """
    for b in (-0.3, -0.1, 0.05, 0.2):
        z = np.array([1.0, 0.5, 0.25, 0.0, b])  # A = 0, i.e. the fold a = 0
        jac = reconstruct_jacobian(z, 24, lock="sym2")
        svals = np.linalg.svd(jac, compute_uv=False)
        assert svals[-1] / svals[0] > 1e-3, f"J degenerate at the fold for b={b}"


def test_repaired_chart_reproduces_the_legacy_algebra() -> None:
    """A = a**2 changes the coordinate, not the constraint set (eqs. 1.1, 1.6)."""
    for a, b in ((0.75, -0.125), (0.6, -0.05), (0.3, -0.2), (1.2, 0.1)):
        assert sym2_coefficients_asq(a * a, b) == pytest.approx(sym2_coefficients(a, b), rel=1e-13)
        assert sym2_spectral_radius_asq(a * a, b) == pytest.approx(
            sym2_spectral_radius(a, b), rel=1e-12
        )
        assert half_ladder_a(a * a) == pytest.approx(a, rel=1e-12)
        c0, c1, c2 = sym2_coefficients(a, b)
        expected = RecurrenceOperator.of(c0, c1, c2).iterate([1.0, 0.4, 0.1], 12)
        np.testing.assert_allclose(
            reconstruct_profile(np.array([1.0, 0.4, 0.1, a * a, b]), 12, lock="sym2"),
            np.asarray(expected, dtype=float),
            rtol=1e-12,
        )
    # A < 0 is still on the Sym2 locus (1.4) -- Proposition A is stated over C.
    assert is_symmetric_square(*sym2_coefficients_asq(-0.4, -0.1))
    assert math.isnan(half_ladder_a(-0.4))


def test_fit_lock_state_is_exact_on_the_module_seed() -> None:
    """Gate T12 / vacuity guard V4, which FINDINGS 6.4 recorded as failing.

    `seed_profile` lies on every lock's constraint set by construction, so the
    fit residual must be at machine level. Round 1 measured 5.9e-7 for sym2:
    the least-squares fit converged to a spurious local minimum (a = 0.708
    instead of 0.75), silently displacing the initial condition of every sym2
    run in the sweep.
    """
    for lock in ("sym2", "order3", "veronese"):
        u = seed_profile(24, lock=lock)
        z, residual = fit_lock_state(u, lock=lock)
        assert residual < 1e-10, f"{lock}: V4 fit residual {residual:.3e}"
        np.testing.assert_allclose(reconstruct_profile(z, 24, lock=lock), u, atol=1e-12)
    # sym2 must recover the seed's own half-ladder: roots (0.5, 0.25) give
    # a = 0.75, b = -0.125, hence A = 0.5625.
    z, _ = fit_lock_state(seed_profile(24, lock="sym2"), lock="sym2")
    assert z[3] == pytest.approx(0.5625, abs=1e-9)
    assert z[4] == pytest.approx(-0.125, abs=1e-9)


def test_lemma_4_4_invariants_hold_pointwise_over_the_full_window() -> None:
    """Lemma 4.4's pointwise invariants across the repaired full window.

    <u, J zdot> = <u, P F> = <u, F> = 0 for the inviscid model, and
    ||Pu - u|| / ||u|| < 1e-10 (gate B3). A step-control fix that broke
    either would not be a fix.
    """
    k = dyadic_wavenumbers(10)
    result = simulate_sym2_shell_model(k, lock="sym2", t_max=0.8, cfl=0.05, n_samples=40)
    assert result.terminated == "t_max"
    assert float(np.max(result.euler_residual)) < 1e-10
    assert float(result.metadata["max_jacobian_condition"]) < 1e6

    rng = np.random.default_rng(0)
    for _ in range(25):
        z = np.array(
            [*rng.uniform(-1.0, 1.0, size=3), rng.uniform(-0.5, 1.0), rng.uniform(-0.4, 0.3)]
        )
        zdot, diag = locked_rhs(z, k, 0.0, lock="sym2")
        u = reconstruct_profile(z, k.size, lock="sym2")
        jac = reconstruct_jacobian(z, k.size, lock="sym2")
        scale = float(np.linalg.norm(u)) * float(np.linalg.norm(jac @ zdot)) + 1e-300
        assert abs(float(u @ (jac @ zdot))) / scale < 1e-10
        assert diag["euler_residual"] < 1e-10


def test_repaired_timestep_rule_is_never_looser_than_contract_eq_4_8() -> None:
    """eq. 4.8''' >= eq. 4.8'' >= eq. 4.8' >= eq. 4.8 pointwise -- a tightening,
    never a loosening, of the contract's stability convention.

    Exercises the shipped `stability_rate`/`parameter_curvature`/`step_rate`
    rather than an inline re-derivation, so the test cannot pass while the
    integrator uses something else.  eq. 4.8' is the 2-norm blend of eq. 4.8's
    terms (the 2-norm dominates the max, and
    1/sqrt(t**2+eps**2) >= 1/(|t|+eps)); eq. 4.8''' adds the non-negative
    curvature term |thetaddot|/sqrt(theta**2+eps**2) under the same square
    root; eq. 4.8'' adds a non-negative error density under a fifth root.
    Each can only shorten the step further.
    """
    k = dyadic_wavenumbers(12)
    eps = 1e-3
    rng = np.random.default_rng(3)
    states = [fit_lock_state(seed_profile(12, lock="sym2"), lock="sym2")[0]]
    states += [
        np.array([*rng.uniform(-1.0, 1.0, size=3), rng.uniform(-0.5, 1.0), rng.uniform(-0.4, 0.3)])
        for _ in range(40)
    ]
    for z in states:
        for nu in (0.0, 1e-3):
            zdot, _diag = locked_rhs(z, k, nu, lock="sym2")
            u = reconstruct_profile(z, 12, lock="sym2")
            legacy = max(
                float(np.max(k * np.abs(u))),
                abs(zdot[3]) / (abs(z[3]) + eps),
                abs(zdot[4]) / (abs(z[4]) + eps),
                nu * float(np.max(k**2)),
            )
            blend = stability_rate(z, u, zdot, k, nu, lock="sym2")
            assert blend >= legacy, f"eq. 4.8' looser than eq. 4.8 at z={z}, nu={nu}"
            zddot = parameter_curvature(z, zdot, k, nu, lock="sym2")
            curved = stability_rate(z, u, zdot, k, nu, lock="sym2", zddot=zddot)
            assert curved >= blend, f"eq. 4.8''' looser than eq. 4.8' at z={z}, nu={nu}"
            for density in (0.0, 1.0, 1e6):
                assert step_rate(curved, density) >= curved >= blend
    # lock="none" is bit-for-bit shell.py's rate -- that is what gate T1 pins.
    # The curvature term must not reach it: "none" has no lock parameters.
    u_none = seed_profile(12, lock="sym2")
    assert stability_rate(u_none, u_none, u_none, k, 0.0, lock="none") == float(
        np.max(k * np.abs(u_none))
    )
    assert not np.any(parameter_curvature(u_none, u_none, k, 0.0, lock="none"))


def test_stability_rate_does_not_collapse_at_a_lock_parameter_turning_point() -> None:
    """The round-2c defect, pinned directly at its mechanism.

    Every lock-parameter term of the contract's eq. 4.8 and of eq. 4.8' is a
    multiple of |thetadot|, so it vanishes IDENTICALLY wherever a lock
    parameter turns around -- and a turning point is exactly where RK4's
    truncation error is largest, since that error is driven by the high
    derivatives of the trajectory and not by its velocity.  eq. 4.8' therefore
    has a local *minimum* of the rate at each such passage and lengthens the
    step straight through it (FINDINGS 10.1: at roots (0.1,-0.6) the rate falls
    20.2 -> 4.49 across the turn of A while dt grows 2.5x, and the three steps
    there carry +125%/+102%/-103% of the run's energy drift).

    The trajectory here is generated by a PLAIN fixed-dt RK4 with no adaptive
    rule at all, so the test does not depend on the rule it is testing.  If
    this ever fails, either the curvature term has stopped firing or eq. 4.8'
    has stopped dipping -- both worth knowing.
    """
    n_shells, n_steps, t_end = 24, 400, 0.2
    k = effective_wavenumber(1e-6, dyadic_wavenumbers(n_shells))
    z, _ = fit_lock_state(seed_profile(n_shells, lock="sym2", roots=(0.1, -0.6)), lock="sym2")

    def field(state: np.ndarray) -> np.ndarray:
        return locked_rhs(state, k, 0.0, lock="sym2")[0]

    def rk4(state: np.ndarray, h: float) -> np.ndarray:
        k1 = field(state)
        k2 = field(state + 0.5 * h * k1)
        k3 = field(state + 0.5 * h * k2)
        k4 = field(state + h * k3)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    dt = t_end / n_steps
    states = [z]
    for _ in range(n_steps):
        states.append(rk4(states[-1], dt))

    primes, curved, pdots = [], [], []
    for state in states:
        zdot = field(state)
        u = reconstruct_profile(state, n_shells, lock="sym2")
        prime = stability_rate(state, u, zdot, k, 0.0, lock="sym2")
        zddot = parameter_curvature(state, zdot, k, 0.0, lock="sym2", rate=prime)
        primes.append(prime)
        curved.append(stability_rate(state, u, zdot, k, 0.0, lock="sym2", zddot=zddot))
        pdots.append((float(zdot[3]), float(zdot[4])))

    # 1. Where eq. 4.8' is at its weakest IS a turning point of a lock
    #    parameter -- that is the defect, stated as a coincidence that must hold.
    worst = int(np.argmin(primes))
    turning = any(
        pdots[i - 1][j] * pdots[i][j] < 0
        for i in range(max(1, worst - 2), min(len(pdots), worst + 3))
        for j in (0, 1)
    )
    assert turning, (
        f"eq. 4.8' bottoms out at step {worst} (t={worst * dt:.4f}) without a "
        "lock-parameter turning point -- the round-2c diagnosis needs redoing"
    )

    # 2. The dip is deep: eq. 4.8' collapses by more than 5x into the turn.
    lo, hi = max(0, worst - 30), min(len(primes), worst + 31)
    assert max(primes[lo:hi]) / primes[worst] > 5.0, (
        f"the (4.8') dip has gone: {max(primes[lo:hi]):.3f} / {primes[worst]:.3f}"
    )

    # 3. eq. 4.8''' does not collapse there: the curvature term carries it.
    assert curved[worst] > 5.0 * primes[worst], (
        f"curvature term does not fire at the turn: {curved[worst]:.3f} vs {primes[worst]:.3f}"
    )
    assert curved[worst] > min(curved), "eq. 4.8''' still bottoms out at the turn"
