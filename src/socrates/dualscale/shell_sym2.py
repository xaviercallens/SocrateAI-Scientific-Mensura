"""Sym2-constrained variant of the dyadic shell model (W1 Step 2).

Implements the interface specified in `docs/W1_sym2_contract.md` section 8:
a family of low-dimensional "locks" on the shell profile --

    "none"     -- unconstrained, u in R^N (control arm, must reproduce shell.py)
    "sym2"     -- u in Sym^2(V) for a half-ladder V = span{lambda^n, mu^n}
                  (order-3 recurrence whose eigenvalues form a geometric
                  progression, eq. 1.4)
    "order3"   -- u obeys *some* order-3 recurrence (dimension-matched control,
                  no Sym^2 restriction)
    "veronese" -- the strict reading u_n = v_n^2 for a half-ladder v

-- integrated by projecting the Katz-Pavlovic field onto the tangent space of
the constraint manifold in the *energy* metric (eq. 4.5): zdot = J(z)^+ F(Phi(z)).
Lemma 4.4 of the contract shows this closure preserves the exact energy
identity dE/dt = -nu*Omega for every lock, so shell.py's energy-drift oracle
remains a pure integration diagnostic here too -- never loosen it to make a
run look clean.

Regime discipline (standing rule 3 of the contract): algebraic identities
(`sym2_coefficients`, `is_symmetric_square`, `certify_sym2_lock`, ...) are
exact and accept `Fraction`; every function with a `float`/`np.ndarray` in its
signature is on the dynamical (float + oracle) side. The two are never mixed
silently.

Tier C caveat (contract section 2): Sym^2(V) is *not* closed under the
Katz-Pavlovic nonlinearity (Observation 2.1/2.2). The Galerkin projection
below is therefore a modelling *choice*, not a consequence of the algebra --
see Lemma 4.4 and section 4.3 of the contract for what is and is not proven.

W1 round 2 -- chart repair (READ THIS BEFORE TOUCHING THE sym2 STATE VECTOR)
--------------------------------------------------------------------------
The sym2 state is ``z = (u0, u1, u2, A, b)`` with ``A = a**2``, NOT
``(u0, u1, u2, a, b)``.  This is the repair of the defect adjudicated in
FINDINGS section 6.2.

Contract eq. 1.2 gives ``(c0, c1, c2) = (-b**3, b(a**2+b), a**2+b)``: the
half-ladder coefficient ``a`` enters *only* through ``a**2``.  The map
``(a, b) -> L_3`` is therefore a 2:1 branched cover of the Sym^2 locus with
branch locus ``a = 0``, and

    dPhi/da = 2a * dPhi/d(a**2)

vanishes identically at ``a = 0``.  The contract's section 1 gauge fix
``a >= 0`` removes the *sign* ambiguity but leaves the branch point.  The
Sym2-locked Galerkin flow drives ``a`` to zero in finite time (measured:
t = 0.1386 at alpha' = 1e-6 and 1e-8, t = 0.2610 at alpha' = 1e-4, N = 24
and N = 30 alike), so ``sigma_min(J) -> 0``, ``zdot = J^+ F`` diverges like
1/a, and RK4 in the ``a``-chart cannot converge across the fold no matter
how small the step -- refining cfl only changes *where* the steps straddle
the branch point.  That, not the timestep rule alone, is what produced the
non-convergent drift table of FINDINGS 6.2.

Using ``A = a**2`` is the honest 1:1 gauge fixing: ``dc/dA = (0, b, 1)`` is
nowhere zero, the fold becomes a regular interior point, and ``A < 0``
(complex half-ladder roots, no *real* order-2 preimage by eq. 1.5) is still
inside the Sym^2 locus (1.4) -- Proposition A is stated over C.  The
constraint set is unchanged; only its parametrisation is.

Consequences for callers: ``reconstruct_profile``/``reconstruct_jacobian``/
``fit_lock_state``/``initial_state`` and ``Sym2ShellResult.lock_parameters``
all carry ``A``, not ``a``.  ``half_ladder_a(A)`` recovers ``a`` where a real
half-ladder exists.  The exact-algebra surface (``sym2_coefficients``,
``is_symmetric_square``, ``certify_sym2_lock``, ``sym2_spectral_radius``)
still speaks in ``(a, b)`` and is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import least_squares

from ..operators.recurrence import (
    RecurrenceOperator,
    closed_form_symmetric_square,
    symmetric_square,
)
from .geometry import effective_wavenumber
from .shell import ShellResult, _rhs, dyadic_wavenumbers

Rational = Fraction | int

LOCKS: tuple[str, ...] = ("none", "sym2", "order3", "veronese")
LOCK_DIM: dict[str, int | None] = {"none": None, "sym2": 5, "order3": 6, "veronese": 4}


def _check_lock(lock: str) -> None:
    if lock not in LOCKS:
        raise ValueError(f"unknown lock {lock!r}; must be one of {LOCKS}")


# ---------------------------------------------------------------------------
# 8.1 Algebra (exact-friendly, no integration)
# ---------------------------------------------------------------------------


def sym2_coefficients(a, b):
    """(c0, c1, c2) for u_{n+3} = c2 u_{n+2} + c1 u_{n+1} + c0 u_n. [eq. 1.1]

    Accepts float or Fraction and returns the same type (whatever arithmetic
    `a`, `b` support). Must equal `closed_form_symmetric_square(a, b)` and
    `symmetric_square(RecurrenceOperator.of(b, a)).coefficients` -- see
    `certify_sym2_lock` (T2).
    """
    return (-(b**3), b * (a * a + b), a * a + b)


def elementary_symmetric(c0, c1, c2):
    """(e1, e2, e3) = (c2, -c1, c0). [eq. 1.2]"""
    return (c2, -c1, c0)


def is_symmetric_square(c0, c1, c2, *, rtol: float = 1e-12) -> bool:
    """Root-free Sym2 membership test e2**3 == e1**3 * e3. [eq. 1.4]

    Exact when c0, c1, c2 are all Fraction/int (rtol is ignored in that
    branch, per the project's "algebra exact, dynamics float" rule);
    otherwise a scale-relative float comparison.
    """
    e1, e2, e3 = elementary_symmetric(c0, c1, c2)
    exact_types = (Fraction, int)
    if isinstance(e1, exact_types) and isinstance(e2, exact_types) and isinstance(e3, exact_types):
        e1, e2, e3 = Fraction(e1), Fraction(e2), Fraction(e3)
        return e2**3 == e1**3 * e3
    e1f, e2f, e3f = float(e1), float(e2), float(e3)
    lhs, rhs = e2f**3, e1f**3 * e3f
    scale = max(abs(lhs), abs(rhs), 1.0)
    return abs(lhs - rhs) <= rtol * scale


def _real_cbrt(x: float) -> float:
    if x == 0:
        return 0.0
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def order3_to_sym2_params(c0, c1, c2) -> tuple[float, float] | None:
    """Inverse map b = -cbrt(e3), a = +sqrt(e1 + cbrt(e3)); a >= 0 gauge. [eq. 1.5]

    Returns None if e1 + cbrt(e3) < 0 (no real preimage).
    """
    e1, e2, e3 = elementary_symmetric(float(c0), float(c1), float(c2))
    cbrt_e3 = _real_cbrt(float(e3))
    b = -cbrt_e3
    radicand = float(e1) + cbrt_e3
    if radicand < 0:
        return None
    return (math.sqrt(radicand), b)


def sym2_spectral_radius(a: float, b: float) -> float:
    """rho_3 = max|root| of L_3, closed form via the discriminant. [eq. 1.6]"""
    disc = a * a + 4 * b
    if disc >= 0:
        return ((abs(a) + math.sqrt(disc)) / 2.0) ** 2
    return abs(b)


# --- repaired sym2 chart: A = a**2 (see module docstring) ------------------


def sym2_coefficients_asq(asq, b):
    """(c0, c1, c2) in the repaired chart coordinate A = a**2. [eq. 1.1]

    Identical to ``sym2_coefficients(a, b)`` whenever ``asq == a*a``; this is
    just eq. 1.1 written in the coordinate that eq. 1.2 actually depends on.
    """
    return (-(b**3), b * (asq + b), asq + b)


def half_ladder_a(asq: float) -> float:
    """Recover the gauge-fixed half-ladder coefficient a = +sqrt(A). [eq. 1.5]

    Returns NaN when A < 0, i.e. when the Sym^2 operator has no *real* order-2
    preimage (eq. 1.5's existence condition e1 + e3**(1/3) >= 0 is exactly
    A >= 0).  Such points are still on the Sym^2 locus (1.4).
    """
    return math.sqrt(asq) if asq >= 0 else float("nan")


def sym2_spectral_radius_asq(asq: float, b: float) -> float:
    """rho_3 in the repaired chart, root-free and valid for A of either sign.

    The three roots of L_3 are in geometric progression with middle root
    ``-b`` (eq. 1.4 / Proposition A), so writing them ``(-b/s, -b, -b s)``
    and matching e1 = A + b gives ``s + 1/s = -(A + 2b)/b``.  Hence
    rho_3 = |b| * (|S| + sqrt(S**2 - 4))/2 when |S| >= 2 and |b| otherwise.
    Agrees with eq. 1.6 wherever both are defined.
    """
    if b == 0.0:
        return abs(asq)
    s_sum = -(asq + 2.0 * b) / b
    if abs(s_sum) >= 2.0:
        return abs(b) * (abs(s_sum) + math.sqrt(s_sum * s_sum - 4.0)) / 2.0
    return abs(b)


def predicted_enstrophy_exponent(rho3: float, *, base: float = 2.0) -> float:
    """p = 0 if rho3 <= 1/base else -1 - log(rho3)/log(base). [eq. 5.3]"""
    if rho3 <= 1.0 / base:
        return 0.0
    return -1.0 - math.log(rho3) / math.log(base)


def cutoff_shell(alpha_prime: float, *, base: float = 2.0) -> float:
    """n_star = -log(alpha_prime) / (2 log base). [eq. 5.1]"""
    return -math.log(alpha_prime) / (2.0 * math.log(base))


# ---------------------------------------------------------------------------
# 8.2 Reconstruction, Jacobian, projection
# ---------------------------------------------------------------------------


def reconstruct_profile(z: np.ndarray, n_shells: int, *, lock: str) -> np.ndarray:
    """u = Phi(z), shape (n_shells,). [eqs. 3.2-3.4]

    lock="none": z is u itself (n_shells,).
    """
    _check_lock(lock)
    z = np.asarray(z, dtype=float)

    if lock == "none":
        if z.shape[0] != n_shells:
            raise ValueError(f"lock='none' requires len(z)=={n_shells}, got {z.shape[0]}")
        return np.array(z, dtype=float)

    if lock == "veronese":
        if z.shape[0] != LOCK_DIM["veronese"]:
            raise ValueError("veronese lock expects z=(v0,v1,a,b)")
        v0, v1, a, b = z
        v = np.zeros(n_shells)
        if n_shells > 0:
            v[0] = v0
        if n_shells > 1:
            v[1] = v1
        for n in range(2, n_shells):
            v[n] = a * v[n - 1] + b * v[n - 2]
        return v**2

    if lock == "sym2":
        if z.shape[0] != LOCK_DIM["sym2"]:
            raise ValueError("sym2 lock expects z=(u0,u1,u2,A,b) with A = a**2")
        u0, u1, u2, asq, b = z
        c0, c1, c2 = sym2_coefficients_asq(asq, b)
    elif lock == "order3":
        if z.shape[0] != LOCK_DIM["order3"]:
            raise ValueError("order3 lock expects z=(u0,u1,u2,c0,c1,c2)")
        u0, u1, u2, c0, c1, c2 = z
    else:  # pragma: no cover - guarded by _check_lock
        raise ValueError(lock)

    u = np.zeros(n_shells)
    if n_shells > 0:
        u[0] = u0
    if n_shells > 1:
        u[1] = u1
    if n_shells > 2:
        u[2] = u2
    for n in range(3, n_shells):
        u[n] = c2 * u[n - 1] + c1 * u[n - 2] + c0 * u[n - 3]
    return u


def reconstruct_jacobian(z: np.ndarray, n_shells: int, *, lock: str) -> np.ndarray:
    """J = dPhi/dz, shape (n_shells, LOCK_DIM[lock]); identity for "none". [eqs. 4.3-4.4]"""
    _check_lock(lock)
    z = np.asarray(z, dtype=float)

    if lock == "none":
        return np.eye(n_shells)

    if lock == "veronese":
        v0, v1, a, b = z
        v = np.zeros(n_shells)
        dv0 = np.zeros(n_shells)
        dv1 = np.zeros(n_shells)
        da = np.zeros(n_shells)
        db = np.zeros(n_shells)
        if n_shells > 0:
            v[0] = v0
            dv0[0] = 1.0
        if n_shells > 1:
            v[1] = v1
            dv1[1] = 1.0
        for n in range(2, n_shells):
            v[n] = a * v[n - 1] + b * v[n - 2]
            dv0[n] = a * dv0[n - 1] + b * dv0[n - 2]
            dv1[n] = a * dv1[n - 1] + b * dv1[n - 2]
            da[n] = a * da[n - 1] + b * da[n - 2] + v[n - 1]
            db[n] = a * db[n - 1] + b * db[n - 2] + v[n - 2]
        jac = np.zeros((n_shells, 4))
        jac[:, 0] = 2 * v * dv0
        jac[:, 1] = 2 * v * dv1
        jac[:, 2] = 2 * v * da
        jac[:, 3] = 2 * v * db
        return jac

    if lock == "sym2":
        _u0, _u1, _u2, asq, b = z
        c0, c1, c2 = sym2_coefficients_asq(asq, b)
        # d(c0,c1,c2)/dA = (0, b, 1) -- eq. 4.4 rewritten in A = a**2.  The
        # legacy a-chart's column was 2a*(0, b, 1), which vanishes at a = 0;
        # that branch point is the round-2 defect (see module docstring).
        param_derivs = [(0.0, b, 1.0), (-3 * b * b, asq + 2 * b, 1.0)]
    elif lock == "order3":
        _u0, _u1, _u2, c0, c1, c2 = z
        param_derivs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    else:  # pragma: no cover
        raise ValueError(lock)

    u = reconstruct_profile(z, n_shells, lock=lock)
    dim = LOCK_DIM[lock]
    assert dim is not None
    jac = np.zeros((n_shells, dim))

    # Amplitude columns j in {0, 1, 2}: G^(j)_i = delta_ij for i<3, then the
    # homogeneous recurrence.
    for j in range(3):
        gj = np.zeros(n_shells)
        if j < n_shells:
            gj[j] = 1.0
        for n in range(3, n_shells):
            gj[n] = c2 * gj[n - 1] + c1 * gj[n - 2] + c0 * gj[n - 3]
        jac[:, j] = gj

    # Parameter columns, eq. 4.3: driven by the same recurrence plus the
    # coefficient-derivative forcing term.
    for pi, (dc0, dc1, dc2) in enumerate(param_derivs):
        d = np.zeros(n_shells)
        for n in range(3, n_shells):
            d[n] = (
                c2 * d[n - 1]
                + c1 * d[n - 2]
                + c0 * d[n - 3]
                + dc2 * u[n - 1]
                + dc1 * u[n - 2]
                + dc0 * u[n - 3]
            )
        jac[:, 3 + pi] = d

    return jac


def _spectral_radius_for(z: np.ndarray, lock: str) -> float:
    if lock == "sym2":
        return sym2_spectral_radius_asq(float(z[-2]), float(z[-1]))
    if lock == "veronese":
        a, b = float(z[-2]), float(z[-1])
        return sym2_spectral_radius(a, b)
    if lock == "order3":
        c0, c1, c2 = float(z[3]), float(z[4]), float(z[5])
        roots = np.roots([1.0, -c2, -c1, -c0])
        return float(np.max(np.abs(roots)))
    return float("nan")


def _locked_zdot(
    z: np.ndarray,
    k: np.ndarray,
    viscosity: float,
    *,
    lock: str,
    closure: str = "galerkin",
    rcond: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """zdot per eq. 4.5 and nothing else: returns (zdot, u, jac, F).

    The bare field, factored out of `locked_rhs` so the RK stages do not pay
    for the per-step diagnostics (a second `lstsq` for the Euler identity plus
    a full SVD).  `locked_rhs` is the public entry point and still returns the
    diagnostics; this one is what the integrator calls 12 times per step.
    Numerically identical -- same `lstsq`, same `rcond`, same arguments.
    """
    n_shells = k.size

    if lock == "none":
        u = np.asarray(z, dtype=float)
        f = _rhs(u, k, viscosity)
        return f, u, np.empty((n_shells, 0)), f

    u = reconstruct_profile(z, n_shells, lock=lock)
    jac = reconstruct_jacobian(z, n_shells, lock=lock)
    f = _rhs(u, k, viscosity)

    if closure == "galerkin":
        zdot, *_ = np.linalg.lstsq(jac, f, rcond=rcond)
    elif closure == "collocation":
        dim = jac.shape[1]
        weight = (k**2) * (u**2)
        if np.count_nonzero(weight) >= dim:
            support = np.argsort(weight)[::-1][:dim]
        else:
            support = np.arange(min(dim, n_shells))
        zdot, *_ = np.linalg.lstsq(jac[support, :], f[support], rcond=rcond)
    elif closure == "enstrophy":
        w = k**2
        jt_w = jac.T * w  # J^T W, W = diag(k^2)
        a_mat = jt_w @ jac
        rhs_vec = jt_w @ f
        zdot = np.linalg.pinv(a_mat, rcond=rcond) @ rhs_vec
    else:
        raise ValueError(f"unknown closure {closure!r}")

    return zdot, u, jac, f


def locked_rhs(
    z: np.ndarray,
    k: np.ndarray,
    viscosity: float,
    *,
    lock: str,
    closure: str = "galerkin",
    rcond: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    """zdot per eq. 4.5, plus per-step diagnostics.

    closure="galerkin" uses the Moore-Penrose pseudo-inverse
    (numpy.linalg.lstsq with rcond). Tikhonov regularisation is FORBIDDEN by
    the contract: it breaks Lemma 4.4 (Pu=u) and turns the energy-drift
    oracle into a measurement of the closure instead of the integrator.
    """
    _check_lock(lock)
    k = np.asarray(k, dtype=float)
    z = np.asarray(z, dtype=float)

    if lock == "none":
        f, _u, _jac, _f = _locked_zdot(z, k, viscosity, lock=lock, closure=closure, rcond=rcond)
        diagnostics = {
            "tangency_defect": 0.0,
            "live_fraction": 1.0,
            "removed_production": 0.0,
            "euler_residual": 0.0,
            "sigma_min": 1.0,
            "sigma_max": 1.0,
            "spectral_radius": float("nan"),
        }
        return f, diagnostics

    zdot, u, jac, f = _locked_zdot(z, k, viscosity, lock=lock, closure=closure, rcond=rcond)

    pf = jac @ zdot
    f_norm = float(np.linalg.norm(f))
    tangency_defect = float(np.linalg.norm(f - pf) / f_norm) if f_norm > 0 else 0.0
    live_fraction = float(np.linalg.norm(pf) / f_norm) if f_norm > 0 else 0.0
    removed_production = float(2.0 * np.dot(k**2 * u, f - pf))

    # Euler-identity check Pu = u (Lemma 4.4), P = J J^+ -- always the
    # *galerkin* projector regardless of `closure`, since this tests whether
    # u lies in range(J) at all, not how zdot was computed.
    jplus_u, *_ = np.linalg.lstsq(jac, u, rcond=rcond)
    pu = jac @ jplus_u
    u_norm = float(np.linalg.norm(u))
    euler_residual = float(np.linalg.norm(pu - u) / u_norm) if u_norm > 0 else 0.0

    svals = np.linalg.svd(jac, compute_uv=False)
    sigma_max = float(svals[0]) if svals.size else 0.0
    sigma_min = float(svals[-1]) if svals.size else 0.0

    diagnostics = {
        "tangency_defect": tangency_defect,
        "live_fraction": live_fraction,
        "removed_production": removed_production,
        "euler_residual": euler_residual,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "spectral_radius": _spectral_radius_for(z, lock),
    }
    return zdot, diagnostics


# ---------------------------------------------------------------------------
# Timestep control (contract section 4.7; eq. 4.8 -> 4.8' -> 4.8'')
# ---------------------------------------------------------------------------

STEP_EPS: float = 1e-3  # eq. 4.8's epsilon, the lock-parameter softening

# eq. 4.8''' -- dimensionless length of the directional-derivative probe used to
# measure the lock-parameter curvature (`parameter_curvature`).  The probe
# displacement is delta = CURVATURE_PROBE / rate', i.e. a fixed *fraction* of
# the state's own natural timescale, which makes it a pure function of the state
# and therefore leaves the whole step rule homogeneous of degree 1 in cfl -- the
# property compensated flatness actually needs (FINDINGS 10.2).  Measured
# insensitivity: the flagship's compensated spread over cfl 0.08/0.04/0.02 is
# 1.000x / 1.008x / 1.012x at CURVATURE_PROBE = 1e-2 / 1e-3 / 1e-4.
CURVATURE_PROBE: float = 1e-3


def parameter_curvature(
    z: np.ndarray,
    zdot: np.ndarray,
    k: np.ndarray,
    viscosity: float,
    *,
    lock: str,
    closure: str = "galerkin",
    rcond: float = 1e-10,
    rate: float | None = None,
    probe: float = CURVATURE_PROBE,
) -> np.ndarray:
    """zddot = d/dt zdot along the flow, by one directional finite difference.

        zddot ~ ( zdot(z + delta*zdot) - zdot(z) ) / delta,
        delta = probe / rate'(z)

    Costs exactly one extra field evaluation per step (out of the thirteen a
    doubled step already takes).  `delta` is set by the state's own timescale
    1/rate', never by cfl, so `zddot` is a pure function of z -- see
    `CURVATURE_PROBE`.  Returns zeros for lock="none" (no lock parameters) and
    whenever the difference is not finite.
    """
    z = np.asarray(z, dtype=float)
    if lock == "none":
        return np.zeros_like(z)
    k = np.asarray(k, dtype=float)
    zdot = np.asarray(zdot, dtype=float)
    if rate is None:
        u = reconstruct_profile(z, k.size, lock=lock)
        rate = stability_rate(z, u, zdot, k, viscosity, lock=lock)
    if not (rate > 0 and math.isfinite(rate)):
        return np.zeros_like(z)
    delta = probe / rate
    zdot_ahead = _locked_zdot(
        z + delta * zdot, k, viscosity, lock=lock, closure=closure, rcond=rcond
    )[0]
    zddot = (zdot_ahead - zdot) / delta
    if not np.all(np.isfinite(zddot)):
        return np.zeros_like(z)
    return zddot


def stability_rate(
    z: np.ndarray,
    u: np.ndarray,
    zdot: np.ndarray,
    k: np.ndarray,
    viscosity: float,
    *,
    lock: str,
    eps: float = STEP_EPS,
    zddot: np.ndarray | None = None,
) -> float:
    """eq. 4.8' (zddot=None) / eq. 4.8''' (zddot given) -- the stability rate.

        rate'   = sqrt( sum_n (k_n u_n)**2
                        + sum_theta thetadot**2 / (theta**2 + eps**2)
                        + (nu max_n k_n**2)**2 )

        rate''' = sqrt( rate'**2
                        + sum_theta |thetaddot| / sqrt(theta**2 + eps**2) )

    eq. 4.8' is a *stability* rate: the fastest nonlinear timescale plus the
    drift timescales of the lock parameters. It is >= eq. 4.8's max() form term
    by term (the 2-norm dominates the max, and
    (theta**2+eps**2)**-1/2 >= (|theta|+eps)**-1), so it never permits a longer
    step than the contract allows. It is NOT an accuracy estimate and must not
    be used as one: see `error_density` and eq. 4.8''.

    WHY THE CURVATURE TERM (FINDINGS section 10, contract Erratum 3).  Every
    lock-parameter term of eq. 4.8 and eq. 4.8' is FIRST order in t: it is a
    multiple of |thetadot|.  It therefore vanishes identically at every
    *turning point* of a lock parameter -- and a turning point is exactly where
    RK4's truncation error is largest, because the truncation error is driven
    by the high derivatives of the trajectory, not by its velocity.  So eq. 4.8'
    has a local *minimum* of the rate at each such passage and lengthens the
    step straight through it. Measured: at roots (0.1,-0.6), A(t) turns over at
    t = 0.1545 and the eq. 4.8' rate dips 20.2 -> 4.49 over eight steps while dt
    *grows* 2.5x; the three steps astride that turn carry +125%, +102% and
    -103% of the whole run's energy drift.  eq. 4.8'''s curvature term is
    5.1e3 there (rate' 4.49 -> 71.6), so the passage is resolved instead.

    sqrt(|thetaddot|/|theta|) is the natural rate attached to a turning point
    (the time for the acceleration to change theta by O(theta)), which is why
    it enters rate**2 linearly.  Adding a non-negative term can only shorten
    the step, so eq. 4.8''' >= eq. 4.8' >= eq. 4.8 still holds pointwise and
    the contract's stability convention is tightened, never relaxed.
    lock="none" has no lock parameters and is untouched -- gate T1 is
    bit-for-bit unaffected.
    """
    if lock == "none":
        rate = float(np.max(k * np.abs(u)))
        if viscosity > 0:
            rate = max(rate, viscosity * float(np.max(k**2)))
        return rate
    rate_sq = float(np.sum((k * u) ** 2))
    n_params = 2 if lock in ("sym2", "veronese") else 3
    for pidx in range(len(z) - n_params, len(z)):
        pv, pd = float(z[pidx]), float(zdot[pidx])
        soft = pv * pv + eps * eps
        rate_sq += pd * pd / soft
        if zddot is not None:
            rate_sq += abs(float(zddot[pidx])) / math.sqrt(soft)
    if viscosity > 0:
        rate_sq += (viscosity * float(np.max(k**2))) ** 2
    return math.sqrt(rate_sq)


def step_rate(stability: float, error_density: float) -> float:
    """eq. 4.8'' -- lift the stability rate by the measured local-error density.

        rate'' = ( rate'**5  +  C )**(1/5)

    `C` is the *relative local-error density* returned by `error_density()`:
    the constant for which one RK4 step of length dt has relative local error
    ~ C*dt**5. Choosing dt = cfl/rate'' therefore equidistributes local error
    at ~cfl**5 per step wherever accuracy binds, and falls back to eq. 4.8'
    wherever stability binds. Because rate'' >= rate' >= eq. 4.8's rate, this
    is a strict tightening of both -- no step is ever longer than the contract
    already permitted.

    WHY this term is necessary (FINDINGS section 8.1). eq. 4.8' is blind to
    accuracy.  Measured on the flagship trajectory at cfl = 0.05: its rate has
    a local minimum of 2.115 at t = 0.182, and the step-doubling error density
    measured at that state is C**(1/5) = 4.05 -- accuracy binds there and
    stability does not.  eq. 4.8' therefore steps dt = 2.4e-2 straight through
    that passage, and that single step carries 105.3% of the whole run's
    energy drift (contract bug signature B2).  Refining cfl only moves where
    the step lands, so the drift is an erratic, sign-flipping function of cfl.
    eq. 4.8'' shortens exactly that step by 2.73x and changes almost nothing
    else: the median rate''/rate' over the run is 1.000.

    The fifth-power blend is the natural one: local error scales as dt**5, so
    adding densities in the 5-norm adds the two step-length constraints in the
    units they are each expressed in.
    """
    return float((stability**5 + max(error_density, 0.0)) ** 0.2)


# Noise floor of the step-doubling estimator, in ulps of ||u||.  Measured on
# the flagship seed (FINDINGS 8.1), sweeping dt down: the gap is 114 ulps at
# dt = 2e-5 with local slope 4.86 (still truncation) and 4-7 ulps at dt = 1e-5
# with slope 4.93 (arithmetic noise), essentially independently of N.  64 ulps
# therefore sits between the two.  See `error_density`.
RICHARDSON_NOISE_ULPS: float = 64.0


def error_density(
    u_full: np.ndarray,
    u_half: np.ndarray,
    dt: float,
    *,
    noise_ulps: float = RICHARDSON_NOISE_ULPS,
) -> float:
    """Relative local-error density C from a step-doubling (Richardson) pair.

    `u_full` is Phi(z) after one RK4 step of length dt, `u_half` after two of
    length dt/2. For a 4th-order method u_full - u_exact = C_abs*dt**5 and
    u_half - u_exact = C_abs*dt**5/16, hence

        C = (16/15) * ||u_full - u_half|| / (||u_half|| * dt**5)

    and a step of length dt has relative local error ~ C*dt**5 (the propagated
    two-half-step solution has 1/16 of that). Measured on the reconstructed
    profile u = Phi(z) rather than on z because ||u||**2 = 2E is the energy
    oracle's own scale, and because u has uniform units where z does not.

    NOISE FLOOR, and why it is not optional.  Below a gap of ~`noise_ulps`
    ulps of ||u|| the difference is floating-point noise, not truncation, and
    C then measures machine epsilon divided by dt**5 -- which *diverges* as dt
    shrinks.  Left unfloored that is a positive feedback loop: at the fixed
    point the map is dt -> cfl*dt / ((16/15)*eps)**(1/5), whose gain is
    ~1333*cfl, so for cfl below ~7.5e-4 dt spirals to zero and the run
    terminates "dt_collapse" with nothing wrong with it.  Measured without the
    floor: fine at cfl = 5e-4, "dt_collapse" after 23 steps at cfl = 2e-4 and
    below.  Returning 0.0 when the estimator has no signal makes eq. 4.8''
    fall back to eq. 4.8' there, which is the self-correcting direction.
    """
    scale = float(np.linalg.norm(u_half))
    if scale <= 0 or dt <= 0:
        return 0.0
    gap = float(np.linalg.norm(np.asarray(u_full) - np.asarray(u_half)))
    if gap <= noise_ulps * float(np.spacing(scale)):
        return 0.0
    return (16.0 / 15.0) * gap / (scale * dt**5)


def _order3_coefficients_from_profile(u_target: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares (c0, c1, c2) with u_m ~ c2 u_{m-1} + c1 u_{m-2} + c0 u_{m-3}.

    An *algebraic* starting point for `fit_lock_state`.  Without it the
    Levenberg-Marquardt fit of a geometric profile routinely converges to a
    spurious local minimum: on the default `seed_profile` (which lies on the
    Sym^2 locus exactly, residual 2.8e-18 at a = 0.75, b = -0.125) the old
    fixed start (a, b) = (0.5, 0.1) returned a = 0.708, b = -0.104 with
    relative residual 5.9e-7 -- the V4/T12 failure recorded in FINDINGS 6.4,
    which silently moved the initial condition of every sym2 run.
    """
    n = u_target.size
    if n < 6:
        return None
    design = np.column_stack([u_target[0:-3], u_target[1:-2], u_target[2:-1]])
    rhs_vec = u_target[3:]
    if not np.any(np.abs(design) > 0):
        return None
    coeffs, *_ = np.linalg.lstsq(design, rhs_vec, rcond=None)
    c0, c1, c2 = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    if not all(math.isfinite(c) for c in (c0, c1, c2)):
        return None
    return (c0, c1, c2)


def _z0_candidates(u_target: np.ndarray, lock: str) -> list[np.ndarray]:
    """Starting points for `fit_lock_state`, algebraic ones first."""
    n = u_target.size
    head = list(u_target[: min(3, n)]) + [0.0] * max(0, 3 - n)
    fitted = _order3_coefficients_from_profile(u_target)
    candidates: list[np.ndarray] = []

    if lock == "sym2":
        if fitted is not None:
            c0, c1, c2 = fitted
            # eq. 1.5 in the repaired chart: b = -cbrt(e3), A = e1 + cbrt(e3).
            cbrt_e3 = _real_cbrt(c0)
            candidates.append(np.array([*head, c2 + cbrt_e3, -cbrt_e3]))
        candidates += [np.array([*head, 0.25, 0.1]), np.array([*head, 1.0, -0.25])]
    elif lock == "order3":
        if fitted is not None:
            candidates.append(np.array([*head, *fitted]))
        candidates.append(np.array([*head, *sym2_coefficients(0.5, 0.1)]))
    elif lock == "veronese":
        v0 = math.copysign(math.sqrt(abs(u_target[0])), 1.0) if n > 0 else 0.1
        v1 = math.copysign(math.sqrt(abs(u_target[1])), 1.0) if n > 1 else 0.1
        if fitted is not None:
            inv = order3_to_sym2_params(*fitted)
            if inv is not None:
                candidates.append(np.array([v0, v1, inv[0], inv[1]]))
        candidates.append(np.array([v0, v1, 0.5, 0.1]))
    else:
        raise ValueError(f"fit_lock_state is not defined for lock={lock!r}")
    return candidates


def _default_z0(u_target: np.ndarray, lock: str) -> np.ndarray:
    return _z0_candidates(u_target, lock)[0]


def fit_lock_state(
    u_target: np.ndarray,
    *,
    lock: str,
    z0: np.ndarray | None = None,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Nonlinear least-squares projection of `u_target` onto the constraint set.

    Uses scipy.optimize.least_squares with the analytic Jacobian from
    `reconstruct_jacobian`, started from the *algebraic* order-3 fit of the
    profile (`_order3_coefficients_from_profile`) and refined from the
    remaining candidates if that leaves a residual above 1e-12.  Returns
    (z, relative_residual); the best candidate wins.

    The multi-start is not decoration: gate T12/V4 requires the seed's fit
    residual to be < 1e-10, and the previous single fixed start missed it by
    four orders of magnitude on the module's own `seed_profile` (see
    `_order3_coefficients_from_profile`).
    """
    if lock == "none":
        raise ValueError("fit_lock_state is not defined for lock='none'")
    _check_lock(lock)
    u_target = np.asarray(u_target, dtype=float)
    n_shells = u_target.size
    starts = [np.asarray(z0, dtype=float)] if z0 is not None else _z0_candidates(u_target, lock)

    def residual(z: np.ndarray) -> np.ndarray:
        return reconstruct_profile(z, n_shells, lock=lock) - u_target

    def jac(z: np.ndarray) -> np.ndarray:
        return reconstruct_jacobian(z, n_shells, lock=lock)

    denom = float(np.linalg.norm(u_target))
    best_z: np.ndarray | None = None
    best_res = float("inf")
    for start in starts:
        try:
            result = least_squares(residual, start, jac=jac, max_nfev=max_iter)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
            continue
        u_fit = reconstruct_profile(result.x, n_shells, lock=lock)
        res = float(np.linalg.norm(u_fit - u_target))
        if res < best_res:
            best_res, best_z = res, result.x
        if denom > 0 and best_res <= 1e-12 * denom:
            break
    if best_z is None:  # pragma: no cover - defensive
        best_z, best_res = starts[0], float(np.linalg.norm(residual(starts[0])))
    return best_z, (best_res / denom if denom > 0 else best_res)


# ---------------------------------------------------------------------------
# 8.3 Result type
# ---------------------------------------------------------------------------


@dataclass
class Sym2ShellResult(ShellResult):
    """ShellResult plus the lock diagnostics.

    All new fields default to empty arrays so the parent's field ordering
    (times, enstrophy, energy, peak_wavenumber, final_state, wavenumbers,
    terminated, label, metadata) is respected.
    """

    spectral_radius: np.ndarray = field(default_factory=lambda: np.array([]))
    lock_parameters: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tangency_defect: np.ndarray = field(default_factory=lambda: np.array([]))
    live_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    removed_production: np.ndarray = field(default_factory=lambda: np.array([]))
    euler_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_min: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_shell_index: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def _peak_index(self) -> int:
        return int(np.argmax(self.enstrophy)) if self.enstrophy.size else 0

    @property
    def t_peak(self) -> float:
        if self.times.size == 0:
            return float("nan")
        return float(self.times[self._peak_index])

    @property
    def rho3_at_peak(self) -> float:
        if self.spectral_radius.size == 0:
            return float("nan")
        idx = min(self._peak_index, self.spectral_radius.size - 1)
        return float(self.spectral_radius[idx])

    @property
    def profile_decay_rate(self) -> float:
        """Empirical exp(d log|u_n| / dn) over 2 <= n <= n_star. [eq. 5.3, empirical]

        n_star is read from metadata["alpha_prime"] via eq. 5.1 when present
        (set by sweep_lock_exponent); otherwise the whole available profile
        (2 <= n <= n_shells-1) is used. This fallback is a Tier C convention
        not pinned down by the contract, since profile_decay_rate is a
        property of the *result*, which does not itself carry alpha'.
        """
        u = self.final_state
        n_shells = u.size
        base = float(self.metadata.get("base", 2.0))
        alpha_prime = self.metadata.get("alpha_prime")
        n_star = cutoff_shell(alpha_prime, base=base) if alpha_prime is not None else n_shells - 1
        lo, hi = 2, int(min(n_star, n_shells - 1))
        if hi <= lo:
            return float("nan")
        ns = np.arange(lo, hi + 1)
        vals = np.abs(u[lo : hi + 1])
        mask = vals > 0
        if np.count_nonzero(mask) < 2:
            return float("nan")
        slope, _intercept = np.polyfit(ns[mask], np.log(vals[mask]), 1)
        return float(np.exp(slope))

    @property
    def top_shell_enstrophy_fraction(self) -> float:  # V3
        u = self.final_state
        k = self.wavenumbers
        weight = k**2 * u**2
        total = float(weight.sum())
        if total <= 0:
            return float("nan")
        top3 = float(np.sort(weight)[-3:].sum())
        return top3 / total

    def summary(self) -> dict[str, object]:
        """Parent summary plus lock diagnostics; see contract section 8.3."""
        base_summary = super().summary()
        lock = str(self.metadata.get("lock", ""))
        closure = str(self.metadata.get("closure", ""))
        base = float(self.metadata.get("base", 2.0))

        nan = float("nan")
        rho3 = self.rho3_at_peak
        predicted = predicted_enstrophy_exponent(rho3, base=base) if math.isfinite(rho3) else nan
        peak_shell_at_peak = (
            float(self.peak_shell_index[self._peak_index]) if self.peak_shell_index.size else nan
        )

        if self.lock_parameters.size:
            param_drift = np.std(self.lock_parameters, axis=0)
            frozen = bool(np.all(param_drift < 1e-6))
        else:
            param_drift = np.array([])
            frozen = False

        off_manifold = bool(self.euler_residual.size and np.max(self.euler_residual) > 1e-8)
        void = bool(frozen or off_manifold)

        max_td = float(np.max(self.tangency_defect)) if self.tangency_defect.size else nan
        min_lf = float(np.min(self.live_fraction)) if self.live_fraction.size else nan
        max_er = float(np.max(self.euler_residual)) if self.euler_residual.size else nan

        extra: dict[str, object] = {
            "lock": lock,
            "closure": closure,
            "rho3_at_peak": rho3,
            "profile_decay_rate": self.profile_decay_rate,
            "predicted_exponent": predicted,
            "peak_shell_at_peak": peak_shell_at_peak,
            "max_tangency_defect": max_td,
            "min_live_fraction": min_lf,
            "max_euler_residual": max_er,
            "param_drift": param_drift,
            "top_shell_enstrophy_fraction": self.top_shell_enstrophy_fraction,
            "void": void,
        }
        return {**base_summary, **extra}


# ---------------------------------------------------------------------------
# 8.4 Simulation and sweep
# ---------------------------------------------------------------------------


def seed_profile(
    n_shells: int,
    *,
    lock: str = "sym2",
    roots: tuple[float, float] = (0.5, 0.25),
    amplitudes: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    normalize_energy: bool = True,
    base: float = 2.0,
) -> np.ndarray:
    """Default seed, shared across locks so the lock is the only causal variable.

    u_n = A l**(2n) + B (l m)**n + C m**(2n) with (l, m) = roots. Renormalised
    to ||u||_2 = 1 (E(0) = 1/2), matching the control arm of FINDINGS 4.
    Defaults give rho_3(0) = 0.25 < 1/kappa = 0.5 (sub-critical) for base=2,
    so enstrophy growth is produced by the dynamics, not handed to it at t=0.

    For lock="veronese" the equal-amplitude choice is replaced by the nearest
    rank-1 profile (A, B, C) = (1, 2, 1), which satisfies B**2 = 4AC [eq. 3.5]
    and keeps the seed on-manifold for that lock (V4).
    """
    _check_lock(lock)
    lam, mu = roots
    a_amp, b_amp, c_amp = (1.0, 2.0, 1.0) if lock == "veronese" else amplitudes
    n = np.arange(n_shells, dtype=float)
    u = a_amp * lam ** (2 * n) + b_amp * (lam * mu) ** n + c_amp * mu ** (2 * n)
    if normalize_energy:
        norm = np.linalg.norm(u)
        if norm > 0:
            u = u / norm
    return u


def simulate_sym2_shell_model(
    wavenumbers: np.ndarray,
    *,
    lock: str = "sym2",
    closure: str = "galerkin",
    t_max: float = 12.0,
    viscosity: float = 0.0,
    initial_profile: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    cfl: float = 0.05,
    max_steps: int = 300_000,
    enstrophy_ceiling: float = 1e30,
    rho_ceiling: float = 1.0,
    min_dt: float = 1e-12,
    n_samples: int = 2000,
    rcond: float = 1e-10,
    cond_ceiling: float = 1e8,
    base: float = 2.0,
    label: str = "",
) -> Sym2ShellResult:
    """Adaptive RK4 on z (eq. 4.5), reconstructing u = Phi(z) at each sample.

    Mirrors simulate_shell_model's contract: same explicit termination
    reporting, same energy-conservation oracle, which Lemma 4.4 shows remains
    a pure integrator diagnostic under closure="galerkin". With lock="none"
    this reduces exactly to simulate_shell_model (gate T1/B4).

    Timestep rule (eq. 4.8'''/4.8'', W1 round 2c).  For lock="none" the rate is
    bit-for-bit `shell.py`'s ``max(max_n k_n|u_n|, nu max_n k_n**2)`` and the
    step is a single RK4 -- that is what gate T1 pins, and nothing below
    touches it.  For a locked run,

        dt = cfl / ( stability_rate(..., zddot=zddot)**5 + C )**(1/5)

    where `stability_rate` with `zddot` is eq. 4.8''' (the eq. 4.8' smooth 2-norm
    blend of eq. 4.8's terms, plus the lock-parameter CURVATURE term) and `C` is the
    relative local-error density measured by step doubling on the *previous*
    step (`error_density`).  The step itself is taken as two half RK4 steps,
    with the single full step retained only as the Richardson estimator -- so
    `C` is available for the next step at no extra field evaluations beyond
    the doubling itself.  `zddot` costs exactly one more.

    eq. 4.8' alone is blind to accuracy.  On the flagship trajectory its rate
    has a local minimum (2.115) at t = 0.182, where the measured local-error
    density is C**(1/5) = 4.05: accuracy binds and stability does not, so
    eq. 4.8' steps dt = 2.4e-2 straight through, and that one step carries
    105.3% of the run's energy drift.  Refining cfl only moves where the step
    lands, which is the erratic, sign-flipping refinement table FINDINGS 7.2
    refuted the previous repair on.  eq. 4.8'' keeps eq. 4.8' as a floor (so
    it remains a strict tightening of the contract's eq. 4.8) and lifts it
    only where the measured error says the step must be shorter -- the median
    rate''/rate' over a run is 1.000.  See `step_rate` and FINDINGS 8.

    eq. 4.8'' was not enough, and FINDINGS 9.2 refuted it on generalisation:
    at 2 of 13 well-conditioned seeds the full-window drift is still not
    compensated-flat.  The reason (FINDINGS 10) is that eq. 4.8's local minima
    are not accidents -- every lock-parameter term of eq. 4.8 and 4.8' is a
    multiple of |thetadot| and therefore vanishes IDENTICALLY at every turning
    point of a lock parameter, which is precisely where the truncation error
    peaks.  eq. 4.8''' adds the curvature term |thetaddot|/sqrt(theta**2+eps**2)
    so the rate cannot collapse there.  The defect is present at the flagship
    too; what is seed-dependent is only how much of the run's drift the
    unresolved passage carries (26% at the flagship, ~100% at the two failing
    seeds).  See `stability_rate` and `parameter_curvature`.

    `cond_ceiling` terminates the run "lock_singular" if cond(J) exceeds it for
    20 consecutive steps.  This is a *new, tighter* guard than the previous
    ``sigma_min < rcond*sigma_max`` test (which at rcond=1e-10 could not see a
    degeneracy until the Jacobian was numerically rank-deficient to machine
    precision).  It is what turns a silent, dt-insensitive energy injection
    into a reported termination.
    """
    _check_lock(lock)
    k = np.asarray(wavenumbers, dtype=float)
    n = k.size

    if initial_state is not None:
        z = np.array(initial_state, dtype=float)
    else:
        profile = (
            np.array(initial_profile, dtype=float)
            if initial_profile is not None
            else seed_profile(n, lock=lock, base=base)
        )
        if lock == "none":
            z = profile
        else:
            z, _resid = fit_lock_state(profile, lock=lock)

    sample_times = np.linspace(0.0, t_max, n_samples)
    next_sample = 0

    times: list[float] = []
    enstrophy: list[float] = []
    energy: list[float] = []
    peak_k: list[float] = []
    spectral_radius: list[float] = []
    lock_parameters: list[np.ndarray] = []
    tangency_defect: list[float] = []
    live_fraction: list[float] = []
    removed_production: list[float] = []
    euler_residual: list[float] = []
    sigma_min: list[float] = []
    peak_shell_index: list[int] = []

    def profile_of(zstate: np.ndarray) -> np.ndarray:
        return zstate if lock == "none" else reconstruct_profile(zstate, n, lock=lock)

    def rhs_of(zstate: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return locked_rhs(zstate, k, viscosity, lock=lock, closure=closure, rcond=rcond)

    def field(zstate: np.ndarray) -> np.ndarray:
        return _locked_zdot(zstate, k, viscosity, lock=lock, closure=closure, rcond=rcond)[0]

    def record(t: float, zstate: np.ndarray) -> None:
        u = profile_of(zstate)
        _zdot, diag = rhs_of(zstate)
        times.append(t)
        enstrophy.append(float(np.sum(k**2 * u**2)))
        energy.append(float(0.5 * np.sum(u**2)))
        weight = k**2 * u**2
        idx = int(np.argmax(weight)) if weight.max() > 0 else 0
        peak_k.append(float(k[idx]))
        peak_shell_index.append(idx)
        spectral_radius.append(diag["spectral_radius"])
        params = np.array(zstate[3:], dtype=float) if lock != "none" else np.array([])
        lock_parameters.append(params)
        tangency_defect.append(diag["tangency_defect"])
        live_fraction.append(diag["live_fraction"])
        removed_production.append(diag["removed_production"])
        euler_residual.append(diag["euler_residual"])
        sigma_min.append(diag["sigma_min"])

    def rk4_step(zstate: np.ndarray, dt: float) -> np.ndarray:
        k1 = field(zstate)
        k2 = field(zstate + 0.5 * dt * k1)
        k3 = field(zstate + 0.5 * dt * k2)
        k4 = field(zstate + dt * k3)
        return zstate + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def doubled_step(zstate: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
        """Two half RK4 steps (propagated) + one full step (Richardson estimator)."""
        z_full = rk4_step(zstate, dt)
        z_half = rk4_step(rk4_step(zstate, 0.5 * dt), 0.5 * dt)
        density = error_density(profile_of(z_full), profile_of(z_half), dt)
        return z_half, density

    record(0.0, z)
    t = 0.0
    terminated = "t_max"
    singular_streak = 0
    max_condition = 0.0
    err_density = 0.0  # eq. 4.8'' -- C, carried over from the previous step
    warmed = False
    n_steps_taken = 0

    for step_index in range(max_steps):
        u = profile_of(z)
        zdot_now = field(z)

        # eq. 4.8'''/4.8''.  lock="none" keeps shell.py's rate bit-for-bit
        # (gate T1) and a plain single RK4 step; a locked run uses the
        # curvature- and error-lifted rate and the doubled step.
        rate = stability_rate(z, u, zdot_now, k, viscosity, lock=lock)
        if lock != "none":
            # eq. 4.8''': (4.8')'s lock-parameter terms are first order in t and
            # so vanish at every turning point of a lock parameter -- exactly
            # where the truncation error peaks.  The curvature term removes that
            # structural blindness; `delta` is set by 1/rate', never by cfl, so
            # the rule stays homogeneous in cfl.  See `stability_rate`.
            zddot = parameter_curvature(
                z, zdot_now, k, viscosity, lock=lock, closure=closure, rcond=rcond, rate=rate
            )
            rate = stability_rate(z, u, zdot_now, k, viscosity, lock=lock, zddot=zddot)
            if not warmed:
                # Startup probe: without it the very first step -- where the
                # error density is largest on this seed -- would be the one
                # step of the run that is not error-controlled.
                dt_probe = cfl / rate if rate > 0 else 0.0
                if dt_probe > 0:
                    if t + dt_probe > t_max:
                        dt_probe = t_max - t
                    _z_probe, err_density = doubled_step(z, dt_probe)
                warmed = True
            rate = step_rate(rate, err_density)

        if rate <= 0 or not np.isfinite(rate):
            terminated = "degenerate" if rate <= 0 else "non_finite"
            break

        dt = cfl / rate
        if dt < min_dt:
            record(t, z)
            terminated = "dt_collapse"
            break
        if t + dt > t_max:
            dt = t_max - t

        if lock == "none":
            z = rk4_step(z, dt)
        else:
            z, err_density = doubled_step(z, dt)
        t += dt
        n_steps_taken += 1
        u = profile_of(z)

        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(u)):
            terminated = "non_finite"
            break

        while next_sample < n_samples and sample_times[next_sample] <= t:
            record(t, z)
            next_sample += 1

        current_enstrophy = float(np.sum(k**2 * u**2))
        if current_enstrophy > enstrophy_ceiling:
            record(t, z)
            terminated = "enstrophy_ceiling"
            break

        if lock != "none":
            _zdot_post, diag_post = rhs_of(z)
            rho_now = diag_post["spectral_radius"]
            if np.isfinite(rho_now) and rho_now > rho_ceiling:
                record(t, z)
                terminated = "rho_ceiling"
                break
            if diag_post["euler_residual"] > 1e-8:
                record(t, z)
                terminated = "off_manifold"
                break
            # Conditioning guard.  The rcond test alone only fires once J is
            # rank-deficient to machine precision; cond_ceiling catches the
            # approach, so a degenerate passage is *reported* rather than
            # silently absorbed into the energy budget (FINDINGS 6.2).
            s_min, s_max = diag_post["sigma_min"], diag_post["sigma_max"]
            if s_min > 0:
                max_condition = max(max_condition, s_max / s_min)
            rank_deficient = s_min < rcond * s_max
            over_conditioned = cond_ceiling > 0 and s_min * cond_ceiling < s_max
            if rank_deficient or over_conditioned:
                singular_streak += 1
            else:
                singular_streak = 0
            if singular_streak >= 20:
                record(t, z)
                terminated = "lock_singular"
                break

        if t >= t_max:
            break
        if step_index == max_steps - 1:
            terminated = "max_steps"

    if not times or times[-1] != t:
        record(t, z)

    final_u = profile_of(z)

    metadata: dict[str, object] = {
        "n_shells": n,
        "viscosity": viscosity,
        "cfl": cfl,
        "k_max": float(k.max()),
        "lock": lock,
        "closure": closure,
        "rcond": rcond,
        "cond_ceiling": cond_ceiling,
        "max_jacobian_condition": max_condition,
        "n_steps": n_steps_taken,
        "base": base,
    }

    return Sym2ShellResult(
        times=np.array(times),
        enstrophy=np.array(enstrophy),
        energy=np.array(energy),
        peak_wavenumber=np.array(peak_k),
        final_state=final_u,
        wavenumbers=k,
        terminated=terminated,
        label=label,
        metadata=metadata,
        spectral_radius=np.array(spectral_radius),
        lock_parameters=np.array(lock_parameters) if lock != "none" else np.zeros((0, 0)),
        tangency_defect=np.array(tangency_defect),
        live_fraction=np.array(live_fraction),
        removed_production=np.array(removed_production),
        euler_residual=np.array(euler_residual),
        sigma_min=np.array(sigma_min),
        peak_shell_index=np.array(peak_shell_index),
    )


def compare_lock(
    n_shells: int = 30,
    alpha_prime: float = 1e-6,
    *,
    locks: tuple[str, ...] = ("none", "sym2", "order3", "veronese"),
    t_max: float = 12.0,
    viscosity: float = 0.0,
    cfl: float = 0.05,
    closure: str = "galerkin",
) -> dict[str, Sym2ShellResult]:
    """Same ladder, same seed, same integrator, same tolerances per lock.

    Direct analogue of compare_regularization(): the lock is the only causal
    variable (contract C4).
    """
    k_eff = effective_wavenumber(alpha_prime, dyadic_wavenumbers(n_shells))
    results: dict[str, Sym2ShellResult] = {}
    for lock in locks:
        seed_lock = lock if lock != "none" else "sym2"
        seed = seed_profile(n_shells, lock=seed_lock)
        results[lock] = simulate_sym2_shell_model(
            k_eff,
            lock=lock,
            closure=closure,
            t_max=t_max,
            viscosity=viscosity,
            initial_profile=seed,
            cfl=cfl,
            label=lock,
        )
    return results


@dataclass
class LockSweep:
    lock: str
    closure: str
    alphas: np.ndarray
    peak_enstrophy: np.ndarray
    exponent: float
    exponent_stderr: float
    rho3_at_peak: np.ndarray
    profile_decay_rate: np.ndarray
    predicted_exponent: np.ndarray
    peak_shell: np.ndarray
    n_star: np.ndarray
    terminated: list
    energy_drift: np.ndarray
    max_tangency_defect: np.ndarray
    min_live_fraction: np.ndarray
    param_drift: np.ndarray
    verdict: str

    def table(self) -> list[dict[str, object]]:
        rows = []
        for i in range(len(self.alphas)):
            rows.append(
                {
                    "alpha_prime": float(self.alphas[i]),
                    "peak_enstrophy": float(self.peak_enstrophy[i]),
                    "rho3_at_peak": float(self.rho3_at_peak[i]),
                    "profile_decay_rate": float(self.profile_decay_rate[i]),
                    "predicted_exponent": float(self.predicted_exponent[i]),
                    "peak_shell": float(self.peak_shell[i]),
                    "n_star": float(self.n_star[i]),
                    "terminated": self.terminated[i],
                    "energy_drift": float(self.energy_drift[i]),
                }
            )
        return rows


def sweep_lock_exponent(
    alphas: np.ndarray | None = None,
    *,
    lock: str = "sym2",
    closure: str = "galerkin",
    n_shells: int = 30,
    t_max: float = 12.0,
    cfl: float = 0.05,
    base: float = 2.0,
) -> LockSweep:
    """The Hypothesis-U measurement, per-lock (contract section 6.1).

    Fits the exponent only over runs that terminated "t_max"; the excluded
    runs are still recorded in `terminated`.
    """
    if alphas is None:
        alphas = np.logspace(-2, -10, 9)
    alphas = np.asarray(alphas, dtype=float)

    peak_enstrophy = []
    rho3 = []
    decay = []
    pred = []
    peak_shell = []
    n_star = []
    terminated = []
    edrift = []
    max_td = []
    min_lf = []
    p_drift = []

    seed_lock = lock if lock != "none" else "sym2"
    for alpha in alphas:
        k_eff = effective_wavenumber(float(alpha), dyadic_wavenumbers(n_shells))
        seed = seed_profile(n_shells, lock=seed_lock, base=base)
        result = simulate_sym2_shell_model(
            k_eff,
            lock=lock,
            closure=closure,
            t_max=t_max,
            cfl=cfl,
            initial_profile=seed,
            base=base,
            label=f"{lock},a'={alpha:g}",
        )
        result.metadata["alpha_prime"] = float(alpha)

        peak_enstrophy.append(result.max_enstrophy)
        rho3.append(result.rho3_at_peak)
        decay.append(result.profile_decay_rate)
        pred.append(
            predicted_enstrophy_exponent(result.rho3_at_peak, base=base)
            if math.isfinite(result.rho3_at_peak)
            else float("nan")
        )
        if result.peak_shell_index.size:
            peak_shell.append(float(result.peak_shell_index[result._peak_index]))
        else:
            peak_shell.append(float("nan"))
        n_star.append(cutoff_shell(float(alpha), base=base))
        terminated.append(result.terminated)
        edrift.append(result.energy_drift)
        if result.tangency_defect.size:
            max_td.append(float(np.max(result.tangency_defect)))
        else:
            max_td.append(float("nan"))
        if result.live_fraction.size:
            min_lf.append(float(np.min(result.live_fraction)))
        else:
            min_lf.append(float("nan"))
        if result.lock_parameters.size:
            p_drift.append(float(np.max(np.std(result.lock_parameters, axis=0))))
        else:
            p_drift.append(0.0)

    peak_enstrophy_arr = np.array(peak_enstrophy)
    ok = np.array([term == "t_max" for term in terminated])
    if np.count_nonzero(ok) >= 2 and np.all(peak_enstrophy_arr[ok] > 0):
        x = np.log10(alphas[ok])
        y = np.log10(peak_enstrophy_arr[ok])
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        exponent = float(coeffs[0])
        exponent_stderr = float(np.sqrt(cov[0, 0]))
    else:
        exponent = float("nan")
        exponent_stderr = float("nan")

    sweep = LockSweep(
        lock=lock,
        closure=closure,
        alphas=alphas,
        peak_enstrophy=peak_enstrophy_arr,
        exponent=exponent,
        exponent_stderr=exponent_stderr,
        rho3_at_peak=np.array(rho3),
        profile_decay_rate=np.array(decay),
        predicted_exponent=np.array(pred),
        peak_shell=np.array(peak_shell),
        n_star=np.array(n_star),
        terminated=terminated,
        energy_drift=np.array(edrift),
        max_tangency_defect=np.array(max_td),
        min_live_fraction=np.array(min_lf),
        param_drift=np.array(p_drift),
        verdict="",
    )
    sweep.verdict = classify_sweep(sweep)
    return sweep


def classify_sweep(sweep: LockSweep, *, arrest_tol: float = 0.05) -> str:
    """Apply (a coarse version of) the decision table of contract section 6.3.

    This is explicitly Tier C protocol (section 10): a heuristic reading of
    A1-A4/D1-D5, not a certified classifier. It never returns "D6_bug" -- bug
    signatures are gates (section 9), not a classification outcome.
    """
    if sweep.param_drift.size and np.any(sweep.param_drift < 1e-6):
        return "void"
    p = sweep.exponent
    if not math.isfinite(p):
        return "void"

    finite_peaks = sweep.peak_enstrophy[np.isfinite(sweep.peak_enstrophy)]
    ratio = (
        float(finite_peaks[-1] / finite_peaks[0])
        if finite_peaks.size >= 2 and finite_peaks[0] != 0
        else float("inf")
    )
    nan = float("nan")
    mean_rho3 = float(np.nanmean(sweep.rho3_at_peak)) if sweep.rho3_at_peak.size else nan
    mean_live = float(np.nanmin(sweep.min_live_fraction)) if sweep.min_live_fraction.size else nan
    has_td = sweep.max_tangency_defect.size
    mean_td = float(np.nanmean(sweep.max_tangency_defect)) if has_td else nan

    if abs(p) < arrest_tol:
        if math.isfinite(mean_live) and mean_live < 0.05:
            return "D5_closure_artifact"
        if ratio < 10 and math.isfinite(mean_rho3) and mean_rho3 <= 1.0 / 2.0 + 0.05:
            return "D1_arrest"
        return "D3_marginal"
    if abs(p + 2.0 / 3.0) < 0.1:
        return "D2p_geometrically_inert" if math.isfinite(mean_td) and mean_td < 0.1 else "D2_inert"
    if p <= -1:
        return "D4_worse"
    return "void"


def separation_test(sym2: LockSweep, order3: LockSweep) -> dict[str, float]:
    """Delta p = p(sym2) - p(order3) with propagated stderr (contract section 6.5).

    `order3_sym2_violation_fraction` (guard B6) needs the per-timestep
    (c0,c1,c2) trace of the order3 run, which LockSweep does not retain (only
    the per-alpha' std of the lock parameters). It is reported as NaN here
    rather than silently fabricated; a caller wanting B6 exactly should check
    `is_symmetric_square` directly against `Sym2ShellResult.lock_parameters`
    of an individual order3 run.
    """
    delta_p = sym2.exponent - order3.exponent
    stderr = math.sqrt(sym2.exponent_stderr**2 + order3.exponent_stderr**2)
    return {
        "delta_p": float(delta_p),
        "delta_p_stderr": float(stderr),
        "order3_sym2_violation_fraction": float("nan"),
    }


# ---------------------------------------------------------------------------
# 8.5 Exact certificate (project convention)
# ---------------------------------------------------------------------------


def _exact_tangency_family_check(n_terms: int = 12) -> bool:
    """Gate T6: the KP field on a pure geometric profile is itself Sym^2, exactly.

    kappa is chosen as a perfect square (4, not the physical base 2) so that
    mu = sqrt(kappa) * lambda^2 is rational whenever lambda is -- this is the
    "rational instance" the contract's T6 asks for; it is independent of, and
    does not stand in for, the base=2 dyadic ladder used elsewhere.
    """
    kappa = Fraction(4)
    lam = Fraction(1, 3)
    r = lam * lam
    mu = 2 * r  # sqrt(kappa) * lambda^2, sqrt(4)=2 exactly
    a = lam + mu
    b = -(lam * mu)
    c0, c1, c2 = sym2_coefficients(a, b)
    l3 = RecurrenceOperator.of(c0, c1, c2)

    kappa_r2 = kappa * r * r
    factor = Fraction(1) / (kappa * r * r) - r
    amp = Fraction(1)
    f_seq = [amp * amp * kappa_r2**n * factor for n in range(n_terms)]
    return l3.annihilates(f_seq)


def certify_sym2_lock(
    a: Rational = Fraction(3, 4),
    b: Rational = Fraction(-1, 8),
    *,
    n_terms: int = 24,
) -> dict[str, object]:
    """Exact-Fraction certificate tying this module to operators/recurrence.py."""
    a, b = Fraction(a), Fraction(b)
    c0, c1, c2 = sym2_coefficients(a, b)
    closed = closed_form_symmetric_square(a, b)
    construction = symmetric_square(RecurrenceOperator.of(b, a)).coefficients

    coefficients_match_closed_form = (c0, c1, c2) == closed
    coefficients_match_construction = (c0, c1, c2) == tuple(construction)

    l3 = RecurrenceOperator.of(c0, c1, c2)
    profile = l3.iterate([Fraction(1), Fraction(2), Fraction(3)], n_terms)
    profile_annihilated = l3.annihilates(profile)

    sym2_locus_holds = is_symmetric_square(c0, c1, c2) is True
    perturbation_detected = is_symmetric_square(c0, c1 + 1, c2) is False

    inv = order3_to_sym2_params(float(c0), float(c1), float(c2))
    inverse_map_roundtrips = (
        inv is not None and abs(inv[0] - abs(float(a))) < 1e-9 and abs(inv[1] - float(b)) < 1e-9
    )

    exact_tangency_family = _exact_tangency_family_check()

    checks = {
        "coefficients_match_closed_form": coefficients_match_closed_form,
        "coefficients_match_construction": coefficients_match_construction,
        "profile_annihilated": profile_annihilated,
        "sym2_locus_holds": sym2_locus_holds,
        "perturbation_detected": perturbation_detected,
        "inverse_map_roundtrips": inverse_map_roundtrips,
        "exact_tangency_family": exact_tangency_family,
    }
    return {
        **checks,
        "certified": all(checks.values()),
        "arithmetic": "exact (Fraction)",
    }
