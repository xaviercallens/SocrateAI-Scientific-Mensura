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
            raise ValueError("sym2 lock expects z=(u0,u1,u2,a,b)")
        u0, u1, u2, a, b = z
        c0, c1, c2 = sym2_coefficients(a, b)
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
        _u0, _u1, _u2, a, b = z
        c0, c1, c2 = sym2_coefficients(a, b)
        param_derivs = [(0.0, 2 * a * b, 2 * a), (-3 * b * b, a * a + 2 * b, 1.0)]
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
    if lock in ("sym2", "veronese"):
        a, b = float(z[-2]), float(z[-1])
        return sym2_spectral_radius(a, b)
    if lock == "order3":
        c0, c1, c2 = float(z[3]), float(z[4]), float(z[5])
        roots = np.roots([1.0, -c2, -c1, -c0])
        return float(np.max(np.abs(roots)))
    return float("nan")


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
    n_shells = k.size
    z = np.asarray(z, dtype=float)

    if lock == "none":
        u = z
        f = _rhs(u, k, viscosity)
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


def _default_z0(u_target: np.ndarray, lock: str) -> np.ndarray:
    n = u_target.size
    head = list(u_target[: min(3, n)]) + [0.0] * max(0, 3 - n)
    if lock == "sym2":
        return np.array([*head, 0.5, 0.1])
    if lock == "order3":
        c0, c1, c2 = sym2_coefficients(0.5, 0.1)
        return np.array([*head, c0, c1, c2])
    if lock == "veronese":
        v0 = math.copysign(math.sqrt(abs(u_target[0])), 1.0) if n > 0 else 0.1
        v1 = math.copysign(math.sqrt(abs(u_target[1])), 1.0) if n > 1 else 0.1
        return np.array([v0, v1, 0.5, 0.1])
    raise ValueError(f"fit_lock_state is not defined for lock={lock!r}")


def fit_lock_state(
    u_target: np.ndarray,
    *,
    lock: str,
    z0: np.ndarray | None = None,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Nonlinear least-squares projection of `u_target` onto the constraint set.

    Uses scipy.optimize.least_squares with the analytic Jacobian from
    `reconstruct_jacobian`. Returns (z, relative_residual).
    """
    if lock == "none":
        raise ValueError("fit_lock_state is not defined for lock='none'")
    _check_lock(lock)
    u_target = np.asarray(u_target, dtype=float)
    n_shells = u_target.size
    if z0 is None:
        z0 = _default_z0(u_target, lock)

    def residual(z: np.ndarray) -> np.ndarray:
        return reconstruct_profile(z, n_shells, lock=lock) - u_target

    def jac(z: np.ndarray) -> np.ndarray:
        return reconstruct_jacobian(z, n_shells, lock=lock)

    result = least_squares(residual, z0, jac=jac, max_nfev=max_iter)
    z = result.x
    u_fit = reconstruct_profile(z, n_shells, lock=lock)
    denom = float(np.linalg.norm(u_target))
    rel_residual = float(np.linalg.norm(u_fit - u_target) / denom) if denom > 0 else float(
        np.linalg.norm(u_fit - u_target)
    )
    return z, rel_residual


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
    base: float = 2.0,
    label: str = "",
) -> Sym2ShellResult:
    """Adaptive RK4 on z (eq. 4.5), reconstructing u = Phi(z) at each sample.

    Mirrors simulate_shell_model's contract: same adaptive-step philosophy
    (eq. 4.8), same explicit termination reporting, same energy-conservation
    oracle, which Lemma 4.4 shows remains a pure integrator diagnostic under
    closure="galerkin". With lock="none" this reduces exactly to
    simulate_shell_model (gate T1/B4).
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
        k1, _ = rhs_of(zstate)
        k2, _ = rhs_of(zstate + 0.5 * dt * k1)
        k3, _ = rhs_of(zstate + 0.5 * dt * k2)
        k4, _ = rhs_of(zstate + dt * k3)
        return zstate + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    record(0.0, z)
    t = 0.0
    terminated = "t_max"
    eps = 1e-3
    singular_streak = 0

    for step_index in range(max_steps):
        u = profile_of(z)
        zdot_now, diag_now = rhs_of(z)

        rate_terms = [float(np.max(k * np.abs(u)))]
        if lock != "none":
            n_params = 2 if lock in ("sym2", "veronese") else 3
            for pidx in range(len(z) - n_params, len(z)):
                pv, pd = z[pidx], zdot_now[pidx]
                rate_terms.append(abs(pd) / (abs(pv) + eps))
        if viscosity > 0:
            rate_terms.append(viscosity * float(np.max(k**2)))
        rate = max(rate_terms)

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

        z = rk4_step(z, dt)
        t += dt
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
            if diag_post["sigma_min"] < rcond * diag_post["sigma_max"]:
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
