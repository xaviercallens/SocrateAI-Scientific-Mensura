"""F3 diagnostic: Benettin-style Lyapunov spectrum -- NumPy reference implementation.

Purpose. docs/MENSURA_BENCHMARK.md scores problems 08 (Lorenz) and 09
(Rossler) against target dimensions 2.05 / 2.01 sourced to [GP83], i.e. to the
original paper of the *baseline method* the benchmark is comparing against.
An external reviewer flagged that as circular. To test it, a METHOD-INDEPENDENT
target is needed. This module computes one: the Kaplan-Yorke (Lyapunov)
dimension

    D_KY = j + (lambda_1 + ... + lambda_j) / |lambda_{j+1}|,
    j = largest index with lambda_1 + ... + lambda_j >= 0,

which is a property of the FLOW (its tangent dynamics), computed without ever
counting a neighbour or a pair, and therefore shares no machinery with either
estimator under test.

Algorithm (Benettin, Galgani, Giorgilli & Strelcyn 1980). Integrate the
trajectory and the full variational system simultaneously,

    xdot = f(x),    Qdot = J(x) Q,   Q(0) = I,

with one RK4 step applied to the augmented (n + n^2)-vector, so the tangent
stages are evaluated at the correct intermediate states. Every `t_reorth` time
units, re-orthonormalise Q by modified Gram-Schmidt and accumulate the logs of
the resulting norms (the diagonal of R). Exponent i is the accumulated log of
column i divided by elapsed time.

THIS FILE IS THE REFERENCE, NOT THE WORKHORSE. It is deliberately plain NumPy
so it can be read and checked line by line; f3_lyap.c re-implements exactly
this algorithm for the long integrations, and f3_crosscheck.py verifies the two
agree to near machine precision on identical short runs. Neither is trusted on
Lorenz or Rossler until the known-answer battery in `validate()` passes.

Standing repository rule applied here: validate the estimator against cases
whose answer is known BEFORE pointing it at the target. The battery is:

  V1  3D linear flow with prescribed non-normal spectrum -> exponents are
      exactly the real parts of the eigenvalues (0.5, -0.2, -1.3).
  V2  2D rotation+contraction -> exactly (-0.3, -0.3): a DEGENERATE pair, the
      case naive Gram-Schmidt orders wrongly if the accumulation is buggy.
  V3  harmonic oscillator -> exactly (0, 0): catches spurious drift, which is
      the failure mode that would matter most here (a fake nonzero lambda_2
      moves D_KY directly).
  V4  Van der Pol limit cycle (mu=1) -> lambda_1 = 0 for a NONLINEAR flow with
      a genuine zero exponent along the trajectory, and lambda_1 + lambda_2 =
      the independently time-averaged divergence of the same orbit.
  V5  Henon map (a=1.4, b=0.3) -> published lambda_1 ~ 0.4192, and the EXACT
      constraint lambda_1 + lambda_2 = ln|det J| = ln(0.3) = -1.2039728...,
      a chaotic system with an analytically known exponent sum.

V5 uses the map code path (`lyapunov_spectrum_map`), which shares the
Gram-Schmidt/accumulation logic with the flow path but not the integrator.

Run: python f3_lyapunov_ref.py
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# =============================================================================
# Core: Benettin algorithm for flows
# =============================================================================


def _mgs(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modified Gram-Schmidt. Returns (orthonormal Q, diagonal of R).

    Columns are processed left to right, each subtracted from the remaining
    ones immediately (modified, not classical, GS -- classical GS loses
    orthogonality on the strongly-contracting columns that matter for
    lambda_3, which is the denominator of D_KY).
    """
    n = Q.shape[1]
    Q = Q.copy()
    diag = np.empty(n)
    for i in range(n):
        norm = float(np.linalg.norm(Q[:, i]))
        diag[i] = norm
        Q[:, i] /= norm
        if i + 1 < n:
            proj = Q[:, i] @ Q[:, i + 1 :]
            Q[:, i + 1 :] -= np.outer(Q[:, i], proj)
    return Q, diag


def _augmented_deriv(
    f: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return f(x), jac(x) @ Q


def _rk4_augmented(f, jac, x: np.ndarray, Q: np.ndarray, dt: float):
    k1x, k1Q = _augmented_deriv(f, jac, x, Q)
    k2x, k2Q = _augmented_deriv(f, jac, x + 0.5 * dt * k1x, Q + 0.5 * dt * k1Q)
    k3x, k3Q = _augmented_deriv(f, jac, x + 0.5 * dt * k2x, Q + 0.5 * dt * k2Q)
    k4x, k4Q = _augmented_deriv(f, jac, x + dt * k3x, Q + dt * k3Q)
    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Q_new = Q + (dt / 6.0) * (k1Q + 2 * k2Q + 2 * k3Q + k4Q)
    return x_new, Q_new


def lyapunov_spectrum(
    f,
    jac,
    x0: np.ndarray,
    *,
    dt: float,
    t_total: float,
    t_reorth: float = 0.5,
    t_transient: float = 0.0,
    n_out: int = 200,
):
    """Full Lyapunov spectrum of a flow by the Benettin/Gram-Schmidt method.

    Returns dict with `exponents` (final), `times`/`traces` (running exponents
    vs elapsed time, for convergence assessment), `mean_divergence` (time
    average of trace J, which must equal the sum of the exponents -- an
    identity, hence a free internal consistency check), and `mean_x`.
    """
    x = np.array(x0, dtype=float)
    n = x.size

    # Transient: trajectory only. The tangent basis is started afterwards so
    # the exponents are not polluted by the approach to the attractor.
    n_trans = int(round(t_transient / dt))
    Ident = np.eye(n)
    for _ in range(n_trans):
        x, _ = _rk4_augmented(f, jac, x, Ident, dt)

    Q = np.eye(n)
    acc = np.zeros(n)
    steps_per_reorth = max(1, int(round(t_reorth / dt)))
    n_reorth = int(round(t_total / (steps_per_reorth * dt)))
    out_every = max(1, n_reorth // n_out)

    div_acc = 0.0
    x_acc = 0.0
    n_steps_done = 0
    times, traces = [], []
    for r in range(1, n_reorth + 1):
        for _ in range(steps_per_reorth):
            div_acc += float(np.trace(jac(x)))
            x_acc += float(x[0])
            n_steps_done += 1
            x, Q = _rk4_augmented(f, jac, x, Q, dt)
        Q, diag = _mgs(Q)
        acc += np.log(diag)
        if r % out_every == 0 or r == n_reorth:
            t_elapsed = r * steps_per_reorth * dt
            times.append(t_elapsed)
            traces.append(acc / t_elapsed)

    t_elapsed = n_reorth * steps_per_reorth * dt
    return {
        "exponents": acc / t_elapsed,
        "times": np.array(times),
        "traces": np.array(traces),
        "mean_divergence": div_acc / n_steps_done,
        "mean_x": x_acc / n_steps_done,
        "t_total": t_elapsed,
    }


def lyapunov_spectrum_map(step, jac, x0: np.ndarray, *, n_iter: int, n_transient: int = 0):
    """Same accumulation logic for a discrete map (used by known-answer case V5)."""
    x = np.array(x0, dtype=float)
    for _ in range(n_transient):
        x = step(x)
    n = x.size
    Q = np.eye(n)
    acc = np.zeros(n)
    logdet_acc = 0.0
    for _ in range(n_iter):
        J = jac(x)
        logdet_acc += math.log(abs(float(np.linalg.det(J))))
        Q = J @ Q
        x = step(x)
        Q, diag = _mgs(Q)
        acc += np.log(diag)
    return {"exponents": acc / n_iter, "mean_log_abs_det": logdet_acc / n_iter}


def kaplan_yorke(exponents: np.ndarray) -> float:
    """D_KY = j + (sum_{i<=j} lambda_i)/|lambda_{j+1}|, j = last index with a
    non-negative partial sum. Returns the full dimension n if the whole
    spectrum sums non-negative (no contraction to interpolate into)."""
    lam = np.sort(np.asarray(exponents, dtype=float))[::-1]
    cum = np.cumsum(lam)
    j = int(np.sum(cum >= 0.0))
    if j == 0:
        return 0.0
    if j >= len(lam):
        return float(len(lam))
    return j + cum[j - 1] / abs(lam[j])


# =============================================================================
# Systems
# =============================================================================

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0  # round-2 problem 08 constants
A_R, B_R, C_R = 0.2, 0.2, 5.7  # round-2 problem 09 constants


def lorenz_f(s):
    x, y, z = s
    return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])


def lorenz_j(s):
    x, y, z = s
    return np.array([[-SIGMA, SIGMA, 0.0], [RHO - z, -1.0, -x], [y, x, -BETA]])


def rossler_f(s):
    x, y, z = s
    return np.array([-y - z, x + A_R * y, B_R + z * (x - C_R)])


def rossler_j(s):
    x, _y, z = s
    return np.array([[0.0, -1.0, -1.0], [1.0, A_R, 0.0], [z, 0.0, x - C_R]])


# --- validation systems ------------------------------------------------------

_P = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]])  # non-normal, det != 0
_D = np.diag([0.5, -0.2, -1.3])
_A_LIN = _P @ _D @ np.linalg.inv(_P)


def lin3_f(s):
    return _A_LIN @ s


def lin3_j(_s):
    return _A_LIN


_A_ROT = np.array([[-0.3, -2.0], [2.0, -0.3]])


def rot_f(s):
    return _A_ROT @ s


def rot_j(_s):
    return _A_ROT


_A_HARM = np.array([[0.0, 1.0], [-1.0, 0.0]])


def harm_f(s):
    return _A_HARM @ s


def harm_j(_s):
    return _A_HARM


MU_VDP = 1.0


def vdp_f(s):
    x, v = s
    return np.array([v, MU_VDP * (1.0 - x * x) * v - x])


def vdp_j(s):
    x, v = s
    return np.array([[0.0, 1.0], [-2.0 * MU_VDP * x * v - 1.0, MU_VDP * (1.0 - x * x)]])


A_HEN, B_HEN = 1.4, 0.3


def henon_step(s):
    x, y = s
    return np.array([1.0 - A_HEN * x * x + y, B_HEN * x])


def henon_jac(s):
    x, _y = s
    return np.array([[-2.0 * A_HEN * x, 1.0], [B_HEN, 0.0]])


# =============================================================================
# Known-answer battery
# =============================================================================


def validate(verbose: bool = True) -> bool:
    ok_all = True

    def report(name, got, expected, tol):
        nonlocal ok_all
        got = np.asarray(got, dtype=float)
        expected = np.asarray(expected, dtype=float)
        err = float(np.max(np.abs(got - expected)))
        ok = err <= tol
        ok_all = ok_all and ok
        if verbose:
            g = "  ".join(f"{v:+.6f}" for v in np.atleast_1d(got))
            e = "  ".join(f"{v:+.6f}" for v in np.atleast_1d(expected))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            print(f"         got      {g}")
            print(f"         expected {e}   (max err {err:.2e}, tol {tol:.0e})")

    if verbose:
        print("=" * 78)
        print("KNOWN-ANSWER VALIDATION (must pass before the targets are touched)")
        print("=" * 78)

    # V1: linear flow, exact spectrum = real parts of the eigenvalues.
    #
    # At finite T the estimate carries an O(1/T) offset: the initial basis is
    # not aligned with the Lyapunov directions, and the fixed log-factor that
    # misalignment costs is divided by T. That offset is a PROPERTY OF THE
    # METHOD, not an error in it, so the known-answer test is stated on the
    # quantity the method actually converges to: two run lengths, a 1/T
    # extrapolation, and the exact spectrum. The 1/T law itself is checked
    # (residual ratio must be T2/T1) rather than assumed, and
    # f3_crosscheck.py additionally confirms the residual is independent of dt,
    # i.e. that it is the basis offset and not integrator truncation.
    t1, t2 = 100.0, 1000.0
    l1 = np.sort(lyapunov_spectrum(lin3_f, lin3_j, np.array([1.0, 1.0, 1.0]),
                                   dt=0.002, t_total=t1, t_reorth=0.1)["exponents"])[::-1]
    l2 = np.sort(lyapunov_spectrum(lin3_f, lin3_j, np.array([1.0, 1.0, 1.0]),
                                   dt=0.002, t_total=t2, t_reorth=0.1)["exponents"])[::-1]
    exact = np.array([0.5, -0.2, -1.3])
    ratio = np.abs(l1 - exact) / np.abs(l2 - exact)
    report("V1a 3D non-normal linear flow: residual decays exactly as 1/T",
           ratio, np.full(3, t2 / t1), 1e-3 * (t2 / t1))
    report("V1b 3D non-normal linear flow: 1/T-extrapolated spectrum is exact",
           (t2 * l2 - t1 * l1) / (t2 - t1), exact, 1e-9)

    # V2: rotation + contraction, exactly degenerate pair. No basis offset
    # here (the whole plane contracts uniformly), so this one is exact at any T.
    r = lyapunov_spectrum(rot_f, rot_j, np.array([1.0, 0.0]), dt=0.001,
                          t_total=200.0, t_reorth=0.1)
    report("V2  2D rotation+contraction (degenerate pair)",
           np.sort(r["exponents"])[::-1], [-0.3, -0.3], 1e-9)

    # V3: harmonic oscillator, both exponents exactly zero. Catches spurious
    # drift -- the failure mode that would matter most here, since a fake
    # nonzero lambda_2 feeds straight into D_KY.
    r = lyapunov_spectrum(harm_f, harm_j, np.array([1.0, 0.0]), dt=0.001,
                          t_total=500.0, t_reorth=0.1)
    report("V3  harmonic oscillator (both exponents exactly 0)",
           np.sort(r["exponents"])[::-1], [0.0, 0.0], 1e-9)

    # V4: Van der Pol limit cycle -- a NONLINEAR flow with a genuine zero
    # exponent along the trajectory, structurally the same situation as
    # lambda_2 on Lorenz/Rossler. lambda_1 -> 0 like 1/T (0.0018 at T=200 here,
    # 1.7e-5 at T=20000 in the C run); the sum identity is the tight test.
    r = lyapunov_spectrum(vdp_f, vdp_j, np.array([2.0, 0.0]), dt=0.001,
                          t_total=200.0, t_reorth=0.1, t_transient=100.0)
    lam = np.sort(r["exponents"])[::-1]
    report("V4a Van der Pol mu=1: lambda_1 -> 0 (flow direction), T=200", lam[0], 0.0, 3e-3)
    report("V4b Van der Pol: sum(lambda) = independently averaged div f",
           lam.sum(), r["mean_divergence"], 1e-5)

    # V5: Henon map -- a genuinely chaotic system with a published lambda_1
    # and an analytically EXACT exponent sum, ln|det J| = ln(b).
    r = lyapunov_spectrum_map(henon_step, henon_jac, np.array([0.1, 0.1]),
                              n_iter=500_000, n_transient=10_000)
    lam = np.sort(r["exponents"])[::-1]
    report("V5a Henon (1.4, 0.3): lambda_1 vs published 0.41922", lam[0], 0.41922, 1e-3)
    report("V5b Henon: sum(lambda) = ln(0.3) exactly (analytic)",
           lam.sum(), math.log(B_HEN), 1e-9)

    if verbose:
        print(f"\n  BATTERY {'PASSED' if ok_all else 'FAILED'}")
    return ok_all


if __name__ == "__main__":
    ok = validate()
    raise SystemExit(0 if ok else 1)
