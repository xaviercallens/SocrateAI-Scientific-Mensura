"""Benchmark: circular restricted three-body problem (CR3BP), planar Lyapunov
periodic orbit near L1 -- poly-algebraic dimension vs traditional dimension.

KNOWN/TRADITIONAL DIMENSION: 1 (exact). A genuinely periodic orbit of the
CR3BP is by definition a smooth closed curve in the (x, y) rotating-frame
plane -- topological / correlation dimension exactly 1. This is only true if
the orbit is *actually* periodic, which is verified below, not assumed.

SYSTEM: Earth-Moon CR3BP, mu = 0.0121505856 (standard textbook value),
primaries at (-mu, 0) [Earth] and (1-mu, 0) [Moon], rotating frame,
dimensionless units (Earth-Moon distance = 1, mean motion = 1). Equations of
motion exactly as given in the assignment brief (equivalent to Koon/Lo/
Marsden/Ross, "Dynamical Systems, the Three-Body Problem and Space Mission
Design", eq. 2.3.9):

    xddot - 2*ydot = x - (1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3
    yddot + 2*xdot = y - (1-mu)*y/r1^3      - mu*y/r2^3

ORBIT CHOICE: a small-amplitude planar Lyapunov periodic orbit around the
collinear libration point L1, found by:
  1. Locating L1 exactly (root of the collinear-equilibrium equation).
  2. Linearizing the CR3BP equations at L1 (numeric Jacobian) and extracting
     the planar oscillatory eigenmode (pure-imaginary eigenvalue pair
     +/- i*omega_p) to get a first-order guess for the initial y-velocity
     at a chosen x-amplitude off L1.
  3. Exploiting the orbit's built-in symmetry: any orbit that starts
     perpendicular to the x-axis (y0=0, vx0=0) and is periodic must cross
     the x-axis perpendicularly again (vx=0) after exactly half a period.
     So the periodicity condition reduces to a single scalar shooting
     problem: find vy0 such that vx=0 at the next y=0 crossing. This is
     solved by root-finding (Brent's method, a bracketed generalization of
     bisection) on the linear guess, first at a coarse integrator step size
     to bracket, then refined at a fine step size for a high-precision root.
  4. The orbit is NOT assumed periodic just because the shooting residual
     is small -- section "SOLVER VERIFICATION" below explicitly integrates
     one full period from the corrected initial condition and checks that
     BOTH position and velocity return to their starting values.

This is a new, small ODE system not otherwise present in this repository;
per the assignment brief, a fresh RK4 integrator is written here rather than
reusing socrates.solvers.

Does NOT modify src/socrates/hypergraph/ or src/socrates/solvers/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from socrates.hypergraph.dimension import local_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

MU = 0.0121505856  # Earth-Moon mass ratio (standard textbook value)
TRADITIONAL_DIMENSION = 1.0
TOLERANCE = 0.2

# Closure gate: how close position/velocity must return to their starting
# values after one confirmed period for the orbit to count as "genuinely
# periodic" (stated explicitly, per the assignment's most important check).
CLOSURE_POS_TOL = 1e-8
CLOSURE_VEL_TOL = 1e-8


# --------------------------------------------------------------------------
# CR3BP dynamics and a fresh RK4 integrator (new ODE system, not in repo).
# --------------------------------------------------------------------------
def rhs(state: np.ndarray) -> np.ndarray:
    """CR3BP equations of motion in the rotating frame, exactly as given in
    the assignment brief."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + MU) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + MU) ** 2 + y ** 2)
    ax = 2 * vy + x - (1 - MU) * (x + MU) / r1 ** 3 - MU * (x - 1 + MU) / r2 ** 3
    ay = -2 * vx + y - (1 - MU) * y / r1 ** 3 - MU * y / r2 ** 3
    return np.array([vx, vy, ax, ay])


def rk4_step(state: np.ndarray, h: float) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + h / 2 * k1)
    k3 = rhs(state + h / 2 * k2)
    k4 = rhs(state + h * k3)
    return state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state0: np.ndarray, dt: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 integration. Returns (times, states) including t=0."""
    states = np.empty((n_steps + 1, 4))
    times = np.empty(n_steps + 1)
    states[0] = state0
    times[0] = 0.0
    s = state0.copy()
    for i in range(n_steps):
        s = rk4_step(s, dt)
        states[i + 1] = s
        times[i + 1] = times[i] + dt
    return times, states


# --------------------------------------------------------------------------
# Step 1: locate L1 exactly (root of the collinear-equilibrium equation).
# --------------------------------------------------------------------------
def find_l1(mu: float) -> float:
    def l1_eq(x: float) -> float:
        r1 = abs(x + mu)
        r2 = abs(x - (1 - mu))
        return x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3

    return brentq(l1_eq, -mu + 1e-9, 1 - mu - 1e-9, xtol=1e-14)


# --------------------------------------------------------------------------
# Step 2: linearize at L1 to get a first-order guess for vy0.
# --------------------------------------------------------------------------
def linear_vy0_guess(x_l1: float, ax_amplitude: float) -> float:
    eq = np.array([x_l1, 0.0, 0.0, 0.0])
    eps = 1e-6
    jac = np.zeros((4, 4))
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        jac[:, i] = (rhs(eq + d) - rhs(eq - d)) / (2 * eps)
    vals, vecs = np.linalg.eig(jac)
    # planar oscillatory mode: the eigenvalue pair with the largest
    # imaginary part and (numerically) zero real part
    osc_idx = np.argmax(np.abs(vals.imag))
    v = vecs[:, osc_idx]
    v = v / v[0].real  # normalize so the dx component is 1 (real)
    # state(t) = Re[v * ax_amplitude * exp(i*omega*t)]; at t=0 this gives
    # (x, y, vx, vy) = (ax_amplitude, 0, 0, Re(v_dvy)*ax_amplitude)
    return float(v[3].real) * ax_amplitude


# --------------------------------------------------------------------------
# Step 3: single-shooting differential correction via the x-axis-crossing
# symmetry (y0=0, vx0=0 -> periodic iff vx=0 at the next y=0 crossing).
# --------------------------------------------------------------------------
def half_period_crossing(
    state0: np.ndarray, dt: float, t_max: float
) -> tuple[float, np.ndarray]:
    """Integrate from state0 (y=0, vx=0 at t=0) until the first y=0 crossing
    at t>0, refine the crossing time via Brent's method, and return
    (t_cross, state_at_crossing)."""
    state = state0.copy()
    t = 0.0
    n_steps = int(t_max / dt)
    for _ in range(n_steps):
        new_state = rk4_step(state, dt)
        if t > 0.05 and np.sign(state[1]) != np.sign(new_state[1]):
            def f(h: float) -> float:
                return rk4_step(state, h)[1]

            h_cross = brentq(f, 0.0, dt, xtol=1e-14, rtol=1e-14)
            cross_state = rk4_step(state, h_cross)
            return t + h_cross, cross_state
        state = new_state
        t += dt
    raise RuntimeError(f"no x-axis crossing found within t_max={t_max}")


def shooting_residual(vy0: float, x0: float, dt: float, t_max: float) -> float:
    state0 = np.array([x0, 0.0, 0.0, vy0])
    _, s_cross = half_period_crossing(state0, dt, t_max)
    return float(s_cross[2])  # vx at crossing; want 0


def find_periodic_orbit(x0: float, vy0_guess: float, dt: float, t_max: float) -> float:
    """Bracket and root-find vy0 so that the trajectory from (x0,0,0,vy0)
    crosses the x-axis perpendicularly (vx=0) after half a period."""
    span = 0.3 * abs(vy0_guess)
    lo, hi = vy0_guess - span, vy0_guess + span
    r_lo = shooting_residual(lo, x0, dt, t_max)
    r_hi = shooting_residual(hi, x0, dt, t_max)
    tries = 0
    while np.sign(r_lo) == np.sign(r_hi) and tries < 6:
        span *= 2
        lo, hi = vy0_guess - span, vy0_guess + span
        r_lo = shooting_residual(lo, x0, dt, t_max)
        r_hi = shooting_residual(hi, x0, dt, t_max)
        tries += 1
    if np.sign(r_lo) == np.sign(r_hi):
        raise RuntimeError(
            f"could not bracket a root: r_lo={r_lo}, r_hi={r_hi} over [{lo}, {hi}]"
        )
    return brentq(
        lambda vy0: shooting_residual(vy0, x0, dt, t_max), lo, hi, xtol=1e-13, rtol=1e-13
    )


def main() -> None:
    x_l1 = find_l1(MU)
    print("=== L1 location ===")
    print(f"  mu = {MU}")
    print(f"  x_L1 = {x_l1:.10f}  (between Earth at -mu and Moon at 1-mu)")

    ax_amplitude = 0.005  # small-amplitude Lyapunov orbit, in normalized units
    x0 = x_l1 + ax_amplitude
    vy0_guess = linear_vy0_guess(x_l1, ax_amplitude)
    print(f"\n=== Linear initial guess ===")
    print(f"  x-amplitude off L1: {ax_amplitude}")
    print(f"  x0 = {x0:.10f}")
    print(f"  vy0 (linear guess) = {vy0_guess:.10f}")

    # Coarse shooting (fast) to get an accurate vy0, then refine with a
    # finer step size for the final high-precision correction.
    print(f"\n=== Differential correction (single shooting, x-axis symmetry) ===")
    vy0_coarse = find_periodic_orbit(x0, vy0_guess, dt=1e-3, t_max=1.8)
    print(f"  vy0 (coarse, dt=1e-3): {vy0_coarse:.12f}")
    vy0_fine = find_periodic_orbit(x0, vy0_coarse, dt=1e-4, t_max=1.8)
    print(f"  vy0 (refined, dt=1e-4): {vy0_fine:.12f}")

    t_half, cross_state = half_period_crossing(
        np.array([x0, 0.0, 0.0, vy0_fine]), dt=1e-5, t_max=1.8
    )
    period = 2.0 * t_half
    print(f"  half-period t_half = {t_half:.10f}")
    print(f"  vx at half-period crossing (should be ~0): {cross_state[2]:.3e}")
    print(f"  full period T = {period:.10f}  (normalized time units)")

    state0 = np.array([x0, 0.0, 0.0, vy0_fine])

    # ---------------------------------------------------------------------
    # SOLVER VERIFICATION (the most important check): integrate one full,
    # confirmed period from the corrected initial condition and check that
    # BOTH position and velocity return close to the starting state.
    # ---------------------------------------------------------------------
    dt_verify = 1e-5
    n_steps_verify = int(round(period / dt_verify))
    dt_actual = period / n_steps_verify  # land exactly on t=T
    _, states_one_period = integrate(state0, dt_actual, n_steps_verify)
    final_state = states_one_period[-1]

    pos_err = float(np.linalg.norm(final_state[:2] - state0[:2]))
    vel_err = float(np.linalg.norm(final_state[2:] - state0[2:]))
    closure_passed = pos_err <= CLOSURE_POS_TOL and vel_err <= CLOSURE_VEL_TOL

    print(f"\n=== Solver verification: orbit closure after one confirmed period ===")
    print(f"  integrator: fixed-step RK4, dt = {dt_actual:.3e}, {n_steps_verify} steps over T")
    print(f"  initial state (x,y,vx,vy): {state0}")
    print(f"  final state   (x,y,vx,vy): {final_state}")
    print(f"  |position error| after 1 period: {pos_err:.3e}  (tol {CLOSURE_POS_TOL:.0e})")
    print(f"  |velocity error| after 1 period: {vel_err:.3e}  (tol {CLOSURE_VEL_TOL:.0e})")
    print(f"  ORBIT CLOSES: {closure_passed}")

    # Secondary sanity check: Jacobi constant (the CR3BP's conserved energy
    # analogue) should be conserved along the trajectory if the integration
    # is trustworthy.
    def jacobi_constant(s: np.ndarray) -> float:
        x, y, vx, vy = s
        r1 = np.sqrt((x + MU) ** 2 + y ** 2)
        r2 = np.sqrt((x - 1 + MU) ** 2 + y ** 2)
        u = 0.5 * (x ** 2 + y ** 2) + (1 - MU) / r1 + MU / r2
        v2 = vx ** 2 + vy ** 2
        return 2 * u - v2

    cj0 = jacobi_constant(state0)
    cj_all = np.array([jacobi_constant(s) for s in states_one_period[::200]])
    cj_drift = float(np.max(np.abs(cj_all - cj0)) / abs(cj0))
    print(f"  Jacobi constant at t=0: {cj0:.12f}")
    print(f"  Jacobi constant max relative drift over 1 period: {cj_drift:.3e}")

    solver_ok = closure_passed and cj_drift < 1e-8

    # ---------------------------------------------------------------------
    # Point cloud: (x, y) positions over >= 2 confirmed periods.
    # ---------------------------------------------------------------------
    n_periods = 3.0
    dt_cloud = 1e-4
    n_steps_cloud = int(round(n_periods * period / dt_cloud))
    times_cloud, states_cloud = integrate(state0, dt_cloud, n_steps_cloud)
    positions = states_cloud[:, :2]

    n_target_points = 400
    stride = max(1, len(positions) // n_target_points)
    sampled = positions[::stride][:n_target_points]
    points = [(float(p[0]), float(p[1])) for p in sampled]

    print(f"\n=== Point cloud ===")
    print(f"  periods covered: {n_periods}")
    print(f"  total RK4 steps in window: {len(positions)}")
    print(f"  sampled points (x, y): {len(points)}  (stride={stride})")

    # ---------------------------------------------------------------------
    # Poly-algebraic dimension: knn_hypergraph + local_dimension/mean_dimension.
    # ---------------------------------------------------------------------
    k = 6  # matches the closed-curve calibration used for the Kepler/circle cases
    hg = knn_hypergraph(points, k=k)

    max_radius = 5
    min_radius = 1
    nodes = sorted(hg.nodes)
    n_samples = 40
    step = max(1, len(nodes) // n_samples)
    sample_nodes = nodes[::step][:n_samples]

    estimates = [
        local_dimension(hg, n, max_radius=max_radius, min_radius=min_radius) for n in sample_nodes
    ]
    well_fit = [e for e in estimates if e.is_well_fit(threshold=0.9)]

    print(f"\n=== Dimension estimation ===")
    print(f"  k (knn_hypergraph): {k}")
    print(f"  max_radius: {max_radius}, min_radius: {min_radius}")
    print(f"  nodes sampled for local_dimension: {len(sample_nodes)} / {len(nodes)} total")
    print(f"  well-fit (r_squared >= 0.9): {len(well_fit)} / {len(sample_nodes)}")

    if well_fit:
        mean_dim = sum(e.dimension for e in well_fit) / len(well_fit)
        mean_r2 = sum(e.r_squared for e in well_fit) / len(well_fit)
    else:
        mean_dim = float("nan")
        mean_r2 = float("nan")

    print(f"  mean dimension (well-fit only): {mean_dim:.4f}")
    print(f"  mean r_squared (well-fit only): {mean_r2:.4f}")

    all_dims = [e.dimension for e in estimates]
    all_r2 = [e.r_squared for e in estimates]
    print(f"  mean dimension (all sampled, unfiltered): {np.mean(all_dims):.4f}")
    print(f"  mean r_squared (all sampled, unfiltered): {np.mean(all_r2):.4f}")

    abs_err = abs(mean_dim - TRADITIONAL_DIMENSION)
    dim_passed = abs_err <= TOLERANCE
    overall_passed = bool(dim_passed and solver_ok)

    print(f"\n=== Result ===")
    print(f"  traditional dimension: {TRADITIONAL_DIMENSION}  (exact, closed curve)")
    print(f"  poly-algebraic dimension: {mean_dim:.4f}")
    print(f"  absolute error: {abs_err:.4f}  (tolerance {TOLERANCE})")
    print(f"  solver verified (closure + Jacobi conservation): {solver_ok}")
    print(f"  PASSED: {overall_passed}")

    if not solver_ok:
        print(
            "\n  CAVEAT: dimension estimate is reported above for transparency, but the "
            "solver did not pass its own closure/conservation checks, so a dimension "
            "match (if any) would NOT count as a genuine pass -- an unverified or "
            "wrong solver does not validate the dimension estimator."
        )


if __name__ == "__main__":
    main()
