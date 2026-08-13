"""Benchmark 10: Driven damped pendulum, mode-locked period-1 regime.

KNOWN/TRADITIONAL DIMENSION: 1 (exact)
SOURCE: Standard nonlinear-dynamics result -- given a genuinely period-1
mode-locked solution (verified below, not assumed), a limit cycle of a
driven-damped system is by definition a smooth closed curve in phase space,
so its topological/correlation dimension is exactly 1. This is the standard
textbook contrast case to chaotic driven systems (Lorenz/Rossler-style):
adding driving and damping does NOT automatically produce chaos -- for most
parameter choices (including the ones used here) it produces a stable
periodic limit cycle instead. The driven pendulum is the classic example
of a system that does either, depending on parameters (see Strogatz,
"Nonlinear Dynamics and Chaos", for the chaotic regime around F~1.5,
gamma~0.5, Omega_drive~2/3, which we deliberately avoid here).

System (standard form):
    thetadot = omega
    omegadot = -gamma*omega - sin(theta) + F*cos(Omega_drive*t)
with gamma=0.5, F=0.5, Omega_drive=0.6 -- well below the classic chaotic
regime (F~1.5) -- chosen to mode-lock to a period-1 limit cycle.

We hand-write an RK4 integrator for this (new, time-dependent) system,
discard a long transient, and VERIFY periodicity via a stroboscopic
(Poincare) section: sampling (theta mod 2pi, omega) at times spaced by
exactly one drive period, over 50 post-transient periods. If the system is
genuinely period-1, these samples converge to a single point; period-N
would give N points; chaos would give a scattered/fractal set. Only after
this verification succeeds do we build the point cloud. NOTE (found during
this run, see the detailed comment at that point in the script below): a
literal "several drive periods of dense samples" point cloud is actually
degenerate for a verified period-1 orbit, because every period retraces the
identical curve -- so instead we densely sample ONE closed period of the
converged limit cycle (confirmed closed via a start/end proximity check),
which is the correct, non-degenerate point cloud for this curve. We then
construct a k-NN proximity graph via
socrates.hypergraph.pointcloud.knn_hypergraph and measure its volume-growth
dimension via socrates.hypergraph.dimension (local_dimension /
mean_dimension), comparing against the exact value of 1.

Do not modify src/socrates/hypergraph/* or src/socrates/solvers/* -- shared
with other agents running in parallel.
"""

from __future__ import annotations

import math

import numpy as np

from socrates.hypergraph.dimension import local_dimension, mean_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

# ---------------------------------------------------------------------------
# 1. Hand-written RK4 integrator for the driven damped pendulum.
#    Parameters deliberately in the non-chaotic, mode-locking regime.
# ---------------------------------------------------------------------------

GAMMA = 0.5
F_DRIVE = 0.5
OMEGA_DRIVE = 0.6


def deriv(theta: float, omega: float, t: float) -> tuple[float, float]:
    thetadot = omega
    omegadot = -GAMMA * omega - math.sin(theta) + F_DRIVE * math.cos(OMEGA_DRIVE * t)
    return thetadot, omegadot


def rk4_step(theta: float, omega: float, t: float, dt: float) -> tuple[float, float]:
    k1t, k1o = deriv(theta, omega, t)
    k2t, k2o = deriv(theta + 0.5 * dt * k1t, omega + 0.5 * dt * k1o, t + 0.5 * dt)
    k3t, k3o = deriv(theta + 0.5 * dt * k2t, omega + 0.5 * dt * k2o, t + 0.5 * dt)
    k4t, k4o = deriv(theta + dt * k3t, omega + dt * k3o, t + dt)
    theta_new = theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
    omega_new = omega + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
    return theta_new, omega_new


T_DRIVE = 2.0 * math.pi / OMEGA_DRIVE  # exact drive period
STEPS_PER_PERIOD = 500                 # dt chosen so period boundaries land exactly on steps
DT = T_DRIVE / STEPS_PER_PERIOD

N_TRANSIENT_PERIODS = 200   # discard, let the system settle onto its attractor
N_CHECK_PERIODS = 50        # stroboscopic samples taken over these periods, post-transient

theta, omega, t = 0.2, 0.0, 0.0  # arbitrary initial condition, away from any fixed point

print(f"Driven damped pendulum: gamma={GAMMA}, F={F_DRIVE}, Omega_drive={OMEGA_DRIVE}, "
      f"T_drive={T_DRIVE:.6f}, dt={DT:.6f} ({STEPS_PER_PERIOD} steps/period)")
print(f"Transient: {N_TRANSIENT_PERIODS} periods; stroboscopic check: {N_CHECK_PERIODS} periods")

# --- transient ---
for _p in range(N_TRANSIENT_PERIODS):
    for _s in range(STEPS_PER_PERIOD):
        theta, omega = rk4_step(theta, omega, t, DT)
        t += DT

finite_ok = math.isfinite(theta) and math.isfinite(omega)
print(f"\nAfter transient: theta={theta:.6f}, omega={omega:.6f}, finite: {finite_ok}")

# ---------------------------------------------------------------------------
# 2. VERIFY the solver: stroboscopic (Poincare) section over N_CHECK_PERIODS
#    drive periods, post-transient.
# ---------------------------------------------------------------------------

strobe_points: list[tuple[float, float]] = []

for p in range(N_CHECK_PERIODS):
    for _s in range(STEPS_PER_PERIOD):
        theta, omega = rk4_step(theta, omega, t, DT)
        t += DT
    # stroboscopic sample: state at an exact multiple of T_drive past the
    # start of the check window (steps_per_period chosen so this lands
    # exactly on a step boundary, no interpolation needed).
    theta_wrapped = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
    strobe_points.append((theta_wrapped, omega))

strobe_arr = np.array(strobe_points)

# Convergence check: compare the last half of the stroboscopic samples --
# if genuinely period-1, they should all sit on top of one another (up to
# integration precision). Look at spread (max-min) in each coordinate.
tail = strobe_arr[len(strobe_arr) // 2:]
theta_spread = float(tail[:, 0].max() - tail[:, 0].min())
omega_spread = float(tail[:, 1].max() - tail[:, 1].min())

print(f"\nStroboscopic section over last {len(tail)} (of {len(strobe_arr)}) check periods:")
print(f"  theta_mod2pi spread: {theta_spread:.3e}")
print(f"  omega spread:        {omega_spread:.3e}")
print(f"  final strobe point: theta_mod2pi={tail[-1,0]:.6f}, omega={tail[-1,1]:.6f}")

PERIOD1_TOL = 1e-6
period1_ok = finite_ok and (theta_spread < PERIOD1_TOL) and (omega_spread < PERIOD1_TOL)
print(f"\nPeriod-1 verification (stroboscopic spread < {PERIOD1_TOL}): "
      f"{'PASSED' if period1_ok else 'FAILED'}")

solver_verified = bool(finite_ok and period1_ok)

# ---------------------------------------------------------------------------
# 3. Build the point cloud.
#
#    IMPORTANT DISCOVERY (worth stating explicitly, not papering over):
#    the prompt for this problem describes the point cloud as "(theta mod
#    2pi, omega) over several drive periods of the CONVERGED limit cycle."
#    We first implemented that literally -- densely sampling the continuous
#    trajectory over N_CLOUD_PERIODS=20 periods post-transient -- and it
#    produced a BAD fit (mean_dimension -> nan, all sampled nodes r_squared
#    < 0.9, typical dimension ~0.86 on the few that fit at all). Diagnosing
#    why: because the orbit is verified period-1 (previous section), every
#    drive period retraces EXACTLY the same closed curve in (theta mod 2pi,
#    omega) space, to within the ~1e-10 stroboscopic convergence precision
#    measured above. So "20 periods of dense samples" is not 20x the
#    distinct points -- it is ~20 near-exact copies of the SAME ~1500-point
#    curve stacked almost on top of each other. knn_hypergraph then connects
#    each point predominantly to its 10-20 near-duplicates from other
#    periods (distance ~1e-9) rather than to its true along-curve
#    neighbours, so ball(radius=1) already saturates to ~2k and shell growth
#    collapses after 1-2 hops -- a real, reproducible degeneracy, not an
#    estimator bug (confirmed directly by inspecting ball sizes per radius).
#
#    Fix: sample ONE drive period (the curve is fully closed after exactly
#    T_drive, confirmed by first/last-point proximity below) at a much
#    finer time resolution, so the point count comes from resolving the
#    single closed curve more finely rather than from re-tracing it. This
#    is the correct point cloud for THIS limit cycle: the full period-1
#    orbit, sampled once around, at high density.
# ---------------------------------------------------------------------------

FINE_STEPS_PER_PERIOD = 1500  # distinct points around the one closed loop
dt_fine = T_DRIVE / FINE_STEPS_PER_PERIOD

cloud_theta = []
cloud_omega = []
theta_start, omega_start = theta, omega
for _s in range(FINE_STEPS_PER_PERIOD):
    theta, omega = rk4_step(theta, omega, t, dt_fine)
    t += dt_fine
    cloud_theta.append(theta)
    cloud_omega.append(omega)

closure_gap = math.hypot(theta - theta_start, omega - omega_start)
print(f"\nSingle-period closure check: |state(t+T_drive) - state(t)| = {closure_gap:.3e} "
      f"(small => the sampled curve is genuinely closed, consistent with period-1)")

cloud_theta_arr = np.array(cloud_theta)
cloud_omega_arr = np.array(cloud_omega)
cloud_theta_wrapped = ((cloud_theta_arr + math.pi) % (2.0 * math.pi)) - math.pi

theta_min, theta_max = cloud_theta_wrapped.min(), cloud_theta_wrapped.max()
print(f"\nPoint-cloud theta_mod2pi range: [{theta_min:.4f}, {theta_max:.4f}] "
      f"(well inside (-pi, pi) => no wrap-around cut through the sampled data)")
wrap_safe = (theta_min > -math.pi + 0.1) and (theta_max < math.pi - 0.1)
print(f"Wrap-around safety check: {'OK' if wrap_safe else 'WARNING -- data near +/-pi boundary'}")

points = list(zip(cloud_theta_wrapped.tolist(), cloud_omega_arr.tolist(), strict=True))
n_points = len(points)
print(f"\nPoint cloud: {n_points} distinct points around ONE closed period-1 limit-cycle loop "
      f"(FINE_STEPS_PER_PERIOD={FINE_STEPS_PER_PERIOD}, dt_fine={dt_fine:.6f})")

K = 10  # matches this module's own calibration tests for a 1D/2D manifold point cloud

hg = knn_hypergraph(points, k=K)

# ---------------------------------------------------------------------------
# 4. Estimate dimension. The limit cycle is a smooth closed 1D curve, so we
#    expect local_dimension ~ 1 everywhere along it, well-fit (no boundary
#    effects to exclude -- there is no boundary on a closed loop, unlike the
#    filled-cube/filled-square calibration cases in this module's tests).
# ---------------------------------------------------------------------------

MAX_RADIUS = 5
SAMPLES = 40

estimates = [local_dimension(hg, i, max_radius=MAX_RADIUS)
             for i in range(0, n_points, max(1, n_points // SAMPLES))]
well_fit = [e for e in estimates if e.is_well_fit(threshold=0.9)]

print(f"\nlocal_dimension over {len(estimates)} sampled nodes "
      f"(max_radius={MAX_RADIUS}): {len(well_fit)} well-fit (r_squared >= 0.9)")
for e in estimates:
    flag = "OK " if e.is_well_fit(threshold=0.9) else "   "
    print(f"  [{flag}] node={e.source:5d} dim={e.dimension:6.3f} r2={e.r_squared:.4f}")

if well_fit:
    mean_dim_manual = sum(e.dimension for e in well_fit) / len(well_fit)
    mean_r2 = sum(e.r_squared for e in well_fit) / len(well_fit)
else:
    mean_dim_manual = float("nan")
    mean_r2 = float("nan")

# Cross-check via the module's own mean_dimension convenience wrapper (uses
# its own r_squared>=0.9 filtering internally).
mean_dim_wrapper = mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS)

print(f"\nMean dimension (manual, well-fit only): {mean_dim_manual:.4f} "
      f"(mean r_squared of well-fit points: {mean_r2:.4f})")
print(f"Mean dimension (module mean_dimension wrapper): {mean_dim_wrapper:.4f}")

# ---------------------------------------------------------------------------
# 5. Compare against the exact traditional value of 1.
# ---------------------------------------------------------------------------

TRADITIONAL_DIMENSION = 1.0
TOLERANCE = 0.2

final_dimension = mean_dim_wrapper
abs_error = abs(final_dimension - TRADITIONAL_DIMENSION)
passed = solver_verified and (abs_error <= TOLERANCE) and not math.isnan(final_dimension)

print(f"\n=== RESULT ===")
print(f"Traditional dimension (exact, period-1 limit cycle => smooth closed curve): "
      f"{TRADITIONAL_DIMENSION}")
print(f"Poly-algebraic (knn_hypergraph + mean_dimension) dimension: {final_dimension:.4f}")
print(f"Mean r_squared of well-fit sampled nodes: {mean_r2:.4f}")
print(f"Absolute error: {abs_error:.4f} (tolerance {TOLERANCE})")
print(f"Solver verified (finite + period-1 stroboscopic convergence): {solver_verified}")
print(f"PASSED: {passed}")
