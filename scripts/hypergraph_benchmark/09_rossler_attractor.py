"""Benchmark 09: Rossler attractor (chaotic).

KNOWN/TRADITIONAL DIMENSION: 2.01 (correlation dimension)
SOURCE: Grassberger & Procaccia (1983), Physica D 9, 189-208, and subsequent
numerical studies commonly cite the Rossler attractor correlation dimension
as approximately 2.01-2.02 at the classic parameters a=0.2, b=0.2, c=5.7
(O. Rossler, 1976, "An equation for continuous chaos", Phys. Lett. A 57).

System:
    xdot = -y - z
    ydot = x + a*y
    zdot = b + z*(x - c)
with a=0.2, b=0.2, c=5.7 (canonical Rossler parameters).

We hand-write an RK4 integrator, verify the trajectory actually shows the
characteristic Rossler band/spiral-with-fold structure (not a fixed point,
not a divergence to infinity), discard a long initial transient, subsample
the post-transient attractor to a tractable point count, build a k-NN
proximity graph via socrates.hypergraph.pointcloud.knn_hypergraph, and
measure its volume-growth dimension via socrates.hypergraph.dimension
(local_dimension / mean_dimension), comparing against the literature
correlation-dimension value above.

Do not modify src/socrates/hypergraph/* or src/socrates/solvers/* -- shared
with other agents running in parallel.
"""

from __future__ import annotations

import math

import numpy as np

from socrates.hypergraph.dimension import local_dimension, mean_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

# ---------------------------------------------------------------------------
# 1. Hand-written RK4 integrator for the Rossler system.
# ---------------------------------------------------------------------------

A, B, C = 0.2, 0.2, 5.7


def rossler_deriv(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array([-y - z, x + A * y, B + z * (x - C)])


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = rossler_deriv(state)
    k2 = rossler_deriv(state + 0.5 * dt * k1)
    k3 = rossler_deriv(state + 0.5 * dt * k2)
    k4 = rossler_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


DT = 0.01
T_TRANSIENT = 100.0       # discard first 100 time units (settles slower than Lorenz)
T_TOTAL = 1100.0          # total integration time -> 1000 time units of post-transient data
N_STEPS = int(round(T_TOTAL / DT))
TRANSIENT_STEPS = int(round(T_TRANSIENT / DT))

state = np.array([1.0, 1.0, 1.0])  # away from the fixed points
trajectory = np.empty((N_STEPS + 1, 3))
trajectory[0] = state
for i in range(N_STEPS):
    state = rk4_step(state, DT)
    trajectory[i + 1] = state

print(f"Integrated Rossler system: a={A}, b={B}, c={C}, dt={DT}, "
      f"T_total={T_TOTAL}, T_transient={T_TRANSIENT}, RK4 steps={N_STEPS}")

# Sanity: no blow-up to infinity, no collapse to a point.
finite_ok = np.all(np.isfinite(trajectory))
overall_range = trajectory.max(axis=0) - trajectory.min(axis=0)
print(f"All finite: {finite_ok}; (x,y,z) range: {overall_range}")

post_transient = trajectory[TRANSIENT_STEPS:]

# ---------------------------------------------------------------------------
# 2. VERIFY the solver: characteristic Rossler band structure.
#    - x, y should spiral outward in the plane roughly until x exceeds ~c,
#      at which point z spikes sharply (the "fold"), then the trajectory
#      resets near z~0 and resumes spiraling.
#    - So z should mostly stay small (near 0) with occasional large spikes,
#      NOT stay uniformly large (that would mean it never returns to the
#      near-planar spiral) and NOT collapse to exactly zero everywhere
#      (that would mean it settled onto a fixed point / 2D limit cycle
#      instead of the folded chaotic band).
#    - Also confirm sensitivity to initial conditions (a hallmark of chaos):
#      two nearby trajectories should diverge exponentially before
#      saturating (bounded by the attractor's finite size).
# ---------------------------------------------------------------------------

z_vals = post_transient[:, 2]
z_median = np.median(z_vals)
z_p95 = np.percentile(z_vals, 95)
z_max = np.max(z_vals)
frac_small_z = np.mean(z_vals < 1.0)
frac_spike_z = np.mean(z_vals > 5.0)

print(f"\nz statistics on post-transient trajectory (n={len(z_vals)} steps):")
print(f"  median z = {z_median:.4f}, 95th pct z = {z_p95:.4f}, max z = {z_max:.4f}")
print(f"  fraction of time with z < 1.0 (near-planar spiral): {frac_small_z:.4f}")
print(f"  fraction of time with z > 5.0 (fold excursion): {frac_spike_z:.4f}")

band_structure_ok = (
    finite_ok
    and z_median < 1.0          # mostly near the x-y plane
    and z_max > 5.0              # but with real excursions (folds)
    and 0.02 < frac_spike_z < 0.5  # spikes are a minority of the time, not absent, not dominant
)

# Exponential sensitivity to initial conditions: perturb by 1e-8 and track
# separation growth over a short window before it saturates at the
# attractor's diameter.
state_a = post_transient[0].copy()
state_b = state_a + np.array([1e-8, 0.0, 0.0])
sep = []
sa, sb = state_a.copy(), state_b.copy()
N_LYAP_STEPS = 3000  # 30 time units
for _ in range(N_LYAP_STEPS):
    sa = rk4_step(sa, DT)
    sb = rk4_step(sb, DT)
    sep.append(np.linalg.norm(sa - sb))
sep = np.array(sep)
# Fit growth in the pre-saturation regime (before separation gets comparable
# to attractor size, roughly the first half where sep is still << 1).
pre_sat = sep < 0.5
if np.sum(pre_sat) > 10:
    t_idx = np.arange(len(sep))[pre_sat]
    log_sep = np.log(sep[pre_sat])
    lyap_slope, _ = np.polyfit(t_idx * DT, log_sep, 1)
else:
    lyap_slope = float("nan")

print(f"\nSensitivity-to-initial-conditions check: initial perturbation 1e-8, "
      f"estimated local exponential growth rate (finite-time Lyapunov proxy) = "
      f"{lyap_slope:.4f} per time unit (positive => chaotic divergence)")
chaos_ok = lyap_slope > 0.0

solver_verified = bool(finite_ok and band_structure_ok and chaos_ok)
print(f"\nSolver verification (finite trajectory: {finite_ok}, "
      f"band/fold structure: {band_structure_ok}, "
      f"positive divergence rate: {chaos_ok}) -> {'PASSED' if solver_verified else 'FAILED'}")

# ---------------------------------------------------------------------------
# 3. Build the point cloud for the k-NN hypergraph.
#    O(n^2) distance computation in knn_hypergraph caps tractable size at a
#    few thousand points. Subsample by STRIDE, evenly spaced in time across
#    the ENTIRE post-transient trajectory (not a truncated prefix/suffix),
#    so the point cloud still covers the full attractor -- both the near-
#    planar spiral region and the folded excursions -- rather than only a
#    slice of trajectory time.
# ---------------------------------------------------------------------------

TARGET_POINTS = 1500
stride = max(1, len(post_transient) // TARGET_POINTS)
sub_traj = post_transient[::stride]
points = [tuple(p) for p in sub_traj]
n_points = len(points)

print(f"\nSubsampled point cloud: stride={stride} (from {len(post_transient)} "
      f"post-transient points), n_points={n_points}")

K = 10  # matches this module's own filled-square (2D) calibration test

hg = knn_hypergraph(points, k=K)

# ---------------------------------------------------------------------------
# 4. Estimate dimension.
#    Try over all sampled points first, matching the discipline used for the
#    Brownian-motion and filled-square calibration cases (no interior
#    restriction needed there); only restrict if boundary/saturation effects
#    turn out to dominate the fit -- exactly the discipline used for the
#    filled-cube case in tests/test_hypergraph_pointcloud.py.
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
# 5. Compare against the literature correlation-dimension value.
# ---------------------------------------------------------------------------

TRADITIONAL_DIMENSION = 2.01
TOLERANCE = 0.5

final_dimension = mean_dim_wrapper
abs_error = abs(final_dimension - TRADITIONAL_DIMENSION)
passed = solver_verified and (abs_error <= TOLERANCE) and not math.isnan(final_dimension)

print(f"\n=== RESULT ===")
print(f"Traditional (Grassberger & Procaccia 1983) correlation dimension: {TRADITIONAL_DIMENSION}")
print(f"Poly-algebraic (knn_hypergraph + mean_dimension) dimension: {final_dimension:.4f}")
print(f"Absolute error: {abs_error:.4f} (tolerance {TOLERANCE})")
print(f"Solver verified (band structure + positive divergence): {solver_verified}")
print(f"PASSED: {passed}")
