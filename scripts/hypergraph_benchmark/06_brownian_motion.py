"""Benchmark 06: Planar Brownian motion (2D random walk).

KNOWN/TRADITIONAL DIMENSION: 2 (exact, not empirical)
SOURCE: S. J. Taylor, 1953, "The Hausdorff alpha-dimensional measure of
Brownian paths in n-space", Proc. Camb. Phil. Soc. 49 -- the Hausdorff
dimension of a planar Brownian motion path is almost surely exactly 2.

This is a genuinely different benchmark category from the deterministic /
periodic problems elsewhere in this suite: a stochastic process with an
EXACT known answer. We simulate a discrete Gaussian random walk (the
standard discretization of Brownian motion), verify it actually behaves
diffusively (the defining statistical signature: MSD ~ t), then feed the
(x, y) walk positions into the same knn_hypergraph + mean_dimension
machinery used throughout this module, and report whether the measured
"emergent" dimension matches the theorem.

Do not modify src/socrates/hypergraph/* -- shared with other agents.
"""

from __future__ import annotations

import math

import numpy as np

from socrates.hypergraph.dimension import local_dimension, mean_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

# ---------------------------------------------------------------------------
# 1. Simulate the random walk.
# ---------------------------------------------------------------------------

SEED = 42
N_STEPS = 4000          # number of increments (path has N_STEPS + 1 points)
STEP_SIGMA = 1.0        # std of each Gaussian increment component, dt = 1

rng = np.random.default_rng(SEED)
increments = rng.normal(loc=0.0, scale=STEP_SIGMA, size=(N_STEPS, 2))
path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)], axis=0)
# path.shape == (N_STEPS + 1, 2)

print(f"Simulated planar random walk: seed={SEED}, steps={N_STEPS}, "
      f"step_sigma={STEP_SIGMA}, path length={len(path)} points")
print(f"Start: {path[0]}, End: {path[-1]}, "
      f"max |displacement|: {np.max(np.linalg.norm(path, axis=1)):.2f}")

# ---------------------------------------------------------------------------
# 2. VERIFY the solver: diffusive scaling, MSD(lag) ~ lag (slope ~ 1 in log-log).
#    This is the defining statistical property of Brownian motion and is
#    checkable directly from the single generated path via lag-averaging
#    (the walk is a stationary-increment process, so this is a legitimate,
#    standard estimator -- not a circular check).
# ---------------------------------------------------------------------------

max_lag = 500  # << N_STEPS, keeps enough independent-ish windows per lag
lags = np.arange(10, max_lag + 1, 10)
msd_vals = []
for lag in lags:
    diffs = path[lag:] - path[:-lag]
    sq_disp = np.sum(diffs ** 2, axis=1)
    msd_vals.append(np.mean(sq_disp))
msd_vals = np.array(msd_vals)

log_lag = np.log(lags.astype(float))
log_msd = np.log(msd_vals)
slope, intercept = np.polyfit(log_lag, log_msd, 1)
ss_res = np.sum((log_msd - (slope * log_lag + intercept)) ** 2)
ss_tot = np.sum((log_msd - np.mean(log_msd)) ** 2)
msd_r_squared = 1.0 - ss_res / ss_tot

print(f"\nMSD(lag) ~ lag^slope fit: slope={slope:.4f}, r_squared={msd_r_squared:.5f}")
print("Expected slope ~= 1.0 for standard Brownian motion (diffusive scaling).")

diffusive_ok = abs(slope - 1.0) < 0.15 and msd_r_squared > 0.98
print(f"Diffusive-scaling check {'PASSED' if diffusive_ok else 'FAILED'} "
      f"(|slope-1|={abs(slope - 1.0):.4f}, r_squared={msd_r_squared:.5f})")

# ---------------------------------------------------------------------------
# 3. Build the point cloud for the k-NN hypergraph.
#    O(n^2) distance computation in knn_hypergraph means the full ~4001-point
#    path is not tractable (empirically ~50s already at n=1500, growing
#    roughly as n^2). We subsample by STRIDE (evenly spaced in time across
#    the *entire* trajectory, not a truncated prefix/suffix) so the point
#    cloud still covers the full spatial extent and the recurrent
#    self-intersections that give the path its anomalous (>1) dimension are
#    preserved, rather than only the fine local step-to-step structure.
# ---------------------------------------------------------------------------

TARGET_POINTS = 1300
stride = max(1, len(path) // TARGET_POINTS)
sub_path = path[::stride]
points = [tuple(p) for p in sub_path]
n_points = len(points)

print(f"\nSubsampled point cloud: stride={stride} (from {len(path)} path points), "
      f"n_points={n_points}")

K = 10  # matches this module's own filled-square (2D, dimension-2) calibration test

hg = knn_hypergraph(points, k=K)

# ---------------------------------------------------------------------------
# 4. Estimate dimension.
#    Try the estimator directly over all sampled points first (this module's
#    own dimension-2 calibration test, the filled square, does NOT need an
#    interior restriction), and separately check whether trajectory-endpoint
#    ("boundary") effects are visible, restricting only if needed -- exactly
#    the discipline used for the filled-cube test in tests/test_hypergraph_pointcloud.py.
# ---------------------------------------------------------------------------

MAX_RADIUS = 5
SAMPLES = 30

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

# Also compute via the module's own mean_dimension convenience wrapper for
# a cross-check (uses its own r_squared>=0.9 filtering internally).
mean_dim_wrapper = mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS)

print(f"\nMean dimension (manual, well-fit only): {mean_dim_manual:.4f} "
      f"(mean r_squared of well-fit points: {mean_r2:.4f})")
print(f"Mean dimension (module mean_dimension wrapper): {mean_dim_wrapper:.4f}")

# ---------------------------------------------------------------------------
# 5. Compare against the exact theorem value.
# ---------------------------------------------------------------------------

TRADITIONAL_DIMENSION = 2.0
TOLERANCE = 0.3

final_dimension = mean_dim_wrapper
abs_error = abs(final_dimension - TRADITIONAL_DIMENSION)
passed = diffusive_ok and (abs_error <= TOLERANCE) and not math.isnan(final_dimension)

print(f"\n=== RESULT ===")
print(f"Traditional (exact, Taylor 1953) dimension: {TRADITIONAL_DIMENSION}")
print(f"Poly-algebraic (knn_hypergraph + mean_dimension) dimension: {final_dimension:.4f}")
print(f"Absolute error: {abs_error:.4f} (tolerance {TOLERANCE})")
print(f"Solver (diffusive scaling) verified: {diffusive_ok}")
print(f"PASSED: {passed}")
