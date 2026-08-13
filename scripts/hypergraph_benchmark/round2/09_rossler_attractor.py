"""Round 2 benchmark: Rossler attractor (a=b=0.2, c=5.7) -- poly-algebraic vs traditional.

KNOWN/TRADITIONAL DIMENSION: 2.01 (correlation dimension, tolerance 0.5).
SOURCE: Grassberger & Procaccia (1983), Physica D 9, 189-208; O. Rossler
(1976), "An equation for continuous chaos", Phys. Lett. A 57.

Compares socrates.hypergraph.dimension (shell-growth on a k-NN proximity
hypergraph) against socrates.hypergraph.baseline (classical Grassberger-
Procaccia correlation-sum dimension) via the apples-to-apples harness in
socrates.hypergraph.comparison, on axis (a) only: compute savings (fewer
points needed for equivalent, STABLY CONVERGED accuracy,
ComparisonResult.compute_savings_fraction, requiring poly_algebraic_wins to
be True, i.e. both methods actually converged).

Axis (b) (density-robustness) is NOT tested for this problem: the Rossler
attractor's natural parametrization (fixed RK4 dt) samples the trajectory
at constant speed-independent TIME density -- there is no close-approach /
slow-fast turning point in this system's own natural sampling the way there
is for e.g. Kepler near perihelion or CR3BP near a primary (the assignment
brief explicitly says so for this problem). Manufacturing an artificial
density variation would not be testing anything physical, so
density_robustness_tested=False and the result is reported as not
applicable, per the assignment's own instruction for this problem.

DIFFERENCES FROM ROUND 1 (scripts/hypergraph_benchmark/09_rossler_attractor.py):
  - Round 1 built a single non-periodic post-transient trajectory (T=1000
    time units after discarding a T=100 transient) and evenly subsampled it
    down to a FIXED 1500-point cloud, then only ever measured the dimension
    at that one point count. That is fine for a single measurement but
    cannot answer "how many points does each method need" -- there was no
    growing-n sweep.
  - This round instead builds a single long non-periodic trajectory at
    CONSTANT time density (points per unit time), long enough that its
    length equals the largest n_grid value, and hands the FULL,
    un-subsampled-further, time-ordered list to
    socrates.hypergraph.comparison.compare(), which evaluates growing
    PREFIXES points[:n] for each n in n_grid -- exactly the discipline used
    in round 2's Kepler/torus/pendulum scripts (see
    tests/test_hypergraph_comparison.py's circle calibration test) and the
    only way to measure "points needed for convergence" honestly. A prefix
    of a constant-time-density sample is itself a shorter, but still
    representative (same density), stretch of trajectory -- not a
    re-subsampled ad hoc set -- so smaller n genuinely means "less
    observation time", which is the real-world quantity a "fewer points
    needed" claim is about.
  - The Rossler attractor is chaotic and NOT a closed orbit, so there is no
    multi-period-duplication risk analogous to closed orbits (finding
    F3/N4/N7) -- a single continuous non-repeating trajectory is used
    directly, as round 1 already did. A duplicate-cluster sanity check is
    still run explicitly below (not just assumed) before trusting that.
  - k and n_grid are both increased from round 1 (which capped at 1500
    points and never tested convergence-vs-n at all) since N5's ~27x
    knn_hypergraph speedup (scipy cKDTree) makes n up to 12800 cheap, and
    the standing rules require giving the traditional method a real chance
    to converge rather than under-testing out of old habit.

Does NOT modify src/socrates/hypergraph/, src/socrates/solvers/, or any
scripts/hypergraph_benchmark/*.py file from round 1.
"""

from __future__ import annotations

import numpy as np

from socrates.hypergraph.baseline import correlation_dimension
from socrates.hypergraph.comparison import compare
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph

# ---------------------------------------------------------------------------
# 1. Rossler system (unchanged physics from round 1) -- hand-written RK4.
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


TRUE_DIMENSION = 2.01
TOLERANCE = 0.5

DT = 0.01
T_TRANSIENT = 100.0  # discard first 100 time units, matches round 1 (settles slower than Lorenz)

K = 15  # matches round 1's choice for this system (fractal D~2 needs more neighbours/shell
        # than the integer-dimension calibration cases; round 1's Lorenz sibling used the same)
N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
MAX_RADIUS = 6  # comparison.py / dimension.py default

N_MAX = N_GRID[-1]
DENSITY = 20.0  # points per unit time (dt_eff=0.05 => stride 5 of the raw dt=0.01 RK4 steps);
                 # fine enough to resolve local curve geometry within a single spiral loop
                 # (the linearized frequency near the origin is omega0 = sqrt(1-(a/2)^2) ~ 0.997,
                 # i.e. one spiral loop takes ~6.3 time units, so 20 pts/time unit gives ~126
                 # points per loop -- comfortably resolved) while keeping stride>=1 vs the raw
                 # RK4 step so consecutive kept points are never near-duplicates.
T_POST = N_MAX / DENSITY  # =640.0 time units of post-transient trajectory needed
T_TOTAL = T_TRANSIENT + T_POST
N_STEPS = int(round(T_TOTAL / DT))
TRANSIENT_STEPS = int(round(T_TRANSIENT / DT))
STRIDE = int(round((1.0 / DENSITY) / DT))


def integrate() -> np.ndarray:
    state = np.array([1.0, 1.0, 1.0])  # away from the fixed points, matches round 1
    trajectory = np.empty((N_STEPS + 1, 3))
    trajectory[0] = state
    for i in range(N_STEPS):
        state = rk4_step(state, DT)
        trajectory[i + 1] = state
    return trajectory


def verify_solver(post_transient: np.ndarray) -> dict[str, object]:
    """Same checks as round 1: characteristic band/fold structure (z mostly
    near 0 with occasional spikes) and positive finite-time Lyapunov proxy
    (exponential sensitivity to a 1e-8 initial perturbation)."""
    z_vals = post_transient[:, 2]
    z_median = float(np.median(z_vals))
    z_max = float(np.max(z_vals))
    frac_spike_z = float(np.mean(z_vals > 5.0))
    finite_ok = bool(np.all(np.isfinite(post_transient)))
    band_structure_ok = (
        finite_ok and z_median < 1.0 and z_max > 5.0 and 0.02 < frac_spike_z < 0.5
    )

    state_a = post_transient[0].copy()
    state_b = state_a + np.array([1e-8, 0.0, 0.0])
    sep = []
    sa, sb = state_a.copy(), state_b.copy()
    for _ in range(3000):  # 30 time units
        sa = rk4_step(sa, DT)
        sb = rk4_step(sb, DT)
        sep.append(np.linalg.norm(sa - sb))
    sep = np.array(sep)
    pre_sat = sep < 0.5
    if np.sum(pre_sat) > 10:
        t_idx = np.arange(len(sep))[pre_sat]
        log_sep = np.log(sep[pre_sat])
        lyap_slope, _ = np.polyfit(t_idx * DT, log_sep, 1)
    else:
        lyap_slope = float("nan")
    chaos_ok = bool(lyap_slope > 0.0)

    passed = bool(finite_ok and band_structure_ok and chaos_ok)
    return {
        "finite": finite_ok,
        "z_median": z_median,
        "z_max": z_max,
        "frac_spike_z": frac_spike_z,
        "band_structure_ok": band_structure_ok,
        "lyapunov_proxy_slope": float(lyap_slope),
        "chaos_ok": chaos_ok,
        "passed": passed,
    }


def main() -> None:
    print(f"=== Rossler attractor: a={A}, b={B}, c={C} ===")
    print(f"DT={DT}, T_transient={T_TRANSIENT}, T_post={T_POST}, T_total={T_TOTAL}, "
          f"N_STEPS={N_STEPS}, DENSITY={DENSITY} pts/time-unit, STRIDE={STRIDE}")

    trajectory = integrate()
    post_transient = trajectory[TRANSIENT_STEPS:]

    solver_check = verify_solver(post_transient)
    print("\n=== Solver verification (band/fold structure + positive Lyapunov proxy) ===")
    for k, v in solver_check.items():
        print(f"  {k}: {v}")
    solver_ok = solver_check["passed"]
    assert solver_ok, "solver verification failed -- not a genuine Rossler trajectory"

    sub = post_transient[::STRIDE][:N_MAX]
    points = [(float(p[0]), float(p[1]), float(p[2])) for p in sub]
    print(f"\n=== Point cloud ===")
    print(f"  post-transient raw steps: {len(post_transient)}, stride: {STRIDE}, "
          f"n_points built: {len(points)} (target N_MAX={N_MAX})")
    assert len(points) >= N_MAX, f"need >= {N_MAX} points, got {len(points)}"

    # Explicit duplicate-cluster sanity check (finding F3/N4/N7) -- confirm,
    # don't assume, that this continuous non-periodic trajectory produced no
    # near-duplicate point clusters at the chosen stride.
    try:
        knn_hypergraph(points, k=K, dedupe=False)
        print("  duplicate check: none found (dedupe=False did not raise)")
    except DuplicatePointsError as exc:
        raise AssertionError(f"unexpected near-duplicate point clusters: {exc}") from exc

    # ---- Compute-savings comparison over the n_grid --------------------
    print(f"\n=== compare() : k={K}, n_grid={N_GRID}, max_radius={MAX_RADIUS}, "
          f"tolerance={TOLERANCE} ===")
    result = compare(
        points,
        true_dimension=TRUE_DIMENSION,
        tolerance=TOLERANCE,
        k=K,
        n_grid=N_GRID,
        max_radius=MAX_RADIUS,
    )
    print(f"  poly_algebraic_min_n: {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n: {result.traditional_min_n}")
    print(f"  poly_algebraic_estimate_at_max_n: {result.poly_algebraic_estimate_at_max_n:.4f}")
    print(f"  traditional_estimate_at_max_n: {result.traditional_estimate_at_max_n:.4f}")
    print(f"  max_n_tested: {result.max_n_tested}")
    print(f"  poly_algebraic_wins: {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction: {result.compute_savings_fraction}")

    # Independent re-derivation trace (do not just trust compare()'s
    # internal min-n search blindly): re-run both estimators by hand at
    # every n_grid stop and print the actual dimension/r_squared, per
    # standing rule 1 ("verify before trusting").
    from socrates.hypergraph.dimension import mean_dimension

    print("\n  --- independent per-n trace ---")
    for n in N_GRID:
        subset = points[:n]
        hg = knn_hypergraph(subset, k=K, dedupe=True)
        poly_dim = mean_dimension(hg, samples=min(n, 40), max_radius=MAX_RADIUS)
        trad_est = correlation_dimension(subset)
        poly_in = "IN " if (np.isfinite(poly_dim) and abs(poly_dim - TRUE_DIMENSION) <= TOLERANCE) else "out"
        trad_in = "IN " if (np.isfinite(trad_est.dimension) and abs(trad_est.dimension - TRUE_DIMENSION) <= TOLERANCE) else "out"
        print(f"    n={n:6d}  poly_dim={poly_dim:7.4f} [{poly_in}]   "
              f"trad_dim={trad_est.dimension:7.4f} r2={trad_est.r_squared:.4f} [{trad_in}]")

    won_on_compute_savings = bool(
        result.poly_algebraic_wins
        and result.compute_savings_fraction is not None
        and result.compute_savings_fraction > 0.10
    )
    print(f"\n  won_on_compute_savings (>10% savings, both converged): {won_on_compute_savings}")

    # ---- Density-robustness: not applicable for this system ------------
    density_robustness_tested = False
    density_robustness_result = (
        "not applicable: no natural density variation in this system's natural sampling"
    )
    won_on_density_robustness = False
    print(f"\n=== Density-robustness ===")
    print(f"  tested: {density_robustness_tested}")
    print(f"  result: {density_robustness_result}")

    overall_win = won_on_compute_savings or won_on_density_robustness

    print(f"\n=== SUMMARY ===")
    print(f"  poly_algebraic_min_n: {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n: {result.traditional_min_n}")
    print(f"  poly_algebraic_wins: {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction: {result.compute_savings_fraction}")
    print(f"  won_on_compute_savings: {won_on_compute_savings}")
    print(f"  won_on_density_robustness: {won_on_density_robustness}")
    print(f"  overall_win: {overall_win}")


if __name__ == "__main__":
    main()
