"""Round 2 hypergraph dimension benchmark: nonlinear pendulum (large amplitude).

Problem: theta'' = -sin(theta), theta(0) = 2.0 (below the pi separatrix),
thetadot(0) = 0. KNOWN/TRADITIONAL DIMENSION: 1 (a bound pendulum below the
separatrix traces a smooth closed periodic orbit in (theta, thetadot) phase
space; its correlation dimension is exactly 1 regardless of the non-constant
speed at which it is traversed).

This problem is flagged for the DENSITY-VARIATION test: a pendulum released
from rest lingers near its turning points (thetadot ~ 0) and sprints through
the bottom (thetadot at its max), so a plain time-uniform sample is *heavily*
non-uniform in arc length -- a real physical effect, not a manufactured one.

Physics/solver: identical setup to round 1's
scripts/hypergraph_benchmark/02_nonlinear_pendulum.py (leapfrog, same THETA0,
same solver-verification gates against the exact elliptic period
4*K(sin^2(theta0/2))). Not re-derived here, only re-verified, per the task's
"read the round-1 script as a reference, do not re-derive" instruction.

Two things are new relative to round 1, both required by this round's goal:

1. SINGLE PERIOD ONLY. Round 1 integrated 3.2 periods and then resampled
   evenly in ARC LENGTH specifically to *avoid* density bias (the round-1
   goal was reporting one clean accuracy number). Round 2's goal (b) is the
   opposite: to directly test robustness *to* that density bias. So this
   script samples exactly one period, time-uniformly, and never rescales by
   arc length. One period also sidesteps finding F3/N4 (near-duplicate
   points from repeated traversals of the same closed curve) at the source,
   per the task instructions, rather than relying on dedupe as a band-aid.

2. BIT-REVERSAL ORDERING for the compare() input. `comparison.compare()`
   evaluates both methods on `points[:n]` prefixes for a grid of n. A plain
   time-ordered single-period trajectory has a pathological property for
   this API: `points[:128]` would cover only the *first* ~1/32 of the
   period in time, not the whole closed curve -- an artifact of slicing, not
   a real small-n measurement. The standard fix (used for progressive/
   multiresolution sampling) is a bit-reversal permutation of a
   power-of-two-length sequence: for M = 2^m samples in time order, the
   first 2^k elements of the bit-reversal-permuted sequence are *exactly*
   the set {0, M/2^k, 2*M/2^k, ...} -- i.e. an evenly-time-spaced,
   full-period-covering subsample, for every k <= m simultaneously. This
   lets one single-period point cloud serve every n in a power-of-two
   n_grid, each level being a genuine time-uniform (density-variation-
   intact) sample of the *entire* orbit, not a truncated arc. Verified
   directly below (`_check_bitreversal_prefix_coverage`) before being
   trusted, not assumed from the textbook property.

Do not modify src/socrates/hypergraph/ or src/socrates/solvers/ or any
round-1 script -- shared with other concurrently running benchmark agents.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ellipk

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.comparison import compare  # noqa: E402
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

THETA0 = 2.0  # rad, well below the pi separatrix (same as round 1)
TRUE_DIMENSION = 1.0
TOLERANCE = 0.15

# n_grid is a power-of-two ladder (see module docstring, point 2) large
# enough that the traditional correlation-sum method has a real chance to
# converge (per standing rule 2 -- round 1's N5 speedup makes n=4096 cheap,
# no reason to under-test). M = max(n_grid) is also the number of
# time-uniform steps integrated over exactly one period.
N_GRID: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096)
M = N_GRID[-1]
K = 6  # knn_hypergraph neighbour count (same choice as round 1)
MAX_RADIUS = 6

# Solver verification gates (identical to round 1's, same method: zero-crossing
# period measurement at a fine, independently-chosen dt, plus a one-period
# state-space return check).
VERIFY_DT = 1e-3
PERIOD_REL_ERR_GATE = 1e-5
RETURN_ERR_GATE = 5e-2
# Separate, tighter closure gate for the actual M-step, dt=period/M cloud-
# generating run (checked independently since it uses a different, coarser dt
# than the verification run).
CLOUD_RETURN_ERR_GATE = 1e-3

# Density-robustness check (goal b): "large enough to be meaningful, e.g.
# traditional_min_n or 1000, whichever is larger" per the task instructions.
DENSITY_CHECK_MIN_N = 1000
DENSITY_ROBUSTNESS_MARGIN = 0.30  # poly error must be >=30% smaller to win


def force(q: np.ndarray) -> np.ndarray:
    return -np.sin(q)


def verify_solver() -> dict[str, object]:
    """Same method as round 1: measured period (zero-crossing) vs the exact
    elliptic period, AND a state-space return check after one exact period."""
    exact_period = float(4.0 * ellipk(np.sin(THETA0 / 2.0) ** 2))
    n_steps = int(1.5 * exact_period / VERIFY_DT)
    run = leapfrog(force, [THETA0], [0.0], dt=VERIFY_DT, n_steps=n_steps)
    theta = run.positions[:, 0]
    thetadot = run.velocities[:, 0]
    times = run.times

    idx = int(np.argmax(theta < 0))
    t_quarter = times[idx - 1] + VERIFY_DT * theta[idx - 1] / (theta[idx - 1] - theta[idx])
    measured_period = 4.0 * float(t_quarter)
    period_rel_err = abs(measured_period - exact_period) / exact_period

    n_one_period = int(round(exact_period / VERIFY_DT))
    theta_at_T = float(theta[n_one_period])
    thetadot_at_T = float(thetadot[n_one_period])
    return_err = math.hypot(theta_at_T - THETA0, thetadot_at_T - 0.0)

    passed = period_rel_err < PERIOD_REL_ERR_GATE and return_err < RETURN_ERR_GATE
    return {
        "exact_period": exact_period,
        "measured_period": measured_period,
        "period_relative_error": period_rel_err,
        "return_error": return_err,
        "passed": passed,
    }


def bit_reversal_permutation(m_bits: int) -> list[int]:
    """perm[i] = bit-reversal of i over m_bits bits, for i in 0..2^m_bits - 1."""
    size = 1 << m_bits
    return [int(f"{i:0{m_bits}b}"[::-1], 2) for i in range(size)]


def check_bitreversal_prefix_coverage(perm: list[int], size: int) -> None:
    """Verify (not assume) that prefixes of the bit-reversal-permuted sequence
    at every power-of-two length are exactly an evenly-strided index set."""
    m_bits = size.bit_length() - 1
    for k in range(0, m_bits + 1):
        n = 1 << k
        prefix = sorted(perm[:n])
        stride = size // n
        expected = list(range(0, size, stride))
        assert prefix == expected, (
            f"bit-reversal prefix-coverage property FAILED at n={n}: "
            f"got {prefix[:5]}..., expected {expected[:5]}..."
        )


def generate_single_period_cloud() -> dict[str, object]:
    """Integrate exactly one period at dt = exact_period / M (time-uniform),
    verify closure at that dt independently, and return both the natural
    time-ordered array and the bit-reversal-permuted array used for compare()."""
    exact_period = float(4.0 * ellipk(np.sin(THETA0 / 2.0) ** 2))
    dt = exact_period / M
    run = leapfrog(force, [THETA0], [0.0], dt=dt, n_steps=M)
    theta = run.positions[:, 0]
    thetadot = run.velocities[:, 0]

    return_err = math.hypot(theta[M] - THETA0, thetadot[M] - 0.0)
    cloud_closure_passed = return_err < CLOUD_RETURN_ERR_GATE

    # Exactly one period, time-uniform, no duplicated start/end point.
    time_ordered = [(float(theta[i]), float(thetadot[i])) for i in range(M)]

    m_bits = M.bit_length() - 1
    perm = bit_reversal_permutation(m_bits)
    check_bitreversal_prefix_coverage(perm, M)
    bitrev_ordered = [time_ordered[perm[i]] for i in range(M)]

    return {
        "dt": dt,
        "return_error": return_err,
        "cloud_closure_passed": cloud_closure_passed,
        "time_ordered": time_ordered,
        "bitrev_ordered": bitrev_ordered,
    }


def natural_time_uniform_subsample(time_ordered: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """A genuine time-uniform (density-variation-intact) subsample of size n
    from the single-period time-ordered trajectory -- evenly strided in TIME,
    not resampled by arc length, so the turning-point clustering survives."""
    size = len(time_ordered)
    idx = np.linspace(0, size, num=n, endpoint=False).astype(int)
    return [time_ordered[i] for i in idx]


def poly_algebraic_dimension_estimate(points: list[tuple[float, float]]) -> float:
    hg = knn_hypergraph(points, k=K, dedupe=True)
    return mean_dimension(hg, samples=40, max_radius=MAX_RADIUS)


def main() -> dict[str, object]:
    print("=== Round 2: Nonlinear pendulum (large amplitude) -- hypergraph dimension benchmark ===\n")

    verification = verify_solver()
    print("--- Solver verification (same method as round 1) ---")
    print(f"  exact period 4*K(m):     {verification['exact_period']:.10f}")
    print(f"  measured period:         {verification['measured_period']:.10f}")
    print(
        f"  period relative error:   {verification['period_relative_error']:.3e}"
        f"  (gate < {PERIOD_REL_ERR_GATE:.0e})"
    )
    print(
        f"  return error (1 period): {verification['return_error']:.3e}"
        f"  (gate < {RETURN_ERR_GATE:.0e})"
    )
    print(f"  SOLVER VERIFIED: {verification['passed']}\n")
    if not verification["passed"]:
        raise RuntimeError("solver verification failed; refusing to trust the point cloud")

    cloud = generate_single_period_cloud()
    print("--- Single-period point cloud (time-uniform, dt = exact_period / M) ---")
    print(f"  M (steps = points in one period): {M}")
    print(f"  dt:                                {cloud['dt']:.6e}")
    print(
        f"  closure return error at this dt:  {cloud['return_error']:.3e}"
        f"  (gate < {CLOUD_RETURN_ERR_GATE:.0e}) -> {cloud['cloud_closure_passed']}"
    )
    if not cloud["cloud_closure_passed"]:
        raise RuntimeError("single-period cloud did not close cleanly; refusing to trust it")
    print("  bit-reversal prefix-coverage property: VERIFIED (checked, not assumed)\n")

    time_ordered = cloud["time_ordered"]
    bitrev_ordered = cloud["bitrev_ordered"]

    print(f"--- compare() : poly-algebraic vs traditional, n_grid={N_GRID}, k={K} ---")
    result = compare(
        bitrev_ordered,
        true_dimension=TRUE_DIMENSION,
        tolerance=TOLERANCE,
        k=K,
        n_grid=N_GRID,
        max_radius=MAX_RADIUS,
    )
    print(f"  poly_algebraic_min_n:        {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n:           {result.traditional_min_n}")
    print(f"  poly_algebraic_wins:         {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction:    {result.compute_savings_fraction}")
    print(f"  poly estimate @ max_n:       {result.poly_algebraic_estimate_at_max_n:.4f}")
    print(f"  traditional estimate @ max_n:{result.traditional_estimate_at_max_n:.4f}")
    print(f"  max_n_tested:                {result.max_n_tested}\n")

    won_on_compute_savings = bool(
        result.poly_algebraic_wins
        and result.compute_savings_fraction is not None
        and result.compute_savings_fraction > 0.10
    )

    # --- Density robustness (goal b) ---
    n_fixed = max(result.traditional_min_n or 0, DENSITY_CHECK_MIN_N)
    n_fixed = min(n_fixed, M)
    density_points = natural_time_uniform_subsample(time_ordered, n_fixed)

    # Confirm the density variation is real and substantial: compare local
    # point spacing near a turning point (|theta| close to THETA0, thetadot
    # ~ 0) vs near the bottom (theta ~ 0, thetadot at its extreme), on this
    # exact sample.
    theta_arr = np.array([p[0] for p in density_points])
    thetadot_arr = np.array([p[1] for p in density_points])
    near_turning = np.argsort(np.abs(np.abs(theta_arr) - THETA0))[:20]
    near_bottom = np.argsort(np.abs(theta_arr))[:20]

    def local_spacing(idxs: np.ndarray) -> float:
        pts = np.array(density_points)
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        d, _ = tree.query(pts[idxs], k=2)
        return float(np.mean(d[:, 1]))

    spacing_turning = local_spacing(near_turning)
    spacing_bottom = local_spacing(near_bottom)

    print(f"--- Density-variation confirmation (n={n_fixed}, natural time-uniform sample) ---")
    print(f"  mean nearest-neighbour spacing near turning point: {spacing_turning:.5f}")
    print(f"  mean nearest-neighbour spacing near bottom (theta~0): {spacing_bottom:.5f}")
    print(f"  density ratio (bottom spacing / turning spacing):  {spacing_bottom / spacing_turning:.2f}x\n")

    try:
        poly_dim = poly_algebraic_dimension_estimate(density_points)
    except DuplicatePointsError as exc:
        raise RuntimeError(f"unexpected duplicates in single-period density sample: {exc}") from exc
    trad_dim = correlation_dimension(density_points).dimension

    poly_error = abs(poly_dim - TRUE_DIMENSION)
    trad_error = abs(trad_dim - TRUE_DIMENSION)
    if trad_error > 0:
        won_on_density_robustness = poly_error <= (1.0 - DENSITY_ROBUSTNESS_MARGIN) * trad_error
    else:
        won_on_density_robustness = poly_error == 0.0

    print(f"--- Density robustness: dimension estimate on natural time-uniform sample (n={n_fixed}) ---")
    print(f"  poly-algebraic estimate:  {poly_dim:.4f}  (abs error {poly_error:.4f})")
    print(f"  traditional estimate:     {trad_dim:.4f}  (abs error {trad_error:.4f})")
    print(f"  WON_ON_DENSITY_ROBUSTNESS (poly error >=30% smaller): {won_on_density_robustness}\n")

    overall_win = bool(won_on_compute_savings or won_on_density_robustness)

    print("=== SUMMARY ===")
    print(f"  poly_algebraic_min_n:        {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n:           {result.traditional_min_n}")
    print(f"  poly_algebraic_wins:         {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction:    {result.compute_savings_fraction}")
    print(f"  won_on_compute_savings:      {won_on_compute_savings}")
    print(f"  won_on_density_robustness:   {won_on_density_robustness}")
    print(f"  OVERALL_WIN:                 {overall_win}")

    return {
        "verification": verification,
        "n_grid": N_GRID,
        "k": K,
        "poly_algebraic_min_n": result.poly_algebraic_min_n,
        "traditional_min_n": result.traditional_min_n,
        "poly_algebraic_wins": result.poly_algebraic_wins,
        "compute_savings_fraction": result.compute_savings_fraction,
        "won_on_compute_savings": won_on_compute_savings,
        "n_fixed_density_check": n_fixed,
        "density_ratio": spacing_bottom / spacing_turning,
        "poly_dim_density_sample": poly_dim,
        "trad_dim_density_sample": trad_dim,
        "poly_error_density_sample": poly_error,
        "trad_error_density_sample": trad_error,
        "won_on_density_robustness": won_on_density_robustness,
        "overall_win": overall_win,
    }


if __name__ == "__main__":
    main()
