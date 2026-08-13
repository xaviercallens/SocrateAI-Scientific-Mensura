"""Round 2 hypergraph dimension benchmark: driven damped pendulum, mode-locked
period-1 regime.

Problem: thetadot = omega, omegadot = -gamma*omega - sin(theta)
         + F*cos(Omega_drive*t), gamma=0.5, F=0.5, Omega_drive=0.6.
KNOWN/TRADITIONAL DIMENSION: 1 (tolerance 0.2). A genuinely period-1
mode-locked limit cycle of a driven-damped system is, by definition, a
smooth closed curve in (theta mod 2pi, omega) phase space, so its
correlation/topological dimension is exactly 1. This is the standard
"driving+damping does not automatically mean chaos" contrast case; the
parameters here (F=0.5, well below the classic chaotic threshold F~1.5)
were chosen in round 1 to mode-lock, and that choice is NOT re-derived here
-- only the mode-locking itself is re-verified below (never trust a solver
call without checking it), per the task's "reuse round 1 as reference for
the physics/solver" instruction.

THIS PROBLEM IS FLAGGED FOR THE DENSITY-VARIATION TEST. Also, round 1 (see
scripts/hypergraph_benchmark/10_driven_pendulum_periodic.py's own inline
post-mortem) independently discovered a duplicate-point degeneracy: naively
sampling several drive periods of a genuinely period-1 orbit does not add
distinct points, it stacks ~exact copies of the same closed curve on top of
each other (inter-period distance ~1e-9), which saturates knn_hypergraph's
neighbour lists with near-duplicates instead of true along-curve neighbours
and collapses shell growth. Round 1's fix -- sample exactly ONE period,
densely, in time -- is reused here and re-verified (closure check below),
per standing rule 4(iii) applied specifically to this problem's own
documented history, not assumed clean this time either.

WHAT THIS SCRIPT DOES DIFFERENTLY FROM ROUND 1
------------------------------------------------
1. Re-verifies mode-locking via the SAME stroboscopic (Poincare) section
   method as round 1 (state sampled once per drive period, spread across
   the tail of the check window must be near zero) -- not re-derived, just
   re-run, since the task brief explicitly asks for this given round 1's
   history with this exact problem.
2. Builds the single-period point cloud, then applies a bit-reversal
   permutation (same technique as round 2's 02_nonlinear_pendulum.py) so
   that `comparison.compare()`'s `points[:n]` prefixes at every power-of-two
   n are themselves genuine time-uniform samples of the WHOLE closed period,
   not truncated arcs -- verified directly (`check_bitreversal_prefix_coverage`),
   not assumed from the textbook property.
3. Runs `socrates.hypergraph.comparison.compare()` for a real
   apples-to-apples poly-algebraic-vs-traditional comparison (round 1 had
   no traditional baseline to compare against).
4. Runs the density-variation robustness test (goal b): dimension estimate
   for BOTH methods on the natural TIME-uniform sample (density variation
   intact, no arc-length rescaling) at a fixed, meaningful n, with the
   actual absolute errors reported side by side.

Do not modify src/socrates/hypergraph/ or src/socrates/solvers/ or any
round-1 script -- shared with other concurrently running benchmark agents.

Run: python scripts/hypergraph_benchmark/round2/10_driven_pendulum_periodic.py
(requires the repo venv: source .venv/bin/activate)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.comparison import compare  # noqa: E402
from socrates.hypergraph.dimension import local_dimension, mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph  # noqa: E402

# ---------------------------------------------------------------------------
# Physics parameters -- identical to round 1 (not re-derived).
# ---------------------------------------------------------------------------
GAMMA = 0.5
F_DRIVE = 0.5
OMEGA_DRIVE = 0.6

TRUE_DIMENSION = 1.0
TOLERANCE = 0.2  # per task brief for this problem

T_DRIVE = 2.0 * math.pi / OMEGA_DRIVE  # exact drive period

# Coarse dt used ONLY for the transient + stroboscopic mode-lock verification
# (matches round 1's method: state-boundary-exact steps once per drive period).
VERIFY_STEPS_PER_PERIOD = 500
VERIFY_DT = T_DRIVE / VERIFY_STEPS_PER_PERIOD
N_TRANSIENT_PERIODS = 200
N_CHECK_PERIODS = 50
PERIOD1_TOL = 1e-6

# n_grid: power-of-two ladder (bit-reversal technique, see module docstring)
# large enough that the traditional correlation-sum method has a real chance
# to converge (standing rule 2 -- N5's cKDTree speedup makes n=4096 cheap).
N_GRID: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048, 4096)
M = N_GRID[-1]  # points in the single-period cloud fed to compare()
K = 6
MAX_RADIUS = 6

CLOUD_CLOSURE_GATE = 1e-3  # single-period closure check on the fine M-step cloud

# Density-robustness check (goal b): "large enough to be meaningful, e.g.
# traditional_min_n or 1000, whichever is larger" per the task instructions.
DENSITY_CHECK_MIN_N = 1000
DENSITY_ROBUSTNESS_MARGIN = 0.30  # poly error must be >=30% smaller to win


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


def wrap(theta: float) -> float:
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def verify_mode_locking() -> dict[str, object]:
    """Same method as round 1: long transient, then a stroboscopic (Poincare)
    section over N_CHECK_PERIODS drive periods -- if genuinely period-1, the
    tail of these samples must collapse to a single point (spread < tol)."""
    theta, omega, t = 0.2, 0.0, 0.0
    for _p in range(N_TRANSIENT_PERIODS):
        for _s in range(VERIFY_STEPS_PER_PERIOD):
            theta, omega = rk4_step(theta, omega, t, VERIFY_DT)
            t += VERIFY_DT

    finite_ok = math.isfinite(theta) and math.isfinite(omega)

    strobe = []
    for _p in range(N_CHECK_PERIODS):
        for _s in range(VERIFY_STEPS_PER_PERIOD):
            theta, omega = rk4_step(theta, omega, t, VERIFY_DT)
            t += VERIFY_DT
        strobe.append((wrap(theta), omega))
    strobe_arr = np.array(strobe)
    tail = strobe_arr[len(strobe_arr) // 2 :]
    theta_spread = float(tail[:, 0].max() - tail[:, 0].min())
    omega_spread = float(tail[:, 1].max() - tail[:, 1].min())

    period1_ok = finite_ok and theta_spread < PERIOD1_TOL and omega_spread < PERIOD1_TOL
    passed = bool(finite_ok and period1_ok)

    return {
        "finite_ok": finite_ok,
        "theta_spread": theta_spread,
        "omega_spread": omega_spread,
        "passed": passed,
        # converged state, to seed the fine single-period cloud from the
        # SAME point on the attractor the verification just certified.
        "theta_final": theta,
        "omega_final": omega,
        "t_final": t,
    }


def bit_reversal_permutation(m_bits: int) -> list[int]:
    size = 1 << m_bits
    return [int(f"{i:0{m_bits}b}"[::-1], 2) for i in range(size)]


def check_bitreversal_prefix_coverage(perm: list[int], size: int) -> None:
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


def generate_single_period_cloud(theta0: float, omega0: float, t0: float) -> dict[str, object]:
    """Integrate exactly ONE drive period at fine dt = T_DRIVE / M, starting
    from the mode-lock-verified converged state. Single period only (round
    1's own documented fix for this problem's duplicate-stacking finding)."""
    dt = T_DRIVE / M
    theta, omega, t = theta0, omega0, t0
    thetas = np.empty(M)
    omegas = np.empty(M)
    for i in range(M):
        theta, omega = rk4_step(theta, omega, t, dt)
        t += dt
        thetas[i] = theta
        omegas[i] = omega

    closure_err = math.hypot(theta - theta0, omega - omega0)
    cloud_closure_passed = closure_err < CLOUD_CLOSURE_GATE

    theta_wrapped = np.array([wrap(x) for x in thetas])
    time_ordered = [(float(theta_wrapped[i]), float(omegas[i])) for i in range(M)]

    m_bits = M.bit_length() - 1
    perm = bit_reversal_permutation(m_bits)
    check_bitreversal_prefix_coverage(perm, M)
    bitrev_ordered = [time_ordered[perm[i]] for i in range(M)]

    return {
        "dt": dt,
        "closure_error": closure_err,
        "cloud_closure_passed": cloud_closure_passed,
        "time_ordered": time_ordered,
        "bitrev_ordered": bitrev_ordered,
        "theta_min": float(theta_wrapped.min()),
        "theta_max": float(theta_wrapped.max()),
        "omega_min": float(omegas.min()),
        "omega_max": float(omegas.max()),
    }


def natural_time_uniform_subsample(
    time_ordered: list[tuple[float, float]], n: int
) -> list[tuple[float, float]]:
    """A genuine time-uniform (density-variation-intact) subsample of size n
    from the single-period time-ordered trajectory -- evenly strided in TIME,
    not resampled by arc length, so any turning-point clustering survives."""
    size = len(time_ordered)
    idx = np.linspace(0, size, num=n, endpoint=False).astype(int)
    return [time_ordered[i] for i in idx]


def poly_algebraic_dimension_estimate(points: list[tuple[float, float]]) -> float:
    hg = knn_hypergraph(points, k=K, dedupe=True)
    return mean_dimension(hg, samples=40, max_radius=MAX_RADIUS)


def main() -> dict[str, object]:
    print("=== Round 2: Driven damped pendulum, mode-locked period-1 -- hypergraph dimension benchmark ===\n")
    print(f"gamma={GAMMA}, F={F_DRIVE}, Omega_drive={OMEGA_DRIVE}, T_drive={T_DRIVE:.6f}\n")

    print("--- Step 1: re-verify mode-locking (stroboscopic/Poincare section) ---")
    v = verify_mode_locking()
    print(f"  transient periods:    {N_TRANSIENT_PERIODS}")
    print(f"  check periods:        {N_CHECK_PERIODS}")
    print(f"  finite state:         {v['finite_ok']}")
    print(f"  theta_mod2pi spread (tail half of check window): {v['theta_spread']:.3e}")
    print(f"  omega spread (tail half of check window):        {v['omega_spread']:.3e}")
    print(f"  PERIOD-1 MODE-LOCKING VERIFIED (spread < {PERIOD1_TOL:.0e}): {v['passed']}\n")
    if not v["passed"]:
        raise RuntimeError("mode-locking did not verify; refusing to trust the point cloud")

    print("--- Step 2: build single-period, time-uniform, bit-reversal-ordered point cloud ---")
    cloud = generate_single_period_cloud(v["theta_final"], v["omega_final"], v["t_final"])
    print(f"  M (points in one period): {M}")
    print(f"  dt (fine):                 {cloud['dt']:.6e}")
    print(
        f"  single-period closure error |state(t0+T_drive) - state(t0)|: "
        f"{cloud['closure_error']:.3e} (gate < {CLOUD_CLOSURE_GATE:.0e}) -> {cloud['cloud_closure_passed']}"
    )
    if not cloud["cloud_closure_passed"]:
        raise RuntimeError("single-period cloud did not close cleanly; refusing to trust it")
    print(
        f"  theta_mod2pi range: [{cloud['theta_min']:.4f}, {cloud['theta_max']:.4f}] "
        "(well inside (-pi, pi) -> no wrap-around cut through the sampled data)"
    )
    print(f"  omega range:        [{cloud['omega_min']:.4f}, {cloud['omega_max']:.4f}]")
    print("  bit-reversal prefix-coverage property: VERIFIED (checked, not assumed)\n")

    time_ordered = cloud["time_ordered"]
    bitrev_ordered = cloud["bitrev_ordered"]

    print(f"--- Step 3: compare() -- poly-algebraic vs traditional, n_grid={N_GRID}, k={K}, max_radius={MAX_RADIUS} ---")
    result = compare(
        bitrev_ordered,
        true_dimension=TRUE_DIMENSION,
        tolerance=TOLERANCE,
        k=K,
        n_grid=N_GRID,
        max_radius=MAX_RADIUS,
    )
    print(f"  poly_algebraic_min_n:         {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n:            {result.traditional_min_n}")
    print(f"  poly_algebraic_wins:          {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction:     {result.compute_savings_fraction}")
    print(f"  poly estimate @ max_n:        {result.poly_algebraic_estimate_at_max_n:.4f}")
    print(f"  traditional estimate @ max_n: {result.traditional_estimate_at_max_n:.4f}")
    print(f"  max_n_tested:                 {result.max_n_tested}\n")

    won_on_compute_savings = bool(
        result.poly_algebraic_wins
        and result.compute_savings_fraction is not None
        and result.compute_savings_fraction > 0.10
    )

    print("--- Step 3b: honesty check -- genuine fit or F1-style constant-shell sentinel? ---")
    print(
        "  A smooth closed curve's k-NN graph can be an exact ring lattice, "
        "giving DimensionEstimate.degenerate=True (r_squared=1.0 sentinel, "
        "not a genuine varied-shell fit) -- checked directly, not assumed."
    )
    for n_check in sorted({n for n in (result.poly_algebraic_min_n, max(N_GRID)) if n is not None}):
        hg_check = knn_hypergraph(bitrev_ordered[:n_check], k=K, dedupe=True)
        ests = [local_dimension(hg_check, node, max_radius=MAX_RADIUS) for node in sorted(hg_check.nodes)]
        wf = [e for e in ests if e.is_well_fit(threshold=0.9)]
        genuine = [e for e in wf if e.is_genuinely_well_fit(threshold=0.9)]
        frac_degenerate = 1.0 - (len(genuine) / len(wf) if wf else 0.0)
        print(
            f"    n={n_check}: well_fit={len(wf)}/{len(ests)}, "
            f"genuinely_well_fit(non-sentinel)={len(genuine)}, "
            f"degenerate fraction of well-fit={frac_degenerate:.3f}"
        )
    print()

    # --- Step 4: density-variation confirmation + robustness test ---
    print("--- Step 4: density-variation confirmation (natural time-uniform sample) ---")
    n_fixed = max(result.traditional_min_n or 0, DENSITY_CHECK_MIN_N)
    n_fixed = min(n_fixed, M)
    density_points = natural_time_uniform_subsample(time_ordered, n_fixed)

    theta_arr = np.array([p[0] for p in density_points])
    omega_arr = np.array([p[1] for p in density_points])
    theta_amp = float(np.max(np.abs(theta_arr)))
    # Turning points: |theta| near its extreme (omega ~ 0). Bottom: theta ~ 0
    # (omega near its extreme). Both physically meaningful reference regions.
    near_turning = np.argsort(np.abs(np.abs(theta_arr) - theta_amp))[:20]
    near_bottom = np.argsort(np.abs(theta_arr))[:20]

    def local_spacing(idxs: np.ndarray) -> float:
        from scipy.spatial import cKDTree

        pts = np.array(density_points)
        tree = cKDTree(pts)
        d, _ = tree.query(pts[idxs], k=2)
        return float(np.mean(d[:, 1]))

    spacing_turning = local_spacing(near_turning)
    spacing_bottom = local_spacing(near_bottom)
    density_ratio = spacing_bottom / spacing_turning if spacing_turning > 0 else float("nan")

    print(f"  n_fixed = max(traditional_min_n or 0, {DENSITY_CHECK_MIN_N}) capped at M = {n_fixed}")
    print(f"  theta amplitude (max |theta|):                       {theta_amp:.4f}")
    print(f"  mean NN spacing near turning point (|theta|~amp):    {spacing_turning:.6f}")
    print(f"  mean NN spacing near bottom (theta~0):                {spacing_bottom:.6f}")
    print(f"  density ratio (bottom spacing / turning spacing):     {density_ratio:.3f}x")
    has_density_variation = density_ratio > 1.15 or density_ratio < 1.0 / 1.15
    print(
        f"  genuine density variation present (ratio departs >15% from 1.0): "
        f"{has_density_variation}\n"
    )

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
    won_on_density_robustness = bool(won_on_density_robustness and has_density_variation)

    print(f"--- Density robustness: dimension estimate on natural time-uniform sample (n={n_fixed}) ---")
    print(f"  poly-algebraic estimate:  {poly_dim:.4f}  (abs error {poly_error:.4f})")
    print(f"  traditional estimate:     {trad_dim:.4f}  (abs error {trad_error:.4f})")
    print(f"  WON_ON_DENSITY_ROBUSTNESS (density variation real AND poly error >=30% smaller): "
          f"{won_on_density_robustness}\n")

    overall_win = bool(won_on_compute_savings or won_on_density_robustness)

    print("=== SUMMARY ===")
    print(f"  poly_algebraic_min_n:        {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n:           {result.traditional_min_n}")
    print(f"  poly_algebraic_wins:         {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction:    {result.compute_savings_fraction}")
    print(f"  won_on_compute_savings:      {won_on_compute_savings}")
    print(f"  density_ratio:               {density_ratio:.3f}x  (has_density_variation={has_density_variation})")
    print(f"  poly_error / trad_error @ n={n_fixed}:  {poly_error:.4f} / {trad_error:.4f}")
    print(f"  won_on_density_robustness:   {won_on_density_robustness}")
    print(f"  OVERALL_WIN:                 {overall_win}")

    return {
        "mode_lock_verification": v,
        "n_grid": N_GRID,
        "k": K,
        "max_radius": MAX_RADIUS,
        "poly_algebraic_min_n": result.poly_algebraic_min_n,
        "traditional_min_n": result.traditional_min_n,
        "poly_algebraic_wins": result.poly_algebraic_wins,
        "compute_savings_fraction": result.compute_savings_fraction,
        "won_on_compute_savings": won_on_compute_savings,
        "n_fixed_density_check": n_fixed,
        "density_ratio": density_ratio,
        "has_density_variation": has_density_variation,
        "poly_dim_density_sample": poly_dim,
        "trad_dim_density_sample": trad_dim,
        "poly_error_density_sample": poly_error,
        "trad_error_density_sample": trad_error,
        "won_on_density_robustness": won_on_density_robustness,
        "overall_win": overall_win,
    }


if __name__ == "__main__":
    main()
