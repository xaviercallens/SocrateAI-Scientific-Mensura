"""Round 2 -- Poly-Algebraic vs traditional, apples-to-apples: Mars vs real JPL Horizons.

KNOWN/TRADITIONAL DIMENSION: 1 (tolerance +/- 0.2). A single-period arc of
Mars's orbit is a smooth curve segment in 3D space -- topological/correlation
dimension exactly 1.

Per the assignment brief for this problem: "no natural density variation ...
report density_robustness_tested=false" -- Mars's orbit (e ~ 0.093) is close
enough to circular that a uniform-in-time sample does not produce the kind of
close-approach/turning-point density spike the density-robustness test is
designed to catch (see 07_restricted_three_body.py / a Kepler-near-perihelion
problem for where that test IS applicable). This script does not attempt to
manufacture one.

WHY THIS SCRIPT GENERATES ITS OWN POINT CLOUD RATHER THAN REUSING THE 182
CACHED EPHEMERIS POINTS DIRECTLY:
comparison.compare() needs an n_grid with real headroom for the traditional
correlation-sum method to have a fair shot at converging (standing rule 2) --
that means thousands of points, not 182. The cached real Horizons ephemeris
(data/horizons_mars_2025.npz) only has 182 daily epochs (182 of Mars's ~687
day period), so it is used here ONLY to (a) supply real initial conditions
and (b) VERIFY the two-body leapfrog solver against genuine external data
(the same ~5e-3 relative-RMS gate scripts/solver_ladder.py level 4 and round
1's 04_mars_horizons.py use) before trusting a single point that solver
produces. Once verified, the *solver* (not a refetch -- CACHE is still the
only data file touched) generates a dense, single-period point cloud for the
actual n_grid comparison. This is exactly what the round-2 task brief asks
for: "regenerate points with any needed fixes, e.g. single-period only".

SINGLE-PERIOD, NOT MULTI-PERIOD (finding N7/F3, standing rule 4-iii):
points are sampled at n_max evenly-spaced times over the half-open interval
[0, T) where T is the orbital period computed from the verified vis-viva
energy of the real initial condition -- NOT [0, T] -- so the last sample
does not nearly coincide with the first (which would otherwise reproduce
the near-duplicate-point corruption the dedupe fix exists for). n_grid
prefixes of this array are then genuinely-nested shorter arcs at the same
time resolution, matching the convention already used in
tests/test_hypergraph_comparison.py's circle tests.

Do not modify src/socrates/hypergraph/ or src/socrates/solvers/, and do not
modify scripts/hypergraph_benchmark/04_mars_horizons.py (round 1, historical
record) -- both are shared with other agents running in parallel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from socrates.hypergraph.comparison import compare  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE = DATA_DIR / "horizons_mars_2025.npz"

GM_SUN = 2.9591220828559093e-4  # AU^3 / day^2 (IAU), same constant as round 1 / solver_ladder.py level 4


def force(q: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(q))
    return -GM_SUN * q / r**3


def verify_solver_against_real_ephemeris(r_true: np.ndarray, v0: np.ndarray, t_days: np.ndarray) -> dict:
    """Same check as round 1: integrate from the real IC and compare to the
    real Horizons arc over the 182 cached days. Standing rule 1 -- verify
    before trusting the solver used to generate the dense point cloud below."""
    substeps = 20  # dt = 0.05 day, identical to round 1 / solver_ladder.py level 4
    run = leapfrog(force, r_true[0], v0, dt=1.0 / substeps, n_steps=(len(t_days) - 1) * substeps)
    r_model = run.positions[::substeps]

    rel_rms = float(
        np.sqrt(np.mean(np.sum((r_model - r_true) ** 2, axis=1)))
        / np.mean(np.linalg.norm(r_true, axis=1))
    )
    passed = rel_rms < 5e-3  # same gate as round 1 / solver_ladder.py level 4
    return {"relative_rms_vs_ephemeris": rel_rms, "passed": passed}


def orbital_period_days(r0: np.ndarray, v0: np.ndarray) -> float:
    """Period from the vis-viva energy of the real initial condition (Kepler's
    third law) -- an independent cross-check, not an assumed textbook value."""
    r0n = float(np.linalg.norm(r0))
    v0n = float(np.linalg.norm(v0))
    energy = v0n**2 / 2 - GM_SUN / r0n
    a = -GM_SUN / (2 * energy)
    return float(2 * np.pi * np.sqrt(a**3 / GM_SUN))


def generate_single_period_cloud(r0: np.ndarray, v0: np.ndarray, period: float, n_max: int,
                                  substeps: int = 20) -> list[tuple[float, float, float]]:
    """n_max points evenly spaced over the half-open interval [0, period) of
    the verified two-body solver, starting from the real Horizons IC."""
    dt_sample = period / n_max
    dt = dt_sample / substeps
    n_steps = n_max * substeps
    run = leapfrog(force, r0, v0, dt=dt, n_steps=n_steps)
    sampled = run.positions[::substeps]  # n_max + 1 points: t=0 .. t=period (inclusive)
    sampled = sampled[:n_max]  # drop the t=period point -- near-duplicate of t=0, single period only
    assert len(sampled) == n_max
    return [tuple(p) for p in sampled]


def main() -> None:
    assert CACHE.exists(), f"expected cached Horizons data at {CACHE}, do not refetch"
    data = np.load(CACHE)
    r_true, v0, t_days = data["r"], data["v0"], data["t"]
    print(f"Loaded cached Horizons data: {len(t_days)} epochs over {t_days[-1] - t_days[0]:.0f} days "
          f"(used for IC + solver verification only)")

    # --- Step 1: verify the solver against real external ephemeris data (standing rule 1) ---
    verification = verify_solver_against_real_ephemeris(r_true, v0, t_days)
    print(f"\nSolver verification vs real Horizons ephemeris (182-day arc):")
    print(f"  relative RMS = {verification['relative_rms_vs_ephemeris']:.3e}  (gate: < 5e-3)")
    print(f"  passed: {verification['passed']}")
    assert verification["passed"], (
        "Solver verification failed -- refusing to trust a dense point cloud from an "
        "unverified solver."
    )

    period = orbital_period_days(r_true[0], v0)
    print(f"\nOrbital period from vis-viva energy of the real IC: {period:.3f} days "
          f"({period / 365.25:.4f} yr) -- matches Mars's known ~686.98 day period")
    assert abs(period - 687.0) < 5.0, "unexpected period -- sanity check on the real IC failed"

    # --- Step 2: generate the dense, single-period point cloud from the verified solver ---
    n_grid = (100, 200, 400, 800, 1600, 3200, 6400)
    k = 6  # k=6 for a 1-manifold target, matching the module's own circle calibration test
    max_radius = 6  # compare()'s own default
    tolerance = 0.2
    true_dimension = 1.0

    n_max = max(n_grid)
    points = generate_single_period_cloud(r_true[0], v0, period, n_max)
    print(f"\nGenerated {len(points)} points over one full period (single period, t in [0, {period:.2f}) days), "
          f"dt_sample = {period / n_max:.5f} days")

    # sanity: consecutive-step distances should be smooth (no jumps), and no
    # near-duplicate clusters from wraparound (single period, half-open interval)
    arr = np.asarray(points)
    step_dists = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    wrap_dist = np.linalg.norm(arr[-1] - arr[0])
    print(f"  step distances: min={step_dists.min():.6f} max={step_dists.max():.6f} AU "
          f"(ratio {step_dists.max() / step_dists.min():.2f})")
    print(f"  wraparound gap (last -> first, should be ~1 step, not ~0): {wrap_dist:.6f} AU")
    assert wrap_dist > 0.5 * step_dists.min(), "unexpected near-duplicate at period wraparound"

    # --- Step 3: apples-to-apples comparison via the shared harness ---
    print(f"\nRunning compare(k={k}, n_grid={n_grid}, max_radius={max_radius}, "
          f"tolerance={tolerance}) ...")
    result = compare(points, true_dimension=true_dimension, tolerance=tolerance,
                      k=k, n_grid=n_grid, max_radius=max_radius)

    print(f"\n=== RESULT ===")
    print(f"  poly_algebraic_min_n = {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n = {result.traditional_min_n}")
    print(f"  poly_algebraic_estimate_at_max_n ({result.max_n_tested}) = "
          f"{result.poly_algebraic_estimate_at_max_n:.4f}")
    print(f"  traditional_estimate_at_max_n ({result.max_n_tested}) = "
          f"{result.traditional_estimate_at_max_n:.4f}")
    print(f"  poly_algebraic_wins = {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction = {result.compute_savings_fraction}")

    # --- Step 4: independent re-derivation of the min_n crossings (standing rule 1) ---
    # Re-run poly-algebraic and traditional dimension estimates directly at
    # each n_grid point (bypassing compare()'s internal loop) to confirm the
    # reported min_n values are not a fluke of the harness's own bookkeeping.
    from socrates.hypergraph.baseline import correlation_dimension
    from socrates.hypergraph.dimension import mean_dimension
    from socrates.hypergraph.pointcloud import knn_hypergraph

    print(f"\n=== Independent re-derivation, per n in n_grid ===")
    for n in n_grid:
        if n > len(points):
            break
        subset = points[:n]
        hg = knn_hypergraph(subset, k=k, dedupe=True)
        poly_dim = mean_dimension(hg, samples=min(n, 40), max_radius=max_radius)
        trad_dim = correlation_dimension(subset).dimension
        print(f"  n={n:5d}: poly={poly_dim:.4f} (err={abs(poly_dim - true_dimension):.4f})   "
              f"trad={trad_dim:.4f} (err={abs(trad_dim - true_dimension):.4f})")

    # --- Step 5: F1 check (standing rule 4-i) -- is poly's near-perfect score genuine? ---
    # A smooth curve sampled near-uniformly in time forms an almost-exact
    # circulant ring lattice in its k-NN graph -- interior points get a
    # CONSTANT shell sequence, which local_dimension flags as .degenerate
    # (an r_squared=1.0 sentinel, not a measurement). mean_dimension()'s own
    # docstring warns callers must check this themselves before citing
    # accuracy -- so check it here rather than taking "1.0000, converged at
    # n=100" at face value.
    from socrates.hypergraph.dimension import local_dimension

    print(f"\n=== F1 check: is the poly-algebraic 'perfect' fit genuine or the degenerate sentinel? ===")
    for n in (100, 3200):
        subset = points[:n]
        hg = knn_hypergraph(subset, k=k, dedupe=True)
        nodes = sorted(hg.nodes)
        step = max(1, len(nodes) // 40)
        sample_nodes = nodes[::step][:40]
        ests = [local_dimension(hg, nd, max_radius=max_radius) for nd in sample_nodes]
        n_well_fit = sum(1 for e in ests if e.is_well_fit(0.9))
        n_degenerate = sum(1 for e in ests if e.degenerate)
        n_genuine = sum(1 for e in ests if e.is_genuinely_well_fit())
        print(f"  n={n}: of {len(ests)} sampled points, {n_well_fit} pass is_well_fit(0.9), "
              f"of which {n_degenerate} are the degenerate constant-shell sentinel and only "
              f"{n_genuine} are genuinely_well_fit().")

    print(f"\ndensity_robustness_tested = False")
    print(f"density_robustness_result = not applicable: no natural density variation in "
          f"this system's natural sampling (assigned by brief; Mars e~0.093 near-circular)")


if __name__ == "__main__":
    main()
