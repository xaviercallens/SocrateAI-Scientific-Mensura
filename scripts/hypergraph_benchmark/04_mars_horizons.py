"""Poly-Algebraic Calculus dimension benchmark: Mars vs JPL Horizons (real ephemeris).

Category: exact_periodic (per the task brief -- a 182-day arc of a periodic
orbit is a smooth curve segment; its topological/correlation dimension is
exactly 1 regardless of the perturbations that make the *orbit itself*
non-closed on this timescale).

KNOWN/TRADITIONAL DIMENSION: 1 (tolerance +/- 0.2)
SOURCE: a 182-day arc of the real (perturbed) Mars orbit is a smooth curve
segment in 3D space -- topological/correlation dimension exactly 1
regardless of the perturbations from Jupiter etc. that solver_ladder.py
level 4 already documents as the source of its ~3.6e-5 residual.

This script:
  1. Loads the SAME cached real JPL Horizons ephemeris data used by
     scripts/solver_ladder.py level 4 (data/horizons_mars_2025.npz) --
     does NOT refetch.
  2. Re-derives the level-4 two-body leapfrog solver run from the same
     initial conditions, and checks its residual against the real ephemeris
     to verify the solver is correct (matching the ~5e-3 relative-RMS gate
     level 4 itself uses) BEFORE trusting anything about the simulated arc.
  3. Builds knn_hypergraph + local_dimension/mean_dimension on the position
     point cloud for BOTH the real ephemeris arc (primary result) and the
     simulated two-body arc (secondary, reported in caveats), following the
     module's own calibration discipline: restrict to *interior* points of
     the arc (a curve has "boundary" at its two endpoints, exactly as a
     filled cube has boundary at its faces/corners in the module's own
     calibration tests in tests/test_hypergraph_pointcloud.py) to avoid the
     known underestimation effect at open-curve endpoints.

Do not modify src/socrates/hypergraph/ or src/socrates/solvers/ -- shared
with other agents running in parallel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from socrates.hypergraph.dimension import local_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE = DATA_DIR / "horizons_mars_2025.npz"

GM_SUN = 2.9591220828559093e-4  # AU^3 / day^2 (IAU), same constant as solver_ladder.py level 4


def force(q: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(q))
    return -GM_SUN * q / r**3


def verify_solver(r_true: np.ndarray, v0: np.ndarray, t_days: np.ndarray) -> dict:
    """Reproduce solver_ladder.py level 4 exactly: integrate two-body Sun-Mars
    from the real initial conditions and compare to the real Horizons arc.

    This is the "confirm your own solver is correct before trusting its
    output" step required by the standing rules -- for a periodic/orbital
    system, checked via residual against independently-sourced ephemeris
    data (a stronger check than internal energy conservation alone, and the
    one the repository's own level-4 gate already uses).
    """
    substeps = 20  # dt = 0.05 day, identical to solver_ladder.py level 4
    run = leapfrog(force, r_true[0], v0, dt=1.0 / substeps, n_steps=(len(t_days) - 1) * substeps)
    r_model = run.positions[::substeps]

    rel_rms = float(
        np.sqrt(np.mean(np.sum((r_model - r_true) ** 2, axis=1)))
        / np.mean(np.linalg.norm(r_true, axis=1))
    )
    passed = rel_rms < 5e-3  # same gate as solver_ladder.py level 4
    return {
        "r_model": r_model,
        "relative_rms_vs_ephemeris": rel_rms,
        "gate": "rel RMS < 5e-3 (residual = real planetary perturbations, per level 4 docstring)",
        "passed": passed,
    }


def interior_indices(n: int, margin: int) -> list[int]:
    """Indices with `margin` points of arc on both sides -- the curve analogue
    of restricting to interior points of a filled cube in the module's own
    calibration tests (endpoints of an open arc have systematically fewer
    true neighbours in one direction than interior points do)."""
    return list(range(margin, n - margin))


def measure_arc_dimension(points: list[tuple[float, float, float]], *, k: int, max_radius: int,
                           min_radius: int, margin: int) -> dict:
    n = len(points)
    hg = knn_hypergraph(points, k=k)
    interior = interior_indices(n, margin)

    estimates = [local_dimension(hg, i, max_radius=max_radius, min_radius=min_radius)
                 for i in interior]
    well_fit = [e for e in estimates if e.is_well_fit(threshold=0.9)]

    if well_fit:
        mean_dim = sum(e.dimension for e in well_fit) / len(well_fit)
        mean_r2 = sum(e.r_squared for e in well_fit) / len(well_fit)
    else:
        mean_dim = float("nan")
        mean_r2 = float("nan")

    return {
        "n_points": n,
        "n_interior": len(interior),
        "k": k,
        "max_radius": max_radius,
        "min_radius": min_radius,
        "n_well_fit": len(well_fit),
        "n_estimates": len(estimates),
        "mean_dimension": mean_dim,
        "mean_r_squared": mean_r2,
        "per_point_dimensions": [round(e.dimension, 4) for e in estimates],
        "per_point_r_squared": [round(e.r_squared, 4) for e in estimates],
    }


def main() -> None:
    assert CACHE.exists(), f"expected cached Horizons data at {CACHE}, do not refetch"
    data = np.load(CACHE)
    r_true, v0, t_days = data["r"], data["v0"], data["t"]
    n = len(t_days)
    print(f"Loaded cached Horizons data: {n} epochs over {t_days[-1] - t_days[0]:.0f} days")

    # --- Step 1: sanity-check the real data shape (a smooth arc, not noise) ---
    step_dists = np.linalg.norm(np.diff(r_true, axis=0), axis=1)
    print(f"Real ephemeris: consecutive-step distances "
          f"min={step_dists.min():.5f} max={step_dists.max():.5f} AU/day "
          f"(smooth, no jumps -> genuine continuous arc)")
    assert step_dists.max() / step_dists.min() < 3.0, "unexpectedly non-smooth arc in real data"

    # --- Step 2: verify the solver (standing rule 1) ---
    verification = verify_solver(r_true, v0, t_days)
    print(f"\nSolver verification (two-body leapfrog vs real Horizons ephemeris):")
    print(f"  relative RMS = {verification['relative_rms_vs_ephemeris']:.3e}")
    print(f"  gate: {verification['gate']}")
    print(f"  passed: {verification['passed']}")
    assert verification["passed"], (
        "Solver verification failed -- would not trust a dimension match from an "
        "unverified/wrong solver."
    )

    r_model = verification["r_model"]

    # --- Step 3: point-cloud dimension estimate, real ephemeris (PRIMARY) ---
    real_points = [tuple(p) for p in r_true]
    # k=6 as in the module's own circle calibration test (a curve/1-manifold
    # target); margin trims a few points off each end of the 182-point arc to
    # avoid the open-endpoint underestimation effect documented in
    # tests/test_hypergraph_pointcloud.py for the filled-cube boundary case.
    k = 6
    margin = 10
    max_radius = 5
    min_radius = 1

    real_result = measure_arc_dimension(
        real_points, k=k, max_radius=max_radius, min_radius=min_radius, margin=margin
    )
    print(f"\n=== REAL Horizons ephemeris arc (PRIMARY) ===")
    print(f"  n_points={real_result['n_points']}, n_interior={real_result['n_interior']}, "
          f"k={k}, max_radius={max_radius}, min_radius={min_radius}")
    print(f"  well-fit estimates: {real_result['n_well_fit']}/{real_result['n_estimates']}")
    print(f"  mean_dimension = {real_result['mean_dimension']:.4f}")
    print(f"  mean_r_squared = {real_result['mean_r_squared']:.4f}")

    # --- Step 4: point-cloud dimension estimate, simulated two-body arc (SECONDARY) ---
    sim_points = [tuple(p) for p in r_model]
    sim_result = measure_arc_dimension(
        sim_points, k=k, max_radius=max_radius, min_radius=min_radius, margin=margin
    )
    print(f"\n=== SIMULATED two-body leapfrog arc (SECONDARY, for comparison) ===")
    print(f"  n_points={sim_result['n_points']}, n_interior={sim_result['n_interior']}, "
          f"k={k}, max_radius={max_radius}, min_radius={min_radius}")
    print(f"  well-fit estimates: {sim_result['n_well_fit']}/{sim_result['n_estimates']}")
    print(f"  mean_dimension = {sim_result['mean_dimension']:.4f}")
    print(f"  mean_r_squared = {sim_result['mean_r_squared']:.4f}")

    # --- Step 5: report against known answer ---
    traditional = 1.0
    tolerance = 0.2
    abs_err = abs(real_result["mean_dimension"] - traditional)
    passed = abs_err <= tolerance and verification["passed"]

    print(f"\n=== RESULT (primary = real ephemeris) ===")
    print(f"  traditional_dimension = {traditional}")
    print(f"  poly_algebraic_dimension (real ephemeris) = {real_result['mean_dimension']:.4f}")
    print(f"  poly_algebraic_dimension (simulated, for comparison) = "
          f"{sim_result['mean_dimension']:.4f}")
    print(f"  absolute_error = {abs_err:.4f}")
    print(f"  tolerance = {tolerance}")
    print(f"  PASSED = {passed}")


if __name__ == "__main__":
    main()
