"""Benchmark: Kepler two-body orbit (e=0.6) -- poly-algebraic dimension vs traditional dimension.

KNOWN/TRADITIONAL DIMENSION: 1 (exact). A bound Kepler orbit is a closed
ellipse -- a smooth closed curve. Its topological / correlation dimension is
exactly 1, regardless of eccentricity.

METHOD:
  1. Reproduce level3_kepler exactly (scripts/solver_ladder.py), with the same
     GM=1, e=0.6, perihelion initial conditions, and the same leapfrog (KDK)
     integrator from socrates.solvers.
  2. Verify the solver the same way level 3 does: energy drift, angular
     momentum drift (KDK is exact for a central force, so this should be at
     the level of floating-point roundoff), and the radial period measured
     from perihelion-return times vs the exact Keplerian period 2*pi
     (GM=a=1).
  3. Sample the (x, y) ORBITAL POSITION (not the full 4D phase space) over
     >= 3 full periods, subsampled to a few hundred well-spaced points. The
     physical claim under test is "the orbit is a 1D curve" -- a curve in
     the 2-D orbital plane -- and (x, y) position captures that directly;
     including (vx, vy) would test a different (true) but different claim
     about the invariant torus/curve in 4D phase space.
  4. Build a k-NN proximity hypergraph over the (x, y) points via
     socrates.hypergraph.pointcloud.knn_hypergraph, and estimate dimension
     with socrates.hypergraph.dimension.local_dimension / mean_dimension,
     exactly as calibrated in tests/test_hypergraph_pointcloud.py's circle
     test (a closed curve is the direct analogue of this problem).

Does NOT modify src/socrates/hypergraph/ or src/socrates/solvers/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from socrates.hypergraph.dimension import local_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

GM = 1.0
ECCENTRICITY = 0.6
TRADITIONAL_DIMENSION = 1.0
TOLERANCE = 0.15


def force(q: np.ndarray) -> np.ndarray:
    r = float(np.linalg.norm(q))
    return -GM * q / r**3


def run_kepler(n_periods: float, dt: float = 5e-4):
    r_peri = 1.0 - ECCENTRICITY
    v_peri = np.sqrt(GM * (1 + ECCENTRICITY) / r_peri)
    n = int(n_periods * 2 * np.pi / dt)
    run = leapfrog(force, [r_peri, 0.0], [0.0, v_peri], dt=dt, n_steps=n)
    return run


def verify_solver(run) -> dict[str, object]:
    """Exactly the level-3 checks from scripts/solver_ladder.py: energy drift,
    angular momentum drift (KDK exact for central force), and radial period
    vs the exact Keplerian value 2*pi (GM=a=1)."""
    drift = run.energy_drift(lambda q: -GM / float(np.linalg.norm(q)))
    angular = (
        run.positions[:, 0] * run.velocities[:, 1] - run.positions[:, 1] * run.velocities[:, 0]
    )
    l_drift = float(np.max(np.abs(angular - angular[0])) / abs(angular[0]))

    radius = np.linalg.norm(run.positions, axis=1)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] < radius[2:])) + 1
    t_minima = []
    dt = float(run.times[1] - run.times[0])
    for i in minima:
        y0, y1, y2 = radius[i - 1], radius[i], radius[i + 1]
        t_minima.append(run.times[i] + dt * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))
    measured_period = float(np.mean(np.diff(t_minima))) if len(t_minima) > 2 else float("nan")
    period_err = abs(measured_period - 2 * np.pi) / (2 * np.pi)

    passed = drift < 1e-5 and l_drift < 1e-9 and period_err < 1e-5
    return {
        "energy_drift": drift,
        "angular_momentum_drift": l_drift,
        "measured_period": measured_period,
        "exact_period": 2 * np.pi,
        "period_relative_error": period_err,
        "n_perihelion_returns_used": len(t_minima),
        "gate": "E drift<1e-5, L drift<1e-9 (KDK exact for central force), T err<1e-5",
        "passed": bool(passed),
    }


def main() -> None:
    n_periods = 6.0  # well over the required >= 3 full periods
    dt = 5e-4  # same resolution as level3_kepler (resolves fast perihelion passage)
    run = run_kepler(n_periods, dt=dt)

    solver_check = verify_solver(run)
    print("=== Solver verification (same checks as solver_ladder.py level 3) ===")
    for k, v in solver_check.items():
        print(f"  {k}: {v}")

    solver_ok = solver_check["passed"]

    # --- Build point cloud: (x, y) orbital position over >=3 periods -----
    # Restrict to the first 3 periods' worth of *time* to keep sampling
    # uniform in time (and hence reasonably uniform in arc-length coverage
    # across repeated passes), then subsample to a few hundred well-spaced
    # points spanning that window.
    period = 2 * np.pi
    t_end_idx = int(3.0 * period / dt)
    positions = run.positions[: t_end_idx + 1]

    n_target_points = 360
    stride = max(1, len(positions) // n_target_points)
    sampled = positions[::stride][:n_target_points]
    points = [(float(p[0]), float(p[1])) for p in sampled]

    print(f"\n=== Point cloud ===")
    print(f"  total leapfrog steps in 3-period window: {len(positions)}")
    print(f"  sampled points (x, y): {len(points)}  (stride={stride})")

    k = 6  # matches the circle-point-cloud calibration test (closed 1D curve)
    hg = knn_hypergraph(points, k=k)

    max_radius = 5
    min_radius = 1
    nodes = sorted(hg.nodes)
    # Sample a spread of nodes across the ellipse rather than every node,
    # matching mean_dimension's own sampling strategy, but call
    # local_dimension directly so we can also report per-node r_squared.
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

    # Also report over ALL sampled nodes (unfiltered) for transparency.
    all_dims = [e.dimension for e in estimates]
    all_r2 = [e.r_squared for e in estimates]
    print(f"  mean dimension (all sampled, unfiltered): {np.mean(all_dims):.4f}")
    print(f"  mean r_squared (all sampled, unfiltered): {np.mean(all_r2):.4f}")

    abs_err = abs(mean_dim - TRADITIONAL_DIMENSION)
    dim_passed = abs_err <= TOLERANCE
    overall_passed = bool(dim_passed and solver_ok)

    print(f"\n=== Result ===")
    print(f"  traditional dimension: {TRADITIONAL_DIMENSION}")
    print(f"  poly-algebraic dimension: {mean_dim:.4f}")
    print(f"  absolute error: {abs_err:.4f}  (tolerance {TOLERANCE})")
    print(f"  solver verified: {solver_ok}")
    print(f"  PASSED: {overall_passed}")


if __name__ == "__main__":
    main()
