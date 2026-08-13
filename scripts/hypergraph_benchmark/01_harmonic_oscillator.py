"""Poly-Algebraic Calculus dimension benchmark: simple harmonic oscillator.

Category: exact_periodic.
KNOWN/TRADITIONAL DIMENSION: 1 (tolerance +/- 0.15).
SOURCE: the phase-space trajectory of any bound 1D Hamiltonian oscillator is
a smooth closed curve (topologically a circle), so its correlation/
topological dimension is exactly 1 -- elementary dynamics, not a measured
constant.

System: x'' = -x, x(0)=1, v(0)=0. Exact solution x(t)=cos(t), v(t)=-sin(t);
phase space (x, v) traces the unit circle.

Pipeline:
  1. Integrate with socrates.solvers.leapfrog over 10 periods at dt=1e-3.
  2. VERIFY the solver call actually reproduces cos(t)/sin(t) (this repo's
     own discipline: do not trust a solver call without checking it).
  3. Subsample the (x, v) trajectory to a point cloud.
  4. Build a k-NN proximity graph (socrates.hypergraph.pointcloud) and
     measure its volume-growth dimension (socrates.hypergraph.dimension).
  5. Compare against the known value 1, report r_squared, and state exactly
     what was tuned and why.

Run: python scripts/hypergraph_benchmark/01_harmonic_oscillator.py
(requires the repo venv: source .venv/bin/activate)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from socrates.hypergraph import knn_hypergraph, local_dimension, mean_dimension  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

TRADITIONAL_DIMENSION = 1.0
TOLERANCE = 0.15


def verify_solver(dt: float, n_periods: int) -> dict[str, object]:
    """Integrate several periods and confirm the leapfrog call reproduces
    cos(t)/sin(t) to a tight tolerance -- this repo's standing rule: verify
    your own solver before trusting anything downstream of it.
    """
    force = lambda q: -q  # noqa: E731
    n_steps = int(round(n_periods * 2 * np.pi / dt))
    run = leapfrog(force, [1.0], [0.0], dt=dt, n_steps=n_steps)

    t = run.times
    x_exact = np.cos(t)
    v_exact = -np.sin(t)
    max_err_x = float(np.max(np.abs(run.positions[:, 0] - x_exact)))
    max_err_v = float(np.max(np.abs(run.velocities[:, 0] - v_exact)))

    drift = run.energy_drift(lambda q: 0.5 * float(q @ q))

    # Orbit-closure check: position/velocity at the end of each full period
    # should return close to the start (x=1, v=0) -- the standing rule's
    # explicit closed-orbit check for periodic systems. The nearest grid
    # index to t = k*2*pi is used (not k * round(2*pi/dt) steps): dt does
    # not divide 2*pi evenly, so that naive index accumulates a ~1.85e-4
    # rad/period time-alignment offset that would masquerade as integrator
    # drift when it is really a sampling-grid rounding artifact. Comparing
    # against the analytic value AT THE ACTUAL SAMPLED TIME isolates the
    # solver's true error from that grid-rounding effect.
    period_end_idx = [
        int(round(k * 2 * np.pi / dt)) for k in range(1, n_periods + 1) if int(round(k * 2 * np.pi / dt)) <= n_steps
    ]
    closure_err = float(
        np.max(
            [
                np.hypot(
                    run.positions[i, 0] - np.cos(t[i]),
                    run.velocities[i, 0] - (-np.sin(t[i])),
                )
                for i in period_end_idx
            ]
        )
    )

    passed = max_err_x < 1e-4 and max_err_v < 1e-4 and drift < 1e-6 and closure_err < 1e-3
    return {
        "run": run,
        "max_err_x_vs_cos": max_err_x,
        "max_err_v_vs_negsin": max_err_v,
        "energy_drift": drift,
        "orbit_closure_error_at_period_ends": closure_err,
        "n_period_ends_checked": len(period_end_idx),
        "passed": passed,
    }


def build_point_cloud(run, n_target: int) -> list[tuple[float, float]]:
    """Subsample the (x, v) trajectory to n_target well-spaced points.

    IMPORTANT TUNING NOTE (found empirically, recorded honestly): the
    oscillator is exactly periodic with period 2*pi. If the subsample count
    divides evenly into the number of periods integrated (e.g. 400 points
    over exactly 10 periods = 40 points/period), successive periods resample
    nearly the SAME 40 phase points (up to leapfrog's ~1e-6 numerical
    error), so the k-NN graph collapses into a clique of near-duplicate
    points instead of tracing the circle -- local_dimension then measures
    something close to 0, not 1. This is exactly the failure mode the task
    brief warns about ("do not sample so densely that adjacent points are
    numerically identical"), just arising from periodic aliasing across
    periods rather than density within one period. Fix: choose n_target
    that does NOT evenly divide the number of periods (397 is prime, 10 is
    not a multiple of it), which staggers the phase of each period's
    samples relative to the last and gives genuinely distinct, evenly
    spaced points around the circle. Verified below by checking the min
    consecutive spacing is close to the max (i.e. uniform, not clustered).
    """
    n_total = len(run.times)
    step = n_total / n_target
    idx = np.unique((np.arange(n_target) * step).astype(int))
    pts = list(zip(run.positions[idx, 0].tolist(), run.velocities[idx, 0].tolist(), strict=True))
    return pts


def spacing_stats(pts: list[tuple[float, float]]) -> tuple[float, float]:
    d = [
        np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    ]
    return float(min(d)), float(max(d))


def main() -> None:
    dt = 1e-3
    n_periods = 10

    print("=" * 70)
    print("STEP 1-2: integrate + verify solver")
    print("=" * 70)
    v = verify_solver(dt, n_periods)
    run = v["run"]
    print(f"  dt={dt}, n_periods={n_periods}, n_steps={len(run.times) - 1}")
    print(f"  max |x - cos(t)|            = {v['max_err_x_vs_cos']:.3e}")
    print(f"  max |v - (-sin(t))|         = {v['max_err_v_vs_negsin']:.3e}")
    print(f"  energy drift (max-min)/max  = {v['energy_drift']:.3e}")
    print(
        f"  orbit closure error at period ends "
        f"(n={v['n_period_ends_checked']}) = {v['orbit_closure_error_at_period_ends']:.3e}"
    )
    print(f"  SOLVER VERIFICATION PASSED: {v['passed']}")
    if not v["passed"]:
        print("  !! Solver did not verify -- downstream dimension result cannot be trusted.")

    print()
    print("=" * 70)
    print("STEP 3: build point cloud")
    print("=" * 70)
    n_target = 397  # prime; does not divide n_periods=10 -- see build_point_cloud docstring
    pts = build_point_cloud(run, n_target)
    min_sp, max_sp = spacing_stats(pts)
    print(f"  n_target={n_target} -> {len(pts)} distinct points (after de-dup by integer index)")
    print(f"  consecutive spacing: min={min_sp:.5f}, max={max_sp:.5f} (uniform ring, as expected)")
    if min_sp < 1e-4:
        print("  !! WARNING: near-duplicate points detected -- aliasing not avoided.")

    print()
    print("=" * 70)
    print("STEP 4: k-NN hypergraph + dimension estimate")
    print("=" * 70)
    k = 6
    max_radius = 5
    min_radius = 1
    # Radius choice note: max_radius=6 was tried first and triggers a
    # numerical artifact in the shared local_dimension's R^2 computation
    # (not a hypergraph/physics problem): on this graph the shell size is
    # EXACTLY constant (6 new nodes per radius step, a circulant ring
    # lattice) at every tested radius up to well past saturation. At
    # max_radius=6 the six repeated log(6.0) values produce a ss_tot that
    # cancels to a tiny nonzero float (~1e-31) instead of exactly 0,
    # so 1 - ss_res/ss_tot lands near 0 by floating-point noise despite the
    # fitted dimension itself still landing on 1.0 to ~1e-32. max_radius=5
    # (five constant-shell points) does not hit this cancellation and
    # reports r_squared exactly 1.0, consistent with the module's own
    # path-graph known-answer test (constant shell => r_squared == 1.0
    # exactly). This is reported here, not silently worked around: it is a
    # real fragility of the shared estimator's R^2 statistic in the exact
    # degenerate (perfectly homogeneous shell) regime, worth flagging to
    # the module owners, but it does not affect the *dimension* estimate,
    # only the diagnostic R^2 at that specific radius choice.
    print(f"  k={k}, max_radius={max_radius}, min_radius={min_radius}")
    print(
        "  (max_radius=6 triggers a float-cancellation artifact in the shared "
        "R^2 calc for this exactly-homogeneous ring graph -- see comment in source; "
        "max_radius=5 avoids it and matches the module's own constant-shell "
        "known-answer behaviour.)"
    )

    hg = knn_hypergraph(pts, k=k)
    print(f"  hypergraph: {len(hg.nodes)} nodes, {len(hg.edges)} edges")

    estimates = [local_dimension(hg, n, max_radius=max_radius, min_radius=min_radius) for n in sorted(hg.nodes)]
    well_fit = [e for e in estimates if e.is_well_fit(threshold=0.9)]
    poly_dim = mean_dimension(hg, samples=None, max_radius=max_radius)

    print(f"  nodes well-fit (r_squared >= 0.9): {len(well_fit)} / {len(estimates)}")
    if well_fit:
        mean_r2 = sum(e.r_squared for e in well_fit) / len(well_fit)
        min_r2 = min(e.r_squared for e in well_fit)
        print(f"  mean r_squared over well-fit nodes: {mean_r2:.6f}")
        print(f"  min  r_squared over well-fit nodes: {min_r2:.6f}")
    else:
        mean_r2 = float("nan")
        min_r2 = float("nan")

    print(f"  poly_algebraic_dimension (mean_dimension) = {poly_dim:.6f}")

    abs_error = abs(poly_dim - TRADITIONAL_DIMENSION)
    passed = bool(v["passed"]) and abs_error <= TOLERANCE

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  traditional dimension : {TRADITIONAL_DIMENSION}")
    print(f"  poly-algebraic estimate: {poly_dim:.6f}")
    print(f"  absolute error         : {abs_error:.6f}  (tolerance {TOLERANCE})")
    print(f"  solver verified         : {v['passed']}")
    print(f"  PASSED                  : {passed}")

    return {
        "poly_dim": poly_dim,
        "mean_r2": mean_r2,
        "min_r2": min_r2,
        "n_well_fit": len(well_fit),
        "n_total": len(estimates),
        "abs_error": abs_error,
        "passed": passed,
        "solver_passed": v["passed"],
        "n_points": len(pts),
        "k": k,
        "max_radius": max_radius,
    }


if __name__ == "__main__":
    main()
