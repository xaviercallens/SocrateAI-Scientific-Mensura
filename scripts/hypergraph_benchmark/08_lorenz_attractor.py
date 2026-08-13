"""Benchmark 08: Lorenz attractor (chaotic) -- poly-algebraic dimension vs
the Grassberger-Procaccia correlation dimension.

KNOWN/TRADITIONAL DIMENSION: 2.05 +/- 0.01
SOURCE: Grassberger, P. & Procaccia, I. (1983), "Measuring the strangeness
of strange attractors", Physica D 9, 189-208. The classic paper that
introduced correlation-dimension estimation and measured D2 = 2.05 for the
Lorenz attractor at the canonical parameters sigma=10, rho=28, beta=8/3.

This is the hardest case in this benchmark suite: the Lorenz attractor is a
genuinely fractal object (its correlation dimension is non-integer), unlike
the closed curves / filled regions / diffusive paths used elsewhere. The
k-NN shell-growth estimator implemented in socrates.hypergraph.dimension was
calibrated on INTEGER-dimension test cases (circle -> 1, filled square -> 2,
filled cube interior -> 3); whether it can resolve a genuinely fractal
non-integer dimension with an honestly-reported r_squared is the actual
question this script answers -- a poor fit here would be informative about
the estimator's limits, not something to hide.

METHOD:
  1. Integrate the Lorenz system with a hand-written RK4 integrator (new
     small ODE system, sigma=10, rho=28, beta=8/3, IC=(1,1,1)).
  2. VERIFY the solver: discard an initial transient, then confirm the
     trajectory visits the neighbourhoods of BOTH non-trivial fixed points
     C+/C- = (+/-sqrt(beta*(rho-1)), +/-sqrt(beta*(rho-1)), rho-1) repeatedly
     (the defining "butterfly" signature -- winds around two lobes, does not
     settle to a fixed point or diverge).
  3. Subsample the post-transient attractor to a tractable point-cloud size
     for the O(n^2) knn_hypergraph construction, build the k-NN graph, and
     estimate dimension via local_dimension / mean_dimension exactly as
     calibrated in tests/test_hypergraph_pointcloud.py.
  4. Compare against 2.05 with a generous (+/- 0.5) tolerance, reporting
     r_squared honestly regardless of outcome.

Does NOT modify src/socrates/hypergraph/ or src/socrates/solvers/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from socrates.hypergraph.dimension import local_dimension, mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical Lorenz parameters (Lorenz 1963; Grassberger & Procaccia 1983).
# ---------------------------------------------------------------------------
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

TRADITIONAL_DIMENSION = 2.05
TOLERANCE = 0.5


def lorenz_rhs(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    return np.array([dx, dy, dz])


def rk4_integrate(state0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Hand-written classical RK4 integrator for the Lorenz ODE system."""
    states = np.empty((n_steps + 1, 3))
    states[0] = state0
    s = state0.copy()
    for i in range(n_steps):
        k1 = lorenz_rhs(s)
        k2 = lorenz_rhs(s + 0.5 * dt * k1)
        k3 = lorenz_rhs(s + 0.5 * dt * k2)
        k4 = lorenz_rhs(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states[i + 1] = s
    return states


def main() -> None:
    # -----------------------------------------------------------------
    # 1. Integrate.
    # -----------------------------------------------------------------
    dt = 0.005
    t_transient = 10.0      # discard this much initial time
    t_total = 100.0         # total integration time (after t=0)
    n_steps = int(t_total / dt)

    state0 = np.array([1.0, 1.0, 1.0])
    print(f"=== Integration ===")
    print(f"  sigma={SIGMA}, rho={RHO}, beta={BETA:.6f}, IC={tuple(state0)}, dt={dt}")
    print(f"  n_steps={n_steps}, t_total={t_total}")

    traj = rk4_integrate(state0, dt, n_steps)
    times = np.arange(n_steps + 1) * dt

    transient_idx = int(t_transient / dt)
    post = traj[transient_idx:]
    post_times = times[transient_idx:]
    print(f"  discarded first t<{t_transient} ({transient_idx} steps) as transient")
    print(f"  post-transient points: {len(post)}  (t in [{post_times[0]:.2f}, {post_times[-1]:.2f}])")

    # -----------------------------------------------------------------
    # 2. VERIFY the solver.
    #    (a) Basic sanity: bounded trajectory (no blow-up / NaN).
    #    (b) Butterfly signature: the two non-trivial fixed points are
    #        C+/- = (+/- sqrt(beta*(rho-1)), +/- sqrt(beta*(rho-1)), rho-1).
    #        Confirm the post-transient trajectory visits the neighbourhood
    #        of BOTH repeatedly (not settling on one, not diverging), by
    #        counting sign changes of x (a standard proxy for lobe-switching
    #        on the Lorenz attractor: x and y share sign near each fixed
    #        point, so a sign change in x indicates a lobe transition) and by
    #        checking closest approach to each fixed point.
    # -----------------------------------------------------------------
    finite_ok = bool(np.all(np.isfinite(post)))
    bounded_ok = bool(np.all(np.abs(post) < 100.0))  # generous bound; Lorenz attractor stays ~O(10-50)

    c_val = np.sqrt(BETA * (RHO - 1))
    fp_plus = np.array([c_val, c_val, RHO - 1])
    fp_minus = np.array([-c_val, -c_val, RHO - 1])
    print(f"\n=== Solver verification ===")
    print(f"  finite trajectory: {finite_ok}")
    print(f"  bounded (|coord| < 100): {bounded_ok}")
    print(f"  fixed points C+={tuple(np.round(fp_plus, 3))}, C-={tuple(np.round(fp_minus, 3))}")

    # NOTE on threshold: C+/C- are repelling fixed points on the Lorenz
    # attractor -- trajectories spiral AROUND them rather than passing
    # through, so "closest approach" is a spiral radius, not ~0. Empirically
    # (checked directly on this trajectory) the 1st-percentile closest
    # approach to each fixed point is ~4-5, well inside the ~24-unit
    # separation between C+ and C-, so a generous but non-trivial threshold
    # of 8.0 (less than half the inter-fixed-point distance) is used, rather
    # than an arbitrarily tight one that would fail on correct trajectories.
    dist_plus = np.linalg.norm(post - fp_plus, axis=1)
    dist_minus = np.linalg.norm(post - fp_minus, axis=1)
    min_dist_plus = float(np.min(dist_plus))
    min_dist_minus = float(np.min(dist_minus))
    interlobe_dist = float(np.linalg.norm(fp_plus - fp_minus))
    print(f"  inter-fixed-point distance |C+ - C-|: {interlobe_dist:.3f}")
    print(f"  closest approach to C+: {min_dist_plus:.3f}")
    print(f"  closest approach to C-: {min_dist_minus:.3f}")
    proximity_threshold = 0.35 * interlobe_dist  # ~8.4 here; well under half-separation
    visits_both_fp_neighbourhoods = (
        min_dist_plus < proximity_threshold and min_dist_minus < proximity_threshold
    )

    # sign changes in x as a lobe-switching proxy
    x = post[:, 0]
    sign_changes = int(np.sum(np.diff(np.sign(x)) != 0))
    frac_x_pos = float(np.mean(x > 0))
    frac_x_neg = float(np.mean(x < 0))
    print(f"  sign changes in x (lobe switches, proxy): {sign_changes}")
    print(f"  fraction of time x>0: {frac_x_pos:.3f}, x<0: {frac_x_neg:.3f} "
          f"(both well above 0 => genuinely visits both lobes, not just switching briefly)")
    switches_lobes = sign_changes >= 10  # should switch many times over t_total-t_transient=90 time units
    balanced_lobes = frac_x_pos > 0.25 and frac_x_neg > 0.25

    # not settled to a fixed point: check the trajectory keeps moving
    # (std of last 20% of the trajectory should be large, not ~0)
    tail = post[int(0.8 * len(post)):]
    tail_std = float(np.std(tail, axis=0).mean())
    print(f"  std of trajectory (last 20%, mean over x,y,z): {tail_std:.3f}")
    not_settled = tail_std > 1.0

    solver_ok = (
        finite_ok and bounded_ok and visits_both_fp_neighbourhoods
        and switches_lobes and balanced_lobes and not_settled
    )
    print(f"  BUTTERFLY CHECK -> visits both lobes: {visits_both_fp_neighbourhoods}, "
          f"switches repeatedly: {switches_lobes}, balanced time in both lobes: {balanced_lobes}, "
          f"not settled/diverged: {not_settled}")
    print(f"  Solver verification PASSED: {solver_ok}")

    # -----------------------------------------------------------------
    # 3. Point cloud: subsample the post-transient attractor.
    #    O(n^2) distance computation in knn_hypergraph rules out using all
    #    ~18000 post-transient points. We take TARGET_POINTS evenly spaced
    #    (by index, i.e. by time) samples across the FULL post-transient
    #    trajectory so both lobes and the full attractor geometry remain
    #    represented (not a truncated prefix, which could bias toward one
    #    lobe or a partial subset of the attractor's structure).
    # -----------------------------------------------------------------
    TARGET_POINTS = 800  # a few hundred to ~1000, per instructions; keeps
                          # knn_hypergraph's O(n^2) distance pass tractable
                          # (comparable to the 800-1300 point counts used in
                          # the other point-cloud benchmarks in this suite)
    stride = max(1, len(post) // TARGET_POINTS)
    sub = post[::stride][:TARGET_POINTS]
    points = [(float(p[0]), float(p[1]), float(p[2])) for p in sub]
    n_points = len(points)

    print(f"\n=== Point cloud ===")
    print(f"  post-transient trajectory points: {len(post)}")
    print(f"  target points: {TARGET_POINTS}, stride: {stride}, n_points used: {n_points}")

    # -----------------------------------------------------------------
    # 4. Build k-NN hypergraph and estimate dimension.
    #    k chosen larger than the integer-dimension calibration tests (which
    #    used k=6..20 for D=1..3) since D2~2.05 needs enough neighbours per
    #    shell to resolve a fractional log-log slope reliably; max_radius
    #    kept modest since the attractor's characteristic "thickness" limits
    #    how far a k-NN ball can grow before saturating/leaving the fractal
    #    scaling regime (the module's own docs recommend excluding
    #    boundary/saturated radii -- local_dimension does this automatically).
    # -----------------------------------------------------------------
    K = 15
    hg = knn_hypergraph(points, k=K)

    max_radius = 5
    min_radius = 1
    nodes = sorted(hg.nodes)
    n_samples = 60
    step = max(1, len(nodes) // n_samples)
    sample_nodes = nodes[::step][:n_samples]

    estimates = [
        local_dimension(hg, n, max_radius=max_radius, min_radius=min_radius) for n in sample_nodes
    ]
    well_fit = [e for e in estimates if e.is_well_fit(threshold=0.9)]

    print(f"\n=== Dimension estimation ===")
    print(f"  k (knn_hypergraph): {K}")
    print(f"  max_radius: {max_radius}, min_radius: {min_radius}")
    print(f"  nodes sampled for local_dimension: {len(sample_nodes)} / {len(nodes)} total")
    print(f"  well-fit (r_squared >= 0.9): {len(well_fit)} / {len(sample_nodes)}")

    if well_fit:
        mean_dim = sum(e.dimension for e in well_fit) / len(well_fit)
        mean_r2 = sum(e.r_squared for e in well_fit) / len(well_fit)
    else:
        mean_dim = float("nan")
        mean_r2 = float("nan")

    all_dims = [e.dimension for e in estimates]
    all_r2 = [e.r_squared for e in estimates]
    print(f"  mean dimension (well-fit only): {mean_dim:.4f}")
    print(f"  mean r_squared (well-fit only): {mean_r2:.4f}")
    print(f"  mean dimension (all sampled, unfiltered): {np.mean(all_dims):.4f}")
    print(f"  mean r_squared (all sampled, unfiltered): {np.mean(all_r2):.4f}")

    # cross-check via the module's own mean_dimension wrapper
    mean_dim_wrapper = mean_dimension(hg, samples=n_samples, max_radius=max_radius)
    print(f"  mean dimension (module mean_dimension wrapper, threshold=0.9 internal): {mean_dim_wrapper:.4f}")

    # Use the manually-computed well-fit mean (matches wrapper's own filter
    # logic but lets us also report mean_r2 for that same subset) as the
    # final reported estimate; fall back to the wrapper's value if our
    # well-fit set is empty.
    final_dimension = mean_dim if well_fit else mean_dim_wrapper

    abs_err = abs(final_dimension - TRADITIONAL_DIMENSION) if not np.isnan(final_dimension) else float("nan")
    dim_within_tol = (not np.isnan(abs_err)) and abs_err <= TOLERANCE
    overall_passed = bool(dim_within_tol and solver_ok)

    print(f"\n=== Result ===")
    print(f"  traditional dimension (Grassberger & Procaccia 1983, D2): {TRADITIONAL_DIMENSION}")
    print(f"  poly-algebraic dimension: {final_dimension:.4f}")
    print(f"  absolute error: {abs_err:.4f}  (tolerance {TOLERANCE})")
    print(f"  fit quality (mean r_squared, well-fit subset): {mean_r2:.4f}")
    print(f"  solver verified (butterfly structure confirmed): {solver_ok}")
    print(f"  PASSED: {overall_passed}")

    if not solver_ok:
        print("\n  CAVEAT: solver verification failed -- a dimension match would not count as passing.")
    if mean_r2 < 0.9 if not np.isnan(mean_r2) else True:
        print("\n  NOTE: fit quality on this genuinely fractal attractor should be read carefully -- "
              "see printed r_squared above; a low value here reflects a real limitation of the simple "
              "shell-growth estimator on non-integer-dimension sets, not a bug.")


if __name__ == "__main__":
    main()
