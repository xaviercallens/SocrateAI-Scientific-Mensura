"""Round 2, Benchmark 5: quasi-periodic 2-torus (Lissajous, omega2/omega1 = sqrt(2)).

KNOWN/TRADITIONAL DIMENSION: 2 (tolerance +/- 0.3).

Physics/solver: unchanged from round 1
(scripts/hypergraph_benchmark/05_quasiperiodic_torus.py), which already verified
(a) the closed form x(t) = cos(t), y(t) = cos(sqrt(2) t) actually solves
x'' = -x, y'' = -2y, and (b) the orbit never closes (irrational frequency
ratio -> genuine quasi-periodicity, not an accidental near-period). Those
are not re-derived here. Round 1 also used t_max = 1800 for a single long
non-repeating trajectory; because the orbit never closes, there is no
multi-period-duplication risk here (unlike closed orbits -- N4/finding N7),
so a single long time series is used directly, exactly as round 1 did.

THIS ROUND'S TASK: use socrates.hypergraph.comparison.compare() -- the
apples-to-apples harness -- to test for a compute-savings win (mechanism
(a) in the round-2 brief). Per the problem brief for this system, natural
(evenly-spaced-in-time) sampling of a Lissajous curve has no natural
TIME-density variation (dt is constant), so the density-robustness
mechanism (b) is reported not-applicable rather than manufactured.

Point-cloud construction: points are sampled at CONSTANT density in time
(matching round 1's ratio of 500 points / 1800 time units, i.e. ~1.7
samples per x-oscillation period), with t_max scaled so that n_max points
span n_max / density time units. compare()'s n_grid then takes PREFIXES of
this single list, so smaller n corresponds to a shorter elapsed time (fewer
periods / less torus coverage) at the SAME time-density, and larger n
corresponds to more elapsed time (more periods / better torus coverage) --
the natural, honest way to ask "how many points does each method need to
correctly detect the dimension" for a manifold that is only asymptotically
2-dimensional as quasi-periodic recurrence fills the torus.

FINDING (verified independently, not just quoted from compare()): the
traditional Grassberger-Procaccia correlation-sum estimator does NOT
stably converge to 2 within tolerance on this system, no matter how large
n gets. Two independent constructions confirm this is a genuine structural
bias of the estimator on this system's invariant measure, not an
insufficient-sample-size problem that a bigger n_grid would fix:

  1. The prefix/growing-span sweep below (n up to 12800): the traditional
     estimate passes through the tolerance band only transiently at small
     n (e.g. ~2.10 at n=100, matching the exact F4/F4b trap this
     benchmark's standing rules warn about -- a transient, not
     convergence), then drifts monotonically down through ~1.85, ~1.81,
     ~1.79, ~1.74 and settles at ~1.64-1.69 by n=3200-12800, i.e. outside
     [1.7, 2.3] and moving further away, not closer.
  2. A second, independent construction (see `_diagnose_structural_bias`):
     large IID-random-time samples of the SAME closed form, at n from 500
     to 16000 (an entirely different sampling order, so this is not an
     artifact of the prefix scheme specifically), give a correlation
     dimension of ~1.59-1.63 with r_squared > 0.999 at every n tested --
     extremely stable, well-fit, and consistently below tolerance.

The mechanism is understood, not just observed: x(t) = cos(t) maps
time-uniform sampling to an arcsine-distributed (turning-point-concentrated)
spatial density on [-1, 1], so the (x, y) invariant measure is strongly
concentrated near the four corners of the square rather than uniform. The
classical correlation integral C(r) is a measure-weighted (Renyi D2)
statistic evaluated over baseline.py's fixed r_min_frac=0.01..r_max_frac=0.2
window of the bounding-box diagonal -- a macroscopic window comparable to
the corner concentration's own scale -- so its log-log slope is measurably
biased below the topological dimension 2 for this non-uniform measure. This
is a property of the DEFAULT r-window applied to this measure, not
something a larger n_grid can fix (confirmed by finding #2 above: far
larger n at excellent r_squared does not move the estimate back toward 2).

Per the round-2 standing rule ("do not report savings when the traditional
method never converged -- that measures the traditional method's failure,
not the poly-algebraic method's success"), this problem is honestly
reported as NOT achieving a compute-savings win, even though the
shell-growth (poly-algebraic) method itself converges cleanly and stably to
~2.0-2.05 (well within tolerance) from n=400 onward in the same sweep, and
independently from n=500 up in the random-time construction too -- see
`_diagnose_structural_bias`. That the local method is unbiased by exactly
the density non-uniformity that biases the global method is the same
mechanism as round 2's formal density-robustness test (b), just not the
form the brief for this problem asked to be tested/counted.
"""

from __future__ import annotations

import math
import random
import time

from socrates.hypergraph.baseline import correlation_dimension
from socrates.hypergraph.comparison import ComparisonResult, compare
from socrates.hypergraph.dimension import mean_dimension
from socrates.hypergraph.pointcloud import knn_hypergraph

OMEGA1 = 1.0
OMEGA2 = math.sqrt(2.0)

TRUE_DIMENSION = 2.0
TOLERANCE = 0.3

# Constant time-density (points per unit time), matching round 1's own
# working configuration (500 points over t_max=1800), extended here to a
# much larger n_grid since N5's speedup makes that cheap and the standing
# rules require giving the traditional method a real chance to converge.
DENSITY = 500 / 1800.0
K = 10
N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
MAX_RADIUS = 6  # comparison.compare()'s own default


def build_point_cloud(n_max: int, t_max: float) -> list[tuple[float, float]]:
    """Evenly-spaced-in-time samples of the closed-form solution, constant
    time-density, single non-repeating trajectory (irrational frequency
    ratio => never closes => no duplicate-point risk, unlike closed
    orbits)."""
    return [
        (math.cos(t_max * i / n_max), math.cos(OMEGA2 * t_max * i / n_max))
        for i in range(n_max)
    ]


def run_comparison() -> tuple[ComparisonResult, list[tuple[int, float]]]:
    n_max = N_GRID[-1]
    t_max = n_max / DENSITY
    points = build_point_cloud(n_max, t_max)

    result = compare(
        points, true_dimension=TRUE_DIMENSION, tolerance=TOLERANCE, k=K, n_grid=N_GRID
    )

    # Diagnostic trace: traditional method's estimate at every n_grid stop,
    # re-derived independently of compare()'s internal min-n search, to see
    # *how* it fails (transient pass then drift away) rather than just
    # whether minimum_points_for_target_accuracy returned None.
    trace = []
    for n in N_GRID:
        est = correlation_dimension(points[:n])
        trace.append((n, est.dimension, est.r_squared))

    return result, trace


def diagnose_structural_bias() -> dict:
    """Independent, differently-ordered construction (IID random times, not
    a growing prefix) to check whether the traditional method's failure
    above is a genuine structural bias of this measure/r-window, or just an
    artifact of the prefix-growing-span sampling order. If both
    constructions agree, it is real.

    Also checks whether the poly-algebraic method independently converges
    to ~2 under this different construction (it is not being scored via
    compare() here -- compare() already did that under the official
    construction -- this is a cross-check only).
    """
    rng = random.Random(42)
    t_span = 200_000.0  # long enough to be an effectively ergodic sample
    trad_trace = []
    poly_trace = []
    for n in (500, 1000, 2000, 4000, 8000, 16000):
        ts = [rng.uniform(0.0, t_span) for _ in range(n)]
        pts = [(math.cos(t), math.cos(OMEGA2 * t)) for t in ts]
        trad = correlation_dimension(pts)
        trad_trace.append((n, trad.dimension, trad.r_squared))
        if n <= 8000:  # keep the cross-check fast; compare() already covered n=12800
            hg = knn_hypergraph(pts, k=K, dedupe=True)
            poly_dim = mean_dimension(hg, samples=40, max_radius=MAX_RADIUS)
            poly_trace.append((n, poly_dim))

    return {"traditional_random_time_trace": trad_trace, "poly_random_time_trace": poly_trace}


def main() -> None:
    print("=== Comparison: poly-algebraic vs traditional, growing-span sweep ===")
    print(f"density={DENSITY:.4f} pts/time-unit, k={K}, n_grid={N_GRID}, max_radius={MAX_RADIUS}")
    t0 = time.time()
    result, trace = run_comparison()
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")
    print()
    print("traditional correlation-dimension trace (n, dimension, r_squared):")
    for n, dim, r2 in trace:
        in_band = "IN-BAND" if abs(dim - TRUE_DIMENSION) <= TOLERANCE else "out-of-band"
        print(f"  n={n:6d}  dim={dim:7.4f}  r2={r2:.4f}  [{in_band}]")
    print()
    print("ComparisonResult:")
    print(f"  poly_algebraic_min_n         = {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n            = {result.traditional_min_n}")
    print(f"  poly_algebraic_wins          = {result.poly_algebraic_wins}")
    print(f"  compute_savings_fraction     = {result.compute_savings_fraction}")
    print(f"  poly_algebraic_estimate@max  = {result.poly_algebraic_estimate_at_max_n:.4f}")
    print(f"  traditional_estimate@max     = {result.traditional_estimate_at_max_n:.4f}")
    print(f"  max_n_tested                 = {result.max_n_tested}")

    # Independent re-derivation of the max-n estimates (rule 1: verify
    # before trusting -- do not just print what compare() returned).
    n_max = N_GRID[-1]
    t_max = n_max / DENSITY
    points = build_point_cloud(n_max, t_max)
    hg = knn_hypergraph(points, k=K, dedupe=True)
    poly_check = mean_dimension(hg, samples=40, max_radius=MAX_RADIUS)
    trad_check = correlation_dimension(points).dimension
    assert abs(poly_check - result.poly_algebraic_estimate_at_max_n) < 1e-9
    assert abs(trad_check - result.traditional_estimate_at_max_n) < 1e-9
    print()
    print("independent re-derivation of estimate@max_n matches compare() exactly: OK")

    print()
    print("=== Structural-bias diagnostic: independent random-time-order construction ===")
    diag = diagnose_structural_bias()
    print("traditional, IID random-time sampling (n, dimension, r_squared):")
    for n, dim, r2 in diag["traditional_random_time_trace"]:
        print(f"  n={n:6d}  dim={dim:7.4f}  r2={r2:.4f}")
    print("poly-algebraic, IID random-time sampling (n, dimension):")
    for n, dim in diag["poly_random_time_trace"]:
        print(f"  n={n:6d}  dim={dim:7.4f}")

    print()
    print("=== Density-robustness mechanism (b) ===")
    print(
        "Not tested: this problem's natural (evenly-spaced-in-time) sampling has"
        " constant time-density (dt fixed), so there is no natural close-approach"
        " / slow-fast-turning-point TIME-clustering to test, per the round-2 brief"
        " for this problem. (A different, unrelated form of SPATIAL non-uniformity"
        " -- arcsine/turning-point concentration in x=cos(t), y=cos(sqrt2 t) -- does"
        " exist and is the mechanism behind the traditional method's structural bias"
        " documented above, but it is not the TIME-density-clustering effect this"
        " benchmark's mechanism (b) is defined to test, and the brief for this"
        " problem explicitly directs reporting mechanism (b) as not applicable"
        " rather than manufacturing a test for it.)"
    )

    won_on_compute_savings = bool(result.poly_algebraic_wins and result.compute_savings_fraction and result.compute_savings_fraction > 0.10)
    won_on_density_robustness = False
    overall_win = won_on_compute_savings or won_on_density_robustness

    print()
    print("=== Summary ===")
    print(f"  poly_algebraic_min_n:      {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n:         {result.traditional_min_n}")
    print(f"  compute_savings_fraction:  {result.compute_savings_fraction}")
    print(f"  won_on_compute_savings:    {won_on_compute_savings}")
    print(f"  won_on_density_robustness: {won_on_density_robustness}")
    print(f"  overall_win:               {overall_win}")


if __name__ == "__main__":
    main()
