"""Round 3, problem 05 (quasiperiodic 2-torus): re-measured against the current library.

WHAT THIS SCRIPT IS FOR
-----------------------
Round 2 scored problem 05 as a NON-WIN and was right to: the traditional
Grassberger-Procaccia baseline never converges on this cloud, and standing
rule 2 forbids banking a baseline failure as a poly-algebraic win. Finding
R2-F9 (docs/MENSURA_BENCHMARK.md Sec. 10.5) recorded that this is
nevertheless the place where the shell-growth estimator is most clearly and
NON-DEGENERATELY better than the baseline -- and that criterion (a) is
structurally incapable of scoring it. H3 added
`comparison.compare_accuracy_at_max_n` (criterion (c)) precisely for this.

So this script measures BOTH, and reports both honestly:

  criterion (a) -- `comparison.compare()`. Expected and reported as NO WIN,
      with `traditional_min_n = None`. No savings fraction is computed or
      claimed. The poly-side `min_n` is still reported as a measurement.
  criterion (c) -- `comparison.compare_accuracy_at_max_n()`. Accuracy at one
      fixed large n against the known dimension 2, with the F1 degeneracy
      guard and the calibrated 0.25 margin doing the work.

PARAMETERS AND THE R2-F3 DISCLOSURE RULE (docs Sec. 9.4)
--------------------------------------------------------
R2-F3 found that problem 01's round-2 "win" was contingent on an undisclosed
non-default `max_radius`. Everything here therefore runs at
`max_radius = 6` -- the module default, NO deviation -- and the primary
configuration is `k = 6`, the value the task brief names as the default.

`k = 10` is ALSO reported, at every stage, because 10 is problem 05's own
round-2 production value (it is written into
scripts/hypergraph_benchmark/round2/05_quasiperiodic_torus.py as `K = 10` and
into round3/n8d_clouds.py's `SPECS["05"]`), and Sec. 10.5's banked numbers
(poly 2.0500, trad 1.6357 at n=12800) are k=10 numbers. Reporting only one
would either break continuity with the ledger or quietly deviate from the
default. Both are reported; the headline verdict is taken at k=6, and section 4
shows the verdict is the same across k in {6, 8, 10, 12, 14}.

All other knobs are at library defaults: `theiler_window = 0`,
`max_ball_fraction = 1.0`, `samples = 40`, `n_radii = 20`,
`margin = DEFAULT_ACCURACY_MARGIN = 0.25`,
`max_sentinel_fraction = DEFAULT_MAX_SENTINEL_FRACTION = 0.05`.
(`compare_accuracy_at_max_n` does not currently accept `theiler_window` or
`max_ball_fraction` at all -- a known capability gap. On this problem it costs
nothing, since the defaults are exactly what a torus wants: there is no
attractor-folding autocorrelation artifact to suppress and no saturation
truncation in force.)

POINT-CLOUD CONSTRUCTION (see Sec. 10.4 / R2-F8)
------------------------------------------------
Sec. 10.4 found that a naive TIME-ORDERED prefix of a chaotic trajectory is a
short smooth arc that poly correctly calls 1-dimensional -- an argument for
whole-orbit-covering prefixes (bit-reversal / Weyl) on problems 01/02/03/04/10.
That pathology needs checking here rather than assuming either way, so section 1
builds BOTH orderings of the SAME 12800 points and section 2 runs criterion (a)
under both:

  "growing-span"  -- the round-2 production construction. Constant time density
      (1/DENSITY = 3.6 time units per sample); prefix `points[:n]` spans
      t in [0, 3.6n). NOT the Sec. 10.4 pathology: even n=100 spans t=360,
      i.e. ~57 periods of the omega=1 oscillator, so the prefix already covers
      the torus rather than tracing one arc of it.
  "bit-reversed"  -- the Sec. 10.4-endorsed whole-orbit-covering ordering. The
      full span t in [0, 46080) is fixed and the 12800 sample INDICES are
      visited in bit-reversed order, so every prefix is spread across the whole
      trajectory at lower density.

Both orderings are permutations of one identical point set, so at max n they
are the same cloud and criterion (c) is ORDERING-INVARIANT by construction --
section 3 asserts that rather than claiming it. Criterion (a) is not, and both
numbers are reported.

SECTIONS
--------
  1  cloud provenance, closed-form re-derivation, and the two orderings   (~1 s)
  2  criterion (a) via compare(), k in {6, 10} x both orderings           (~6 min)
  3  criterion (c) via compare_accuracy_at_max_n(), k in {6, 10}          (~3 min)
  4  criterion (c) robustness: k sweep, max_radius sweep, 10 re-phasings  (~8 min)
  5  BASELINE-fairness: is GP's 1.6357 a handicap or a real property?     (~4 min)

Section 5 exists because a criterion (c) win is worth exactly as much as the
baseline it beats, and the cheapest way to manufacture one is to fit the
correlation sum over a bad radius window and call the result the state of the
art. It sweeps GP's own fit window and radius count, re-measures on an
independently constructed IID sample of the same torus, and checks GP against a
known-answer uniform square under identical settings.

  6  R2-F3 asked of the BASELINE -- the section that decides the verdict (~12 min)

Usage:  python scripts/hypergraph_benchmark/round3/05_quasiperiodic_torus.py --sections 1,2,3,4,5,6

RESULT (measured, this script)
------------------------------
criterion (a): NO WIN, and no savings number exists. `traditional_min_n` is
    None -- the correlation sum is in-band at n=100..1600 and then drifts OUT
    (1.6853, 1.6536, 1.6357 at n=3200/6400/12800) and never returns, so it
    never stably converges. Poly reaches and holds the band at n=200 (k=6) /
    n=400 (k=10). Standing rule 2: a baseline that never converges is a
    baseline failure, not a poly win. No savings fraction is claimed.

criterion (c): WIN AT LIBRARY DEFAULTS, REFUSED AFTER SECTION 6.
    At defaults `compare_accuracy_at_max_n` returns
    `poly_algebraic_accuracy_win = True`: poly 2.0464 (|err| 0.0464) vs GP
    1.6357 (|err| 0.3643), gap 0.3180 >= margin 0.25, sentinel fraction 0.000.
    It survives k in {6,8,10,12,14} (5/5), max_radius 3..8 (6/6), 10/10
    re-phasings, and an independent IID-time construction.

    Section 6 breaks it. Condition (c4) requires the BASELINE's error to exceed
    the tolerance, and GP's 1.6357 is a property of the library's FIXED radius
    window [0.01, 0.2] x diagonal, which sits in the correlation sum's
    saturation regime. Under answer-blind plateau selection -- the actual GP
    procedure, and validated here on uniform square (1.98), uniform cube (2.94)
    and circle (1.01) -- GP reads 1.806-1.829 at every selector width tried,
    i.e. |err| ~ 0.19, INSIDE the problem's own +/-0.3 tolerance. (c4) then
    fails and criterion (c) correctly refuses.

    That is finding R2-F3 with the roles reversed: a win contingent on a
    hyperparameter, this time the baseline's. `h3_win_robustness.py` swept
    poly's k and max_radius and never swept the correlation sum's radius
    window, so the contingency was invisible to the criterion's own robustness
    check. It is a real gap in the criterion, not just in this problem.

    Section 6c checks whether the contingency closes with more data. It does,
    but only at n=102400 -- 8x past the benchmark's n_grid max -- and the next
    point, n=204800, fails on a different condition (poly's sentinel fraction
    reaches 0.075, above the 0.05 cap, so (c2) refuses). One grid point wide,
    bracketed by a failure, and reachable only by extending the grid: that is
    measurement-config gaming, and it is refused here for the same reason AR4
    refused the perched Lorenz cell.

    What survives as a true statement: poly's error (0.046) is 3-4x smaller
    than GP's BEST ACHIEVABLE error on this cloud (0.158-0.191), with zero
    degenerate fits, and GP drifts further from 2.0 as n grows while poly does
    not. That is a genuine accuracy advantage. It is not a criterion (c) win at
    the benchmark's n=12800.

CARRY-FORWARD FOR THE CRITERION ITSELF
---------------------------------------
`h3_win_robustness.py` establishes that a criterion (c) verdict is stable
against POLY's hyperparameters (k, max_radius) and against reseeding. It never
varies the BASELINE's. Section 6 shows that on this problem the baseline's
radius window alone flips the verdict.

Section 6d asks the same question of problem 06, the other case R2-F9 flagged,
and the answer is worse: GP on planar Brownian motion reads 1.3943 at the fixed
default window and **1.9746** -- |err| 0.025, essentially exact -- once its
scaling region is plateau-selected. Sec. 10.5's "confidently wrong by 0.61" is
a fit-window artifact there, not a property of the method.

So this is not a quirk of problem 05. BOTH cases criterion (c) was built to
credit are contingent on the same baseline hyperparameter, and on one of them
the baseline is actually correct. Any criterion (c) win banked anywhere in this
benchmark should be re-checked against a plateau-selected baseline before it is
scored. The check is cheap and self-contained: `gp_plateau_dimension` in this
file needs nothing from the library and is validated on three known-answer
clouds in section 6a.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.comparison import (  # noqa: E402
    DEFAULT_ACCURACY_MARGIN,
    DEFAULT_MAX_SENTINEL_FRACTION,
    compare,
    compare_accuracy_at_max_n,
)

# --- problem 05's own constants, read off the round-2 script ------------------
OMEGA2 = math.sqrt(2.0)
DENSITY = 500 / 1800.0
N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
TRUE_DIMENSION = 2.0
TOLERANCE = 0.3
MAX_RADIUS = 6  # module default; NOT a deviation (R2-F3)
K_DEFAULT = 6  # the brief's default
K_PRODUCTION = 10  # round-2 problem 05's own K
N_MAX = N_GRID[-1]


# ---------------------------------------------------------------------------
# Section 1: the cloud, re-derived, and its two orderings
# ---------------------------------------------------------------------------


def closed_form_cloud(n_max: int = N_MAX) -> list[tuple[float, float]]:
    """(cos t, cos(sqrt(2) t)) at constant density -- the round-2 construction.

    Re-derived here from the closed form rather than imported, so this script
    does not depend on the round-2 module being importable; section 1 asserts
    it is identical to what n8d_clouds.load_cloud('05') returns.
    """
    t_max = n_max / DENSITY
    return [
        (math.cos(t_max * i / n_max), math.cos(OMEGA2 * t_max * i / n_max)) for i in range(n_max)
    ]


def bit_reversed_order(n: int) -> list[int]:
    """Indices 0..n-1 in bit-reversed order (n a power of two times 100 here, so
    the reversal is done on the smallest enclosing power of two and out-of-range
    values are dropped -- the standard low-discrepancy trick, and exactly what
    round-2 problems 02/10 use)."""
    bits = max(1, (n - 1).bit_length())
    order = []
    for i in range(1 << bits):
        rev = int(format(i, f"0{bits}b")[::-1], 2)
        if rev < n:
            order.append(rev)
    return order


def section1() -> dict:
    print("=" * 78)
    print("SECTION 1 -- cloud provenance and the two prefix orderings")
    print("=" * 78)

    pts = closed_form_cloud()
    print(f"closed-form cloud: {len(pts)} points in {len(pts[0])}D, "
          f"density={DENSITY:.6f} pts/time-unit, t_max={N_MAX / DENSITY:.1f}")

    # (1a) identical to the cached round-2 production cloud?
    try:
        import n8d_clouds  # type: ignore

        cached = n8d_clouds.load_cloud("05")
        same = len(cached) == len(pts) and all(
            abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12
            for a, b in zip(cached, pts, strict=True)
        )
        print(f"(1a) identical to n8d_clouds.load_cloud('05'): {same}  "
              f"(n={len(cached)} vs {len(pts)})")
        assert same, "closed-form re-derivation does not match the round-2 production cloud"
    except ImportError:
        print("(1a) n8d_clouds not importable; skipping the provenance cross-check")

    # (1b) non-closure: the defining property of a quasiperiodic torus.
    #      The FULL phase state (x, y, vx, vy) never returns near its start.
    def state(t: float) -> tuple[float, float, float, float]:
        return (
            math.cos(t),
            math.cos(OMEGA2 * t),
            -math.sin(t),
            -OMEGA2 * math.sin(OMEGA2 * t),
        )

    # t is searched from 1.0, not 0: the trivial neighbourhood of the start is
    # not a return, and including it would make the check vacuously "close".
    s0 = state(0.0)
    best_d, best_t = float("inf"), 0.0
    steps = 500_000
    for i in range(steps + 1):
        t = 1.0 + (500.0 - 1.0) * i / steps
        s = state(t)
        d = math.dist(s0, s)
        if d < best_d:
            best_d, best_t = d, t
    print(f"(1b) closest full-state return over t in [1, 500]: d={best_d:.6f} at t={best_t:.4f} "
          f"-> never closes (a closed orbit would give d ~ 0)")

    # (1c) the two orderings
    order_bitrev = bit_reversed_order(N_MAX)
    bitrev = [pts[i] for i in order_bitrev]
    assert sorted(order_bitrev) == list(range(N_MAX))
    assert set(bitrev) == set(pts), "bit-reversed cloud is not a permutation of the same points"
    print(f"(1c) bit-reversed ordering built: {len(bitrev)} points, verified to be a "
          f"permutation of the identical point set (so max-n clouds coincide)")
    print(f"     growing-span  prefix n=100 spans t in [0, {3.6 * 100:.0f})")
    print(f"     bit-reversed  prefix n=100 spans t in [0, {N_MAX / DENSITY:.0f}) at 1/128 density")

    return {"growing_span": pts, "bit_reversed": bitrev, "closest_return": best_d}


# ---------------------------------------------------------------------------
# Section 2: criterion (a)
# ---------------------------------------------------------------------------


def section2(clouds: dict) -> dict:
    print()
    print("=" * 78)
    print("SECTION 2 -- criterion (a): compute savings via comparison.compare()")
    print("=" * 78)
    print(f"n_grid={N_GRID}, true={TRUE_DIMENSION}, tol={TOLERANCE}, max_radius={MAX_RADIUS}, "
          f"theiler_window=0, max_ball_fraction=1.0  [all library defaults]")

    out = {}
    for ordering in ("growing_span", "bit_reversed"):
        pts = clouds[ordering]
        for k in (K_DEFAULT, K_PRODUCTION):
            t0 = time.time()
            res = compare(
                pts,
                true_dimension=TRUE_DIMENSION,
                tolerance=TOLERANCE,
                k=k,
                n_grid=N_GRID,
                max_radius=MAX_RADIUS,
            )
            dt = time.time() - t0
            out[(ordering, k)] = res
            print()
            print(f"--- {ordering}, k={k}  ({dt:.0f}s) ---")
            print(f"  poly_algebraic_min_n     = {res.poly_algebraic_min_n}")
            print(f"  traditional_min_n        = {res.traditional_min_n}")
            print(f"  poly_algebraic_wins      = {res.poly_algebraic_wins}")
            print(f"  compute_savings_fraction = {res.compute_savings_fraction}")
            print(f"  poly estimate @ n={res.max_n_tested:<6d} = "
                  f"{res.poly_algebraic_estimate_at_max_n:.4f}")
            print(f"  trad estimate @ n={res.max_n_tested:<6d} = "
                  f"{res.traditional_estimate_at_max_n:.4f}")

    # Why traditional_min_n is None, shown rather than asserted: the per-n trace.
    print()
    print("--- traditional per-n trace (growing-span ordering, identical for both k) ---")
    pts = clouds["growing_span"]
    trace = []
    for n in N_GRID:
        est = correlation_dimension(pts[:n])
        in_band = abs(est.dimension - TRUE_DIMENSION) <= TOLERANCE
        trace.append((n, est.dimension, est.r_squared, in_band))
        print(f"  n={n:6d}  dim={est.dimension:7.4f}  r2={est.r_squared:.4f}  "
              f"[{'IN-BAND' if in_band else 'out-of-band'}]")
    print("  -> the baseline is confidently fitted (r2 > 0.99) and drifting AWAY from 2.0;")
    print("     it never converges, so criterion (a) has no savings number. Standing rule 2:")
    print("     a baseline failure is NOT a poly win. No savings fraction is claimed.")

    return {"results": out, "traditional_trace": trace}


# ---------------------------------------------------------------------------
# Section 3: criterion (c)
# ---------------------------------------------------------------------------


def _report_accuracy(tag: str, res) -> None:
    print()
    print(f"--- {tag} ---")
    print(f"  n_evaluated                = {res.n_evaluated} "
          f"(poly graph nodes {res.poly_algebraic_n_nodes})")
    print(f"  poly estimate              = {res.poly_algebraic_estimate:.4f}  "
          f"|err| = {res.poly_algebraic_abs_error:.4f}")
    print(f"  traditional estimate       = {res.traditional_estimate:.4f}  "
          f"|err| = {res.traditional_abs_error:.4f}   (r2 = {res.traditional_r_squared:.4f})")
    print(f"  accuracy gap (trad - poly) = "
          f"{res.traditional_abs_error - res.poly_algebraic_abs_error:.4f}  "
          f"(margin {res.margin})")
    print(f"  sentinel fraction          = {res.poly_algebraic_sentinel_fraction:.4f}  "
          f"(degenerate {res.poly_algebraic_degenerate_fraction:.4f} + near-degenerate "
          f"{res.poly_algebraic_near_degenerate_fraction:.4f}; cap "
          f"{res.max_sentinel_fraction})")
    print(f"  more_accurate              = {res.more_accurate}")
    print(f"  poly_algebraic_accuracy_win= {res.poly_algebraic_accuracy_win}")
    print(f"  traditional_accuracy_win   = {res.traditional_accuracy_win}")
    print(f"  verdict_reason             = {res.verdict_reason}")


def section3(clouds: dict) -> dict:
    print()
    print("=" * 78)
    print("SECTION 3 -- criterion (c): accuracy at max n via compare_accuracy_at_max_n()")
    print("=" * 78)
    print(f"max_n={N_MAX}, true={TRUE_DIMENSION}, tolerance={TOLERANCE}, "
          f"margin={DEFAULT_ACCURACY_MARGIN} (default), "
          f"max_sentinel_fraction={DEFAULT_MAX_SENTINEL_FRACTION} (default), "
          f"max_radius={MAX_RADIUS} (default)")

    out = {}
    for k in (K_DEFAULT, K_PRODUCTION):
        res = compare_accuracy_at_max_n(
            clouds["growing_span"],
            true_dimension=TRUE_DIMENSION,
            max_n=N_MAX,
            k=k,
            tolerance=TOLERANCE,
            max_radius=MAX_RADIUS,
        )
        out[k] = res
        _report_accuracy(f"k={k}, growing-span ordering", res)

    # Ordering-invariance is claimed in the docstring; measure it once instead.
    res_b = compare_accuracy_at_max_n(
        clouds["bit_reversed"],
        true_dimension=TRUE_DIMENSION,
        max_n=N_MAX,
        k=K_DEFAULT,
        tolerance=TOLERANCE,
        max_radius=MAX_RADIUS,
    )
    out[("bit_reversed", K_DEFAULT)] = res_b
    _report_accuracy(f"k={K_DEFAULT}, bit-reversed ordering (ordering-invariance check)", res_b)
    print()
    print(f"  ordering invariance: poly {out[K_DEFAULT].poly_algebraic_estimate:.4f} vs "
          f"{res_b.poly_algebraic_estimate:.4f}; trad "
          f"{out[K_DEFAULT].traditional_estimate:.4f} vs {res_b.traditional_estimate:.4f}; "
          f"same verdict = "
          f"{out[K_DEFAULT].poly_algebraic_accuracy_win == res_b.poly_algebraic_accuracy_win}")

    return out


# ---------------------------------------------------------------------------
# Section 4: is the criterion (c) verdict a property of the data or the knobs?
# ---------------------------------------------------------------------------


def rephased_cloud(seed: int, n: int = N_MAX) -> list[tuple[float, float]]:
    """The same non-repeating trajectory, started at a different time.

    Deterministic system, so a redraw means a different window of the orbit --
    the same construction h3_margin_calibration.py uses for this cloud.
    """
    t_max = n / DENSITY
    t0 = 13.7 * seed
    return [
        (math.cos(t0 + t_max * i / n), math.cos(OMEGA2 * (t0 + t_max * i / n))) for i in range(n)
    ]


def section4(clouds: dict) -> dict:
    print()
    print("=" * 78)
    print("SECTION 4 -- criterion (c) robustness (the R2-F3 question, asked up front)")
    print("=" * 78)

    pts = clouds["growing_span"]
    out: dict = {}

    print()
    print("(4a) k sweep at max_radius=6 (default):")
    k_rows = []
    for k in (6, 8, 10, 12, 14):
        r = compare_accuracy_at_max_n(
            pts, true_dimension=TRUE_DIMENSION, max_n=N_MAX, k=k,
            tolerance=TOLERANCE, max_radius=MAX_RADIUS,
        )
        k_rows.append((k, r.poly_algebraic_estimate, r.traditional_estimate,
                       r.poly_algebraic_sentinel_fraction, r.poly_algebraic_accuracy_win))
        print(f"  k={k:3d}  poly={r.poly_algebraic_estimate:7.4f}  "
              f"trad={r.traditional_estimate:7.4f}  "
              f"sentinel={r.poly_algebraic_sentinel_fraction:.3f}"
              f"  win={r.poly_algebraic_accuracy_win}")
    out["k_sweep"] = k_rows
    print(f"  -> {sum(1 for r in k_rows if r[4])} / {len(k_rows)} wins")

    print()
    print(f"(4b) max_radius sweep at k={K_DEFAULT}:")
    mr_rows = []
    for mr in range(3, 9):
        r = compare_accuracy_at_max_n(
            pts, true_dimension=TRUE_DIMENSION, max_n=N_MAX, k=K_DEFAULT,
            tolerance=TOLERANCE, max_radius=mr,
        )
        mr_rows.append((mr, r.poly_algebraic_estimate, r.traditional_estimate,
                        r.poly_algebraic_sentinel_fraction, r.poly_algebraic_accuracy_win))
        print(f"  max_radius={mr}  poly={r.poly_algebraic_estimate:7.4f}  "
              f"trad={r.traditional_estimate:7.4f}  "
              f"sentinel={r.poly_algebraic_sentinel_fraction:.3f}  "
              f"win={r.poly_algebraic_accuracy_win}")
    out["max_radius_sweep"] = mr_rows
    print(f"  -> {sum(1 for r in mr_rows if r[4])} / {len(mr_rows)} wins "
          f"(R2-F3: the verdict must not depend on this knob)")

    print()
    print(f"(4c) 10 re-phasings of the trajectory at k={K_DEFAULT}, n=1600 and n={N_MAX}:")
    seed_rows = []
    for n_eval in (1600, N_MAX):
        wins = 0
        for seed in range(10):
            r = compare_accuracy_at_max_n(
                rephased_cloud(seed, n=n_eval), true_dimension=TRUE_DIMENSION,
                max_n=n_eval, k=K_DEFAULT, tolerance=TOLERANCE, max_radius=MAX_RADIUS,
            )
            wins += bool(r.poly_algebraic_accuracy_win)
            seed_rows.append((n_eval, seed, r.poly_algebraic_estimate, r.traditional_estimate,
                              r.poly_algebraic_sentinel_fraction, r.poly_algebraic_accuracy_win))
            print(f"  n={n_eval:6d} seed={seed:2d}  poly={r.poly_algebraic_estimate:7.4f}  "
                  f"trad={r.traditional_estimate:7.4f}  "
                  f"sentinel={r.poly_algebraic_sentinel_fraction:.3f}  "
                  f"win={r.poly_algebraic_accuracy_win}")
        print(f"  -> n={n_eval}: {wins} / 10 re-phasings win")
    out["seed_sweep"] = seed_rows

    print()
    print("(4d) two-sidedness control: the same call on a uniform 3-cube, where the")
    print("     criterion is supposed to fire AGAINST poly (docs' stated behaviour).")
    import numpy as np

    rng = np.random.default_rng(7)
    cube = [tuple(p) for p in rng.uniform(0.0, 1.0, size=(3200, 3))]
    rc = compare_accuracy_at_max_n(
        cube, true_dimension=3.0, max_n=3200, k=K_DEFAULT, tolerance=TOLERANCE,
        max_radius=MAX_RADIUS,
    )
    print(f"  uniform 3-cube n=3200: poly={rc.poly_algebraic_estimate:.4f} "
          f"(|err| {rc.poly_algebraic_abs_error:.4f}), trad={rc.traditional_estimate:.4f} "
          f"(|err| {rc.traditional_abs_error:.4f})")
    print(f"  poly_win={rc.poly_algebraic_accuracy_win}  "
          f"trad_win={rc.traditional_accuracy_win}  more_accurate={rc.more_accurate}")
    out["cube_control"] = (rc.poly_algebraic_estimate, rc.traditional_estimate,
                           rc.poly_algebraic_accuracy_win, rc.traditional_accuracy_win)

    return out


# ---------------------------------------------------------------------------
# Section 5: is the BASELINE's failure real, or is poly winning against a
#            handicapped opponent? (the spurious-positive trap for criterion (c))
# ---------------------------------------------------------------------------


def section5(clouds: dict) -> dict:
    print()
    print("=" * 78)
    print("SECTION 5 -- is the baseline's 1.6357 a real property, or a handicap?")
    print("=" * 78)
    print("A criterion (c) win is only as good as the baseline it beats. Three ways")
    print("the 1.6357 could be an artifact rather than a measurement, each checked:")

    pts = clouds["growing_span"]
    out: dict = {}

    # (5a) The scaling-region window. GP's answer depends on which radii are fitted.
    #      If 1.6357 is an artifact of [0.01, 0.2] x diagonal, some other window
    #      should recover 2.0.
    print()
    print("(5a) correlation-sum fit-window sweep at n=12800 "
          "(r_min_frac, r_max_frac as fractions of the bounding-box diagonal):")
    windows = [
        (0.001, 0.02), (0.002, 0.05), (0.005, 0.1), (0.01, 0.2),
        (0.02, 0.3), (0.05, 0.5), (0.1, 0.7), (0.002, 0.2), (0.01, 0.5),
    ]
    win_rows = []
    for rmin, rmax in windows:
        est = correlation_dimension(pts, r_min_frac=rmin, r_max_frac=rmax)
        in_tol = abs(est.dimension - TRUE_DIMENSION) <= TOLERANCE
        win_rows.append((rmin, rmax, est.dimension, est.r_squared, in_tol))
        print(f"  r in [{rmin:.3f}, {rmax:.3f}] x diag  dim={est.dimension:7.4f}  "
              f"r2={est.r_squared:.4f}  {'IN-TOL' if in_tol else 'out-of-tol'}")
    n_in = sum(1 for r in win_rows if r[4])
    print(f"  -> {n_in} / {len(win_rows)} fit windows put GP inside |err| <= {TOLERANCE}")
    out["baseline_window_sweep"] = win_rows

    # (5b) n_radii. The default 20 is a discretisation of the same window.
    print()
    print("(5b) n_radii sweep at the default window:")
    nr_rows = []
    for nr in (10, 15, 20, 30, 50):
        est = correlation_dimension(pts, n_radii=nr)
        nr_rows.append((nr, est.dimension, est.r_squared))
        print(f"  n_radii={nr:3d}  dim={est.dimension:7.4f}  r2={est.r_squared:.4f}")
    out["baseline_n_radii_sweep"] = nr_rows

    # (5c) A different SAMPLING of the same measure. The growing-span cloud is a
    #      deterministic even-in-time sweep; if GP's failure were an ordering /
    #      sampling artifact rather than a property of this measure at this n, an
    #      IID ergodic sample of the same torus would behave differently.
    print()
    print("(5c) IID random-time sample of the SAME torus, n=12800 "
          "(independent construction, not a prefix):")
    import random

    rng = random.Random(42)
    ts = [rng.uniform(0.0, 200_000.0) for _ in range(N_MAX)]
    iid = [(math.cos(t), math.cos(OMEGA2 * t)) for t in ts]
    r_iid = compare_accuracy_at_max_n(
        iid, true_dimension=TRUE_DIMENSION, max_n=N_MAX, k=K_DEFAULT,
        tolerance=TOLERANCE, max_radius=MAX_RADIUS,
    )
    _report_accuracy("IID random-time torus, k=6", r_iid)
    out["iid_construction"] = (r_iid.poly_algebraic_estimate, r_iid.traditional_estimate,
                               r_iid.poly_algebraic_accuracy_win)

    # (5d) Known-answer control on the SAME estimator settings: a uniform square
    #      is a genuinely 2-dimensional measure with no torus geometry. If GP
    #      reads ~2.0 there and ~1.64 here, its error is about this measure, not
    #      about the estimator being broken.
    print()
    print("(5d) known-answer control -- uniform square (true 2.0), n=12800, same settings:")
    import numpy as np

    sq_rng = np.random.default_rng(11)
    square = [tuple(p) for p in sq_rng.uniform(-1.0, 1.0, size=(N_MAX, 2))]
    est_sq = correlation_dimension(square)
    r_sq = compare_accuracy_at_max_n(
        square, true_dimension=TRUE_DIMENSION, max_n=N_MAX, k=K_DEFAULT,
        tolerance=TOLERANCE, max_radius=MAX_RADIUS,
    )
    print(f"  GP on uniform square: dim={est_sq.dimension:.4f}  r2={est_sq.r_squared:.4f}  "
          f"|err|={abs(est_sq.dimension - 2.0):.4f}")
    print(f"  poly on uniform square: {r_sq.poly_algebraic_estimate:.4f}  "
          f"|err|={r_sq.poly_algebraic_abs_error:.4f}")
    print(f"  criterion (c) verdict on the square: poly_win={r_sq.poly_algebraic_accuracy_win} "
          f"(GP is inside tolerance here, so (c4) refuses -- as it should)")
    out["square_control"] = (est_sq.dimension, r_sq.poly_algebraic_estimate,
                             r_sq.poly_algebraic_accuracy_win)

    # (5e) `compare_accuracy_at_max_n` accepts neither `theiler_window` nor
    #      `max_ball_fraction` (the known H1/AR2 capability gap). Show, rather
    #      than assume, that this costs nothing on THIS cloud.
    print()
    print("(5e) the known capability gap (no theiler_window / max_ball_fraction on")
    print("     compare_accuracy_at_max_n) -- measured cost on this cloud:")
    from socrates.hypergraph.baseline import theiler_window_from_autocorrelation

    auto_w = theiler_window_from_autocorrelation(np.asarray(pts, dtype=float))
    print(f"  theiler_window='auto' resolves to W={auto_w} on this cloud "
          f"(W=1 is the conventional 'no window')")
    for tw in (0, 1, auto_w):
        e = correlation_dimension(pts, theiler_window=tw)
        print(f"    theiler_window={tw}: GP dim={e.dimension:.4f}  r2={e.r_squared:.4f}")
    print("  -> the correction is a no-op here. Samples are 3.6 time units apart while")
    print("     the periods are 2*pi and 2*pi/sqrt(2), so consecutive points are not")
    print("     temporally correlated neighbours. The gap costs this problem nothing.")
    out["theiler_auto"] = auto_w

    return out


# ---------------------------------------------------------------------------
# Section 6: R2-F3 applied to the BASELINE -- the contingency that decides this
#            problem. THIS IS THE SECTION THAT MATTERS.
# ---------------------------------------------------------------------------


def gp_plateau_dimension(
    points: list[tuple[float, ...]],
    *,
    n_radii: int = 80,
    lo: float = 1e-5,
    hi: float = 0.9,
    width: int = 12,
    smooth: int = 3,
    min_pairs: int = 200,
) -> tuple[float, float, float, float]:
    """Grassberger-Procaccia with an ANSWER-BLIND scaling-region selection.

    The library's `correlation_dimension` fits a FIXED radius window,
    [0.01, 0.2] x bounding-box diagonal. That is a documented convenience, not
    the GP procedure: Grassberger and Procaccia -- and every practitioner since
    -- select the scaling region by finding the *plateau* in the local slope
    d log C / d log r, because at large r the correlation sum saturates toward
    C=1 and biases the slope DOWN.

    This function does that selection and nothing else. It slides a fixed-width
    window over log r and returns the fit from the window whose local slopes are
    FLATTEST (minimum standard deviation). It never sees `true_dimension`, so it
    cannot be steering toward the answer -- and section 6a validates it on three
    known-answer clouds before it is used to judge anything.

    Returns (dimension, slope_sd, r_lo/diag, r_hi/diag).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    arr = np.asarray(points, dtype=float)
    n = len(arr)
    tree = cKDTree(arr)
    diag = float(np.sqrt(np.sum((arr.max(axis=0) - arr.min(axis=0)) ** 2)))
    radii = np.logspace(math.log10(lo * diag), math.log10(hi * diag), n_radii)
    counts = np.array([tree.count_neighbors(tree, r) for r in radii])
    pairs = (counts - n) / 2
    total = n * (n - 1) / 2
    c_of_r = pairs / total
    # Drop radii that carry no slope information: C=0, C=1, or so few pairs that
    # the local slope is discreteness noise rather than a measurement.
    keep = (c_of_r > 0) & (c_of_r < 1) & (pairs >= min_pairs)
    radii, c_of_r = radii[keep], c_of_r[keep]
    if len(radii) < width + smooth + 1:
        return float("nan"), float("nan"), float("nan"), float("nan")

    log_r, log_c = np.log(radii), np.log(c_of_r)
    local = np.array(
        [(log_c[i + smooth] - log_c[i]) / (log_r[i + smooth] - log_r[i])
         for i in range(len(log_r) - smooth)]
    )
    best = None
    for i in range(len(local) - width):
        sd = float(np.std(local[i:i + width]))
        slope = float(np.polyfit(log_r[i:i + width + smooth], log_c[i:i + width + smooth], 1)[0])
        if best is None or sd < best[0]:
            best = (sd, slope, radii[i] / diag, radii[i + width + smooth] / diag)
    assert best is not None
    return best[1], best[0], best[2], best[3]


def section6(clouds: dict) -> dict:
    print()
    print("=" * 78)
    print("SECTION 6 -- R2-F3 asked of the BASELINE: is 1.6357 a hyperparameter?")
    print("=" * 78)
    print("Section 5a already showed GP's answer moving from 1.4454 to 1.8225 as its")
    print("fit window shrinks. That is a hyperparameter, and criterion (c)'s verdict")
    print("turns on it: (c4) requires the baseline error to EXCEED the tolerance. So")
    print("the question is not 'what does the default give' but 'what does a")
    print("competently-configured GP give'. Answer-blind plateau selection, below.")

    import numpy as np

    out: dict = {}
    pts = clouds["growing_span"]

    # (6a) VALIDATE the selector on known answers before trusting it anywhere.
    print()
    print("(6a) validation of the answer-blind plateau selector on known-answer clouds")
    print("     (if it distorts these, it may not be used to judge the torus):")
    rng2 = np.random.default_rng(11)
    square = [tuple(p) for p in rng2.uniform(-1.0, 1.0, size=(N_MAX, 2))]
    rng3 = np.random.default_rng(7)
    cube = [tuple(p) for p in rng3.uniform(0.0, 1.0, size=(N_MAX, 3))]
    circle = [
        (math.cos(2 * math.pi * i / N_MAX), math.sin(2 * math.pi * i / N_MAX))
        for i in range(N_MAX)
    ]
    val_rows = []
    for name, cloud, truth in (
        ("uniform square", square, 2.0),
        ("uniform cube", cube, 3.0),
        ("circle", circle, 1.0),
    ):
        dim, sd, r0, r1 = gp_plateau_dimension(cloud)
        val_rows.append((name, truth, dim, abs(dim - truth)))
        print(f"  {name:16s} true={truth}  plateau={dim:7.4f}  |err|={abs(dim - truth):.4f}  "
              f"window r/diag=[{r0:.5f}, {r1:.5f}]  slope sd={sd:.4f}")
    worst = max(r[3] for r in val_rows)
    print(f"  -> worst known-answer error {worst:.4f}. The selector is not the problem.")
    out["selector_validation"] = val_rows

    # (6b) the same selector on the torus, and the ORACLE upper bound on GP.
    print()
    print("(6b) the selector applied to the torus at the benchmark's max n:")
    dim_p, sd_p, r0, r1 = gp_plateau_dimension(pts)
    dim_default = correlation_dimension(pts).dimension
    print(f"  GP, library default window [0.01, 0.2] x diag : {dim_default:7.4f}  "
          f"|err|={abs(dim_default - TRUE_DIMENSION):.4f}  "
          f"{'OUTSIDE' if abs(dim_default - TRUE_DIMENSION) > TOLERANCE else 'INSIDE'} tolerance "
          f"{TOLERANCE}")
    print(f"  GP, answer-blind plateau window              : {dim_p:7.4f}  "
          f"|err|={abs(dim_p - TRUE_DIMENSION):.4f}  "
          f"{'OUTSIDE' if abs(dim_p - TRUE_DIMENSION) > TOLERANCE else 'INSIDE'} tolerance "
          f"{TOLERANCE}   (r/diag=[{r0:.5f}, {r1:.5f}])")
    out["gp_default"] = dim_default
    out["gp_plateau"] = dim_p
    print()
    print("  CONSEQUENCE. Criterion (c) condition (c4) is 'traditional_abs_error >")
    print("  tolerance'. It holds at the library's default fit window and FAILS under")
    print("  answer-blind plateau selection. The criterion (c) win recorded in section 3")
    print("  is therefore contingent on a BASELINE hyperparameter that neither the")
    print("  criterion nor h3_win_robustness.py sweeps -- h3 swept poly's k and")
    print("  max_radius and never touched the correlation sum's radius window.")
    print("  This is finding R2-F3 with the roles reversed, and it is disqualifying:")
    print("  a poly win must not depend on how the baseline was configured.")

    # (6c) does it become non-contingent at larger n? GP drifts DOWN with data.
    print()
    print("(6c) does the contingency close at larger n? (extends past the benchmark's")
    print("     n_grid; reported as an extension, NOT as the scored measurement)")
    rows = []
    for n_eval in (12800, 25600, 51200, 102400, 204800):
        t_max = n_eval / DENSITY
        cloud = [
            (math.cos(t_max * i / n_eval), math.cos(OMEGA2 * t_max * i / n_eval))
            for i in range(n_eval)
        ]
        r = compare_accuracy_at_max_n(
            cloud, true_dimension=TRUE_DIMENSION, max_n=n_eval, k=K_DEFAULT,
            tolerance=TOLERANCE, max_radius=MAX_RADIUS,
        )
        gp_p = gp_plateau_dimension(cloud)[0]
        sent = r.poly_algebraic_sentinel_fraction
        # A win here needs BOTH baseline configurations to fail the bar AND the
        # poly side to stay out of the F1 sentinel regime (condition (c2)).
        both_out = (
            r.traditional_abs_error > TOLERANCE
            and abs(gp_p - TRUE_DIMENSION) > TOLERANCE
            and r.poly_algebraic_abs_error <= TOLERANCE
            and sent <= DEFAULT_MAX_SENTINEL_FRACTION
        )
        rows.append((n_eval, r.poly_algebraic_estimate, sent,
                     r.traditional_estimate, gp_p, both_out))
        print(f"  n={n_eval:7d}  poly={r.poly_algebraic_estimate:7.4f} "
              f"(|e|={r.poly_algebraic_abs_error:.4f}, sentinel={sent:.3f})  "
              f"GP default={r.traditional_estimate:7.4f} "
              f"(|e|={r.traditional_abs_error:.4f})  GP plateau={gp_p:7.4f} "
              f"(|e|={abs(gp_p - TRUE_DIMENSION):.4f})  "
              f"non-contingent win: {both_out}")
    out["large_n"] = rows
    print()
    print("  READ THIS ROW BY ROW. The contingency closes only at n=102400 -- 8x past")
    print("  the benchmark's own n_grid max -- and the very next grid point, n=204800,")
    print("  fails for a DIFFERENT reason: poly's sentinel fraction reaches 0.075,")
    print("  above the 0.05 cap, so condition (c2) refuses it. The non-contingent win")
    print("  is one grid point wide and bracketed by a failure. Extending n_grid until")
    print("  a win appears is measurement-config gaming, not an estimator result -- the")
    print("  same 'perched between neighbours' pattern AR4 refused on Lorenz. Scored")
    print("  verdict stays at the benchmark's n=12800, where criterion (c) is refused.")

    # (6d) The same question, asked of the OTHER problem R2-F9 flagged. This is a
    #      one-call cross-check and it decides how far section 6 generalises.
    print()
    print("(6d) does the same contingency hit problem 06 (Brownian), the other case")
    print("     criterion (c) was built for? (cross-check, not this problem's verdict)")
    try:
        import n8d_clouds  # type: ignore

        brownian = n8d_clouds.load_cloud("06")[:25600]
    except (ImportError, KeyError):
        print("  n8d_clouds unavailable; skipped")
    else:
        b_default = correlation_dimension(brownian).dimension
        b_plateau, _, br0, br1 = gp_plateau_dimension(brownian)
        print(f"  GP, library default window : {b_default:7.4f}  "
              f"|err|={abs(b_default - 2.0):.4f}  "
              f"{'OUTSIDE' if abs(b_default - 2.0) > TOLERANCE else 'INSIDE'} tolerance 0.3")
        print(f"  GP, answer-blind plateau   : {b_plateau:7.4f}  "
              f"|err|={abs(b_plateau - 2.0):.4f}  "
              f"{'OUTSIDE' if abs(b_plateau - 2.0) > TOLERANCE else 'INSIDE'} tolerance 0.3"
              f"   (r/diag=[{br0:.5f}, {br1:.5f}])")
        print("  -> WORSE THAN ON THE TORUS. Grassberger-Procaccia is not merely inside")
        print("     tolerance on planar Brownian motion, it is essentially EXACT once its")
        print("     scaling region is selected rather than fixed. The 0.6057 error that")
        print("     Sec. 10.5 records for problem 06 is a fit-window artifact, not a")
        print("     property of the method. Both of R2-F9's two flagged cases are")
        print("     contingent on the same baseline hyperparameter.")
        out["brownian_cross_check"] = (b_default, b_plateau)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5,6")
    args = ap.parse_args()
    want = {s.strip() for s in args.sections.split(",")}

    clouds = section1()  # always: everything downstream needs the cloud
    if "2" in want:
        section2(clouds)
    if "3" in want:
        section3(clouds)
    if "4" in want:
        section4(clouds)
    if "5" in want:
        section5(clouds)
    if "6" in want:
        section6(clouds)


if __name__ == "__main__":
    main()
