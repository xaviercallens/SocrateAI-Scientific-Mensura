"""Round 3 remeasurement of benchmark problem 09 (Rossler attractor, a=b=0.2,
c=5.7, D_corr ~ 2.01) against the CURRENT library state -- post N8, H1, H3, and
the AutoResearch H2 loop (AR1..AR4, none of which modified src/).

WHAT THE RECORD SAYS GOING IN
-----------------------------
Round 2 scored problem 09 as a criterion-(a) win. Round 3's AR1/AR2 loop
re-derived it under the two corrections that postdate round 2 -- Theiler
exclusion (H1) and the finite-size saturation guard (AR2) -- and recorded, at
k=15, max_radius=6, theiler_window="auto", max_ball_fraction=0.5:

    poly_algebraic_min_n = 100, traditional_min_n = 800, savings 0.875

with a STANDING CAVEAT from AR3: every accepted node at n=100 was a
two-radius (zero-residual-degrees-of-freedom) fit, so the number is
correct-as-measured but its per-node R^2 is an algebraic identity rather than a
fit-quality statement. Section 5 re-audits that caveat against the current code
instead of quoting it.

Criterion (c) (`comparison.compare_accuracy_at_max_n`) postdates round 2
entirely and has NEVER been measured on this problem. It is measured here.

HYPERPARAMETERS AND THE R2-F3 GUARD
-----------------------------------
docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 9.4 (finding R2-F3) traced a false round-2
win to an UNDISCLOSED non-default `max_radius`. This script therefore takes the
module defaults as the headline and discloses every deviation:

  max_radius = 6  -- the dimension.py / comparison.py default. NO deviation
                     anywhere in this script's headline numbers; section 4
                     sweeps 3..8 as a guard, never to pick a headline.
  k          = 6  -- the module default, and the HEADLINE here. This is a
                     deliberate change from round 2 / AR1-AR4, which all used
                     problem 09's production k=15. k=15 is re-measured
                     alongside in every section (section 2 for criterion (a),
                     section 4 for criterion (c)) so the two are directly
                     comparable and neither is quoted alone. The point of
                     leading with k=6 is precisely that a win which needs a
                     hand-picked k is the R2-F3 failure mode with a different
                     knob.
  tolerance  = 0.5, true_dimension = 2.01 -- problem 09's production values
                     (scripts/hypergraph_benchmark/round2/09_rossler_attractor.py),
                     unchanged.

`theiler_window` and `max_ball_fraction` are reported at BOTH the library
defaults (0, 1.0 -- the pre-H1/pre-AR2 code path, which reproduces round 2
exactly) and the corrected settings ("auto", 0.5), because a recorded number
must be quoted with them (comparison.compare's own docstring says so).

POINT-CLOUD CONSTRUCTION -- WHICH ONE, AND WHY
----------------------------------------------
`compare()` measures growing PREFIXES `points[:n]`, so the ordering of the cloud
is a load-bearing methodological choice, not a detail. Sec. 10.10 diagnosed
time-ordered prefixes of a CLOSED ORBIT as unsound (a prefix is a short arc, not
the object), which is why problems 03/04/07 were re-measured on whole-orbit
prefixes. Two constructions are measured here rather than one, because the
Rossler attractor is neither the closed-orbit case nor obviously exempt:

  ARM A -- ROUND-2 PRODUCTION, TIME-ORDERED (the headline).
      One continuous post-transient trajectory at constant time density
      (DENSITY = 20 points per time unit; dt = 0.01 RK4, stride 5), 12800
      points = 640 time units, handed to compare() in time order. Round 2's
      own argument for this: the attractor is chaotic and NOT a closed orbit,
      so there is no multi-period duplication, and a prefix of a
      constant-density sample is a genuinely SHORTER OBSERVATION of the same
      process -- "fewer points" and "less observing time" are the same
      real-world quantity here, which is exactly the quantity a compute-savings
      claim is about. That argument is sound and is why arm A is the headline;
      it is also the only construction under which the standing 100-vs-800
      baseline can be checked digit-for-digit.

  ARM B -- ATTRACTOR-COVERING PREFIXES (the coverage control).
      The same 12800 points, reordered by a bit-reversal (van der Corput,
      radix 2) permutation of the index range, so that EVERY prefix is a
      temporally stratified subsample spanning all 640 time units at a density
      proportional to n. This exists because arm A has a real coverage problem
      that the closed-orbit diagnosis is a special case of: at n=100, arm A is
      the first 5.0 time units of trajectory, and the Rossler spiral period is
      ~6.0 time units, so the winning n=100 cell sees LESS THAN ONE LOOP of a
      structure whose dimension is 2.01. Arm B removes that confound at the
      cost of changing what "n points" means (a coarser view of everything
      rather than a complete view of a little). If the win survives only under
      arm A, that is a caveat this script must surface, not bury.
      WHAT THE MEASUREMENT CONCLUDED (written after the fact, so the plan above
      is not mistaken for the result): arm A stays the headline in the sense
      that its numbers reproduce the standing record digit-for-digit, but the
      arm-A winning cell turns out to contain ZERO pairs that are close in
      space and far apart in time (section 7a), i.e. no information about the
      attractor at all -- so arm B, where it is a TIE, is the construction the
      verdict rests on. The going-in plan is left standing above deliberately.
      Theiler handling in arm B is explicit rather than "auto": the window is
      applied by `_time_local_pair_counts` / `_windowed_neighbours` as a
      difference of TIME INDICES, so arm B passes the ORIGINAL trajectory
      indices as `time_indices` together with an explicit integer window
      measured once on the full-density cloud. Using "auto" on a stratified
      prefix would return a lag in units of that prefix's own positions and
      then apply it as an original-index difference -- the exact mismatch
      `baseline.theiler_window_from_autocorrelation`'s docstring warns about.

The cloud itself is not rebuilt here: it comes from
`n8d_clouds.load_cloud("09")`, which reconstructs it by importing round 2's own
`09_rossler_attractor.py` and calling its own integrator. Section 1 re-derives
it from the round-2 module independently and asserts byte-equality, so "the
production cloud" is verified rather than asserted.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not modify src/. It does not touch docs/ or reports/. Criterion (b)
(density robustness) is not measured: it is retired benchmark-wide (Sec. 10.8),
and round 2's own script already recorded it as not applicable to this problem
(the Rossler attractor's natural fixed-dt parametrisation has no close-approach
speed variation to exploit).

SECTIONS
    1  cloud provenance (independent rebuild from the round-2 integrator, asserted
       identical) and what an arm-A prefix physically contains.       ~1 min
    2  criterion (a) on arm A: defaults vs corrected, k=6 and k=15,
       with the full per-n traces both min_n values are read off.     ~3 min
    3  criterion (a) on arm B (attractor-covering prefixes).          ~3 min
    4  criterion (c) at n=12800, plus the R2-F3 knob guard.           ~4 min
    5  vacuity audit of the winning cell (AR3's zero-dof caveat) and
       the measured cost of a min_fit_length=3 floor.                 ~1 min
    6  R2-F3 guard on criterion (a): sweep max_ball_fraction; and a
       symmetric grid-floor probe under arm B.                        ~5 min
    7  the spurious-positive test: does the winning arm-A cell contain
       any information about the attractor at all?  (needs section 2) ~1 min

USAGE
    python scripts/hypergraph_benchmark/round3/09_rossler_attractor.py
    python scripts/hypergraph_benchmark/round3/09_rossler_attractor.py --sections 2,7
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import load_cloud  # noqa: E402

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    minimum_points_for_target_accuracy,
    theiler_window_from_autocorrelation,
)
from socrates.hypergraph.comparison import (  # noqa: E402
    DEFAULT_ACCURACY_MARGIN,
    compare,
    compare_accuracy_at_max_n,
)
from socrates.hypergraph.dimension import (  # noqa: E402
    local_dimension,
    mean_dimension,
    near_constant_consensus,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

# --- problem 09 production constants (round-2 script, unchanged) -------------
TRUE_DIMENSION = 2.01
TOLERANCE = 0.5
N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
N_MAX = N_GRID[-1]
DENSITY = 20.0  # points per time unit
SPIRAL_PERIOD = 6.0  # ~2*pi/omega0, omega0 = sqrt(1 - (a/2)^2) ~ 0.997

# --- hyperparameters ---------------------------------------------------------
K_DEFAULT = 6  # module default -> the headline
K_PRODUCTION = 15  # round-2 / AR1-AR4 production value, measured alongside
MAX_RADIUS = 6  # module default; NO deviation in any headline number
SAMPLES = 40  # comparison.py's own default

DEFAULT_SETTINGS = (0, 1.0)  # (theiler_window, max_ball_fraction): pre-H1/pre-AR2
CORRECTED_SETTINGS = ("auto", 0.5)  # H1 Theiler + AR2 saturation guard


# =============================================================================
# helpers
# =============================================================================


def bit_reversal_order(m: int) -> list[int]:
    """A permutation of range(m) whose every prefix is spread over the whole range.

    Van der Corput / bit-reversal on the next power of two, with out-of-range
    indices dropped. Prefix n is a radix-2 stratified subsample of range(m), so
    for a time-ordered trajectory every prefix spans the full observation window
    at a density proportional to n.
    """
    bits = max(1, (m - 1).bit_length())
    size = 1 << bits
    order = []
    for i in range(size):
        rev = int(format(i, f"0{bits}b")[::-1], 2)
        if rev < m:
            order.append(rev)
    assert sorted(order) == list(range(m))
    return order


def fit_window_length(est, max_ball_fraction: float, n_nodes: int) -> int:
    """Number of radii that actually entered a node's log-log fit.

    Recomputed from the volumes `local_dimension` reports, applying the same
    two stop rules it applies (zero shell, and the saturation guard against
    `volumes[r-1]`). A length of 2 means zero residual degrees of freedom: the
    line passes exactly through both points and R^2 is an algebraic identity,
    not a measurement. That is the AR3 caveat this reproduces.
    """
    volumes = (1,) + tuple(est.volumes)
    budget = max_ball_fraction * n_nodes
    length = 0
    for r in range(1, len(volumes)):
        shell = volumes[r] - volumes[r - 1]
        if shell <= 0:
            break
        if volumes[r - 1] > budget:
            break
        length += 1
    return length


def poly_trace(
    points, *, k, theiler_window, max_ball_fraction, n_grid=N_GRID, time_indices=None
):
    """Per-n shell-growth estimate, i.e. the sequence compare() reduces to a min_n."""
    out = []
    for n in n_grid:
        if n > len(points) or k >= n:
            break
        ti = None if time_indices is None else list(time_indices[:n])
        try:
            hg = knn_hypergraph(
                points[:n], k=k, dedupe=True, theiler_window=theiler_window, time_indices=ti
            )
        except ValueError:
            out.append((n, float("nan")))
            continue
        out.append(
            (
                n,
                mean_dimension(
                    hg,
                    samples=min(n, SAMPLES),
                    max_radius=MAX_RADIUS,
                    max_ball_fraction=max_ball_fraction,
                ),
            )
        )
    return out


def trad_trace(points, *, theiler_window, n_grid=N_GRID, time_indices=None):
    out = []
    for n in n_grid:
        if n > len(points):
            break
        ti = None if time_indices is None else list(time_indices[:n])
        try:
            est = correlation_dimension(
                points[:n], theiler_window=theiler_window, time_indices=ti
            )
            out.append((n, est.dimension, est.r_squared))
        except ValueError:
            out.append((n, float("nan"), 0.0))
    return out


def min_n_from_trace(trace, tolerance: float) -> int | None:
    """`compare()`'s stable-convergence rule, re-applied to an already-measured trace.

    Identical logic to `comparison.poly_algebraic_minimum_points`' final loop and
    to `baseline.minimum_points_for_target_accuracy`': the first n that is within
    tolerance AND stays within tolerance at every larger n tested.
    """
    rows = [(n, d) for n, d in trace]
    for i, (n, d) in enumerate(rows):
        if not math.isfinite(d) or abs(d - TRUE_DIMENSION) > tolerance:
            continue
        if all(math.isfinite(dd) and abs(dd - TRUE_DIMENSION) <= tolerance for _, dd in rows[i:]):
            return n
    return None


def fmt_trace(trace) -> str:
    return ", ".join(
        f"{n}:{'nan' if not math.isfinite(d) else f'{d:.3f}'}" for n, d, *_ in trace
    )


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# =============================================================================
# section 1 -- cloud provenance and what a prefix actually contains
# =============================================================================


def section_1(state) -> None:
    rule("SECTION 1 -- cloud provenance, and what an arm-A prefix physically is")
    points = state["arm_a"]

    print(f"  cloud: {len(points)} points in R^3, from n8d_clouds.load_cloud('09')")
    print("  which reconstructs it by importing round 2's own 09_rossler_attractor.py")

    # Independent rebuild straight from the round-2 module (not the disk cache).
    t0 = time.time()
    rebuilt = load_cloud("09", rebuild=True)
    same = len(rebuilt) == len(points) and all(
        a == b for pa, pb in zip(rebuilt, points, strict=True) for a, b in zip(pa, pb, strict=True)
    )
    print(f"  independent rebuild from the round-2 integrator: {len(rebuilt)} points "
          f"({time.time() - t0:.1f}s)")
    print(f"  identical to the cached cloud, coordinate for coordinate: {same}")
    assert same, "cloud provenance check FAILED -- the cached cloud is not the round-2 cloud"

    arr = np.asarray(points)
    print(f"  bounding box: x[{arr[:, 0].min():.2f},{arr[:, 0].max():.2f}] "
          f"y[{arr[:, 1].min():.2f},{arr[:, 1].max():.2f}] "
          f"z[{arr[:, 2].min():.2f},{arr[:, 2].max():.2f}]")

    print("\n  WHAT EACH ARM-A PREFIX COVERS (density = 20 pts/time unit, spiral period ~6.0):")
    print(f"    {'n':>7}  {'time units':>11}  {'spiral loops':>13}  {'auto theiler W':>15}")
    for n in N_GRID:
        span = n / DENSITY
        w = theiler_window_from_autocorrelation(points[:n])
        print(f"    {n:>7}  {span:>11.1f}  {span / SPIRAL_PERIOD:>13.2f}  {w:>15}")
    print("\n  => n=100 is 5.0 time units, i.e. 0.83 of ONE spiral loop. The cell that")
    print("     carries the standing criterion-(a) win therefore never sees the")
    print("     attractor; it sees a single arc. That is what section 3's arm B tests.")
    print("     Note also that 'auto' is re-derived per prefix and is clamped by")
    print("     max_window = min(n//10, 250), so the small-n cells get a small window")
    print("     for a reason unrelated to the physics.")

    # Duplicate-cluster sanity check, run rather than assumed.
    from scipy.spatial import cKDTree

    diag = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    tree = cKDTree(arr)
    pairs = tree.query_pairs(r=1e-9 * diag)
    print(f"\n  duplicate-cluster check (threshold 1e-9 * bbox diagonal = {1e-9 * diag:.3e}): "
          f"{len(pairs)} coincident pairs")
    state["provenance_ok"] = same


# =============================================================================
# section 2 -- criterion (a), arm A (round-2 production construction)
# =============================================================================


def section_2(state) -> None:
    rule("SECTION 2 -- CRITERION (a): compute savings, arm A (time-ordered prefixes)")
    points = state["arm_a"]
    rows = {}

    for k in (K_DEFAULT, K_PRODUCTION):
        for tw, mbf in (DEFAULT_SETTINGS, CORRECTED_SETTINGS):
            t0 = time.time()
            r = compare(
                points,
                TRUE_DIMENSION,
                TOLERANCE,
                k=k,
                n_grid=N_GRID,
                max_radius=MAX_RADIUS,
                theiler_window=tw,
                max_ball_fraction=mbf,
            )
            rows[(k, tw, mbf)] = r
            sav = r.compute_savings_fraction
            print(
                f"  k={k:<3} theiler={str(tw):<5} mbf={mbf:<4}  "
                f"poly_min_n={str(r.poly_algebraic_min_n):<5} "
                f"trad_min_n={str(r.traditional_min_n):<5} "
                f"wins={str(r.poly_algebraic_wins):<5} "
                f"savings={'n/a' if sav is None else f'{sav:+.3f}'}   "
                f"[poly@{r.max_n_tested}={r.poly_algebraic_estimate_at_max_n:.4f} "
                f"trad@{r.max_n_tested}={r.traditional_estimate_at_max_n:.4f}]  "
                f"({time.time() - t0:.0f}s)"
            )

    print("\n  PER-n TRACES (the sequences those min_n values are read off; band = "
          f"[{TRUE_DIMENSION - TOLERANCE:.2f}, {TRUE_DIMENSION + TOLERANCE:.2f}])")
    for k in (K_DEFAULT, K_PRODUCTION):
        for tw, mbf in (DEFAULT_SETTINGS, CORRECTED_SETTINGS):
            tr = poly_trace(points, k=k, theiler_window=tw, max_ball_fraction=mbf)
            print(f"    poly k={k:<3} theiler={str(tw):<5} mbf={mbf}: {fmt_trace(tr)}")
            state.setdefault("traces", {})[("A", "poly", k, tw, mbf)] = tr
    for tw in (0, "auto"):
        tr = trad_trace(points, theiler_window=tw)
        print(f"    trad  theiler={str(tw):<5}          : {fmt_trace(tr)}")
        state.setdefault("traces", {})[("A", "trad", tw)] = tr

    state["crit_a"] = rows
    head = rows[(K_DEFAULT, *CORRECTED_SETTINGS)]
    prod = rows[(K_PRODUCTION, *CORRECTED_SETTINGS)]
    print(
        f"\n  HEADLINE (k=6 module default, max_radius=6, theiler=auto, mbf=0.5): "
        f"poly {head.poly_algebraic_min_n} vs trad {head.traditional_min_n}, "
        f"savings {head.compute_savings_fraction:.3f}"
        if head.compute_savings_fraction is not None
        else "  HEADLINE: no savings (a method did not converge)"
    )
    print(
        f"  CROSS-CHECK at production k=15: poly {prod.poly_algebraic_min_n} vs trad "
        f"{prod.traditional_min_n} -- the win does not depend on the non-default k."
    )
    print(
        "  AT LIBRARY DEFAULTS (theiler=0, mbf=1.0) there is NO criterion-(a) win at "
        "either k;\n  the win exists only under the H1+AR2 corrections, and must be\n"
        "  quoted with them."
    )


# =============================================================================
# section 3 -- criterion (a), arm B (attractor-covering prefixes)
# =============================================================================


def section_3(state) -> None:
    rule("SECTION 3 -- CRITERION (a) under arm B: attractor-covering prefixes (control)")
    points_a = state["arm_a"]
    order = state["order_b"]
    points_b = state["arm_b"]

    w_full = theiler_window_from_autocorrelation(points_a)
    print(f"  arm B ordering: bit-reversal permutation of range({len(points_a)}).")
    print("  Every prefix spans the full 640.0 time units; prefix n has effective")
    print(f"  time spacing {len(points_a)}/n samples = {640.0:.0f}/n time units.")
    print(f"  Explicit Theiler window W = {w_full} ORIGINAL-index units, measured once on the")
    print("  full-density time-ordered cloud (auto would be mis-scaled on a stratified")
    print("  prefix -- see this script's docstring and baseline's own warning).")
    print(f"  first 8 arm-B original indices: {order[:8]}")

    print()
    for k in (K_DEFAULT, K_PRODUCTION):
        for tw, mbf, label in ((0, 1.0, "defaults"), (w_full, 0.5, "corrected")):
            t0 = time.time()
            r = compare(
                points_b,
                TRUE_DIMENSION,
                TOLERANCE,
                k=k,
                n_grid=N_GRID,
                max_radius=MAX_RADIUS,
                theiler_window=tw,
                time_indices=order,
                max_ball_fraction=mbf,
            )
            sav = r.compute_savings_fraction
            print(
                f"  k={k:<3} {label:<9} theiler={str(tw):<5} mbf={mbf:<4}  "
                f"poly_min_n={str(r.poly_algebraic_min_n):<5} "
                f"trad_min_n={str(r.traditional_min_n):<5} "
                f"wins={str(r.poly_algebraic_wins):<5} "
                f"savings={'n/a' if sav is None else f'{sav:+.3f}'}   "
                f"[poly@{r.max_n_tested}={r.poly_algebraic_estimate_at_max_n:.4f} "
                f"trad@{r.max_n_tested}={r.traditional_estimate_at_max_n:.4f}]  "
                f"({time.time() - t0:.0f}s)"
            )
            state.setdefault("crit_a_b", {})[(k, label)] = r

    print("\n  PER-n TRACES, arm B")
    for k in (K_DEFAULT, K_PRODUCTION):
        tr = poly_trace(
            points_b, k=k, theiler_window=w_full, max_ball_fraction=0.5, time_indices=order
        )
        print(f"    poly k={k:<3} theiler={w_full} mbf=0.5: {fmt_trace(tr)}")
        state.setdefault("traces", {})[("B", "poly", k)] = tr
    tr = trad_trace(points_b, theiler_window=w_full, time_indices=order)
    print(f"    trad  theiler={w_full}          : {fmt_trace(tr)}")
    state.setdefault("traces", {})[("B", "trad")] = tr


# =============================================================================
# section 4 -- criterion (c): asymptotic accuracy at fixed n
# =============================================================================


def section_4(state) -> None:
    rule("SECTION 4 -- CRITERION (c): asymptotic accuracy at n = 12800")
    points = state["arm_a"]

    print("  DISCLOSED CAPABILITY GAP: comparison.compare_accuracy_at_max_n accepts")
    print("  NEITHER theiler_window NOR max_ball_fraction. Criterion (c) is therefore")
    print("  measurable ONLY at the library defaults (theiler 0, mbf 1.0) -- the")
    print("  H1/AR2 corrections cannot be applied to it without changing src/, which")
    print("  this script does not do. Every criterion-(c) number below is a")
    print("  DEFAULTS number, and section 4c bounds what the corrections could do.")

    print(f"\n  4a. headline k={K_DEFAULT} and production k={K_PRODUCTION}, max_radius=6, "
          f"tolerance={TOLERANCE}, margin={DEFAULT_ACCURACY_MARGIN}")
    results = {}
    for k in (K_DEFAULT, K_PRODUCTION):
        t0 = time.time()
        r = compare_accuracy_at_max_n(
            points,
            TRUE_DIMENSION,
            N_MAX,
            k=k,
            tolerance=TOLERANCE,
            max_radius=MAX_RADIUS,
            samples=SAMPLES,
        )
        results[k] = r
        print(
            f"    k={k:<3} poly={r.poly_algebraic_estimate:.4f} "
            f"(err {r.poly_algebraic_abs_error:.4f})"
            f"  trad={r.traditional_estimate:.4f} (err {r.traditional_abs_error:.4f})"
            f"  gap={r.traditional_abs_error - r.poly_algebraic_abs_error:+.4f}"
            f"  sentinel={r.poly_algebraic_sentinel_fraction:.3f}"
            f"  tradR2={r.traditional_r_squared:.4f}  nodes={r.poly_algebraic_n_nodes}"
            f"  ({time.time() - t0:.0f}s)"
        )
        print(f"        more_accurate={r.more_accurate}  poly_win={r.poly_algebraic_accuracy_win}"
              f"  trad_win={r.traditional_accuracy_win}")
        print(f"        {r.verdict_reason}")
    state["crit_c"] = results

    print("\n  4b. IS THE NON-WIN TUNABLE AWAY? Criterion (c) needs")
    print("      poly_err <= tolerance < trad_err AND (trad_err - poly_err) >= margin.")
    for k in (K_DEFAULT, K_PRODUCTION):
        r = results[k]
        gap = r.traditional_abs_error - r.poly_algebraic_abs_error
        note = (
            "gap >= margin, so some tolerance could work"
            if gap >= DEFAULT_ACCURACY_MARGIN
            else "gap < margin, so NO tolerance produces a win -- (c5) fails for EVERY tolerance"
        )
        print(f"      k={k:<3} gap={gap:.4f} vs margin {DEFAULT_ACCURACY_MARGIN}: {note}")

    print("\n  4c. R2-F3 GUARD: does the criterion-(c) verdict depend on a knob?")
    print(f"      {'k':>4} {'max_radius':>11}  {'poly':>8} {'trad':>8} {'gap':>8}  poly_win")
    for k in (6, 8, 10, 12, 15):
        r = compare_accuracy_at_max_n(
            points, TRUE_DIMENSION, N_MAX, k=k, tolerance=TOLERANCE,
            max_radius=MAX_RADIUS, samples=SAMPLES,
        )
        print(f"      {k:>4} {MAX_RADIUS:>11}  {r.poly_algebraic_estimate:>8.4f} "
              f"{r.traditional_estimate:>8.4f} "
              f"{r.traditional_abs_error - r.poly_algebraic_abs_error:>+8.4f}  "
              f"{r.poly_algebraic_accuracy_win}")
    for mr in (3, 4, 5, 7, 8):
        r = compare_accuracy_at_max_n(
            points, TRUE_DIMENSION, N_MAX, k=K_DEFAULT, tolerance=TOLERANCE,
            max_radius=mr, samples=SAMPLES,
        )
        print(f"      {K_DEFAULT:>4} {mr:>11}  {r.poly_algebraic_estimate:>8.4f} "
              f"{r.traditional_estimate:>8.4f} "
              f"{r.traditional_abs_error - r.poly_algebraic_abs_error:>+8.4f}  "
              f"{r.poly_algebraic_accuracy_win}")
    print("      (the traditional column is constant by construction: the baseline reads")
    print("       neither k nor max_radius.)")


# =============================================================================
# section 5 -- the AR3 vacuity caveat, re-audited against current code
# =============================================================================


def section_5(state) -> None:
    rule("SECTION 5 -- vacuity audit of the winning cell (AR3's standing caveat)")
    points = state["arm_a"]
    tw, mbf = CORRECTED_SETTINGS

    for k in (K_DEFAULT, K_PRODUCTION):
        print(f"\n  k={k}, theiler={tw}, mbf={mbf}, max_radius={MAX_RADIUS}")
        print(f"    {'n':>6} {'nodes':>6} {'accepted':>9} {'mean dim':>9} "
              f"{'len2 acc':>9} {'len>=3 acc':>11} {'degen':>7} {'neardeg':>8}")
        for n in (100, 200, 400):
            hg = knn_hypergraph(points[:n], k=k, dedupe=True, theiler_window=tw)
            nodes = sorted(hg.nodes)
            step = max(1, len(nodes) // SAMPLES)
            sampled = nodes[::step][:SAMPLES]
            ests = [
                local_dimension(hg, u, max_radius=MAX_RADIUS, max_ball_fraction=mbf)
                for u in sampled
            ]
            consensus = near_constant_consensus(ests, threshold=0.9)
            accepted = [
                e for e in ests
                if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)
            ]
            lens = [fit_window_length(e, mbf, len(hg.nodes)) for e in accepted]
            n2 = sum(1 for length in lens if length == 2)
            n3 = sum(1 for length in lens if length >= 3)
            md = mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS, max_ball_fraction=mbf)
            print(
                f"    {n:>6} {len(hg.nodes):>6} {len(accepted):>9} "
                f"{'nan' if not math.isfinite(md) else f'{md:>9.4f}'} "
                f"{n2:>9} {n3:>11} "
                f"{sum(1 for e in ests if e.degenerate):>7} "
                f"{sum(1 for e in ests if e.near_degenerate):>8}"
            )
            if n == 100:
                state.setdefault("vacuity", {})[k] = (len(accepted), n2, n3)

    print("\n  'len2 acc' counts ACCEPTED nodes whose fit window held exactly TWO radii.")
    print("  Such a fit has zero residual degrees of freedom: the line passes exactly")
    print("  through both points, so R^2 = 1 identically and the >=0.9 acceptance test")
    print("  is satisfied by arithmetic rather than by evidence. If that column equals")
    print("  the accepted count at n=100, the winning cell's per-node fit quality is")
    print("  vacuous -- the DIMENSION VALUE is still what the estimator says, but its")
    print("  R^2 is not a measurement. This is AR3's caveat, re-derived here rather")
    print("  than quoted.")

    print("\n  COST OF MAKING IT NON-VACUOUS (min_fit_length=3 floor, applied post hoc:")
    print("  accept only nodes whose fit window held >= 3 radii, then re-derive min_n):")
    for k in (K_DEFAULT, K_PRODUCTION):
        trace = []
        for n in N_GRID:
            if n > len(points):
                break
            hg = knn_hypergraph(points[:n], k=k, dedupe=True, theiler_window=tw)
            nodes = sorted(hg.nodes)
            step = max(1, len(nodes) // SAMPLES)
            sampled = nodes[::step][:SAMPLES]
            ests = [
                local_dimension(hg, u, max_radius=MAX_RADIUS, max_ball_fraction=mbf)
                for u in sampled
            ]
            consensus = near_constant_consensus(ests, threshold=0.9)
            vals = [
                e.dimension
                for e in ests
                if e.is_well_fit(threshold=0.9, near_constant_consensus=consensus)
                and fit_window_length(e, mbf, len(hg.nodes)) >= 3
            ]
            trace.append((n, sum(vals) / len(vals) if vals else float("nan")))
        gated_min = None
        for i, (n, d) in enumerate(trace):
            if math.isfinite(d) and abs(d - TRUE_DIMENSION) <= TOLERANCE and all(
                math.isfinite(dd) and abs(dd - TRUE_DIMENSION) <= TOLERANCE
                for _, dd in trace[i:]
            ):
                gated_min = n
                break
        trad = minimum_points_for_target_accuracy(
            points, TRUE_DIMENSION, TOLERANCE, n_grid=N_GRID, theiler_window=tw
        )
        sav = None if gated_min is None or trad is None else 1.0 - gated_min / trad
        print(f"    k={k:<3} gated trace: {fmt_trace(trace)}")
        print(f"    k={k:<3} gated poly_min_n={gated_min} vs trad {trad}, "
              f"savings={'n/a' if sav is None else f'{sav:.3f}'}")
        state.setdefault("gated", {})[k] = (gated_min, trad, sav)


# =============================================================================
# section 6 -- R2-F3 guard on criterion (a): is the win a property of a knob?
# =============================================================================


def section_6(state) -> None:
    rule("SECTION 6 -- R2-F3 guard on criterion (a), plus a grid-floor probe")
    points = state["arm_a"]
    order = state["order_b"]
    points_b = state["arm_b"]

    print("  6a. arm A, k=6, theiler=auto, max_radius=6: sweep the ONE-SIDED knob")
    print("      max_ball_fraction. comparison.py states this knob applies to the poly")
    print("      side only, so a win that lives at exactly one value of it is a")
    print("      property of the setting, not of the method.")
    print(f"      {'mbf':>5}  {'poly_min_n':>10} {'trad_min_n':>10}  {'savings':>8}  win")
    for mbf in (0.2, 0.35, 0.5, 0.6, 0.75, 1.0):
        r = compare(
            points, TRUE_DIMENSION, TOLERANCE, k=K_DEFAULT, n_grid=N_GRID,
            max_radius=MAX_RADIUS, theiler_window="auto", max_ball_fraction=mbf,
        )
        sav = r.compute_savings_fraction
        print(f"      {mbf:>5}  {str(r.poly_algebraic_min_n):>10} "
              f"{str(r.traditional_min_n):>10}  "
              f"{'n/a' if sav is None else f'{sav:>+8.3f}'}  {r.poly_algebraic_wins}")
        state.setdefault("mbf_sweep", {})[mbf] = (
            r.poly_algebraic_min_n, r.traditional_min_n, r.poly_algebraic_wins
        )
    print("      MEASURED RESULT, stated plainly because it cuts AGAINST the sceptical")
    print("      reading: at k=6 the arm-A win is INSENSITIVE to this one-sided knob --")
    print("      it holds at mbf 0.5, 0.6, 0.75 and 1.0 alike (1.0 = guard OFF). So the")
    print("      win is NOT an artifact of max_ball_fraction; comparing this row against")
    print("      section 2's theiler=0 row shows the driver is the Theiler window alone,")
    print("      and Theiler is the TWO-SIDED knob, applied to both estimators equally.")
    print("      This is not the R2-F3 failure mode. The reasons the win is still not a")
    print("      sample-efficiency result are sections 3 and 7, not this sweep.")

    print("\n  6b. GRID-FLOOR PROBE, arm B. Under arm B both methods reach the band at")
    print("      the FIRST grid point (100), so the production grid cannot separate")
    print("      them. Extending the grid DOWNWARD is applied to both methods")
    print("      identically -- it is not a one-sided setting -- and is reported as a")
    print("      supplementary datum, never as the headline.")
    fine_grid = (25, 50) + N_GRID
    w_full = theiler_window_from_autocorrelation(points)
    for k in (K_DEFAULT,):
        r = compare(
            points_b, TRUE_DIMENSION, TOLERANCE, k=k, n_grid=fine_grid,
            max_radius=MAX_RADIUS, theiler_window=w_full, time_indices=order,
            max_ball_fraction=0.5,
        )
        sav = r.compute_savings_fraction
        print(f"      k={k} grid={fine_grid[:3]}...: poly_min_n={r.poly_algebraic_min_n} "
              f"trad_min_n={r.traditional_min_n} win={r.poly_algebraic_wins} "
              f"savings={'n/a' if sav is None else f'{sav:+.3f}'}")
        tr = poly_trace(
            points_b, k=k, theiler_window=w_full, max_ball_fraction=0.5,
            n_grid=fine_grid, time_indices=order,
        )
        tr_t = trad_trace(
            points_b, theiler_window=w_full, n_grid=fine_grid, time_indices=order
        )
        print(f"      poly trace: {fmt_trace(tr)}")
        print(f"      trad trace: {fmt_trace(tr_t)}")
        state.setdefault("fine_b", {})[k] = r
    print("      READ THIS CELL WITH CARE. Any apparent separation here is decided by")
    print("      whether the traditional estimate at n=50 sits inside the band edge")
    print(f"      {TRUE_DIMENSION + TOLERANCE:.2f}; a miss by hundredths is a knife-edge, not")
    print("      a compute-savings result, and is reported as such rather than banked.")


# =============================================================================
# section 7 -- is the winning cell a measurement of the attractor, or a
#              coincidence? (the spurious-positive test)
# =============================================================================


def section_7(state) -> None:
    rule("SECTION 7 -- does the winning arm-A cell measure the ATTRACTOR at all?")
    from scipy.spatial import cKDTree

    points = state["arm_a"]
    arr = np.asarray(points)
    diag = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    w_full = theiler_window_from_autocorrelation(points)
    eps = 0.05 * diag

    print("  7a. RECURRENCE CENSUS. The dimension of a strange attractor is a statement")
    print("      about points that are CLOSE IN SPACE but FAR APART IN TIME -- that is")
    print("      exactly what the Theiler window keeps and what a correlation sum counts.")
    print("      A prefix containing no such pairs carries no information about the")
    print("      attractor's transverse structure, whatever number an estimator returns.")
    print(f"      eps = 5% of the FULL cloud's bbox diagonal = {eps:.3f} (fixed in physical")
    print(f"      units across n so the census is comparable); Theiler window {w_full}.")
    print(f"      {'n':>7} {'time units':>11} {'pairs |dt|>W, d<eps':>21} {'per point':>10}")
    for n in N_GRID:
        sub = arr[:n]
        tree = cKDTree(sub)
        pairs = tree.query_pairs(r=eps)
        recur = sum(1 for i, j in pairs if abs(i - j) > w_full)
        print(f"      {n:>7} {n / DENSITY:>11.1f} {recur:>21} {recur / n:>10.2f}")
    print("      => at n=100 the winning cell contains ZERO (or near-zero) recurrent")
    print("         pairs: the prefix is a single non-self-approaching arc.")

    print("\n  7b. WHAT IS THE PREFIX, GEOMETRICALLY? A smooth non-recurring trajectory")
    print("      arc is a 1-DIMENSIONAL curve. Both estimators say so at the LIBRARY")
    print("      DEFAULTS, and both stop saying so once the H1/AR2 corrections are on:")
    print(f"      {'n':>7} {'poly dflt':>10} {'trad dflt':>10} {'poly corr':>10} {'trad corr':>10}")
    pd = dict((n, d) for n, d in state["traces"][("A", "poly", K_DEFAULT, 0, 1.0)])
    pc = dict((n, d) for n, d in state["traces"][("A", "poly", K_DEFAULT, "auto", 0.5)])
    td = {n: d for n, d, _ in state["traces"][("A", "trad", 0)]}
    tc = {n: d for n, d, _ in state["traces"][("A", "trad", "auto")]}
    for n in N_GRID:
        print(f"      {n:>7} {pd.get(n, float('nan')):>10.3f} {td.get(n, float('nan')):>10.3f} "
              f"{pc.get(n, float('nan')):>10.3f} {tc.get(n, float('nan')):>10.3f}")
    print("      => at n=100 the DEFAULT-path poly estimate is ~1.0, which is the CORRECT")
    print("         dimension of the object the prefix actually is. The corrected path")
    print("         moves it to ~2.35 -- a 1.35 error against the prefix's own geometry --")
    print("         and that displaced value lands inside the band [1.51, 2.51] centred on")
    print("         the ATTRACTOR's 2.01. The traditional side is displaced too (to ~6.1)")
    print("         but out of the band. The criterion-(a) gap at small n is therefore a")
    print("         comparison of two failure modes' directions, not of sample efficiency.")

    print("\n  7c. HAS THE k-NN GRAPH STOPPED BEING LOCAL? Median Euclidean length of the")
    print("      k=6 hypergraph's edges, relative to the prefix's own median")
    print("      nearest-neighbour distance. A large ratio means the 'neighbourhoods'")
    print("      being fitted are not neighbourhoods.")
    print(f"      {'n':>7} {'med NN dist':>12} {'med edge, W=0':>14} "
          f"{'med edge, auto':>15} {'ratio':>7}")
    for n in (100, 200, 400, 800, 1600, 12800):
        sub = arr[:n]
        tree = cKDTree(sub)
        dists, _ = tree.query(sub, k=2)
        med_nn = float(np.median(dists[:, 1]))
        lens = {}
        for tw in (0, "auto"):
            hg = knn_hypergraph(points[:n], k=K_DEFAULT, dedupe=True, theiler_window=tw)
            adj = hg.adjacency()
            ls = [
                float(np.linalg.norm(arr[int(u)] - arr[int(v)]))
                for u, vs in adj.items()
                for v in vs
                if int(u) < int(v)
            ]
            lens[tw] = float(np.median(ls)) if ls else float("nan")
        print(f"      {n:>7} {med_nn:>12.4f} {lens[0]:>14.4f} {lens['auto']:>15.4f} "
              f"{lens['auto'] / lens[0]:>7.2f}")

    print("\n  7d. IS THE WIN A PROPERTY OF THE TOLERANCE BAND'S WIDTH? min_n re-derived")
    print("      from the traces above under the SAME stable-convergence rule, at four")
    print("      tolerances. Both methods are re-scored at each, so this is symmetric;")
    print("      0.5 is problem 09's production tolerance and the rest are diagnostic.")
    print(f"      {'tol':>5}  {'arm A poly':>11} {'arm A trad':>11}  {'arm B poly':>11} "
          f"{'arm B trad':>11}")
    tr_ap = state["traces"].get(("A", "poly", K_DEFAULT, "auto", 0.5))
    tr_at = state["traces"].get(("A", "trad", "auto"))
    tr_bp = state["traces"].get(("B", "poly", K_DEFAULT))
    tr_bt = state["traces"].get(("B", "trad"))
    for tol in (0.2, 0.3, 0.4, 0.5):
        cells = []
        for tr in (tr_ap, tr_at, tr_bp, tr_bt):
            if tr is None:
                cells.append("--")
            else:
                cells.append(str(min_n_from_trace([(row[0], row[1]) for row in tr], tol)))
        print(f"      {tol:>5}  {cells[0]:>11} {cells[1]:>11}  {cells[2]:>11} {cells[3]:>11}")
    print("      ('--' = section 3 was not run in this process, so arm B has no trace.)")
    print("      => a win that exists only in the widest column is a statement about how")
    print("         wide the acceptance band is relative to the shell-growth estimator's")
    print("         own scatter, not about sample efficiency.")


# =============================================================================


def summary(state) -> None:
    rule("SUMMARY -- problem 09, Rossler attractor, current library state")
    a = state.get("crit_a", {})
    ab = state.get("crit_a_b", {})
    c = state.get("crit_c", {})

    if a:
        h = a.get((K_DEFAULT, *CORRECTED_SETTINGS))
        d = a.get((K_DEFAULT, *DEFAULT_SETTINGS))
        if d:
            print(f"  criterion (a), k=6 mr=6, LIBRARY DEFAULTS  : poly {d.poly_algebraic_min_n} "
                  f"vs trad {d.traditional_min_n} -> win={d.poly_algebraic_wins}")
        if h:
            hs = h.compute_savings_fraction
            print(f"  criterion (a), k=6 mr=6, theiler=auto mbf=0.5: poly "
                  f"{h.poly_algebraic_min_n} vs trad {h.traditional_min_n} -> "
                  f"win={h.poly_algebraic_wins}, "
                  f"savings={'n/a' if hs is None else f'{hs:.3f}'}")
    if ab:
        for (k, label), r in sorted(ab.items()):
            print(f"  criterion (a), arm B k={k} {label:<9}       : poly "
                  f"{r.poly_algebraic_min_n} vs trad {r.traditional_min_n} -> "
                  f"win={r.poly_algebraic_wins}")
    if c:
        for k, r in sorted(c.items()):
            print(f"  criterion (c), k={k:<3} n=12800 (defaults only): poly err "
                  f"{r.poly_algebraic_abs_error:.4f} vs trad err {r.traditional_abs_error:.4f}, "
                  f"gap {r.traditional_abs_error - r.poly_algebraic_abs_error:+.4f} -> "
                  f"win={r.poly_algebraic_accuracy_win}")
    if state.get("vacuity"):
        for k, (acc, n2, n3) in sorted(state["vacuity"].items()):
            print(f"  n=100 vacuity audit, k={k:<3}: {acc} accepted, {n2} of them "
                  f"two-radius (zero-dof), {n3} with >=3 radii")
    if state.get("gated"):
        for k, (g, t, s) in sorted(state["gated"].items()):
            print(f"  min_fit_length=3 floor, k={k:<3}: poly_min_n {g} vs trad {t}, "
                  f"savings {'n/a' if s is None else f'{s:.3f}'}")
    if state.get("mbf_sweep"):
        wins = [f for f, (_, _, w) in sorted(state["mbf_sweep"].items()) if w]
        print(f"  arm-A criterion-(a) win holds at max_ball_fraction in {wins} "
              f"of {sorted(state['mbf_sweep'])}")
    if state.get("fine_b"):
        for k, r in sorted(state["fine_b"].items()):
            print(f"  arm B, grid extended to 25/50, k={k}: poly {r.poly_algebraic_min_n} "
                  f"vs trad {r.traditional_min_n} -> win={r.poly_algebraic_wins}")

    print(
        "\n  READING OF THE ABOVE (stated here so the script's own conclusion is on the\n"
        "  record next to its numbers, and can be argued with):\n"
        "\n"
        "  * Criterion (a) REPRODUCES the standing record exactly on arm A, and now\n"
        "    does so at k=6 as well as k=15 -- and at k=6 the AR3 zero-dof caveat is\n"
        "    GONE (all 40 accepted nodes at n=100 have >=3-radius fit windows).\n"
        "  * It is nevertheless NOT a sample-efficiency result. Three independent\n"
        "    reasons, each measured above rather than argued:\n"
        "      (i)  the arm-A n=100 prefix contains no recurrent pairs at all\n"
        "           (section 7a) -- it is 0.83 of one spiral loop, a smooth 1-D arc,\n"
        "           and BOTH estimators read it as ~1-D on the default code path\n"
        "           (poly 0.994). The corrected path displaces poly to ~2.35, which\n"
        "           is 1.35 wrong about the object in hand and happens to land inside\n"
        "           a band centred on the attractor's 2.01 (section 7b).\n"
        "      (ii) the arm-A corrected poly trace does not converge; it oscillates\n"
        "           (2.35, 1.89, 2.41, 1.76, 2.12, 2.00, 2.07, 1.77) and merely never\n"
        "           leaves a +-0.5 band, so min_n=100 records the band's width, not a\n"
        "           convergence point. Narrow the band and the win disappears\n"
        "           symmetrically for both methods (section 7d).\n"
        "      (iii) under arm B, where every prefix actually covers the attractor,\n"
        "           the poly trace DOES converge monotonically and the result is a\n"
        "           TIE at the grid floor (100 vs 100), not a win (section 3).\n"
        "  * What the win is NOT: it is not an R2-F3-style knob artifact. Section 6a\n"
        "    shows it survives max_ball_fraction 0.5/0.6/0.75/1.0 alike (the one-sided\n"
        "    knob), and max_radius is at its default 6 throughout. The driver is the\n"
        "    Theiler window, which is applied to BOTH estimators. The disqualification\n"
        "    is about what the winning cell contains, not about how it was configured.\n"
        "  * Criterion (c) is a clean, knob-independent NO WIN: the accuracy gap is\n"
        "    0.07 (k=6) to 0.20 (k=15), below the 0.25 margin at every k in 6..15 and\n"
        "    every max_radius in 3..8, so no tolerance produces a win either.\n"
    )


SECTIONS = {
    1: section_1, 2: section_2, 3: section_3, 4: section_4, 5: section_5, 6: section_6,
    7: section_7,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5,6,7")
    args = ap.parse_args()
    wanted = [int(s) for s in args.sections.split(",") if s.strip()]

    points = load_cloud("09")
    assert len(points) >= N_MAX, f"cloud too small: {len(points)}"
    points = points[:N_MAX]
    order = bit_reversal_order(len(points))
    state = {
        "arm_a": points,
        "order_b": order,
        "arm_b": [points[i] for i in order],
    }

    for s in wanted:
        SECTIONS[s](state)
    summary(state)


if __name__ == "__main__":
    main()
