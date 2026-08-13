"""Round 3 re-measurement of benchmark problem 08 (Lorenz attractor).

WHAT THIS SCRIPT IS FOR
=======================
Re-measure problem 08 against the CURRENT library state (post-N8 near-constant
consensus gate, post-H1 Theiler window, post-H3 criterion (c), and after the
AutoResearch H2 loop, which kept no source change: dimension.py, comparison.py,
baseline.py and pointcloud.py are the AR2/AR3 end state).  Nothing below
modifies the library; every number is produced by calling it.

Both criteria are measured:

  criterion (a) -- compute savings.  `comparison.compare()`: the smallest n in
                   the grid at which each method is STABLY within tolerance of
                   the literature dimension.  `ComparisonResult.poly_algebraic_wins`
                   requires BOTH methods to converge in the grid AND poly's
                   min_n to be STRICTLY smaller; a tie is not a win, and a
                   baseline that never converges is not a win either (rule 2).
  criterion (c) -- asymptotic accuracy.  `comparison.compare_accuracy_at_max_n()`:
                   both estimators run once at the largest available n and
                   scored on |error| against the same tolerance, with the F1
                   degeneracy guard and the calibrated 0.25 margin.

HYPERPARAMETERS, AND THE ONE DISCLOSED DEVIATION (finding R2-F3)
================================================================
R2-F3 recorded a round-2 "win" that turned out to be contingent on an
UNDISCLOSED non-default `max_radius`.  So, explicitly:

  max_radius = 6   -- the module default (`dimension.local_dimension`,
                      `comparison.compare`).  NOT a deviation.  Every number
                      below uses 6 and nothing sweeps it except the disclosed
                      robustness sweep in section 5, which is reported in full
                      including the cells that disagree.

  k = 10           -- a DELIBERATE, DISCLOSED deviation from the module default
                      k=6.  Reason: 10 is problem 08's PRODUCTION setting, fixed
                      in round 1 and used by every recorded 08 number
                      (docs/POLY_ALGEBRAIC_BENCHMARK.md Sec. 9.4 table row 08,
                      round2/08_lorenz_attractor.py `K = 10`, and
                      round3/n8d_clouds.py `CloudSpec("08", ..., 10, 6, ...)`).
                      Re-measuring the baselines of record at a different k
                      would compare two different experiments.  The deviation is
                      neutralised rather than merely announced: section 2 runs
                      the ENTIRE criterion (a) measurement at k=6 as well, and
                      section 4 does the same for criterion (c), so the verdict
                      can be read off the module default alone if preferred.
                      Both k are reported whatever they say.

  samples = 40, n_radii = 20 -- library defaults, untouched.

THE POINT-CLOUD QUESTION, STATED UP FRONT
=========================================
`compare()` scores a method at n by handing it the PREFIX `points[:n]`, so the
ORDER of the cloud is part of the experiment, not a detail.  For a chaotic
attractor there are two defensible orders and they measure different things:

  (A) TIME-ORDERED (`--order time`).  The post-transient trajectory as
      integrated, dt=0.005.  This is the round-2 production construction and
      the construction behind every 08 number on the record (the AR1-AR4
      iterations all load it via `n8d_clouds.load_cloud("08")`).  Its prefix at
      n=100 is 0.5 time units of trajectory -- a short arc, NOT the attractor.
      docs Sec. 10.4 measured exactly this and called it an artifact: "a
      time-ordered prefix of a chaotic trajectory is a short smooth arc, and
      poly correctly calls it 1-dimensional ... That is right about the sample
      and wrong about the attractor."

  (B) BIT-REVERSAL COVERING (`--order bitrev`).  The same trajectory points,
      reordered by the van der Corput / bit-reversal permutation, so that every
      prefix is a low-discrepancy subsample of the WHOLE attractor.  This is the
      construction docs Sec. 10.4 endorses in that same sentence ("an argument
      for the whole-orbit-covering prefix constructions (bit-reversal, Weyl)")
      and the one whose uniform application moved the round-2 count from 2 to 4
      (Sec. 10.10 correction).  Time indices are carried alongside the reordered
      points and passed to BOTH estimators, so the Theiler window stays a
      statement about trajectory time and not about array position.

Neither is silently preferred.  Sections 2 and 3 run the identical criterion (a)
measurement under both; section 4 runs criterion (c) under both.  (B) is the
methodologically endorsed one and (A) is the one the recorded baselines used;
if they disagree, that disagreement is the finding and is reported as such.
The point cloud itself -- which points exist -- is IDENTICAL between the two;
only the order in which prefixes reveal them differs, and both estimators
always see the same prefix.

CORRECTION SETTINGS
===================
Two library capabilities are off by default and are reported both off and on,
because a recorded number must be quoted with them:

  theiler_window=0 / "auto"   H1.  Applied to BOTH estimators by one knob.
  max_ball_fraction=1.0 / 0.5 AR2 saturation guard, poly side only (1.0 = off).

"defaults" = (0, 1.0) -- the pre-correction code path, bit-for-bit.
"corrected" = ("auto", 0.5) -- the AutoResearch end-state setting.  0.5 is
`dimension.SATURATION_BALL_FRACTION`, held fixed, NOT recalibrated here.

KNOWN CAPABILITY GAP, MEASURED NOT ASSUMED
==========================================
`compare_accuracy_at_max_n` accepts NEITHER `theiler_window` NOR
`max_ball_fraction`.  Criterion (c) therefore has exactly one setting: defaults.
Section 4 reports that library number as THE criterion (c) result.  It also
reports, clearly labelled as NOT criterion (c) and NOT scoreable, a hand-rolled
corrected variant assembled from the same public functions, so the size of the
gap is on the record rather than guessed at.

RESULTS AS MEASURED (round 3, current library, this script)
===========================================================
CRITERION (a) -- NO WIN, in every cell run.  `poly_algebraic_wins` needs
poly_min_n STRICTLY below trad_min_n; it is never below and is sometimes above.

  ordering       settings    k    poly_min_n  trad_min_n  savings   win
  time-ordered   defaults    10      800         200      -3.000    False
  time-ordered   corrected   10      800         800       0.000    False
  time-ordered   defaults     6     1600         200      -7.000    False
  time-ordered   corrected    6      800         800       0.000    False
  bit-reversal   defaults    10      200         100      -1.000    False
  bit-reversal   corrected   10      100         100       0.000    False
  bit-reversal   defaults     6      200         100      -1.000    False
  bit-reversal   corrected    6      100         100       0.000    False

  Section 5 adds k in {6,8,10,12,15} and max_radius in {4,5,6,7,8}: 14 further
  cells, savings 0.000 in all 14, win False in all 14.  The verdict does not
  depend on a tuning knob in either direction.

  The time-ordered k=10 corrected row reproduces the recorded baseline of
  record digit-for-digit (poly per-n 2.318, 1.988, nan, 1.930, 2.157, 2.152,
  2.156, 2.126; poly 800 / trad 800, tie).

  WHY criterion (a) has no headroom left on this problem: under the endorsed
  bit-reversal ordering the BASELINE already converges at n=100, the first
  point of the grid.  Nothing poly can do beats the grid floor, so the ceiling
  on 08 is a tie, not a win, unless the grid itself is changed -- and changing
  the grid floor to manufacture a gap would be measurement-config gaming.

CRITERION (c) -- NO WIN, in every cell run, and not a near miss.
  ordering      k   poly     |err|    trad     |err|   gap     sentinel  win
  time-ordered  10  2.0375   0.0125   1.9429   0.1071  +0.0946  0.050    False
  time-ordered   6  1.9768   0.0732   1.9429   0.1071  +0.0339  0.125    False
  bit-reversal  10  2.0704   0.0204   1.9362   0.1138  +0.0934  0.000    False
  bit-reversal   6  2.0023   0.0477   1.9362   0.1138  +0.0661  0.000    False

  Poly is closer to 2.05 than the baseline in all four cells, and it is still
  not a win, for two independent reasons: (c4) fails because the BASELINE's
  error (0.107-0.114) is comfortably inside problem 08's own tolerance 0.5 --
  both methods answer this problem correctly at n=12800, so the difference is
  not decision-relevant -- and (c5) fails anyway because the gap (0.034-0.095)
  is well under the calibrated 0.25 margin, i.e. inside the two estimators'
  measured run-to-run spread.  The time-ordered k=6 cell fails (c2) first
  (sentinel 0.125 > 0.05) on top of that.

  This is not a tolerance that can be moved: (c3) needs poly_err <= tolerance
  and (c4) needs tolerance < trad_err, so a win requires the two errors to
  straddle by at least the margin.  Here they differ by at most 0.095.

WHAT THIS PROBLEM ACTUALLY SHOWS (worth recording even though it is not a win)
=============================================================================
The largest effect measured here is not an estimator effect at all -- it is the
point-cloud construction.  Switching from time-ordered to bit-reversal prefixes,
with the estimators and every hyperparameter untouched, moves poly's min_n from
800 to 100 and the baseline's from 200/800 to 100, and it removes the n=400 nan
that AR3 and AR4 spent two iterations proving was not a hop-count, parity or
fit-window artifact.  Both are correct: the nan is a real property of the
time-ordered n=400 PREFIX, which is a smooth single-lobe arc (section 1
measures it: 0 lobe switches and 100% of points on one lobe at n<=400), and it
simply is not present when the prefix samples the attractor.  The AR3/AR4
conclusion "Lorenz min_n=800 is a property of the data" should be read as a
property of that CLOUD ORDERING, not of the Lorenz attractor.

USAGE
    python scripts/hypergraph_benchmark/round3/08_lorenz_attractor.py --sections 1
    python scripts/hypergraph_benchmark/round3/08_lorenz_attractor.py --sections 2,3
    python scripts/hypergraph_benchmark/round3/08_lorenz_attractor.py --sections 4,5
Section 1 ~1 min, 2 ~8 min, 3 ~8 min, 4 ~3 min, 5 ~10 min.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import load_cloud  # noqa: E402

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    theiler_window_from_autocorrelation,
)
from socrates.hypergraph.comparison import compare, compare_accuracy_at_max_n  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    SATURATION_BALL_FRACTION,
    degenerate_fraction,
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

# ---- problem 08 production configuration (round 1 / round 2, unchanged) -----
K_PRODUCTION = 10  # disclosed deviation from the module default 6; see docstring
K_MODULE_DEFAULT = 6
MAX_RADIUS = 6  # module default; NOT a deviation
N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
TRUE_DIMENSION = 2.05  # Grassberger & Procaccia 1983, D_2 = 2.05 +- 0.01
TOLERANCE = 0.5
SAMPLES = 40

CORRECTED_BALL_FRACTION = SATURATION_BALL_FRACTION  # 0.5, held fixed, not tuned

CONFIGS = {
    "defaults": {"theiler_window": 0, "max_ball_fraction": 1.0},
    "corrected": {"theiler_window": "auto", "max_ball_fraction": CORRECTED_BALL_FRACTION},
}


# =============================================================================
# Point-cloud construction
# =============================================================================


def bit_reversal_order(size: int) -> list[int]:
    """Indices 0..size-1 in van der Corput / bit-reversal order.

    Reversal is over enough bits to cover `size`; values landing past the end
    are dropped, which keeps the sequence a permutation of range(size) for any
    size, not only powers of two.  The prefix-coverage property is verified
    empirically in `check_prefix_coverage` rather than assumed.
    """
    bits = max(1, (size - 1).bit_length())
    out = []
    for i in range(1 << bits):
        r = int(f"{i:0{bits}b}"[::-1], 2)
        if r < size:
            out.append(r)
    assert sorted(out) == list(range(size))
    return out


def check_prefix_coverage(order: list[int], size: int, n_grid: tuple[int, ...]) -> list[tuple]:
    """How well does each prefix cover the trajectory in TIME?

    Reported as the largest gap between consecutive time indices in the sorted
    prefix, normalised by the ideal even-stride gap size/n.  1.0 is a perfect
    even stride; the time-ordered cloud gives size/n (the whole tail is one gap).
    """
    rows = []
    for n in n_grid:
        if n > size:
            break
        idx = sorted(order[:n])
        gaps = [b - a for a, b in zip(idx, idx[1:], strict=False)] + [size - idx[-1] + idx[0]]
        rows.append((n, max(gaps), max(gaps) / (size / n), idx[-1]))
    return rows


def lorenz_clouds():
    """Return (time_ordered_points, bitrev_points, bitrev_time_indices).

    The points are `n8d_clouds.load_cloud("08")` verbatim -- the round-2
    production cloud, RK4 at dt=0.005, first 10 time units discarded as
    transient -- so the time-ordered arm reproduces the recorded baselines
    exactly.  The bitrev arm is a REORDERING of that same list: identical
    points, identical count, different prefix order.
    """
    points = load_cloud("08")
    order = bit_reversal_order(len(points))
    bitrev = [points[i] for i in order]
    return points, bitrev, order


# =============================================================================
# Section 1 -- provenance, sanity, and what the two orderings actually are
# =============================================================================


def section1() -> None:
    print("=" * 78)
    print("SECTION 1 -- cloud provenance and the two prefix orderings")
    print("=" * 78)

    points, bitrev, order = lorenz_clouds()
    arr = np.asarray(points)
    print("\nsource: n8d_clouds.load_cloud('08')  (round-2 production cloud)")
    print(f"  points: {len(points)}  dim: {arr.shape[1]}  dt=0.005, transient t<10 dropped")
    print(f"  x range [{arr[:,0].min():.3f}, {arr[:,0].max():.3f}]")
    print(f"  z range [{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]")

    # Both lobes visited, and the trajectory is not settling into one.
    lobe = np.sign(arr[:, 0])
    switches = int(np.sum(lobe[1:] != lobe[:-1]))
    frac_pos = float(np.mean(lobe > 0))
    print(f"  lobe switches: {switches}   lobe balance: {frac_pos*100:.1f}% / "
          f"{(1-frac_pos)*100:.1f}%  (a settled orbit would be ~100/0)")

    # Duplicate check at the largest grid point actually used.
    try:
        knn_hypergraph(points[: N_GRID[-1]], k=K_PRODUCTION, dedupe=False)
        print(f"  duplicate check at n={N_GRID[-1]}: none (dedupe=False did not raise)")
    except Exception as exc:  # noqa: BLE001
        print(f"  duplicate check at n={N_GRID[-1]}: {type(exc).__name__}: {exc}")

    print("\nTheiler window that 'auto' resolves to, per prefix (time-ordered):")
    for n in N_GRID:
        w_time = theiler_window_from_autocorrelation(points[:n])
        w_br = theiler_window_from_autocorrelation(bitrev[:n], time_indices=order[:n])
        print(f"  n={n:6d}   time-ordered W={w_time:4d}    bitrev (time_indices) W={w_br:4d}")

    print("\nPrefix coverage in trajectory time (max gap / ideal even-stride gap;")
    print("1.0 = perfect stride, large = the prefix is a short arc, not the attractor):")
    rows_t = check_prefix_coverage(list(range(len(points))), len(points), N_GRID)
    rows_b = check_prefix_coverage(order, len(points), N_GRID)
    print(f"  {'n':>6}  {'time-ordered':>14}  {'bitrev':>10}   {'time-ordered lobes':>20}"
          f"  {'bitrev lobes':>14}")
    for (n, _, ratio_t, _), (_, _, ratio_b, _) in zip(rows_t, rows_b, strict=True):
        lt = np.sign(np.asarray(points[:n])[:, 0])
        lb = np.sign(np.asarray(bitrev[:n])[:, 0])
        st = f"{int(np.sum(lt[1:] != lt[:-1]))} sw, {np.mean(lt>0)*100:.0f}% +"
        sb = f"{np.mean(lb>0)*100:.0f}% +"
        print(f"  {n:6d}  {ratio_t:14.1f}  {ratio_b:10.1f}   {st:>20}  {sb:>14}")

    print("\nREADING: the time-ordered prefix at n=100 spans 0.5 time units of a")
    print("single lobe -- it is a smooth arc, not a sample of the attractor.  The")
    print("bitrev prefix at the same n covers the whole trajectory (ratio ~1-2).")
    print("This is the docs Sec. 10.4 artifact, reproduced here as a measurement.")


# =============================================================================
# Criterion (a)
# =============================================================================


def poly_trace(points, time_indices, k, cfg):
    """Per-n shell-growth estimate -- the numbers behind compare()'s min_n."""
    out = []
    for n in N_GRID:
        if n > len(points) or k >= n:
            break
        ti = None if time_indices is None else time_indices[:n]
        try:
            hg = knn_hypergraph(
                points[:n], k=k, dedupe=True,
                theiler_window=cfg["theiler_window"], time_indices=ti,
            )
        except Exception:  # noqa: BLE001
            out.append((n, float("nan")))
            continue
        out.append((n, mean_dimension(
            hg, samples=min(n, SAMPLES), max_radius=MAX_RADIUS,
            max_ball_fraction=cfg["max_ball_fraction"],
        )))
    return out


def trad_trace(points, time_indices, cfg):
    out = []
    for n in N_GRID:
        if n > len(points):
            break
        ti = None if time_indices is None else time_indices[:n]
        try:
            est = correlation_dimension(
                points[:n], theiler_window=cfg["theiler_window"], time_indices=ti,
            )
            out.append((n, est.dimension))
        except Exception:  # noqa: BLE001
            out.append((n, float("nan")))
    return out


def _fmt_trace(trace) -> str:
    return "  ".join(
        f"{n}:{'nan' if not math.isfinite(d) else f'{d:.3f}'}" for n, d in trace
    )


def _in_band(d) -> str:
    if not math.isfinite(d):
        return " "
    return "*" if abs(d - TRUE_DIMENSION) <= TOLERANCE else " "


def criterion_a(points, time_indices, label: str) -> None:
    band = (TRUE_DIMENSION - TOLERANCE, TRUE_DIMENSION + TOLERANCE)
    print(f"\nordering: {label}   true={TRUE_DIMENSION}  tol={TOLERANCE}  "
          f"band=[{band[0]:.2f}, {band[1]:.2f}]  max_radius={MAX_RADIUS}")
    print(f"n_grid = {N_GRID}")

    for k in (K_PRODUCTION, K_MODULE_DEFAULT):
        tag = "PRODUCTION (disclosed deviation)" if k == K_PRODUCTION else "module default"
        for cname, cfg in CONFIGS.items():
            print(f"\n--- k={k} [{tag}], settings={cname} "
                  f"(theiler={cfg['theiler_window']!r}, "
                  f"max_ball_fraction={cfg['max_ball_fraction']}) ---")
            res = compare(
                points, TRUE_DIMENSION, TOLERANCE,
                k=k, n_grid=N_GRID, max_radius=MAX_RADIUS,
                theiler_window=cfg["theiler_window"],
                time_indices=time_indices,
                max_ball_fraction=cfg["max_ball_fraction"],
            )
            print(f"  compare(): poly_min_n={res.poly_algebraic_min_n}  "
                  f"trad_min_n={res.traditional_min_n}  "
                  f"savings={res.compute_savings_fraction}  "
                  f"WIN={res.poly_algebraic_wins}")
            print(f"  at max n={res.max_n_tested}: poly={res.poly_algebraic_estimate_at_max_n:.4f}"
                  f"  trad={res.traditional_estimate_at_max_n:.4f}")

            pt = poly_trace(points, time_indices, k, cfg)
            tt = trad_trace(points, time_indices, cfg)
            print(f"  poly per-n : {_fmt_trace(pt)}")
            print(f"  trad per-n : {_fmt_trace(tt)}")
            print("  in band    : poly " + "".join(f"{_in_band(d)}{n}" for n, d in pt))
            print("             : trad " + "".join(f"{_in_band(d)}{n}" for n, d in tt))


def section2() -> None:
    print("=" * 78)
    print("SECTION 2 -- criterion (a) on the TIME-ORDERED cloud (ordering of record)")
    print("=" * 78)
    points, _, _ = lorenz_clouds()
    criterion_a(points, None, "time-ordered (round-2 production, dt=0.005 prefixes)")


def section3() -> None:
    print("=" * 78)
    print("SECTION 3 -- criterion (a) on the BIT-REVERSAL cloud (Sec. 10.4-endorsed)")
    print("=" * 78)
    _, bitrev, order = lorenz_clouds()
    criterion_a(bitrev, order, "bit-reversal covering prefixes, time_indices carried")


# =============================================================================
# Section 4 -- criterion (c)
# =============================================================================


def _corrected_accuracy_variant(points, time_indices, k, max_n):
    """NOT criterion (c).  The same two estimates with the H1 window and the AR2
    saturation guard switched on, assembled from public functions because
    `compare_accuracy_at_max_n` cannot accept either.  Reported only to size the
    capability gap; it is not scoreable and is not banked.
    """
    prefix = points[:max_n]
    ti = None if time_indices is None else time_indices[:max_n]
    try:
        hg = knn_hypergraph(prefix, k=k, dedupe=True, theiler_window="auto", time_indices=ti)
        poly = mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS,
                              max_ball_fraction=CORRECTED_BALL_FRACTION)
        deg = degenerate_fraction(hg, samples=SAMPLES, max_radius=MAX_RADIUS,
                                  max_ball_fraction=CORRECTED_BALL_FRACTION)
        ndeg = near_degenerate_fraction(hg, samples=SAMPLES, max_radius=MAX_RADIUS,
                                        max_ball_fraction=CORRECTED_BALL_FRACTION)
    except Exception:  # noqa: BLE001
        poly = deg = ndeg = float("nan")
    trad = correlation_dimension(prefix, theiler_window="auto", time_indices=ti).dimension
    return poly, trad, deg, ndeg


def section4() -> None:
    print("=" * 78)
    print("SECTION 4 -- criterion (c): asymptotic accuracy at max n")
    print("=" * 78)
    points, bitrev, order = lorenz_clouds()
    max_n = N_GRID[-1]

    for label, pts, ti in (
        ("time-ordered", points, None),
        ("bit-reversal", bitrev, order),
    ):
        for k in (K_PRODUCTION, K_MODULE_DEFAULT):
            print(f"\n--- ordering={label}  k={k}  n={max_n}  tolerance={TOLERANCE} "
                  f"(problem 08's own bar)  margin=0.25 ---")
            res = compare_accuracy_at_max_n(
                pts, TRUE_DIMENSION, max_n, k=k, tolerance=TOLERANCE, max_radius=MAX_RADIUS,
            )
            print(f"  n_evaluated={res.n_evaluated}  poly_nodes={res.poly_algebraic_n_nodes}")
            print(f"  poly={res.poly_algebraic_estimate:.4f}  |err|="
                  f"{res.poly_algebraic_abs_error:.4f}   "
                  f"trad={res.traditional_estimate:.4f}  |err|="
                  f"{res.traditional_abs_error:.4f}  trad R2={res.traditional_r_squared:.4f}")
            print(f"  sentinel fraction={res.poly_algebraic_sentinel_fraction:.3f} "
                  f"(degen {res.poly_algebraic_degenerate_fraction:.3f} + near-degen "
                  f"{res.poly_algebraic_near_degenerate_fraction:.3f})")
            print(f"  gap (trad|err| - poly|err|) = "
                  f"{res.traditional_abs_error - res.poly_algebraic_abs_error:+.4f}")
            print(f"  more_accurate={res.more_accurate}   "
                  f"CRITERION (c) WIN={res.poly_algebraic_accuracy_win}   "
                  f"traditional_accuracy_win={res.traditional_accuracy_win}")
            print(f"  reason: {res.verdict_reason}")

            poly_c, trad_c, deg_c, ndeg_c = _corrected_accuracy_variant(pts, ti, k, max_n)
            print(f"  [NOT criterion (c) -- capability gap probe, theiler='auto' + "
                  f"max_ball_fraction={CORRECTED_BALL_FRACTION}]")
            print(f"      poly={poly_c:.4f} |err|={abs(poly_c - TRUE_DIMENSION):.4f}   "
                  f"trad={trad_c:.4f} |err|={abs(trad_c - TRUE_DIMENSION):.4f}   "
                  f"sentinel={deg_c + ndeg_c:.3f}")

    print("\nNOTE: with problem 08's stated tolerance 0.5, condition (c4) requires the")
    print("BASELINE error to exceed 0.5 at max n.  Whether it does is a measurement,")
    print("reported above.  The tolerance is not adjustable to manufacture a win --")
    print("(c3) and (c4) pull in opposite directions in it (see comparison.py).")


# =============================================================================
# Section 5 -- R2-F3 robustness: does any verdict depend on a tuning knob?
# =============================================================================


def section5() -> None:
    print("=" * 78)
    print("SECTION 5 -- R2-F3 robustness sweep (k and max_radius), all cells reported")
    print("=" * 78)
    points, bitrev, order = lorenz_clouds()

    cfg = CONFIGS["corrected"]

    print("\nPurpose: sections 2-4 report a LOSS.  This sweep asks the R2-F3")
    print("question in the only direction that matters for a loss -- is there a")
    print("knob setting that WOULD have won, which the reported cells happened to")
    print("miss?  Every cell run is printed, including the non-default max_radius")
    print("cells, which are context and are NOT banked under any circumstances.")

    for label, pts, ti in (("time-ordered", points, None), ("bit-reversal", bitrev, order)):
        print(f"\n--- ordering={label}  settings=corrected "
              f"(theiler='auto', max_ball_fraction={CORRECTED_BALL_FRACTION}) ---")
        print(f"  {'k':>4} {'max_radius':>11}  {'poly_min_n':>10} {'trad_min_n':>10} "
              f"{'savings':>8}  win")
        for k in (6, 8, 10, 12, 15):
            res = compare(
                pts, TRUE_DIMENSION, TOLERANCE, k=k, n_grid=N_GRID,
                max_radius=MAX_RADIUS, theiler_window=cfg["theiler_window"],
                time_indices=ti, max_ball_fraction=cfg["max_ball_fraction"],
            )
            sv = res.compute_savings_fraction
            print(f"  {k:>4} {MAX_RADIUS:>11}  {str(res.poly_algebraic_min_n):>10} "
                  f"{str(res.traditional_min_n):>10} "
                  f"{'n/a' if sv is None else f'{sv:.3f}':>8}  "
                  f"{res.poly_algebraic_wins}")

    print(f"\n--- ordering=bit-reversal  settings=corrected  max_radius sweep at "
          f"k={K_PRODUCTION} ---")
    print("    (NON-DEFAULT max_radius: context only, NOT banked -- this is the exact")
    print("     knob finding R2-F3 was about)")
    _, bitrev2, order2 = points, bitrev, order
    for mr in (4, 5, 7, 8):
        res = compare(
            bitrev2, TRUE_DIMENSION, TOLERANCE, k=K_PRODUCTION, n_grid=N_GRID,
            max_radius=mr, theiler_window=cfg["theiler_window"],
            time_indices=order2, max_ball_fraction=cfg["max_ball_fraction"],
        )
        sv = res.compute_savings_fraction
        print(f"  {K_PRODUCTION:>4} {mr:>11}  {str(res.poly_algebraic_min_n):>10} "
              f"{str(res.traditional_min_n):>10} "
              f"{'n/a' if sv is None else f'{sv:.3f}':>8}  {res.poly_algebraic_wins}")


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4, 5: section5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="1,2,3,4",
                    help="comma-separated section numbers (1-5)")
    args = ap.parse_args()
    for s in [int(x) for x in args.sections.split(",")]:
        SECTIONS[s]()
        print()


if __name__ == "__main__":
    main()
