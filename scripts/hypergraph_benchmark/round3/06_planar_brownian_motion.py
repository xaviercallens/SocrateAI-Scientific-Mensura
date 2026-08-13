"""Round-3 remeasurement of benchmark problem 06 (planar Brownian motion).

WHAT THIS SCRIPT IS FOR
-----------------------
Problem 06 asks whether the shell-growth ("poly-algebraic") dimension estimator
recovers the Hausdorff dimension of a planar Brownian path, which is *exactly*
2 by a theorem (S. J. Taylor 1953, [T53]) rather than by a literature estimate.
Round 2 recorded it as a criterion-(a) case; finding R2-F9 later showed that
criterion (a) cannot score it at all, because the Grassberger-Procaccia
baseline never converges to within tolerance anywhere on the grid, and standing
rule 2 forbids banking a baseline failure as a win.

This script therefore remeasures BOTH criteria against the current, unmodified
library (post N8 / H1 / H3 / AutoResearch-H2):

  * criterion (a) -- sample efficiency -- via `comparison.compare`
  * criterion (c) -- asymptotic accuracy at fixed n -- via
    `comparison.compare_accuracy_at_max_n` (the H3 function)

and then attacks the criterion-(c) result with the sweeps that finding R2-F3
demands of any recorded win.

HYPERPARAMETERS, DISCLOSED (R2-F3 discipline)
---------------------------------------------
R2-F3 recorded a false win caused by an undisclosed `max_radius` deviation, so
every hyperparameter here is stated and every non-default one is justified in
place:

  max_radius = 6      -- the library default. NOT deviated from anywhere.
  k          = 6 AND 10 -- both are reported for every headline number. 6 is the
                        instructed default; 10 is problem 06's round-2
                        production value and the one the standing
                        `poly_algebraic_min_n = 400` baseline was measured at.
                        Reporting only one of them would make the comparison
                        against the standing baseline non-like-for-like, so
                        both are always shown and the verdict is required to
                        agree across them.
  samples    = 40     -- library default.
  n_grid     = problem 06's production grid, 100 ... 25600.
  tolerance  = 0.30   -- problem 06's own benchmark tolerance (Sec. 2 of the
                        ledger). 0.15 (the tolerance h3_win_robustness.py used)
                        is reported alongside it so the verdict can be seen not
                        to depend on which of the two is picked.
  theiler_window   = 0   (library default, i.e. correction OFF)
  max_ball_fraction = 1.0 (library default, i.e. saturation guard OFF)

The last two are defaults, but for THIS problem they are also a deliberate
choice with a stated reason, and section 4 measures what happens when they are
turned on rather than merely asserting that they should not be. See section 4C.

POINT CLOUD
-----------
`n8d_clouds.load_cloud("06")` -- the round-2 production cloud, reproduced
literally from `round2/06_brownian_motion.py`'s module-level code: numpy
default_rng(seed=42), 80,000 unit-variance 2-D increments, cumulative sum from
the origin, stride `len(path)//25_600 = 3`, giving 26,667 points. This is the
same cloud every standing problem-06 number in the ledger was measured on, so
the numbers here are directly comparable to it. Because every grid point uses
the PREFIX `points[:n]` and the path is a cumulative sum, the n-point cloud is
a genuine prefix of the 25,600-point cloud -- no reconstruction per n.

Sections 4A/4B build *independent* Brownian draws (fresh seeds, same
construction) so the single-cloud result can be separated from a property of
seed 42.

RUN
---
    python scripts/hypergraph_benchmark/round3/06_planar_brownian_motion.py
    python scripts/hypergraph_benchmark/round3/06_planar_brownian_motion.py --sections 1,2
    python scripts/hypergraph_benchmark/round3/06_planar_brownian_motion.py --sections 3,4

Timings (this machine): section 1 ~2 s, section 2 ~60 s, section 3 ~35 s,
section 4 ~7 min. Exit code is 0 always -- this is a measurement, not a gate.
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

from socrates.hypergraph import comparison  # noqa: E402
from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    theiler_window_from_autocorrelation,
)
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

PROBLEM = "06"
TRUE_DIM = 2.0
TOLERANCE = 0.30  # problem 06's own benchmark tolerance
ALT_TOLERANCE = 0.15  # the tolerance h3_win_robustness.py used, for continuity
N_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12_800, 25_600)
MAX_N = N_GRID[-1]
MAX_RADIUS = 6  # library default; never deviated from
SAMPLES = 40  # library default
K_VALUES = (6, 10)  # 6 = instructed default, 10 = round-2 production value


# ---------------------------------------------------------------------------
# section 1 -- provenance and data verification
# ---------------------------------------------------------------------------


def brownian_cloud(seed: int, n_points: int = 25_600) -> list[tuple[float, ...]]:
    """A fresh draw with problem 06's exact construction, differing only in seed.

    Identical code path to `n8d_clouds._build_06` (80,000 increments, stride
    `len(path)//25_600`); asserted against it for seed 42 in section 1.
    """
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.0, scale=1.0, size=(80_000, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(increments, axis=0)], axis=0)
    stride = max(1, len(path) // 25_600)
    return [tuple(float(c) for c in p) for p in path[::stride]][:n_points]


def msd_slope(points: list[tuple[float, ...]]) -> tuple[float, float]:
    """log-log slope of mean squared displacement vs lag: 1.0 == diffusive."""
    arr = np.asarray(points, dtype=float)
    lags = np.unique(np.logspace(0, math.log10(len(arr) // 4), 25).astype(int))
    log_lag, log_msd = [], []
    for lag in lags:
        d = arr[lag:] - arr[:-lag]
        log_lag.append(math.log(float(lag)))
        log_msd.append(math.log(float(np.mean(np.sum(d * d, axis=1)))))
    x = np.asarray(log_lag)
    y = np.asarray(log_msd)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(slope), 1.0 - ss_res / ss_tot


def section1(points: list[tuple[float, ...]]) -> None:
    print("=" * 78)
    print("SECTION 1 -- point cloud provenance and data verification")
    print("=" * 78)
    print(f"cloud source : n8d_clouds.load_cloud('{PROBLEM}') (round-2 production cloud)")
    print(f"points       : {len(points)} in {len(points[0])} dimensions")
    print(f"first point  : {points[0]}  (path starts at the origin)")
    print(f"n_grid       : {N_GRID}  (every grid point is a PREFIX of this cloud)")

    rebuilt = brownian_cloud(42)
    same = all(a == b for a, b in zip(points[: len(rebuilt)], rebuilt, strict=False))
    print(
        f"\nself-check   : the seed-42 draw built by this script's own "
        f"`brownian_cloud` matches load_cloud('{PROBLEM}') point-for-point over "
        f"its first {len(rebuilt)} points: {same}"
    )
    print(
        "               (so section 4's fresh-seed draws differ from the "
        "production cloud in the seed and nothing else)"
    )

    slope, r2 = msd_slope(points)
    print(
        f"\nMSD check    : log-log slope {slope:.4f} (r^2 {r2:.4f}); 1.0 is diffusive "
        f"scaling, i.e. this really is a Brownian path"
    )
    print(
        "known dim    : 2 EXACTLY, by theorem (Taylor 1953, [T53]) -- not a "
        "literature estimate with its own error bar"
    )


# ---------------------------------------------------------------------------
# section 2 -- criterion (a), compute savings
# ---------------------------------------------------------------------------


def section2(points: list[tuple[float, ...]]) -> dict[int, comparison.ComparisonResult]:
    print()
    print("=" * 78)
    print("SECTION 2 -- criterion (a): compute savings, via comparison.compare")
    print("=" * 78)
    print(
        f"defaults throughout: max_radius={MAX_RADIUS}, samples={SAMPLES}, "
        f"theiler_window=0, max_ball_fraction=1.0; tolerance={TOLERANCE}"
    )

    print("\n2A -- per-n estimates, both methods (this is what compare() searches over)")
    print(f"{'n':>7}  {'poly k=6':>9} {'in tol':>7}  {'poly k=10':>9} {'in tol':>7}  ")
    poly_per_n: dict[int, dict[int, float]] = {k: {} for k in K_VALUES}
    for n in N_GRID:
        if n > len(points):
            break
        row = f"{n:7d}  "
        for k in K_VALUES:
            hg = knn_hypergraph(points[:n], k=k, dedupe=True)
            d = mean_dimension(hg, samples=min(n, SAMPLES), max_radius=MAX_RADIUS)
            poly_per_n[k][n] = d
            ok = math.isfinite(d) and abs(d - TRUE_DIM) <= TOLERANCE
            row += f"{d:9.4f} {'IN' if ok else 'OUT':>7}  "
        print(row)

    print(f"\n{'n':>7}  {'traditional':>11} {'r^2':>7} {'in tol':>7}")
    trad_per_n: dict[int, float] = {}
    for n in N_GRID:
        if n > len(points):
            break
        est = correlation_dimension(points[:n])
        trad_per_n[n] = est.dimension
        ok = math.isfinite(est.dimension) and abs(est.dimension - TRUE_DIM) <= TOLERANCE
        print(f"{n:7d}  {est.dimension:11.4f} {est.r_squared:7.4f} {'IN' if ok else 'OUT':>7}")
    print(
        "\nNote the traditional column: high r^2 throughout (a well-fit straight "
        "line) while\nthe slope itself is ~0.6 below the true dimension. Its "
        "fit-quality number does not\nflag its error -- the same behaviour Sec. "
        "10.5 of the ledger records."
    )

    print("\n2B -- comparison.compare() verdicts")
    results: dict[int, comparison.ComparisonResult] = {}
    for k in K_VALUES:
        t0 = time.time()
        r = comparison.compare(points, TRUE_DIM, TOLERANCE, k=k, n_grid=N_GRID)
        results[k] = r
        savings = r.compute_savings_fraction
        print(
            f"  k={k:<3} poly_min_n={r.poly_algebraic_min_n}  "
            f"traditional_min_n={r.traditional_min_n}  "
            f"savings={'None' if savings is None else f'{savings:.3f}'}  "
            f"poly_wins={r.poly_algebraic_wins}   [{time.time() - t0:.0f}s]"
        )
        print(
            f"       at max_n={r.max_n_tested}: poly={r.poly_algebraic_estimate_at_max_n:.4f}, "
            f"traditional={r.traditional_estimate_at_max_n:.4f}"
        )
    print(
        "\nCriterion (a) reading: `traditional_min_n is None` means the baseline "
        "never gets\nwithin +-0.30 of 2.0 anywhere on the grid, so there is no "
        "denominator for a savings\nfraction and `poly_algebraic_wins` is False "
        "BY DESIGN (standing rule 2: a baseline\nfailure is not a poly win). "
        "This is finding R2-F9 reproduced against current code,\nnot a new "
        "result -- and it is precisely why criterion (c) exists."
    )
    return results


# ---------------------------------------------------------------------------
# section 3 -- criterion (c), accuracy at fixed n
# ---------------------------------------------------------------------------


def _refusal(reason: str) -> str:
    """One-phrase summary of which criterion (c) condition refused a no-win."""
    if reason.startswith("win"):
        return "-"
    if "sentinel" in reason:
        return "c2 sentinel/degenerate"
    if "poly's own error" in reason:
        return "c3 poly inaccurate"
    if "also within the tolerance" in reason:
        return "c4 trad also in tol"
    if "below the margin" in reason:
        return "c5 gap < margin"
    return "c1 no estimate"


def _crit_c_row(label: str, r: comparison.AccuracyComparisonResult) -> str:
    gap = r.traditional_abs_error - r.poly_algebraic_abs_error
    return (
        f"{label:>22}  {r.poly_algebraic_estimate:8.4f} {r.poly_algebraic_abs_error:7.4f}  "
        f"{r.traditional_estimate:8.4f} {r.traditional_abs_error:7.4f}  {gap:7.4f}  "
        f"{r.poly_algebraic_sentinel_fraction:6.3f}  "
        f"{'WIN' if r.poly_algebraic_accuracy_win else ' no':>3}  "
        f"{_refusal(r.verdict_reason)}"
    )


CRIT_C_HEADER = (
    f"{'config':>22}  {'poly':>8} {'|err|':>7}  {'trad':>8} {'|err|':>7}  "
    f"{'gap':>7}  {'sentl':>6}  win  refused by"
)


def section3(points: list[tuple[float, ...]]) -> dict[tuple[int, float], object]:
    print()
    print("=" * 78)
    print("SECTION 3 -- criterion (c): accuracy at max n, via compare_accuracy_at_max_n")
    print("=" * 78)
    print(
        f"max_n={MAX_N}, max_radius={MAX_RADIUS}, samples={SAMPLES}, "
        f"margin={comparison.DEFAULT_ACCURACY_MARGIN}, "
        f"max_sentinel_fraction={comparison.DEFAULT_MAX_SENTINEL_FRACTION}"
    )
    print(CRIT_C_HEADER)
    out: dict[tuple[int, float], object] = {}
    for tol in (TOLERANCE, ALT_TOLERANCE):
        for k in K_VALUES:
            r = comparison.compare_accuracy_at_max_n(
                points,
                TRUE_DIM,
                MAX_N,
                k=k,
                tolerance=tol,
                max_radius=MAX_RADIUS,
                samples=SAMPLES,
            )
            out[(k, tol)] = r
            print(_crit_c_row(f"k={k}, tol={tol}", r))
    print()
    for (k, tol), r in out.items():
        print(f"  k={k}, tol={tol}: {r.verdict_reason}")
    print(
        f"\n  n_evaluated={out[(6, TOLERANCE)].n_evaluated}, "
        f"poly_algebraic_n_nodes={out[(6, TOLERANCE)].poly_algebraic_n_nodes} (k=6) -- "
        f"no points were dropped as near-duplicates, so both methods saw the same "
        f"{out[(6, TOLERANCE)].n_evaluated} points."
    )
    return out


# ---------------------------------------------------------------------------
# section 4 -- adversarial robustness of the criterion (c) verdict
# ---------------------------------------------------------------------------


def section4(points: list[tuple[float, ...]], n_seeds: int = 10) -> None:
    print()
    print("=" * 78)
    print("SECTION 4 -- is the criterion (c) verdict a property of the data or the knobs?")
    print("=" * 78)
    print(
        "R2-F3 is the standing example of a win that existed only at one "
        "hyperparameter\nsetting. Every knob that could have produced this one "
        "is swept below."
    )

    print(f"\n4A -- hyperparameter sweeps on the production cloud (tol={TOLERANCE})")
    print(CRIT_C_HEADER)
    k_wins = 0
    for k in (6, 8, 10, 12, 14):
        r = comparison.compare_accuracy_at_max_n(
            points, TRUE_DIM, MAX_N, k=k, tolerance=TOLERANCE, max_radius=MAX_RADIUS
        )
        k_wins += bool(r.poly_algebraic_accuracy_win)
        print(_crit_c_row(f"k={k}", r))
    print(f"  -> k sweep: {k_wins}/5 wins")

    mr_wins = 0
    for mr in range(3, 9):
        r = comparison.compare_accuracy_at_max_n(
            points, TRUE_DIM, MAX_N, k=6, tolerance=TOLERANCE, max_radius=mr
        )
        mr_wins += bool(r.poly_algebraic_accuracy_win)
        print(_crit_c_row(f"max_radius={mr}", r))
    print(f"  -> max_radius sweep: {mr_wins}/6 wins (max_radius=6 is the default)")

    print(f"\n4B -- {n_seeds} INDEPENDENT Brownian draws (fresh seed, same construction)")
    print(CRIT_C_HEADER)
    seed_wins = 0
    for seed in range(n_seeds):
        pts = brownian_cloud(1000 + seed)
        r = comparison.compare_accuracy_at_max_n(
            pts, TRUE_DIM, MAX_N, k=6, tolerance=TOLERANCE, max_radius=MAX_RADIUS
        )
        seed_wins += bool(r.poly_algebraic_accuracy_win)
        print(_crit_c_row(f"seed={1000 + seed}", r))
    print(f"  -> seed sweep: {seed_wins}/{n_seeds} wins at n={MAX_N}, k=6")
    print(
        "     Read the 'refused by' column on any no-win row. On the seeds "
        "measured here the\n     refusals are (c4) -- poly stayed inside "
        "tolerance (|err| ~0.04) and the BASELINE\n     also happened to land "
        "inside +-0.30 on that draw, so the two answers stopped\n     being "
        "decision-relevantly different. That is the criterion refusing itself "
        "on a\n     lucky baseline draw, not poly failing: no refusal here is "
        "a (c2) degeneracy or a\n     (c3) poly inaccuracy. "
        "(h3_win_robustness.py measures 16/20 at n=1600, k=10,\n     tol=0.15; "
        "its refusals are (c3)/(c5) at that much smaller n.)"
    )

    print("\n4C -- the two knobs `compare_accuracy_at_max_n` does NOT accept")
    print(
        "This is the standing capability gap: criterion (c) cannot be asked for a\n"
        "theiler-corrected or saturation-guarded reading. Both are therefore "
        "measured here\nby hand, from the same library primitives the function "
        "itself calls, so the\nverdict's dependence on them is on the record "
        "rather than untested."
    )

    print("\n  (i) theiler_window='auto' -- and why the default 0 is the right reading here")
    print(f"      {'n':>7}  {'auto window W':>14}  {'W / n':>7}")
    for n in N_GRID:
        w = theiler_window_from_autocorrelation(points[:n])
        print(f"      {n:7d}  {w:14d}  {w / n:7.4f}")
    print(
        "      W equals the clamp exactly at n<=800 (max_window = n//10: 8, 20, "
        "40, 80) and\n      pins to the other clamp arm, 250, at n>=3200; only "
        "n=1600 crosses 1/e at all,\n      and it crosses at 153 against a "
        "clamp of 160. So W is essentially the clamp\n      `min(n//10, 250)` "
        "rather than a measured decorrelation time -- which is what a\n"
        "      NON-STATIONARY signal does: a Brownian path has variance growing "
        "linearly in t,\n      so its pooled autocorrelation decays only over "
        "the whole record and there is no\n      decorrelation time to read "
        "off. The heuristic's own stated local model (AR(1) /\n      "
        "Ornstein-Uhlenbeck, i.e. stationary) does not hold here. A window that "
        "scales with\n      the record length is a clamp artifact, so "
        "theiler_window=0 (the library default)\n      is the honest setting "
        "for problem 06."
    )
    print("\n      What the corrected reading would be anyway, stated rather than hidden:")
    print(
        f"      {'setting':>22}  {'poly':>8}  {'trad':>8}  "
        f"{'poly |err|':>10}  {'trad |err|':>10}"
    )
    prefix = points[:MAX_N]
    for label, window in (("theiler=0 (default)", 0), ("theiler='auto'", "auto")):
        hg = knn_hypergraph(prefix, k=6, dedupe=True, theiler_window=window)
        poly = mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS)
        trad = correlation_dimension(prefix, theiler_window=window).dimension
        print(
            f"      {label:>22}  {poly:8.4f}  {trad:8.4f}  "
            f"{abs(poly - TRUE_DIM):10.4f}  {abs(trad - TRUE_DIM):10.4f}"
        )
    print(
        "      Read honestly: under the clamp-derived window the verdict does "
        "not merely\n      weaken, it REVERSES -- poly goes to ~2.68 (worse) "
        "and the baseline to ~1.73\n      (better, and inside +-0.30), which "
        "is a criterion-(c) win for the TRADITIONAL\n      side. The "
        "criterion-(c) win recorded here is therefore contingent on\n"
        "      theiler_window=0. That contingency is defensible for the reason "
        "above (the\n      window is a clamp, not a measurement, on a "
        "non-stationary path) and it is the\n      library default, but it is "
        "a real contingency and is reported as one."
    )

    print("\n  (ii) max_ball_fraction (AR2 saturation guard; default 1.0 = off)")
    print(f"      {'fraction':>22}  {'poly':>8}  {'poly |err|':>10}")
    hg = knn_hypergraph(prefix, k=6, dedupe=True)
    for frac in (1.0, 0.5, 0.35, 0.2):
        poly = mean_dimension(hg, samples=SAMPLES, max_radius=MAX_RADIUS, max_ball_fraction=frac)
        print(f"      {frac:22.2f}  {poly:8.4f}  {abs(poly - TRUE_DIM):10.4f}")
    print(
        "      The traditional side is unaffected by this knob (it is one-sided "
        "by design),\n      so any row whose poly |err| stays <= 0.30 leaves "
        "the section-3 verdict standing."
    )


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="1,2,3,4")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    wanted = {s.strip() for s in args.sections.split(",")}

    points = load_cloud(PROBLEM)
    if "1" in wanted:
        section1(points)
    if "2" in wanted:
        section2(points)
    if "3" in wanted:
        section3(points)
    if "4" in wanted:
        section4(points, n_seeds=args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
