"""F3 diagnostic: re-score benchmark problems 08 and 09 against the in-house
Kaplan-Yorke targets instead of the [GP83] targets.

The experiment is deliberately minimal: EVERYTHING is held at the recorded
configuration and at `comparison`'s module defaults, and the ONLY thing that
changes between the two arms is the number passed as `true_dimension`
(2.05 -> D_KY(Lorenz), 2.01 -> D_KY(Rossler)). Any verdict change is then
attributable to the target and to nothing else.

Clouds come from `round3/n8d_clouds.load_cloud`, i.e. the round-2 production
clouds verbatim, so the GP arm reproduces the recorded numbers rather than
re-deriving a similar-looking cloud.

Configurations swept, because one configuration is not a result:
  * k = the recorded production k (10 for 08, 15 for 09) AND k = 6, the
    module default;
  * n_grid = comparison.compare's own default (100..3200) AND the recorded
    production grid (100..12800);
  * tolerance = the recorded 0.5, plus a ladder of tighter tolerances down to
    0.01 -- because the question "does the target choice change anything?"
    has a quantitative answer (the tolerance at which the two targets first
    disagree) that is more informative than a yes/no at 0.5 alone.

Everything else -- theiler_window=0, max_ball_fraction=1.0, max_radius=6,
samples=40, margin=0.25, max_sentinel_fraction=0.05 -- is left at the module
default and stated here so the numbers are quotable with their settings.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "hypergraph_benchmark" / "round3"))

from n8d_clouds import load_cloud  # noqa: E402
from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.comparison import (  # noqa: E402
    DEFAULT_ACCURACY_MARGIN,
    DEFAULT_MAX_SENTINEL_FRACTION,
    compare,
    compare_accuracy_at_max_n,
)

KY_PATH = Path(
    "/tmp/claude-1000/-home-xavkal-xdev/cb774970-2d62-4712-aaa9-94dafb5c745b"
    "/scratchpad/f3_ky_targets_final.json"
)

DEFAULT_GRID = (100, 200, 400, 800, 1600, 3200)  # comparison.compare's own default
PROD_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800)  # recorded production grid

PROBLEMS = {
    "08": {"label": "Lorenz", "k_prod": 10, "gp": 2.05},
    "09": {"label": "Rossler", "k_prod": 15, "gp": 2.01},
}

TOLERANCE_LADDER = (0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01)

# Debug switch. F3_FAST=1 shrinks every point-count grid so the control flow can
# be exercised in seconds; it is NOT a configuration any reported number may
# come from, and the banner below says so on every fast run.
FAST = os.environ.get("F3_FAST") == "1"
if FAST:
    DEFAULT_GRID = (100, 200, 400)
    PROD_GRID = (100, 200, 400, 800)
RESOLVING_NS = (400, 800) if FAST else (1600, 3200, 6400, 12800)
RESOLVING_KS = (6, 10) if FAST else (6, 8, 10, 12, 15)
BASELINE_NS = (400, 800) if FAST else (800, 1600, 3200, 6400, 12800)


def main() -> None:
    with KY_PATH.open() as fh:
        ky = json.load(fh)

    print("=" * 78)
    print("F3 RE-SCORING: [GP83] target vs in-house Kaplan-Yorke target")
    print("=" * 78)
    if FAST:
        print("  *** F3_FAST=1: SHRUNK GRIDS, DEBUG ONLY, NOT REPORTABLE ***")
    print(f"  module defaults in force: theiler_window=0, max_ball_fraction=1.0,")
    print(f"  max_radius=6, samples=40, margin={DEFAULT_ACCURACY_MARGIN}, "
          f"max_sentinel_fraction={DEFAULT_MAX_SENTINEL_FRACTION}")

    for prob, meta in PROBLEMS.items():
        pts = load_cloud(prob)
        gp = meta["gp"]
        ky_val = ky[meta["label"].lower()]["D_KY"]
        ky_sd = ky[meta["label"].lower()]["sd"]
        print()
        print("#" * 78)
        print(f"# PROBLEM {prob} -- {meta['label']}   cloud: {len(pts)} points")
        print(f"#   [GP83] target      {gp}")
        print(f"#   in-house D_KY      {ky_val:.5f} +- {ky_sd:.5f}")
        print(f"#   |D_KY - GP|        {abs(ky_val - gp):.5f}")
        print("#" * 78)

        for k in (meta["k_prod"], 6):
            for grid_name, grid in (("default(->3200)", DEFAULT_GRID),
                                    ("production(->12800)", PROD_GRID)):
                grid = tuple(n for n in grid if n <= len(pts))
                max_n = grid[-1]
                print(f"\n--- k={k}  n_grid={grid_name}  max_n={max_n} ---")

                # criterion (a): identical except for the target value
                res = {}
                for tag, target in (("GP", gp), ("KY", ky_val)):
                    r = compare(pts, target, 0.5, k=k, n_grid=grid)
                    res[tag] = r
                    print(f"  compare()  target={tag} ({target:.5f}) tol=0.5:  "
                          f"poly_min_n={r.poly_algebraic_min_n}  "
                          f"trad_min_n={r.traditional_min_n}  "
                          f"poly@max={r.poly_algebraic_estimate_at_max_n:.4f}  "
                          f"trad@max={r.traditional_estimate_at_max_n:.4f}  "
                          f"poly_wins={r.poly_algebraic_wins}")
                same_a = (
                    res["GP"].poly_algebraic_min_n == res["KY"].poly_algebraic_min_n
                    and res["GP"].traditional_min_n == res["KY"].traditional_min_n
                    and res["GP"].poly_algebraic_wins == res["KY"].poly_algebraic_wins
                )
                print(f"  -> criterion (a) verdict CHANGED: {not same_a}")

                # Criterion (c). The MEASUREMENTS here (both estimates, the
                # degeneracy fractions, the baseline's R^2) do not depend on
                # the target at all -- only the verdict does. So the estimator
                # is run ONCE and the single measured result is re-scored
                # against both targets with `dataclasses.replace`, which is
                # both cheaper and stricter: it makes it impossible for a
                # verdict difference to come from re-running the estimator and
                # landing on a different sample of nodes.
                measured = compare_accuracy_at_max_n(pts, gp, max_n, k=k, tolerance=0.5)
                acc = {}
                for tag, target in (("GP", gp), ("KY", ky_val)):
                    a = replace(measured, true_dimension=target)
                    acc[tag] = a
                    print(f"  accuracy() target={tag} ({target:.5f}) tol=0.5:  "
                          f"poly={a.poly_algebraic_estimate:.4f} "
                          f"(|err| {a.poly_algebraic_abs_error:.4f})  "
                          f"trad={a.traditional_estimate:.4f} "
                          f"(|err| {a.traditional_abs_error:.4f})  "
                          f"more_accurate={a.more_accurate}  "
                          f"poly_win={a.poly_algebraic_accuracy_win}  "
                          f"trad_win={a.traditional_accuracy_win}")
                    print(f"      sentinel={a.poly_algebraic_sentinel_fraction:.3f}  "
                          f"trad_r2={a.traditional_r_squared:.4f}  "
                          f"n_nodes={a.poly_algebraic_n_nodes}")
                same_c = (
                    acc["GP"].poly_algebraic_accuracy_win == acc["KY"].poly_algebraic_accuracy_win
                    and acc["GP"].traditional_accuracy_win == acc["KY"].traditional_accuracy_win
                    and acc["GP"].more_accurate == acc["KY"].more_accurate
                )
                print(f"  -> criterion (c) verdict CHANGED: {not same_c}")

                # the benchmark's own headline pass/fail (|estimate - target| <= tol)
                for tag, target in (("GP", gp), ("KY", ky_val)):
                    e = acc[tag].poly_algebraic_estimate
                    print(f"  ledger-style pass/fail vs {tag}: poly={e:.4f} "
                          f"|err|={abs(e - target):.4f} <= 0.5 -> "
                          f"{'PASS' if abs(e - target) <= 0.5 else 'FAIL'}")

                # How tight would the tolerance have to be for the two targets
                # to disagree at all? This is the quantitative version of the
                # question; a yes/no at 0.5 alone hides how far from the
                # decision boundary the two candidate targets sit.
                print("  tolerance ladder (does the GP/KY choice flip anything?):")
                poly = acc["GP"].poly_algebraic_estimate
                trad = acc["GP"].traditional_estimate
                for tol in TOLERANCE_LADDER:
                    row = []
                    for _tag, target in (("GP", gp), ("KY", ky_val)):
                        row.append((abs(poly - target) <= tol, abs(trad - target) <= tol))
                    flip = row[0] != row[1]
                    print(f"    tol={tol:<5}  GP: poly {'P' if row[0][0] else 'F'} "
                          f"trad {'P' if row[0][1] else 'F'}   "
                          f"KY: poly {'P' if row[1][0] else 'F'} "
                          f"trad {'P' if row[1][1] else 'F'}   "
                          f"{'<-- DIFFERS' if flip else ''}")

        # What does the repo's own GP baseline actually say on this cloud?
        # (Tests the reviewer's 'the baseline is guaranteed to look correct'
        # half of the claim directly.)
        print(f"\n--- repo baseline (Grassberger-Procaccia) vs the [GP83] number it is "
              f"scored against ---")
        for n in BASELINE_NS:
            if n > len(pts):
                continue
            cd = correlation_dimension(pts[:n])
            print(f"    n={n:<6} D2_repo={cd.dimension:.4f}  r2={cd.r_squared:.4f}  "
                  f"|D2_repo - GP83|={abs(cd.dimension - gp):.4f}  "
                  f"|D2_repo - D_KY|={abs(cd.dimension - ky_val):.4f}")

        # ------------------------------------------------------------------
        # RESOLVING POWER. Both candidate targets sit within ~0.06 of the
        # integer 2. The question a fractal-dimension benchmark is supposed to
        # answer is whether the estimator can see that EXCESS -- so measure
        # the excess against the estimator's own spread across the settings
        # nobody has a principled reason to prefer between, and against the
        # tolerance the ledger actually applies.
        # ------------------------------------------------------------------
        print("\n--- resolving power: can either method see the fractal excess at all? ---")
        excess_gp, excess_ky = gp - 2.0, ky_val - 2.0
        print(f"    fractal excess over the integer 2:  GP {excess_gp:+.4f}   "
              f"KY {excess_ky:+.4f}   ledger tolerance +-0.5")
        polys, trads = [], []
        for k in RESOLVING_KS:
            for n in RESOLVING_NS:
                if n > len(pts):
                    continue
                a = compare_accuracy_at_max_n(pts, gp, n, k=k, tolerance=0.5)
                polys.append((k, n, a.poly_algebraic_estimate))
                trads.append((n, a.traditional_estimate))
        pv = [p for _, _, p in polys]
        tv = sorted({(n, t) for n, t in trads})
        print(f"    poly estimate over k in {RESOLVING_KS} x n in {RESOLVING_NS}: "
              f"n={len(pv)} values, mean {statistics.fmean(pv):.4f}, "
              f"sd {statistics.stdev(pv):.4f}, range [{min(pv):.4f}, {max(pv):.4f}]")
        print(f"    spread / |fractal excess|:  vs GP {statistics.stdev(pv)/abs(excess_gp):.1f}x   "
              f"vs KY {statistics.stdev(pv)/abs(excess_ky):.1f}x")
        print(f"    trad estimate by n: " + "  ".join(f"n={n}:{t:.4f}" for n, t in tv))
        for name, target in (("GP", gp), ("KY", ky_val)):
            print(f"    null estimator 'always answer exactly 2.0' vs {name} target "
                  f"{target:.4f}: |err|={abs(2.0 - target):.4f} <= 0.5 -> "
                  f"{'PASSES' if abs(2.0 - target) <= 0.5 else 'fails'}")
        print(f"    null estimator 'always answer exactly 2.0' also passes every other "
              f"tolerance down to {min(t for t in TOLERANCE_LADDER if t >= abs(2.0-ky_val)):.2f}")


if __name__ == "__main__":
    main()
