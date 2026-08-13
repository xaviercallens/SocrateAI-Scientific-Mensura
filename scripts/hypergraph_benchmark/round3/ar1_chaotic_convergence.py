"""AutoResearch iteration 1: convergence of the shell-growth estimator on the
two chaotic-attractor problems (08 Lorenz, 09 Rossler).

Measures, against whatever `src/socrates/hypergraph/` currently contains:

  * `comparison.compare()` on each problem's PRODUCTION cloud / k / max_radius /
    n_grid / tolerance (read from n8d_clouds.SPECS, which was transcribed from
    the round-2 scripts), giving poly_algebraic_min_n vs traditional_min_n;
  * the per-n dimension trace behind that verdict, so a change of min_n can be
    attributed to a specific n rather than trusted from the summary;
  * the compute-savings spot check on problems 02 and 10, which the round-3
    ledger records at poly_algebraic_min_n = 64 and must not regress.

Run with no arguments. `--label` only tags the printed header so before/after
runs are distinguishable in a transcript.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, load_cloud  # noqa: E402

from socrates.hypergraph.baseline import correlation_dimension  # noqa: E402
from socrates.hypergraph.comparison import compare  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    mean_dimension,
    near_degenerate_fraction,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

CHAOTIC = ("08", "09")
SPOT_CHECK = ("02", "10")


def per_n_trace(problem: str) -> None:
    spec = SPECS[problem]
    points = load_cloud(problem)
    print(f"  {'n':>6} | {'poly_dim':>9} {'in?':>4} {'neardeg':>8} | {'trad_dim':>9} {'in?':>4}")
    for n in spec.n_grid:
        if n > len(points) or spec.k >= n:
            break
        sub = points[:n]
        hg = knn_hypergraph(sub, k=spec.k, dedupe=True)
        pd = mean_dimension(hg, samples=min(n, spec.samples), max_radius=spec.max_radius)
        nd = near_degenerate_fraction(hg, samples=min(n, spec.samples), max_radius=spec.max_radius)
        td = correlation_dimension(sub).dimension
        tol = spec.tolerance
        p_in = "IN" if math.isfinite(pd) and abs(pd - spec.true_dimension) <= tol else "--"
        t_in = "IN" if math.isfinite(td) and abs(td - spec.true_dimension) <= tol else "--"
        print(f"  {n:>6} | {pd:>9.4f} {p_in:>4} {nd:>8.3f} | {td:>9.4f} {t_in:>4}")


def run(problem: str, *, trace: bool) -> None:
    spec = SPECS[problem]
    points = load_cloud(problem)
    result = compare(
        points,
        true_dimension=spec.true_dimension,
        tolerance=spec.tolerance,
        k=spec.k,
        n_grid=spec.n_grid,
        max_radius=spec.max_radius,
    )
    print(
        f"problem {problem} ({spec.label}): k={spec.k}, max_radius={spec.max_radius}, "
        f"true={spec.true_dimension}, tol={spec.tolerance}"
    )
    print(f"  poly_algebraic_min_n = {result.poly_algebraic_min_n}")
    print(f"  traditional_min_n    = {result.traditional_min_n}")
    print(f"  poly@max_n={result.max_n_tested}: {result.poly_algebraic_estimate_at_max_n:.4f}   "
          f"trad@max_n: {result.traditional_estimate_at_max_n:.4f}")
    print(f"  poly_algebraic_wins  = {result.poly_algebraic_wins}   "
          f"savings = {result.compute_savings_fraction}")
    if trace:
        per_n_trace(problem)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="current working tree")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated problem ids")
    args = ap.parse_args()

    print(f"=== AR iteration 1 convergence measurement :: {args.label} ===\n")
    wanted = args.only.split(",") if args.only else list(CHAOTIC) + list(SPOT_CHECK)
    for problem in wanted:
        run(problem, trace=(not args.no_trace) and problem in CHAOTIC)


if __name__ == "__main__":
    main()
