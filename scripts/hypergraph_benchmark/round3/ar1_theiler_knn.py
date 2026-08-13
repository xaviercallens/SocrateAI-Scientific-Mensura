"""AR iteration 1: a Theiler window on the k-NN GRAPH, measured end to end.

THE PROPOSED CHANGE. `pointcloud.knn_hypergraph` gains `theiler_window` /
`time_indices`, mirroring `baseline.correlation_dimension`'s already-verified
arguments of the same names, and `comparison.compare()` /
`poly_algebraic_minimum_points` forward them. Default 0 everywhere = the
original code path verbatim.

WHY. Theiler (1986) is normally cited against the correlation sum. The k-NN
graph has the same disease in a sharper form, because k-NN is a RANKING rather
than a count: when the along-trajectory spacing is finer than the transverse
spacing between successive passes of the attractor, a point's k nearest
neighbours are all its own temporal neighbours, and the "proximity graph" is a
1-dimensional chain whatever the underlying set is. Section 1 measures that
directly instead of assuming it.

WHY IT IS OPT-IN RATHER THAN AUTOMATIC. Section 3 turns the window on for all
ten round-2 clouds and shows it is NOT safe as a default: on a 1-D closed orbit
sampled once (problems 03/04/07, two of which are verified wins) the temporal
neighbours ARE the genuine spatial neighbours, so the exclusion replaces local
geometry with long-range chords and convergence is lost outright. Section 4
measures the statistic that discriminates the two regimes -- the factor by
which the window inflates the k-th neighbour distance -- and shows that on this
evidence its must-fire and must-not-fire sides are only 1.55x and 2.39x apart.
That bracket is too narrow to promote into an automatic rule here (compare the
3.4x / 18x / 37x separations dimension.py's NEAR_CONSTANT_* constants are held
to), so the window stays a caller decision and every recorded number stays
reproducible. Widening or replacing that statistic is the natural next
iteration; section 4 is the data it would start from.

Run: python scripts/hypergraph_benchmark/round3/ar1_theiler_knn.py
Sections can be selected with --sections 1,2,3,4 (default: all).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, load_cloud  # noqa: E402

from socrates.hypergraph.baseline import (  # noqa: E402
    correlation_dimension,
    theiler_window_from_autocorrelation,
)
from socrates.hypergraph.comparison import compare  # noqa: E402
from socrates.hypergraph.dimension import mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

CHAOTIC = ("08", "09")


def grid_for(problem: str, points) -> list[int]:
    spec = SPECS[problem]
    return [n for n in spec.n_grid if n <= len(points) and spec.k < n]


# ---------------------------------------------------------------------------
# 1. Is the k-NN graph on these trajectories actually a temporal chain?
# ---------------------------------------------------------------------------
def section_1() -> None:
    print("=" * 78)
    print("1. TEMPORAL DOMINANCE OF THE UNCORRECTED k-NN GRAPH")
    print("   'adjacent' = the two endpoints are within k of each other in trajectory")
    print("   time index, i.e. the edge records the sampling rate, not the geometry.")
    print("=" * 78)
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(f"\n  problem {problem} ({spec.label}), k={spec.k}")
        print(f"  {'n':>7} {'adjacent-edge frac':>19} {'poly dim (window off)':>22}")
        for n in grid_for(problem, points)[:5]:
            arr = np.asarray(points[:n], float)
            _, idx = cKDTree(arr).query(arr, k=spec.k + 1)
            frac = float(np.mean(np.abs(idx[:, 1:] - np.arange(n)[:, None]) <= spec.k))
            hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True)
            dim = mean_dimension(hg, samples=min(n, spec.samples), max_radius=spec.max_radius)
            print(f"  {n:>7} {frac:>19.3f} {dim:>22.4f}")
        print(f"  (true dimension of this attractor: {spec.true_dimension})")


# ---------------------------------------------------------------------------
# 2. The headline: compare() before and after, on 08 and 09
# ---------------------------------------------------------------------------
def _series(problem: str, window):
    spec = SPECS[problem]
    points = load_cloud(problem)
    poly, trad, used = [], [], []
    for n in grid_for(problem, points):
        sub = points[:n]
        w = window if window != "auto" else theiler_window_from_autocorrelation(
            np.asarray(sub, float)
        )
        used.append(w)
        try:
            hg = knn_hypergraph(sub, k=spec.k, dedupe=True, theiler_window=window)
            poly.append((n, mean_dimension(hg, samples=min(n, spec.samples),
                                           max_radius=spec.max_radius)))
        except ValueError:
            poly.append((n, float("nan")))
        trad.append((n, correlation_dimension(sub, theiler_window=window).dimension))
    return poly, trad, used


def section_2() -> None:
    print("\n" + "=" * 78)
    print("2. compare() WITH THE WINDOW OFF (recorded state) AND ON ('auto')")
    print("   ONE knob, applied to BOTH estimators -- the window is a statement about")
    print("   what trajectory data means, not an advantage handed to one side.")
    print("=" * 78)
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(f"\n  problem {problem} ({spec.label}): k={spec.k}, "
              f"true={spec.true_dimension}, tol={spec.tolerance}")
        for window, label in ((0, "window OFF (before)"), ("auto", "window AUTO (after)")):
            result = compare(
                points,
                true_dimension=spec.true_dimension,
                tolerance=spec.tolerance,
                k=spec.k,
                n_grid=spec.n_grid,
                max_radius=spec.max_radius,
                theiler_window=window,
            )
            poly, trad, used = _series(problem, window)
            print(f"\n    {label}")
            print(f"      poly_algebraic_min_n = {result.poly_algebraic_min_n}      "
                  f"traditional_min_n = {result.traditional_min_n}")
            print(f"      windows used per n: {used}")
            head = " ".join(f"{n:>7d}" for n, _ in poly)
            print(f"      {'n':>10} {head}")
            print(f"      {'poly':>10} " + " ".join(f"{d:7.3f}" for _, d in poly))
            print(f"      {'trad':>10} " + " ".join(f"{d:7.3f}" for _, d in trad))


# ---------------------------------------------------------------------------
# 3. Collateral: what happens if the window is turned on EVERYWHERE
# ---------------------------------------------------------------------------
def _min_n(series, true_d, tol):
    for i, (n, d) in enumerate(series):
        if not math.isfinite(d) or abs(d - true_d) > tol:
            continue
        if all(math.isfinite(x) and abs(x - true_d) <= tol for _, x in series[i:]):
            return n
    return None


def section_3() -> None:
    print("\n" + "=" * 78)
    print("3. COLLATERAL -- why the window must NOT be a default")
    print("   poly_algebraic_min_n on all ten round-2 clouds, window off vs 'auto'.")
    print("=" * 78)
    print(f"\n  {'prob':>5} {'label':>32} {'off':>7} {'auto':>7}   verdict")
    for problem in sorted(SPECS):
        spec = SPECS[problem]
        points = load_cloud(problem)
        rows = {}
        for window in (0, "auto"):
            series = []
            for n in grid_for(problem, points):
                try:
                    hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True, theiler_window=window)
                    series.append((n, mean_dimension(hg, samples=min(n, spec.samples),
                                                     max_radius=spec.max_radius)))
                except ValueError:
                    series.append((n, float("nan")))
            rows[window] = _min_n(series, spec.true_dimension, spec.tolerance)
        off, auto = rows[0], rows["auto"]
        if off == auto:
            verdict = "unchanged"
        elif auto is None:
            verdict = "DESTROYED by the window"
        elif off is None:
            verdict = "rescued by the window"
        else:
            verdict = f"{'improved' if auto < off else 'WORSENED'} {off} -> {auto}"
        print(f"  {problem:>5} {spec.label:>32} {str(off):>7} {str(auto):>7}   {verdict}")


# ---------------------------------------------------------------------------
# 4. The discriminating statistic, and why it is not yet an automatic rule
# ---------------------------------------------------------------------------
def _inflation(points, k, window) -> float:
    """Median over points of d_k(with window) / d_k(without).

    ~1 means there were plenty of spatially near, temporally distant points, so
    the window removes redundancy. >>1 means the temporal neighbours WERE the
    only genuine near neighbours, so the window is trading local geometry for
    long-range chords.
    """
    arr = np.asarray(points, float)
    n = len(arr)
    d, idx = cKDTree(arr).query(arr, k=min(n, k + 2 * window + 1))
    ratios = []
    for i in range(n):
        pairs = list(zip(d[i], idx[i], strict=True))
        plain = [dd for dd, jj in pairs if int(jj) != i][:k]
        win = [dd for dd, jj in pairs if abs(int(jj) - i) >= window][:k]
        if len(plain) == k and len(win) == k and plain[-1] > 0:
            ratios.append(win[-1] / plain[-1])
    return float(np.median(ratios)) if ratios else float("nan")


def section_4() -> None:
    print("\n" + "=" * 78)
    print("4. THE DISCRIMINATOR, AND THE BRACKET IT DOES NOT YET EARN")
    print("   median d_k inflation caused by the window. Small = the window is safe.")
    print("=" * 78)
    print(f"\n  {'prob':>5} {'label':>32}   inflation per n (n:W:ratio)")
    for problem in sorted(SPECS):
        spec = SPECS[problem]
        points = load_cloud(problem)
        cells = []
        for n in grid_for(problem, points):
            sub = points[:n]
            w = theiler_window_from_autocorrelation(np.asarray(sub, float))
            cells.append(f"{n}:W{w}:{_inflation(sub, spec.k, w):.2f}")
        print(f"  {problem:>5} {spec.label:>32}   {' '.join(cells)}")
    print(
        "\n  Reading it: 03/04/07 (1-D closed orbits, W large because a periodic\n"
        "  autocorrelation crosses 1/e on a phase shift rather than on genuine\n"
        "  decorrelation) inflate 4x-84x -- the window has nothing legitimate to\n"
        "  find there. 08/09 sit at 1.0-1.7 wherever the window helps. The two\n"
        "  sides are separated by only ~1.55x (Rossler n=800) to 2.39x (Brownian\n"
        "  n=400), so any threshold between them is a knife edge calibrated on the\n"
        "  very clouds it would then be scored on. Hence: opt-in."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4")
    args = ap.parse_args()
    wanted = {s.strip() for s in args.sections.split(",")}
    for name, fn in (("1", section_1), ("2", section_2), ("3", section_3), ("4", section_4)):
        if name in wanted:
            fn()


if __name__ == "__main__":
    main()
