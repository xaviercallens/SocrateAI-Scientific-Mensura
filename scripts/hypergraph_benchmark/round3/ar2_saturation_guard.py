"""AutoResearch iteration 2: the finite-size saturation guard on the fit window.

THE ONE CHANGE THIS SCRIPT MEASURES
-----------------------------------
`dimension.local_dimension` gains `max_ball_fraction` (default 1.0 = OFF =
pre-AR2 code path verbatim, since a ball can never exceed the whole graph).
When enabled, the shell at radius `r` enters the log-log fit only if the ball
at radius `r - 1` covered at most that fraction of the graph's nodes.
`mean_dimension`, `degenerate_fraction`, `near_degenerate_fraction`,
`comparison.poly_algebraic_minimum_points` and `comparison.compare` forward it.

WHY. `local_dimension` previously ended its fit window only at an EXACTLY zero
shell. On the two chaotic-attractor clouds at small n the ball engulfs the
whole sample several radii before that: the trailing shells shrink because the
sample ran out, and they sit at the high-leverage end of the log-log fit, so
they drag the estimate down. Section 1 shows this directly. This is the
standard "fit inside the scaling region" discipline of correlation-dimension
practice (Grassberger & Procaccia 1983; Theiler 1986 Sec. IV) transposed to
graph-hop radius, and the graph-side form of the manifold-adaptive idea
(Farahmand, Szepesvari & Audibert 2007) -- the neighbourhood the estimate is
read off adapts to the sample in hand rather than being a fixed max_radius at
every n.

SECTIONS (select with --sections, default all)
  1  the defect: what fraction of the cloud each ball covers, per radius, on
     08 and 09, and the shell sequences that produces
  2  before/after compare() on 08 and 09, over the 2x2 of
     theiler_window in {0, "auto"} x max_ball_fraction in {1.0, 0.5},
     with the per-n dimension trace behind each verdict
  3  calibration: max_ball_fraction in {1.0, 0.75, 0.6, 0.5, 0.35} over ALL TEN
     round-2 production clouds, with the Theiler window off and on
  4  regression spot check: problems 02 and 10 must stay at
     poly_algebraic_min_n = 64, and the DEFAULT settings must reproduce the
     recorded numbers on every problem bit-for-bit

Section 3 is the expensive one (~1h). Sections 1, 2 and 4 run in a few minutes.
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

from socrates.hypergraph.comparison import compare, poly_algebraic_minimum_points  # noqa: E402
from socrates.hypergraph.dimension import (  # noqa: E402
    SATURATION_BALL_FRACTION,
    local_dimension,
    mean_dimension,
)
from socrates.hypergraph.pointcloud import DuplicatePointsError, knn_hypergraph  # noqa: E402

CHAOTIC = ("08", "09")
SPOT_CHECK = ("02", "10")

# The numbers docs/MENSURA_BENCHMARK.md records, at DEFAULT settings.
RECORDED_DEFAULT_MIN_N = {
    "01": 50, "02": 64, "03": 100, "04": 100, "05": 400,
    "06": 400, "07": 100, "08": 800, "09": 1600, "10": 64,
}

FRACTIONS = (1.0, 0.75, 0.6, 0.5, 0.35)


def _shells(estimate) -> tuple[int, ...]:
    volumes = (1,) + estimate.volumes
    return tuple(volumes[i] - volumes[i - 1] for i in range(1, len(volumes)))


def _sampled(hg, samples: int):
    nodes = sorted(hg.nodes)
    if samples < len(nodes):
        step = max(1, len(nodes) // samples)
        nodes = nodes[::step][:samples]
    return nodes


# --- section 1 ---------------------------------------------------------------


def section1() -> None:
    print("=" * 78)
    print("SECTION 1 -- the defect: balls that outrun the sample")
    print("=" * 78)
    print(
        "For each n, the MEDIAN over sampled nodes of ball(r) / N, the fraction of\n"
        "the graph the ball at radius r already covers. Any radius whose PREVIOUS\n"
        "ball is past max_ball_fraction is measuring the remaining budget, not the\n"
        "geometry. Theiler window on ('auto'), which is the corrected setting AR1\n"
        "left as the best state.\n"
    )
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(f"problem {problem} ({spec.label}): k={spec.k}, max_radius={spec.max_radius}")
        header = "  ".join(f"r={r}" for r in range(1, spec.max_radius + 1))
        print(f"  {'n':>6} | {header}")
        for n in spec.n_grid[:4]:
            if n > len(points) or spec.k >= n:
                break
            try:
                hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True, theiler_window="auto")
            except ValueError as exc:
                print(f"  {n:>6} | (no graph: {type(exc).__name__})")
                continue
            nodes = _sampled(hg, min(n, spec.samples))
            ests = [local_dimension(hg, u, max_radius=spec.max_radius) for u in nodes]
            n_nodes = len(hg.nodes)
            cols = []
            for r in range(1, spec.max_radius + 1):
                vals = sorted(e.volumes[r - 1] / n_nodes for e in ests)
                cols.append(f"{vals[len(vals) // 2]:.2f}")
            print(f"  {n:>6} | " + "  ".join(f"{c:>4}" for c in cols))
            for e in ests[:3]:
                print(f"         shells={_shells(e)} dim={e.dimension:.3f} r2={e.r_squared:.3f}")
        print()


# --- section 2 ---------------------------------------------------------------


def _trace(problem: str, theiler, fraction: float) -> list[tuple[int, float]]:
    spec = SPECS[problem]
    points = load_cloud(problem)
    out: list[tuple[int, float]] = []
    for n in spec.n_grid:
        if n > len(points) or spec.k >= n:
            break
        try:
            hg = knn_hypergraph(points[:n], k=spec.k, dedupe=True, theiler_window=theiler)
        except DuplicatePointsError:
            continue
        except ValueError:
            out.append((n, float("nan")))
            continue
        out.append(
            (
                n,
                mean_dimension(
                    hg,
                    samples=min(n, spec.samples),
                    max_radius=spec.max_radius,
                    max_ball_fraction=fraction,
                ),
            )
        )
    return out


def section2() -> None:
    print("=" * 78)
    print("SECTION 2 -- before/after on the two chaotic attractors")
    print("=" * 78)
    for problem in CHAOTIC:
        spec = SPECS[problem]
        points = load_cloud(problem)
        print(
            f"problem {problem} ({spec.label}): k={spec.k}, max_radius={spec.max_radius}, "
            f"true={spec.true_dimension}, tol={spec.tolerance}"
        )
        for theiler in (0, "auto"):
            for fraction in (1.0, SATURATION_BALL_FRACTION):
                result = compare(
                    points,
                    true_dimension=spec.true_dimension,
                    tolerance=spec.tolerance,
                    k=spec.k,
                    n_grid=spec.n_grid,
                    max_radius=spec.max_radius,
                    theiler_window=theiler,
                    max_ball_fraction=fraction,
                )
                trace = _trace(problem, theiler, fraction)
                cells = []
                for n, dim in trace:
                    if not math.isfinite(dim):
                        cells.append(f"{n}:nan")
                    else:
                        mark = "*" if abs(dim - spec.true_dimension) <= spec.tolerance else " "
                        cells.append(f"{n}:{dim:.3f}{mark}")
                print(
                    f"  theiler={str(theiler):>6} max_ball_fraction={fraction:<4} -> "
                    f"poly_min_n={str(result.poly_algebraic_min_n):>5}  "
                    f"trad_min_n={str(result.traditional_min_n):>5}  "
                    f"wins={result.poly_algebraic_wins}"
                )
                print("      trace: " + " ".join(cells))
        print()


# --- section 3 ---------------------------------------------------------------


def _min_n(problem: str, theiler, fraction: float) -> int | None:
    spec = SPECS[problem]
    return poly_algebraic_minimum_points(
        load_cloud(problem),
        spec.true_dimension,
        spec.tolerance,
        k=spec.k,
        n_grid=spec.n_grid,
        max_radius=spec.max_radius,
        samples=spec.samples,
        theiler_window=theiler,
        max_ball_fraction=fraction,
    )


def section3() -> None:
    print("=" * 78)
    print("SECTION 3 -- calibration of max_ball_fraction over all ten clouds")
    print("=" * 78)
    print(
        "poly_algebraic_min_n; '-' = never converges. 1.0 is the guard OFF, i.e. the\n"
        "row every other row must be read against. A SMALLER number is not always\n"
        "better: problems 02 and 10 dropping from 64 to 32 is the vacuous acceptance\n"
        "dimension.py's near-constant consensus rule was built to remove, so it counts\n"
        "as a regression, not a win.\n"
    )
    problems = list(SPECS)
    for theiler in (0, "auto"):
        print(f"  theiler_window={theiler!r}")
        print(f"  {'frac':>6} | " + " ".join(f"{p:>7}" for p in problems))
        for fraction in FRACTIONS:
            cells = []
            for problem in problems:
                value = _min_n(problem, theiler, fraction)
                cells.append(str(value) if value is not None else "-")
            print(f"  {fraction:>6} | " + " ".join(f"{c:>7}" for c in cells))
        print()


# --- section 4 ---------------------------------------------------------------


def section4() -> None:
    print("=" * 78)
    print("SECTION 4 -- regression spot check")
    print("=" * 78)
    print("(a) DEFAULT settings must reproduce the recorded numbers on every problem.")
    ok = True
    for problem, recorded in RECORDED_DEFAULT_MIN_N.items():
        got = _min_n(problem, 0, 1.0)
        flag = "OK" if got == recorded else "REGRESSED"
        ok &= got == recorded
        print(f"  problem {problem}: recorded {recorded:>5}  got {str(got):>5}   {flag}")
    print(f"  -> defaults unchanged: {ok}\n")

    print("(b) The compute-savings wins on 02 and 10 under the guard.")
    for problem in SPOT_CHECK:
        spec = SPECS[problem]
        points = load_cloud(problem)
        for fraction in (1.0, SATURATION_BALL_FRACTION):
            result = compare(
                points,
                true_dimension=spec.true_dimension,
                tolerance=spec.tolerance,
                k=spec.k,
                n_grid=spec.n_grid,
                max_radius=spec.max_radius,
                max_ball_fraction=fraction,
            )
            print(
                f"  problem {problem} max_ball_fraction={fraction:<4}: "
                f"poly={str(result.poly_algebraic_min_n):>5} "
                f"trad={str(result.traditional_min_n):>5} "
                f"wins={result.poly_algebraic_wins} "
                f"savings={result.compute_savings_fraction}"
            )
    print()


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4")
    args = ap.parse_args()
    print(f"SATURATION_BALL_FRACTION = {SATURATION_BALL_FRACTION}\n")
    for token in args.sections.split(","):
        SECTIONS[int(token.strip())]()


if __name__ == "__main__":
    main()
