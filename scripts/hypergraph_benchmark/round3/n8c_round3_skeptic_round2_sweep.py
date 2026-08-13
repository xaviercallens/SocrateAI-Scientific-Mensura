"""Round-3 independent skeptic, part 4: do any RECORDED round-2 headline
numbers move in the current tree?

docs/POLY_ALGEBRAIC_BENCHMARK.md records `poly_algebraic_min_n = 400` for both
problem 05 (quasiperiodic torus) and problem 06 (Brownian). Both run
`compare()` at max_radius=6, i.e. fit windows of length 6, where the
near-constant branch is LIVE (length 6 >= NEAR_CONSTANT_MIN_FIT_LENGTH = 5).
The round-2 repair re-derived only problem 01 and argued the rest were safe
because "this change can only remove acceptances" -- true relative to round
1's N8, but the recorded numbers were produced by the PRE-N8 tree, against
which N8 only ADDS acceptances.

Each problem's point cloud is rebuilt here from its own script's construction,
and `poly_algebraic_minimum_points` is run through the real production path
three ways: pre-N8 (pure R^2), N8 round-1 (no length gate), N8b (as shipped).

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_round2_sweep.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from n8c_round3_skeptic_prob06 import well_fit_mode  # noqa: E402

from socrates.hypergraph import comparison  # noqa: E402
from socrates.hypergraph.dimension import local_dimension, mean_dimension  # noqa: E402
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402

MODES = ("pre-N8", "N8-round1", "N8b-shipped", "N8d-shipped")


def cloud_p05() -> tuple[list[tuple[float, float]], int, tuple[int, ...], float, float]:
    """round2/05_quasiperiodic_torus.py's own build_point_cloud."""
    w = math.sqrt(2.0)
    n_grid = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
    n_max = n_grid[-1]
    t_max = n_max / (500 / 1800.0)
    pts = [(math.cos(t_max * i / n_max), math.cos(w * t_max * i / n_max)) for i in range(n_max)]
    return pts, 10, n_grid, 2.0, 0.3


def cloud_p06() -> tuple[list[tuple[float, float]], int, tuple[int, ...], float, float]:
    """round2/06_brownian_motion.py's own path + coarse-graining."""
    rng = np.random.default_rng(42)
    inc = rng.normal(loc=0.0, scale=1.0, size=(80_000, 2))
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(inc, axis=0)], axis=0)
    stride = max(1, len(path) // 25_600)
    pts = [tuple(map(float, p)) for p in path[::stride]]
    n_grid = (100, 200, 400, 800, 1600, 3200, 6400, 12_800, 25_600)
    return pts, 10, n_grid, 2.0, 0.3


def report(name: str, recorded: int | None, builder) -> bool:
    pts, k, n_grid, truth, tol = builder()
    print(f"\n=== {name} ===")
    print(
        f"  cloud {len(pts)} points, k={k}, max_radius=6 (compare() default), "
        f"true_dim={truth}, tolerance={tol}"
    )
    print(f"  RECORDED in docs/POLY_ALGEBRAIC_BENCHMARK.md: poly_algebraic_min_n = {recorded}")

    got = {}
    for mode in MODES:
        with well_fit_mode(mode):
            got[mode] = comparison.poly_algebraic_minimum_points(
                pts, truth, tol, k=k, n_grid=n_grid
            )
        print(f"    {mode:<12} -> {got[mode]}")

    print(f"  {'n':>7} " + " ".join(f"{m:>11}" for m in MODES) + "   admitted")
    for n in n_grid:
        if n > len(pts) or k >= n:
            break
        hg = knn_hypergraph(pts[:n], k=k, dedupe=True)
        vals = {}
        for mode in MODES:
            with well_fit_mode(mode):
                vals[mode] = mean_dimension(hg, samples=min(n, 40), max_radius=6)
        nodes = sorted(hg.nodes)
        stp = max(1, len(nodes) // 40)
        ests = [local_dimension(hg, u, max_radius=6) for u in nodes[::stp][:40]]
        adm = [e for e in ests if e.near_degenerate and e.r_squared < 0.9]
        marks = " ".join(
            f"{vals[m]:11.4f}" if abs(vals[m] - truth) <= tol else f"{vals[m]:10.4f}*"
            for m in MODES
        )
        print(f"  {n:7d} {marks}   {len(adm)}")
    print("  (* = outside tolerance)")

    # "shipped" is whatever the working tree actually does, i.e. the last mode
    # in MODES -- N8d, the graph-level consensus rule -- not the historical
    # node-local behaviour this script was written against.
    shipped = MODES[-1]
    ok = got[shipped] == recorded
    print(f"  --> shipped tree ({shipped}) {'MATCHES' if ok else 'DOES NOT MATCH'} "
          "the recorded number")
    if got["N8b-shipped"] != recorded:
        print(f"      (the round-2 node-local gate gave {got['N8b-shipped']}, "
              f"pre-N8 gave {got['pre-N8']})")
    if not ok and got["pre-N8"] == recorded:
        print("      and the pre-N8 tree DOES match it, so the near-constant gate is the cause.")
    return ok


def main() -> int:
    print("=" * 92)
    print("PART 4: RECORDED ROUND-2 HEADLINE NUMBERS, PRE-N8 vs SHIPPED")
    print("=" * 92)
    results = {
        "round-2 problem 05 (quasiperiodic torus)": report(
            "round-2 problem 05 (quasiperiodic torus)", 400, cloud_p05
        ),
        "round-2 problem 06 (Brownian motion)": report(
            "round-2 problem 06 (Brownian motion)", 400, cloud_p06
        ),
    }
    print("\n" + "=" * 92)
    for name, ok in results.items():
        print(f"  {name:<44} {'unchanged' if ok else 'CHANGED -- regression'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
