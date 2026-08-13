"""Round-3 (N8d) step 1: BROADEN the calibration set for a graph-level consensus rule.

The round-2 skeptic proposed gating the near-constant branch on
`near_degenerate_fraction` (a decision made once per GRAPH) instead of purely
per node, measuring 0.98 on problem 01's ring versus <= 0.01 on every damaged
graph -- but flagged that the must-fire side rested on exactly ONE example.

This script measures that fraction, UNDER THE CURRENTLY SHIPPED per-node gate,
on every round-2 production cloud at its production k / max_radius, at every n
in its production n_grid, plus the documented R2-F4 anchor graph. Problems 01,
02, 03, 04, 07 and 10 have true dimension 1 (the must-fire side); problems 05,
06, 08 and 09 have true dimension ~2 (the must-not-fire side).

It also reports, per graph, the statistic the eventual rule is built on: what
the graph's CONFIDENT nodes (r_squared >= 0.9, which includes the exactly
constant/degenerate ones) say the dimension is. That is the quantity a
near-constant node's "this neighbourhood is ~1D" claim can be checked against.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from n8d_clouds import SPECS, anchor_cloud, load_cloud, perfect_ring_cloud  # noqa: E402
from n8d_policy import R2_THRESHOLD, measure_graph, measure_grid  # noqa: E402

ONE_D = {"01", "02", "03", "04", "07", "10"}

HEADER = (
    f"  {'n':>7} {'sampled':>7} {'near_deg':>9} {'deg':>7} {'R2_pass':>8} "
    f"{'n_conf':>6} {'conf_mean':>10} {'|cm-1|':>7} {'nd_mean':>8} "
    f"{'mean_pre':>9} {'mean_n8b':>9}"
)


def row(m) -> str:
    conf = [e for e in m.estimates if e.r_squared >= R2_THRESHOLD]
    nd = [e for e in m.estimates if e.near_degenerate]
    cm = sum(e.dimension for e in conf) / len(conf) if conf else float("nan")
    ndm = sum(e.dimension for e in nd) / len(nd) if nd else float("nan")
    return (
        f"  {m.n:>7} {m.n_sampled:>7} {m.near_degenerate_fraction:>9.4f} "
        f"{m.degenerate_fraction:>7.4f} {m.r2_pass_fraction:>8.4f} {len(conf):>6} "
        f"{cm:>10.4f} {abs(cm - 1.0):>7.4f} {ndm:>8.4f} "
        f"{m.mean(policy='pre_n8'):>9.4f} {m.mean(policy='n8b'):>9.4f}"
    )


def main() -> None:
    print("=" * 118)
    print("N8d STEP 1: graph-level statistics on every round-2 production cloud")
    print("(measured with the CURRENTLY SHIPPED code -- min-fit-length gate ON,")
    print(" no consensus rule yet)")
    print("=" * 118)

    print("\n--- DOCUMENTED R2-F4 ANCHOR: problem 01's cloud, n=200, k=6, max_radius=6 ---")
    print(HEADER)
    anchor = measure_graph(anchor_cloud(), k=6, max_radius=6, samples=200)
    assert anchor is not None
    print(row(anchor))
    hits = [e for e in anchor.estimates if e.volumes == (7, 12, 17, 23, 29, 35)]
    print(f"  anchor nodes with volumes (7,12,17,23,29,35): {len(hits)}; "
          f"near_degenerate={[e.near_degenerate for e in hits]}, "
          f"dim={[round(e.dimension, 4) for e in hits]}, "
          f"R^2={[round(e.r_squared, 4) for e in hits]}")

    print("\n--- CONTROL: a PERFECTLY uniform 200-point circle "
          "(the exactly-constant N1 regime) ---")
    print(HEADER)
    ring = measure_graph(perfect_ring_cloud(200), k=6, max_radius=6, samples=200)
    assert ring is not None
    print(row(ring))

    for problem, spec in SPECS.items():
        points = load_cloud(problem)
        rows = measure_grid(points, spec)
        truth = (
            "1D -- branch MUST be able to fire" if problem in ONE_D else "NOT 1D -- must not fire"
        )
        print(
            f"\n--- problem {problem}: {spec.label} | k={spec.k} max_radius={spec.max_radius} "
            f"true_dim={spec.true_dimension} tol={spec.tolerance} | {truth} ---"
        )
        print(HEADER)
        for m in rows:
            print(row(m))


if __name__ == "__main__":
    main()
