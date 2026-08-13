"""Round-3 independent skeptic, part 1: rebuild the documented anchor case
from scratch and confirm the min-fit-length gate (N8b) did not disturb it.

Deliberately shares NO code with the round-1/round-2 scripts: own BFS, own
least-squares fit via numpy, own re-implementation of the three gate
conditions. The repo's `local_dimension` is called only at the very end, to
compare against numbers derived without it.

Run: python scripts/hypergraph_benchmark/round3/n8c_round3_skeptic_anchor.py
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph.dimension import (  # noqa: E402
    NEAR_CONSTANT_CV,
    NEAR_CONSTANT_MIN_FIT_LENGTH,
    NEAR_CONSTANT_SLOPE_BOUND,
    local_dimension,
    mean_dimension,
)
from socrates.hypergraph.pointcloud import knn_hypergraph  # noqa: E402
from socrates.solvers import leapfrog  # noqa: E402

# ---------------------------------------------------------------------------
# Independent primitives
# ---------------------------------------------------------------------------


def my_bfs_volumes(adj: dict[int, set[int]], source: int, max_radius: int) -> list[int]:
    """Plain textbook BFS by levels; returns volumes[0..max_radius]."""
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        if dist[u] >= max_radius:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    counts = [0] * (max_radius + 1)
    for d in dist.values():
        counts[d] += 1
    volumes, running = [], 0
    for d in range(max_radius + 1):
        running += counts[d]
        volumes.append(running)
    return volumes


def my_fit(radii: list[int], shells: list[int]) -> dict:
    """numpy least squares on log(shell) vs log(radius) plus my own
    re-derivation of the three gate conditions."""
    x = np.log(np.asarray(radii, dtype=float))
    y = np.log(np.asarray(shells, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    s = np.asarray(shells, dtype=float)
    cv = float(s.std()) / float(s.mean())  # population std, matches repo
    band = float(y.max() - y.min())
    var_x = float(np.sum((x - x.mean()) ** 2))
    bound = band * float(np.sum(np.abs(x - x.mean()))) / (2.0 * var_x)
    near = (
        len(radii) >= NEAR_CONSTANT_MIN_FIT_LENGTH
        and cv <= NEAR_CONSTANT_CV
        and bound <= NEAR_CONSTANT_SLOPE_BOUND
    )
    return {
        "dimension": float(slope) + 1.0,
        "r_squared": r2,
        "cv": cv,
        "slope_bound": bound,
        "near_degenerate": bool(near),
    }


# ---------------------------------------------------------------------------
# Problem 01 point cloud, rebuilt from the solver
# ---------------------------------------------------------------------------


def build_problem01_points(n: int, dt: float = 2e-4) -> list[tuple[float, float]]:
    force = lambda q: -q  # noqa: E731
    n_steps = int(round(2 * np.pi / dt))
    run = leapfrog(force, [1.0], [0.0], dt=dt, n_steps=n_steps)

    t = run.times
    max_err_x = float(np.max(np.abs(run.positions[:, 0] - np.cos(t))))
    max_err_v = float(np.max(np.abs(run.velocities[:, 0] + np.sin(t))))
    print(f"  solver check: max|x-cos t|={max_err_x:.3e}  max|v+sin t|={max_err_v:.3e}")
    assert max_err_x < 1e-4 and max_err_v < 1e-4, "solver does not reproduce the exact solution"

    n_steps_total = len(run.times) - 1
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    fracs = (np.arange(n) * phi) % 1.0
    idx = np.clip(np.floor(fracs * n_steps_total).astype(int), 0, n_steps_total)
    assert len(np.unique(idx)) == n, "Weyl indices collided"
    return [(float(run.positions[i, 0]), float(run.velocities[i, 0])) for i in idx]


def main() -> int:
    print("=" * 78)
    print("PART 1: DOCUMENTED ANCHOR CASE (problem 01, n=200, k=6, max_radius=6)")
    print("=" * 78)
    print(
        f"  shipped constants: MIN_FIT_LENGTH={NEAR_CONSTANT_MIN_FIT_LENGTH} "
        f"CV={NEAR_CONSTANT_CV} SLOPE_BOUND={NEAR_CONSTANT_SLOPE_BOUND}"
    )

    points = build_problem01_points(200)
    hg = knn_hypergraph(points, k=6)
    adj = {u: set(vs) for u, vs in hg.adjacency().items()}
    nodes = sorted(hg.nodes)
    print(f"  point cloud: {len(points)} points -> hypergraph with {len(nodes)} nodes")

    target_shells = (6, 5, 5, 6, 6, 6)
    hits = []
    for node in nodes:
        vols = my_bfs_volumes(adj, node, 6)
        shells = tuple(vols[i] - vols[i - 1] for i in range(1, 7))
        if shells == target_shells:
            hits.append((node, vols))
    print(f"  nodes with shells {target_shells} found by my own BFS: {len(hits)}")
    assert hits, "documented anchor shell sequence not present"

    node, vols = hits[0]
    print(f"  node {node}: volumes(r=1..6) = {tuple(vols[1:])}  (doc: (7,12,17,23,29,35))")
    assert tuple(vols[1:]) == (7, 12, 17, 23, 29, 35)

    mine = my_fit([1, 2, 3, 4, 5, 6], list(target_shells))
    print(
        f"  MY numpy fit : dim={mine['dimension']:.10f} R^2={mine['r_squared']:.10f} "
        f"cv={mine['cv']:.6f} bound={mine['slope_bound']:.6f} near={mine['near_degenerate']}"
    )
    print("  doc says     : dim=1.0333        R^2=0.0550")

    est = local_dimension(hg, node, max_radius=6)
    print(
        f"  REPO         : dim={est.dimension:.10f} R^2={est.r_squared:.10f} "
        f"cv={est.shell_cv:.6f} bound={est.slope_bound:.6f} "
        f"near={est.near_degenerate} degenerate={est.degenerate}"
    )
    print(
        f"    is_well_fit(0.9)={est.is_well_fit(0.9)}  "
        f"is_genuinely_well_fit(0.9)={est.is_genuinely_well_fit(0.9)}"
    )

    assert abs(est.dimension - mine["dimension"]) < 1e-12
    assert abs(est.r_squared - mine["r_squared"]) < 1e-12
    assert abs(est.shell_cv - mine["cv"]) < 1e-12
    assert abs(est.slope_bound - mine["slope_bound"]) < 1e-12
    assert est.near_degenerate is True, "ANCHOR LOST: the length gate evicted it"
    assert est.is_well_fit(0.9) is True
    assert est.r_squared < 0.9, "anchor must be kept BY the gate, not by R^2"
    assert abs(est.dimension - 1.0333) < 1e-3
    assert abs(est.r_squared - 0.0550) < 1e-3

    # Whole-graph consequence.
    ests = [local_dimension(hg, x, max_radius=6) for x in nodes]
    r2_only = [e for e in ests if e.r_squared >= 0.9]
    kept = [e for e in ests if e.is_well_fit(0.9)]
    md = mean_dimension(hg, max_radius=6)
    worst = max(abs(e.dimension - 1.0) for e in kept)
    print(f"  R^2-only baseline keeps {len(r2_only)}/{len(ests)}")
    print(f"  with the gate  keeps    {len(kept)}/{len(ests)}")
    print(f"  mean_dimension(max_radius=6) = {md:.6f}   (truth 1.0)")
    print(f"  worst |dim-1| among kept     = {worst:.6f}  (problem tolerance +/-0.15)")
    assert len(r2_only) == 0
    assert len(kept) == 196, f"expected 196 kept, got {len(kept)}"
    assert not math.isnan(md) and abs(md - 1.0) < 0.15
    assert worst < 0.15

    print("\n  PART 1 RESULT: anchor reproduced from scratch and PRESERVED by N8b.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
