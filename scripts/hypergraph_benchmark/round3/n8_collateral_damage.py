"""N8 collateral-damage probe: does the near-constant gate admit spurious
dimension~1 nodes into measurements whose true dimension is NOT 1?

This is the sharpest attack on N8's justification. `slope_bound` rigorously
bounds |dimension - 1|. That is an ERROR bound only if the true dimension is
1. If a genuinely 2D or 3D structure can produce a near-constant shell
sequence over the fit window, N8 would accept a dim~1 node that is badly
wrong, and `mean_dimension` would be dragged toward 1 exactly where it must
not be. The repair agent only checked the Lorenz attractor.

Every non-trivial point cloud below has a KNOWN true dimension != 1.

Run: python scripts/hypergraph_benchmark/round3/n8_collateral_damage.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from socrates.hypergraph import knn_hypergraph, local_dimension  # noqa: E402


def grid2d(n_side: int):
    return [(float(i), float(j)) for i in range(n_side) for j in range(n_side)]


def cube3d(n_side: int):
    return [
        (float(i), float(j), float(m))
        for i in range(n_side)
        for j in range(n_side)
        for m in range(n_side)
    ]


def disc2d(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * math.pi
    return list(zip((r * np.cos(th)).tolist(), (r * np.sin(th)).tolist(), strict=True))


def torus2d(n: int):
    """Lissajous 2-torus, benchmark problem 05: true dimension 2."""
    t = np.arange(n) * 0.01
    return list(
        zip(
            (np.cos(t) + 0.5 * np.cos(math.sqrt(2) * t)).tolist(),
            (np.sin(t) + 0.5 * np.sin(math.sqrt(2) * t)).tolist(),
            strict=True,
        )
    )


def brownian2d(n: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    return list(map(tuple, np.cumsum(rng.normal(size=(n, 2)), axis=0).tolist()))


def lorenz(n: int):
    dt = 0.01
    x = np.array([1.0, 1.0, 1.0])
    out = []
    for i in range(n * 5 + 5000):
        dx = np.array(
            [10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - (8.0 / 3.0) * x[2]]
        )
        x = x + dt * dx
        if i >= 5000 and (i - 5000) % 5 == 0:
            out.append(tuple(x.tolist()))
    return out[:n]


CASES = [
    ("2D grid 20x20", grid2d(20), 2.0),
    ("2D grid 30x30", grid2d(30), 2.0),
    ("2D random disc n=800", disc2d(800), 2.0),
    ("2D random disc n=2000", disc2d(2000), 2.0),
    ("2-torus (problem 05) n=1500", torus2d(1500), 2.0),
    ("2D Brownian (problem 06) n=1500", brownian2d(1500), 2.0),
    ("3D cube 10x10x10", cube3d(10), 3.0),
    ("3D cube 14x14x14", cube3d(14), 3.0),
    ("Lorenz attractor n=1500", lorenz(1500), 2.06),
]


def main() -> None:
    print("=" * 100)
    print("N8 COLLATERAL DAMAGE: near_degenerate nodes in structures whose TRUE dimension is not 1")
    print("=" * 100)
    print(
        f"{'case':<32} {'k':>2} {'R':>2} {'nodes':>6} {'near_deg':>9} "
        f"{'mean PRE':>9} {'mean POST':>9} {'delta':>8} {'true':>5}"
    )
    any_damage = False
    worst_delta = 0.0
    for label, pts, truth in CASES:
        for k in (6, 8, 12):
            for max_radius in (3, 4, 6):
                try:
                    hg = knn_hypergraph(pts, k=k, dedupe=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {label}: knn failed: {exc}")
                    continue
                nodes = sorted(hg.nodes)
                step = max(1, len(nodes) // 200)
                nodes = nodes[::step][:200]
                ests = [local_dimension(hg, u, max_radius=max_radius) for u in nodes]
                near = [e for e in ests if e.near_degenerate]
                pre = [e.dimension for e in ests if e.r_squared >= 0.9]
                post = [e.dimension for e in ests if e.is_well_fit(threshold=0.9)]
                mpre = sum(pre) / len(pre) if pre else float("nan")
                mpost = sum(post) / len(post) if post else float("nan")
                delta = abs(mpost - mpre) if (pre and post) else float("nan")
                flag = ""
                if near:
                    # Damage = a near_degenerate node was admitted into a
                    # measurement whose truth is far from 1.
                    any_damage = True
                    flag = f"  <-- {len(near)} admitted, truth={truth}"
                if math.isfinite(delta):
                    worst_delta = max(worst_delta, delta)
                print(
                    f"{label:<32} {k:>2} {max_radius:>2} {len(ests):>6} {len(near):>9} "
                    f"{mpre:>9.4f} {mpost:>9.4f} {delta:>8.4f} {truth:>5.2f}{flag}"
                )
    print()
    print(f"  any near_degenerate node admitted in a non-1D structure: {any_damage}")
    print(f"  worst |mean_POST - mean_PRE| over all configurations      : {worst_delta:.6f}")


if __name__ == "__main__":
    main()
